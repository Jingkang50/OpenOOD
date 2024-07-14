import numpy as np
import sys
import torch
import torch.nn as nn
from functools import partial

from torch.autograd import Function
from torch.functional import F

from torch.utils.data import DataLoader
from tqdm import tqdm
import openood.utils.comm as comm
from openood.utils import Config
from .lr_scheduler import cosine_annealing

import subprocess
import importlib.util


class ISHTrainer:
    def __init__(self, net: nn.Module, train_loader: DataLoader,
                 config: Config) -> None:
        self.net = net
        self.train_loader = train_loader
        self.config = config
        self.optimizer = torch.optim.SGD(
            [{
                'params': list(net.parameters())[:-2]
            }, {
                'params': list(net.parameters())[-2:],
                'weight_decay': config.optimizer.weight_decay_fc
            }],
            config.optimizer.lr,
            momentum=config.optimizer.momentum,
            weight_decay=config.optimizer.weight_decay,
            nesterov=config.optimizer.nesterov,
        )

        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer,
            lr_lambda=lambda step: cosine_annealing(
                step,
                config.optimizer.num_epochs * len(train_loader),
                1,
                1e-6 / config.optimizer.lr,
            ),
        )

        self.net = to_ish(self.net,
                          strategy=config.trainer.trainer_args.mode,
                          param=config.trainer.trainer_args.param,
                          layer=config.trainer.trainer_args.layer)

    def train_epoch(self, epoch_idx):
        self.net.train()

        loss_avg = 0.0

        train_dataiter = iter(self.train_loader)

        with tqdm(range(1,
                        len(train_dataiter) + 1),
                  desc='Epoch {:03d}'.format(epoch_idx),
                  position=0,
                  leave=True,
                  disable=not comm.is_main_process()) as tepoch:

            for train_step in tepoch:
                batch = next(train_dataiter)
                data = batch['data'].cuda()
                target = batch['label'].cuda()

                # forward
                logits_classifier, feature = self.net(data,
                                                      return_feature=True)
                loss = F.cross_entropy(logits_classifier, target)

                # backward
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                self.scheduler.step()

                # exponential moving average, show smooth values
                with torch.no_grad():
                    loss_avg = loss_avg * 0.8 + float(loss) * 0.2

        # comm.synchronize()

        metrics = {}
        metrics['epoch_idx'] = epoch_idx
        metrics['loss'] = self.save_metrics(loss_avg)

        return self.net, metrics

    def save_metrics(self, loss_avg):
        all_loss = comm.gather(loss_avg)
        total_losses_reduced = np.mean([x for x in all_loss])

        return total_losses_reduced


class _ISHTLinear(Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, weight: nn.Parameter, bias: nn.Parameter,
                ish_reshaper):
        ctx.ish_reshaper = ish_reshaper
        ctx.x_shape = x.shape
        ctx.has_bias = bias is not None
        ctx.save_for_backward(ish_reshaper.select(x, ctx), weight)
        return F.linear(x, weight, bias)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        x, weight = ctx.saved_tensors
        grad_bias = torch.sum(grad_output, list(
            range(grad_output.dim() - 1))) if ctx.has_bias else None
        ic, oc = weight.shape
        x = ctx.ish_reshaper.pad(x, ctx)
        grad_weight = grad_output.view(-1, ic).T.mm(x.view(-1, oc))
        grad_input = torch.matmul(grad_output, weight, out=x.view(ctx.x_shape))
        return grad_input, grad_weight, grad_bias, None


_linear_forward = _ISHTLinear.apply


def linear_forward(self, x):
    if self.training:
        x = _linear_forward(x, self.weight, self.bias, self.ish_reshaper)
    else:
        x = F.linear(x, self.weight, self.bias)
    return x


supports = {
    nn.Linear: linear_forward,
}


class ISHReshaper(object):
    def __init__(self, strategy, param):
        self.param = param
        self.reserve = 1 - param

        self.select = getattr(self, f'cache_{strategy}')
        self.pad = getattr(self, f'load_{strategy}')

    def cache_minksample_expscale(self, x: torch.Tensor, ctx=None):
        shape = x.shape
        x = x.reshape(shape[0], -1)
        # calculate the sum of the input per sample
        s1 = x.sum(dim=[1])

        x, idxs = x.abs().topk(int(x.shape[1] * self.reserve),
                               dim=1,
                               sorted=False)
        x.dropped = True  # provide a flag for act judges

        # calculate new sum of the input per sample after pruning
        s2 = x.sum(dim=[1])

        # apply sharpening
        scale = s1 / s2
        x = x * torch.exp(scale[:, None])

        ctx.idxs = idxs
        ctx.shape = shape
        return x

    def load_minksample_expscale(self, x, ctx=None):
        return torch.zeros(ctx.shape, device=x.device,
                           dtype=x.dtype).scatter_(1, ctx.idxs, x)

    def cache_expscale(self, x: torch.Tensor, ctx=None):
        input = x.clone()
        shape = x.shape
        x = x.reshape(shape[0], -1)
        # calculate the sum of the input per sample
        s1 = x.sum(dim=[1])

        x, idxs = x.abs().topk(int(x.shape[1] * self.reserve),
                               dim=1,
                               sorted=False)
        x.dropped = True  # provide a flag for act judges

        # calculate new sum of the input per sample after pruning
        s2 = x.sum(dim=[1])

        # apply sharpening
        scale = s1 / s2

        if len(shape) == 4:
            input = input * torch.exp(scale[:, None, None, None])
        elif len(shape) == 2:
            input = input * torch.exp(scale[:, None])
        else:
            raise NotImplementedError

        ctx.idxs = idxs
        ctx.shape = shape
        return input

    def load_expscale(self, x, ctx=None):
        return x

    def cache_minksample_lnscale(self, x: torch.Tensor, ctx=None):
        shape = x.shape
        x = x.reshape(shape[0], -1)
        # calculate the sum of the input per sample
        s1 = x.sum(dim=[1])

        x, idxs = x.abs().topk(int(x.shape[1] * self.reserve),
                               dim=1,
                               sorted=False)
        x.dropped = True  # provide a flag for act judges

        # calculate new sum of the input per sample after pruning
        s2 = x.sum(dim=[1])

        # apply sharpening
        scale = s1 / s2
        x = x * scale[:, None]

        ctx.idxs = idxs
        ctx.shape = shape
        return x

    def load_minksample_lnscale(self, x, ctx=None):
        return torch.zeros(ctx.shape, device=x.device,
                           dtype=x.dtype).scatter_(1, ctx.idxs, x)

    @staticmethod
    def transfer(model, strategy, gamma, autocast):
        _type = type(model)
        ish_reshaper = ISHReshaper(strategy, gamma)
        model.forward = partial(supports[_type], model)
        model.ish_reshaper = ish_reshaper
        print(f'{_type}.forward => ish.{strategy}.{_type}.forward')

        for child in model.children():
            ISHReshaper.transfer(child, strategy, gamma, autocast)
        return model


def to_ish(model: nn.Module,
           strategy: str,
           param: float,
           autocast: bool = False,
           layer=None):
    if layer == 'r1':
        if hasattr(model, 'module'):
            ISHReshaper.transfer(model.module.fc, strategy, param, autocast)
        else:
            ISHReshaper.transfer(model.fc, strategy, param, autocast)

    elif layer == 'all':
        ISHReshaper.transfer(model, strategy, param, autocast)

    return model
