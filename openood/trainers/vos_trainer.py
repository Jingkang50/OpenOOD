import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions.multivariate_normal import MultivariateNormal
from tqdm import tqdm

import openood.utils.comm as comm
from openood.utils import Config


def cosine_annealing(step, total_steps, lr_max, lr_min):
    return lr_min + (lr_max -
                     lr_min) * 0.5 * (1 + np.cos(step / total_steps * np.pi))


class VOSTrainer:
    def __init__(self, net, train_loader, config: Config):
        self.train_loader = train_loader
        self.config = config
        self.net = net
        weight_energy = torch.nn.Linear(config.num_classes, 1).cuda()
        torch.nn.init.uniform_(weight_energy.weight)
        self.logistic_regression = torch.nn.Linear(1, 2).cuda()
        self.optimizer = torch.optim.SGD(
            list(net.parameters()) + list(weight_energy.parameters()) +
            list(self.logistic_regression.parameters()),
            config.optimizer['lr'],
            momentum=config.optimizer['momentum'],
            weight_decay=config.optimizer['weight_decay'],
            nesterov=True)
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer,
            lr_lambda=lambda step: cosine_annealing(
                step, config.optimizer['num_epochs'] * len(train_loader), 1,
                1e-6 / config.optimizer['lr']))
        self.number_dict = {}
        for i in range(self.config['num_classes']):
            self.number_dict[i] = 0
        self.data_dict = torch.zeros(self.config['num_classes'],
                                     self.config['sample_number'],
                                     self.config['feature_dim']).cuda()

    def train_epoch(self, epoch_idx):
        self.net.train()
        loss_avg = 0.0
        sample_number = self.config['sample_number']
        num_classes = self.config['num_classes']
        train_dataiter = iter(self.train_loader)
        eye_matrix = torch.eye(self.config['feature_dim'], device='cuda')

        for train_step in tqdm(range(1,
                                     len(train_dataiter) + 1),
                               desc='Epoch {:03d}'.format(epoch_idx),
                               position=0,
                               leave=True,
                               disable=not comm.is_main_process()):
            batch = next(train_dataiter)
            images = batch['data'].cuda()
            labels = batch['label'].cuda()

            x, output = self.net.forward(images, return_feature=True)

            sum_temp = 0
            for index in range(num_classes):
                sum_temp += self.number_dict[index]
            lr_reg_loss = torch.zeros(1).cuda()[0]
            if (sum_temp == num_classes * sample_number
                    and epoch_idx < self.config['start_epoch']):
                target_numpy = labels.cpu().data.numpy()
                for index in range(len(labels)):
                    dict_key = target_numpy[index]
                    self.data_dict[dict_key] = torch.cat(
                        (self.data_dict[dict_key][1:],
                         output[index].detach().view(1, -1)), 0)
            elif (sum_temp == num_classes * sample_number
                  and epoch_idx >= self.config['start_epoch']):
                target_numpy = labels.cpu().data.numpy()
                for index in range(len(labels)):
                    dict_key = target_numpy[index]
                    self.data_dict[dict_key] = torch.cat(
                        (self.data_dict[dict_key][1:],
                         output[index].detach().view(1, -1)), 0)
                for index in range(num_classes):
                    if index == 0:
                        X = self.data_dict[index] - self.data_dict[index].mean(
                            0)
                        mean_embed_id = self.data_dict[index].mean(0).view(
                            1, -1)
                    else:
                        X = torch.cat((X, self.data_dict[index] -
                                       self.data_dict[index].mean(0)), 0)
                        mean_embed_id = torch.cat(
                            (mean_embed_id, self.data_dict[index].mean(0).view(
                                1, -1)), 0)

                temp_precision = torch.mm(X.t(), X) / len(X)
                temp_precision += 0.0001 * eye_matrix
                for index in range(num_classes):
                    new_dis = MultivariateNormal(
                        loc=mean_embed_id[index],
                        covariance_matrix=temp_precision)
                    negative_samples = new_dis.rsample(
                        (self.config['sample_from'], ))
                    prob_density = new_dis.log_prob(negative_samples)
                    cur_samples, index_prob = torch.topk(
                        -prob_density, self.config['select'])
                    if index == 0:
                        ood_samples = negative_samples[index_prob]
                    else:
                        ood_samples = torch.cat(
                            (ood_samples, negative_samples[index_prob]), 0)
                if len(ood_samples) != 0:

                    energy_score_for_fg = log_sum_exp(x,
                                                      num_classes=num_classes,
                                                      dim=1)
                    try:
                        predictions_ood = self.net.fc(ood_samples)
                    except AttributeError:
                        predictions_ood = self.net.module.fc(ood_samples)

                    energy_score_for_bg = log_sum_exp(predictions_ood,
                                                      num_classes=num_classes,
                                                      dim=1)

                    input_for_lr = torch.cat(
                        (energy_score_for_fg, energy_score_for_bg), -1)
                    labels_for_lr = torch.cat(
                        (torch.ones(len(output)).cuda(),
                         torch.zeros(len(ood_samples)).cuda()), -1)

                    output1 = self.logistic_regression(input_for_lr.view(
                        -1, 1))

                    lr_reg_loss = F.cross_entropy(output1,
                                                  labels_for_lr.long())
            else:
                target_numpy = labels.cpu().data.numpy()
                for index in range(len(labels)):
                    dict_key = target_numpy[index]

                    if self.number_dict[dict_key] < sample_number:
                        self.data_dict[dict_key][self.number_dict[
                            dict_key]] = output[index].detach()
                        self.number_dict[dict_key] += 1
            self.optimizer.zero_grad()
            loss = F.cross_entropy(x, labels)
            loss += self.config.trainer['loss_weight'] * lr_reg_loss
            loss.backward()

            self.optimizer.step()
            self.scheduler.step()

            loss_avg = loss_avg * 0.8 + float(loss) * 0.2

        metrics = {}
        metrics['loss'] = loss_avg
        metrics['epoch_idx'] = epoch_idx
        return self.net, metrics


def log_sum_exp(value, num_classes=10, dim=None, keepdim=False):
    """Numerically stable implementation of the operation."""
    value.exp().sum(dim, keepdim).log()

    # TODO: torch.max(value, dim=None) threw an error at time of writing
    weight_energy = torch.nn.Linear(num_classes, 1).cuda()
    if dim is not None:
        m, _ = torch.max(value, dim=dim, keepdim=True)
        value0 = value - m
        if keepdim is False:
            m = m.squeeze(dim)

        output = m + torch.log(
            torch.sum(F.relu(weight_energy.weight) * torch.exp(value0),
                      dim=dim,
                      keepdim=keepdim))
        # set lower bound
        out_list = output.cpu().detach().numpy().tolist()
        for i in range(len(out_list)):
            if out_list[i] < -1:
                out_list[i] = -1
            else:
                continue
        output = torch.Tensor(out_list).cuda()
        return output
    else:
        m = torch.max(value)
        sum_exp = torch.sum(torch.exp(value - m))
        # if isinstance(sum_exp, Number):
        #     return m + math.log(sum_exp)
        # else:
        return m + torch.log(sum_exp)
