"""Adapted from: https://github.com/facebookresearch/odin."""
from typing import Any

import torch
import torch.nn as nn

from .base_postprocessor import BasePostprocessor
from openood.preprocessors.transform import normalization_dict


class ODINPostprocessor(BasePostprocessor):
    def __init__(self, config):
        super().__init__(config)
        self.args = self.config.postprocessor.postprocessor_args

        self.temperature = self.args.temperature
        self.noise = self.args.noise
        try:
            self.input_std = normalization_dict[self.config.dataset.name][1]
        except KeyError:
            self.input_std = [0.5, 0.5, 0.5]
        self.args_dict = self.config.postprocessor.postprocessor_sweep

    def postprocess(self, net: nn.Module, data: Any):
        data.requires_grad = True
        output = net(data)

        # Calculating the perturbation we need to add, that is,
        # the sign of gradient of cross entropy loss w.r.t. input
        criterion = nn.CrossEntropyLoss()

        labels = output.detach().argmax(axis=1)

        # Using temperature scaling
        output = output / self.temperature

        loss = criterion(output, labels)
        loss.backward()

        # Normalizing the gradient to binary in {0, 1}
        gradient = torch.ge(data.grad.detach(), 0)
        gradient = (gradient.float() - 0.5) * 2

        # Scaling values taken from original code
        gradient[:, 0] = (gradient[:, 0]) / self.input_std[0]
        gradient[:, 1] = (gradient[:, 1]) / self.input_std[1]
        gradient[:, 2] = (gradient[:, 2]) / self.input_std[2]

        # Adding small perturbations to images
        tempInputs = torch.add(data.detach(), gradient, alpha=-self.noise)
        output = net(tempInputs)
        output = output / self.temperature

        # Calculating the confidence after adding perturbations
        nnOutput = output.detach()
        nnOutput = nnOutput - nnOutput.max(dim=1, keepdims=True).values
        nnOutput = nnOutput.exp() / nnOutput.exp().sum(dim=1, keepdims=True)

        conf, pred = nnOutput.max(dim=1)

        return pred, conf

    def set_hyperparam(self, hyperparam: list):
        self.temperature = hyperparam[0]
        self.noise = hyperparam[1]

    def get_hyperparam(self):
        return [self.temperature, self.noise]
