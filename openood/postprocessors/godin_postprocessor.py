from typing import Any

import torch
import torch.nn as nn

from .base_postprocessor import BasePostprocessor
from openood.preprocessors.transform import normalization_dict


class GodinPostprocessor(BasePostprocessor):
    def __init__(self, config):
        super(GodinPostprocessor, self).__init__(config)
        self.args = self.config.postprocessor.postprocessor_args

        self.score_func = self.args.score_func
        self.noise_magnitude = self.args.noise_magnitude
        try:
            self.input_std = normalization_dict[self.config.dataset.name][1]
        except KeyError:
            self.input_std = [0.5, 0.5, 0.5]

    def postprocess(self, net: nn.Module, data: Any):
        data.requires_grad = True
        output = net(data, inference=True)

        # Calculating the perturbation we need to add, that is,
        # the sign of gradient of cross entropy loss w.r.t. input
        max_scores, _ = torch.max(output, dim=1)
        max_scores.backward(torch.ones(len(max_scores)).cuda())

        # Normalizing the gradient to binary in {0, 1}
        gradient = torch.ge(data.grad.detach(), 0)
        gradient = (gradient.float() - 0.5) * 2

        # Scaling values taken from original code
        gradient[:, 0] = (gradient[:, 0]) / self.input_std[0]
        gradient[:, 1] = (gradient[:, 1]) / self.input_std[1]
        gradient[:, 2] = (gradient[:, 2]) / self.input_std[2]

        # Adding small perturbations to images
        tempInputs = torch.add(data.detach(),
                               gradient,
                               alpha=self.noise_magnitude)

        # calculate score
        output = net(tempInputs, inference=True, score_func=self.score_func)

        # Calculating the confidence after adding perturbations
        nnOutput = output.detach()
        nnOutput = nnOutput - nnOutput.max(dim=1, keepdims=True).values
        nnOutput = nnOutput.exp() / nnOutput.exp().sum(dim=1, keepdims=True)

        conf, pred = nnOutput.max(dim=1)

        return pred, conf
