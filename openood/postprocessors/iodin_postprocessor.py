from typing import Any

import torch
import torch.nn as nn

from .odin_postprocessor import ODINPostprocessor


class IODINPostprocessor(ODINPostprocessor):
    def __init__(self, config):
        super().__init__(config)
        self.args = self.config.postprocessor.postprocessor_args

    def get_mask(self, grad):
        batch_size, channels, height, width = grad.shape
        abs_grad = abs(grad).view(batch_size, -1)
        max_indices = torch.argmax(abs_grad, dim=1, keepdim=True)
        # or use topk operation at the cost of an extra hyperparameter
        # max_indices = torch.topk(abs_grad, k=no_of_pixels, dim=1)[1]
        mask = torch.zeros_like(abs_grad, dtype=torch.uint8)
        mask.scatter_(1, max_indices, 1)
        mask = mask.view(batch_size, channels, height, width)
        return mask

    def postprocess(self, net: nn.Module, data: Any):
        data.requires_grad = True
        output = net(data)
        criterion = nn.CrossEntropyLoss()
        labels = output.detach().argmax(axis=1)
        output = output / self.temperature
        loss = criterion(output, labels)
        loss.backward()
        grad = data.grad.detach()

        mask = self.get_mask(grad)
        gradient = torch.ge(grad, 0)
        gradient = (gradient.float() - 0.5) * 2

        gradient[:, 0] = (gradient[:, 0]) / self.input_std[0]
        gradient[:, 1] = (gradient[:, 1]) / self.input_std[1]
        gradient[:, 2] = (gradient[:, 2]) / self.input_std[2]

        tempInputs = torch.add(data.detach(),
                               gradient * mask,
                               alpha=-self.noise)
        output = net(tempInputs)
        output = output / self.temperature

        nnOutput = output.detach()
        nnOutput = nnOutput - nnOutput.max(dim=1, keepdims=True).values
        nnOutput = nnOutput.exp() / nnOutput.exp().sum(dim=1, keepdims=True)

        conf, pred = nnOutput.max(dim=1)

        return pred, conf
