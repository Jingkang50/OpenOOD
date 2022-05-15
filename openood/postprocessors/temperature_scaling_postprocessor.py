from .base_postprocessor import BasePostprocessor
from torch import nn, optim
import torch
from typing import Any


class TemperatureScalingPostprocessor(BasePostprocessor):
    """ 
    A decorator which wraps a model with temperature scaling, internalize T as part of a net model.
    """
    def __init__(self, config):
        super(TemperatureScalingPostprocessor, self).__init__()
        self.config = config
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)  # initialize T


    def setup(self, net: nn.Module, id_loader_dict, ood_loader_dict):
        assert 'val' in id_loader_dict.keys(), "No validation dataset found!" # make sure that validation set exists
        val_dl = id_loader_dict['val']
        nll_criterion = nn.CrossEntropyLoss().cuda()

        logits_list = []   # fit in whole dataset at one time to back prop
        labels_list = []
        with torch.no_grad():   # fix other params of the net, only learn temperature
            for batch in val_dl:
                data = batch['data'].cuda()
                labels = batch['target']
                logits = net(data)
                logits_list.append(logits)
                labels_list.append(labels)
            logits = torch.cat(logits_list).cuda()   # convert a list of many tensors (each of a batch) to one tensor
            labels = torch.cat(labels_list).cuda()  

        optimizer = optim.LBFGS([self.temperature], lr=0.01, max_iter=50)  

        def eval():  # make sure only temperature parameter will be learned, fix other parameters of the network
            optimizer.zero_grad()
            loss = nll_criterion(self._temperature_scale(logits), labels)
            loss.backward()
            return loss
        optimizer.step(eval)


    def _temperature_scale(self, logits):
        temperature = self.temperature.unsqueeze(1).expand(logits.size()[0], logits.size()[1])
        return logits / temperature


    def postprocess(self, net: nn.Module, data: Any):
        logits = net(data)
        logits_ts = self._temperature_scale(logits)
        score = torch.softmax(logits_ts, dim=1)
        conf, pred = torch.max(score, dim=1)
        return pred, conf