from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from .base_postprocessor import BasePostprocessor
# from .gram_tools import detect
from .gram_tools import Detector


class GRAMPostprocessor(BasePostprocessor):
    def __init__(self, config):
        self.config = config
        self.postprocessor_args = config.postprocessor.postprocessor_args

    def setup(self, net: nn.Module, id_loader_dict, ood_loader_dict):
        data_train = id_loader_dict['train']
        data_test = id_loader_dict['test']
        self.train_preds = []
        self.train_confs = []
        self.train_logits = []
        for idx in range(0, len(data_train), 128):
            batch = torch.squeeze(torch.stack([x[0] for x in data_train[idx:idx+128]]), dim=1).cpu()
            
            logits = net(batch)
            confs = F.softmax(logits, dim=1).cpu().detach().numpy()
            preds = np.argmax(confs, axis=1)
            logits = (logits.cpu().detach().numpy())

            self.train_confs.extend(np.max(confs, axis=1))    
            self.train_preds.extend(preds)
            self.train_logits.extend(logits)
        print("Done")

        self.test_preds = []
        self.test_confs = []
        self.test_logits = []

        for idx in range(0, len(data_test), 128):
            batch = torch.squeeze(torch.stack([x[0] for x in data_test[idx:idx+128]]), dim=1).cpu()
            
            logits = net(batch)
            confs = F.softmax(logits, dim=1).cpu().detach().numpy()
            preds = np.argmax(confs, axis=1)
            logits = (logits.cpu().detach().numpy())

            self.test_confs.extend(np.max(confs, axis=1))   
            self.test_preds.extend(preds)
            self.test_logits.extend(logits)
        print("Done")
        
        self.detector = Detector(net, self.train_confs, self.train_logits,
                                 self.train_preds, self.test_confs,
                                 self.test_logits, self.test_preds, data_test)
        self.detector.compute_minmaxs(data_train, POWERS=range(1, 11))
        self.detector.compute_test_deviations(POWERS=range(1, 11))

    def postprocess(self, net: nn.Module, data: Any):
        data = list(data)
        
        results = self.detector.compute_ood_deviations(data, POWERS=range(1, 11))
        pred = results.max(1)[1]
        return pred, results
