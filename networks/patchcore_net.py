import logging
from re import X
import torch
import torch.nn as nn
from torch._C import import_ir_module
logger = logging.getLogger(__name__)

class patchcore_net(nn.Module):

    def __init__(self, backbone):
        super(patchcore_net, self).__init__()

        def hook_t(module, input, output):
            self.features.append(output)

        self.module = backbone
        self.module.layer2[-1].register_forward_hook(hook_t)
        self.module.layer3[-1].register_forward_hook(hook_t)
        
        for param in self.parameters():
            param.requires_grad = False
        self.module.to("cuda:0")
        self.criterion = torch.nn.MSELoss(reduction='sum')

    def init_features(self):
        self.features = []

    def forward(self, x_t, return_feature):
        x_t = x_t.cuda()
        self.init_features()
        _ = self.module(x_t)

        return self.features
            

        
        