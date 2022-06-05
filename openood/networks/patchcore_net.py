import torch
import torch.nn as nn


class PatchcoreNet(nn.Module):
    def __init__(self, backbone):
        super(PatchcoreNet, self).__init__()

        # def hook_t(module, input, output):
        #     self.features.append(output)

        # path = '/home/pengyunwang/.cache/torch/hub/vision-0.9.0'
        # module = torch.hub._load_local(path,
        #                                'wide_resnet50_2',
        #                                pretrained=True)
        # self.module = module
        # self.module.layer2[-1].register_forward_hook(hook_t)
        # self.module.layer3[-1].register_forward_hook(hook_t)

        self.backbone = backbone

        for param in self.parameters():
            param.requires_grad = False
        # self.module.cuda()
        backbone.cuda()
        self.criterion = torch.nn.MSELoss(reduction='sum')

    def forward(self, x, return_feature):
        _, feature_list = self.backbone(x, return_feature_list=True)
        return [feature_list[-3], feature_list[-2]]

    # def init_features(self):
    #     self.features = []

    # def forward(self, x_t, return_feature):
    #     x_t = x_t.cuda()
    #     self.init_features()
    #     _ = self.module(x_t)

    #     import pdb
    #     pdb.set_trace()

    #     return self.features
