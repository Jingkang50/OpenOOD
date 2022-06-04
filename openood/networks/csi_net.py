import torch.nn as nn


class CSINet(nn.Module):
    def __init__(self,
                 backbone,
                 feature_size,
                 num_classes=10,
                 simclr_dim=128,
                 shift_trans_type='rotation'):
        super(CSINet, self).__init__()
        self.backbone = backbone
        self.linear = nn.Linear(feature_size, num_classes)
        self.simclr_layer = nn.Sequential(
            nn.Linear(feature_size, feature_size),
            nn.ReLU(),
            nn.Linear(feature_size, simclr_dim),
        )
        self.feature_size = feature_size
        self.joint_distribution_layer = nn.Linear(feature_size,
                                                  4 * num_classes)

        self.K_shift = get_shift_module(shift_trans_type)
        self.shift_cls_layer = nn.Linear(feature_size, self.K_shift)

    def forward(self,
                inputs,
                penultimate=False,
                simclr=False,
                shift=False,
                joint=False):
        _aux = {}
        _return_aux = False

        _, features = self.backbone(inputs, return_feature=True)

        output = self.linear(features)

        if penultimate:
            _return_aux = True
            _aux['penultimate'] = features

        if simclr:
            _return_aux = True
            _aux['simclr'] = self.simclr_layer(features)

        if shift:
            _return_aux = True
            _aux['shift'] = self.shift_cls_layer(features)

        if joint:
            _return_aux = True
            _aux['joint'] = self.joint_distribution_layer(features)

        if _return_aux:
            return output, _aux

        return output


def get_shift_module(shift_trans_type):

    if shift_trans_type == 'rotation':
        K_shift = 4
    elif shift_trans_type == 'cutperm':
        K_shift = 4
    else:
        K_shift = 1

    return K_shift
