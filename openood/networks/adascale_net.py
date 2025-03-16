import torch
import torch.nn as nn


class AdaScaleANet(nn.Module):
    def __init__(self, backbone):
        super(AdaScaleANet, self).__init__()
        self.backbone = backbone
        self.logit_scaling = False

    def forward(self, x, return_feature=False, return_feature_list=False):
        try:
            return self.backbone(x, return_feature, return_feature_list)
        except TypeError:
            return self.backbone(x, return_feature)

    def forward_threshold(self, feature, percentiles):
        scale = ada_scale(torch.relu(feature), percentiles)
        if self.logit_scaling:
            logits_cls = self.backbone.get_fc_layer()(feature)
            logits_cls *= (scale**2.0)
        else:
            feature *= torch.exp(scale)
            logits_cls = self.backbone.get_fc_layer()(feature)
        return logits_cls


class AdaScaleLNet(AdaScaleANet):
    def __init__(self, backbone):
        super(AdaScaleLNet, self).__init__(backbone)
        self.logit_scaling = True


def ada_scale(x, percentiles):
    assert x.dim() == 2
    b, c = x.shape
    assert percentiles.shape == (b, )
    assert torch.all(0 < percentiles) and torch.all(percentiles < 100)
    n = x.shape[1:].numel()
    ks = n - torch.round(n * percentiles.cuda() / 100.0).to(torch.int)
    max_k = ks.max()
    values, _ = torch.topk(x, max_k, dim=1)
    mask = torch.arange(max_k, device=x.device)[None, :] < ks[:, None]
    batch_sums = x.sum(dim=1, keepdim=True)
    masked_values = values * mask
    topk_sums = masked_values.sum(dim=1, keepdim=True)
    scales = batch_sums / topk_sums
    return scales
