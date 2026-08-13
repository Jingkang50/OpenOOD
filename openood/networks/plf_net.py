import torch


class PLFNet(torch.nn.Module):
    def __init__(self, backbone):
        super(PLFNet, self).__init__()
        self.backbone = backbone

    def forward(self, x, return_feature=False, return_feature_list=False):
        try:
            return self.backbone(x, return_feature, return_feature_list)
        except TypeError:
            return self.backbone(x, return_feature)

    def forward_threshold(self, x, y_start, y_end, x1, y1, m1, x2, m2):
        y_start = torch.tensor(y_start, device=x.device)
        y_end = torch.tensor(y_end, device=x.device)
        x1 = torch.tensor(x1, device=x.device)
        y1 = torch.tensor(y1, device=x.device)
        m1 = torch.tensor(m1, device=x.device)
        x2 = torch.tensor(x2, device=x.device)
        m2 = torch.tensor(m2, device=x.device)

        _, features = self.backbone(x, return_feature=True)
        x_abs = torch.abs(features)
        m0 = (y_end - y_start) / x1

        seg1 = y_start + m0 * x_abs
        seg_jump = torch.full_like(x_abs, y1)
        seg2 = y1 + m1 * (x_abs - x1)
        seg3 = (y1 + m1 * (x2 - x1)) + m2 * (x_abs - x2)

        y_pos = torch.where(
            x_abs < x1, seg1,
            torch.where(x_abs == x1, seg_jump,
                        torch.where(x_abs <= x2, seg2, seg3)))
        features = torch.where(features < 0, -y_pos, y_pos)
        features = features.view(features.size(0), -1)
        logits_cls = self.backbone.get_fc_layer()(features)
        return logits_cls

    def get_fc(self):
        fc = self.backbone.fc
        return fc.weight.cpu().detach().numpy(), fc.bias.cpu().detach().numpy()
