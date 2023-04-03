from torchvision.models.swin_transformer import SwinTransformer


class Swin_T(SwinTransformer):
    def __init__(self,
                 patch_size=[4, 4],
                 embed_dim=96,
                 depths=[2, 2, 6, 2],
                 num_heads=[3, 6, 12, 24],
                 window_size=[7, 7],
                 stochastic_depth_prob=0.2,
                 num_classes=1000):
        super(Swin_T,
              self).__init__(patch_size=patch_size,
                             embed_dim=embed_dim,
                             depths=depths,
                             num_heads=num_heads,
                             window_size=window_size,
                             stochastic_depth_prob=stochastic_depth_prob,
                             num_classes=num_classes)
        self.feature_size = embed_dim * 2**(len(depths) - 1)

    def forward(self, x, return_feature=False):
        x = self.features(x)
        x = self.norm(x)
        x = self.permute(x)
        x = self.avgpool(x)
        x = self.flatten(x)

        if return_feature:
            return self.head(x), x
        else:
            return self.head(x)

    def forward_threshold(self, x, threshold):
        x = self.features(x)
        x = self.norm(x)
        x = self.permute(x)
        x = self.avgpool(x)
        x = self.flatten(x)
        feature = x.clip(max=threshold)
        feature = feature.view(feature.size(0), -1)
        logits_cls = self.head(feature)

        return logits_cls

    def get_fc(self):
        fc = self.head
        return fc.weight.cpu().detach().numpy(), fc.bias.cpu().detach().numpy()

    def get_fc_layer(self):
        return self.head
