from mmcls.models import CLASSIFIERS, ImageClassifier


@CLASSIFIERS.register_module()
class ImageClassifierWithReturnFeature(ImageClassifier):
    def forward(self, x, *args, **kwargs):
        if 'return_feature' in kwargs:
            return self.backbone(x)[0][-1]
        else:
            return super().forward(x, *args, **kwargs)
