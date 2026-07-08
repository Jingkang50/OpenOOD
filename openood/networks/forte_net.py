from collections import OrderedDict

import torch
import torch.nn as nn
import torchvision.transforms as tvs_trans

# Normalization constants of the OpenAI CLIP image processor. Following the
# original Forte pipeline (Ganguly et al., ICLR 2025), a single CLIP-style
# preprocessing is shared by all backbones.
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)

DEFAULT_FORTE_BACKBONES = ('clip', 'vitmsn', 'dinov2')


class Convert:
    def __init__(self, mode='RGB'):
        self.mode = mode

    def __call__(self, image):
        return image.convert(self.mode)


class ForteNet(nn.Module):
    """Frozen self-supervised feature extractors used by the Forte OOD
    detector (https://openreview.net/forum?id=7XNgVPxCiA).

    This module is not a classifier: `forward` returns the concatenation of
    the backbone embeddings. The Forte postprocessor accesses the individual
    representation spaces through `get_features`, which returns one embedding
    per backbone; PRDC summary statistics are computed per representation
    space and only then concatenated.

    All backbones consume the same CLIP-preprocessed inputs (see
    `ForteNet.get_preprocessor`), matching the reference implementation.
    """
    def __init__(self, backbones=DEFAULT_FORTE_BACKBONES):
        super().__init__()
        from transformers import AutoModel, CLIPModel, ViTMSNModel

        self.backbone_names = list(backbones)
        modules = {}
        for name in self.backbone_names:
            if name == 'clip':
                modules[name] = CLIPModel.from_pretrained(
                    'openai/clip-vit-base-patch32')
            elif name == 'vitmsn':
                modules[name] = ViTMSNModel.from_pretrained(
                    'facebook/vit-msn-base')
            elif name == 'dinov2':
                modules[name] = AutoModel.from_pretrained(
                    'facebook/dinov2-base')
            else:
                raise ValueError(f'Unsupported Forte backbone: {name}')
        self.backbones = nn.ModuleDict(modules)
        for p in self.parameters():
            p.requires_grad_(False)

    @staticmethod
    def get_preprocessor():
        """Image transform equivalent to the CLIP image processor."""
        return tvs_trans.Compose([
            Convert('RGB'),
            tvs_trans.Resize(
                224, interpolation=tvs_trans.InterpolationMode.BICUBIC),
            tvs_trans.CenterCrop(224),
            tvs_trans.ToTensor(),
            tvs_trans.Normalize(CLIP_MEAN, CLIP_STD),
        ])

    @torch.no_grad()
    def get_features(self, x):
        """Returns an OrderedDict mapping backbone name -> (B, D) features."""
        feats = OrderedDict()
        for name in self.backbone_names:
            model = self.backbones[name]
            if name == 'clip':
                out = model.get_image_features(pixel_values=x)
                if not torch.is_tensor(out):
                    # transformers>=5 returns the vision model output; the
                    # CLIP image embedding is the projected pooled output
                    out = model.visual_projection(out.pooler_output)
                feats[name] = out
            else:
                # contiguous() materializes the CLS slice; keeping the view
                # would retain every full hidden-state tensor downstream
                cls = model(pixel_values=x).last_hidden_state[:, 0, :]
                feats[name] = cls.contiguous()
        return feats

    def forward(self, x, return_feature=False):
        feats = torch.cat(list(self.get_features(x).values()), dim=1)
        if return_feature:
            return feats, feats
        return feats
