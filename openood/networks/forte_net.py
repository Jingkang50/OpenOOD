from collections import OrderedDict

import torch
import torch.nn as nn
import torchvision.transforms as tvs_trans

# Normalization constants of the OpenAI CLIP image processor. Following the
# original Forte pipeline (Ganguly et al., ICLR 2025), a single CLIP-style
# preprocessing is shared by all backbones.
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)

# The three representation spaces of the paper.
DEFAULT_FORTE_BACKBONES = ('clip', 'vitmsn', 'dinov2')

# Named encoders. `pool` selects the image embedding: 'image_features'
# (projected embedding of CLIP-style models), 'cls' (class token) or 'mean'
# (average of the patch tokens, for models without a class token).
# `shared_clip_norm` keeps the CLIP normalization for that encoder, which
# is what the paper does for all three of its backbones; other encoders use
# the normalization published in their image processor config. Every
# encoder here is pretrained without ID class labels.
FORTE_BACKBONES = {
    # paper configuration
    'clip': dict(hub='openai/clip-vit-base-patch32',
                 pool='image_features',
                 shared_clip_norm=True),
    'vitmsn': dict(hub='facebook/vit-msn-base',
                   pool='cls',
                   shared_clip_norm=True),
    'dinov2': dict(hub='facebook/dinov2-base',
                   pool='cls',
                   shared_clip_norm=True),
    # alternative encoders (not part of the paper configuration)
    'dinov3': dict(hub='facebook/dinov3-vitb16-pretrain-lvd1689m',
                   pool='cls'),
    'siglip2': dict(hub='google/siglip2-base-patch16-224',
                    pool='image_features'),
    'dino': dict(hub='facebook/dino-vitb16', pool='cls'),
    'mae': dict(hub='facebook/vit-mae-base',
                pool='cls',
                model_kwargs=dict(mask_ratio=0.0)),
    'ijepa': dict(hub='facebook/ijepa_vith14_1k', pool='mean'),
}


class Convert:
    def __init__(self, mode='RGB'):
        self.mode = mode

    def __call__(self, image):
        return image.convert(self.mode)


class FloorCenterCrop:
    """Center crop with floor-rounded offsets, as in the Hugging Face CLIP
    image processor. torchvision's `CenterCrop` rounds the offset to nearest
    instead, which shifts the crop by one pixel whenever the excess is odd.
    """
    def __init__(self, size):
        self.size = size

    def __call__(self, image):
        width, height = image.size
        left = (width - self.size) // 2
        top = (height - self.size) // 2
        return image.crop((left, top, left + self.size, top + self.size))


def resolve_forte_backbone(spec):
    """Resolves a backbone spec into (name, settings).

    A spec is either a key of `FORTE_BACKBONES` or a Hugging Face hub id
    with an optional pooling suffix, e.g. `facebook/dinov2-large` or
    `facebook/ijepa_vith16_1k:mean`. Hub ids default to class-token
    pooling. Except for the paper's three backbones, which share the CLIP
    normalization, encoders use the normalization stored in their image
    processor config.
    """
    if spec in FORTE_BACKBONES:
        name, cfg = spec, dict(FORTE_BACKBONES[spec])
    elif '/' in spec:
        hub, _, pool = spec.partition(':')
        # the name ends up in feature-cache file names
        name = spec.replace('/', '--').replace(':', '-')
        cfg = dict(hub=hub, pool=pool or 'cls')
    else:
        raise ValueError(
            f'Unsupported Forte backbone: {spec}. Use one of '
            f'{sorted(FORTE_BACKBONES)} or a Hugging Face hub id.')
    pool = cfg['pool']
    if pool not in ('image_features', 'cls', 'mean'):
        raise ValueError(f'Unsupported pooling "{pool}" in backbone {spec}')
    if cfg.pop('shared_clip_norm', False):
        cfg['norm'] = None
    else:
        from transformers import AutoImageProcessor
        processor = AutoImageProcessor.from_pretrained(cfg['hub'])
        cfg['norm'] = (tuple(processor.image_mean),
                       tuple(processor.image_std))
    return name, cfg


class ForteNet(nn.Module):
    """Frozen feature extractors used by the Forte OOD detector
    (https://openreview.net/forum?id=7XNgVPxCiA).

    This module is not a classifier: `forward` returns the concatenation of
    the backbone embeddings. The Forte postprocessor accesses the individual
    representation spaces through `get_features`, which returns one embedding
    per backbone; PRDC summary statistics are computed per representation
    space and only then concatenated.

    The default backbones are the paper's (CLIP ViT-B/32, ViT-MSN-B,
    DINOv2-B), all consuming the same CLIP-preprocessed inputs as in the
    reference implementation. Other encoders can be selected by name (see
    `FORTE_BACKBONES`) or by Hugging Face hub id (see
    `resolve_forte_backbone`). Inputs are always the 224x224 CLIP-normalized
    images of `ForteNet.get_preprocessor`; encoders expecting a different
    normalization are re-normalized internally.
    """
    def __init__(self, backbones=DEFAULT_FORTE_BACKBONES):
        super().__init__()
        from transformers import AutoModel

        self.backbone_names = []
        self._pool = {}
        modules = {}
        for spec in backbones:
            name, cfg = resolve_forte_backbone(spec)
            if name in modules:
                raise ValueError(f'Duplicate Forte backbone: {spec}')
            modules[name] = AutoModel.from_pretrained(
                cfg['hub'], **cfg.get('model_kwargs', {}))
            self.backbone_names.append(name)
            self._pool[name] = cfg['pool']
            if cfg.get('norm') is not None:
                # x_clip * scale + shift == (x_raw - mean) / std
                mean, std = (torch.tensor(v).view(1, 3, 1, 1)
                             for v in cfg['norm'])
                clip_mean = torch.tensor(CLIP_MEAN).view(1, 3, 1, 1)
                clip_std = torch.tensor(CLIP_STD).view(1, 3, 1, 1)
                self.register_buffer(f'_scale_{len(modules)}',
                                     clip_std / std,
                                     persistent=False)
                self.register_buffer(f'_shift_{len(modules)}',
                                     (clip_mean - mean) / std,
                                     persistent=False)
        self.backbones = nn.ModuleDict(modules)
        for p in self.parameters():
            p.requires_grad_(False)

    @staticmethod
    def get_preprocessor():
        """Image transform reproducing the CLIP image processor exactly
        (shortest edge to 224 with bicubic resampling, floor-offset center
        crop, CLIP normalization)."""
        return tvs_trans.Compose([
            Convert('RGB'),
            tvs_trans.Resize(
                224, interpolation=tvs_trans.InterpolationMode.BICUBIC),
            FloorCenterCrop(224),
            tvs_trans.ToTensor(),
            tvs_trans.Normalize(CLIP_MEAN, CLIP_STD),
        ])

    @torch.no_grad()
    def get_features(self, x):
        """Returns an OrderedDict mapping backbone name -> (B, D) features."""
        feats = OrderedDict()
        for i, name in enumerate(self.backbone_names, start=1):
            model = self.backbones[name]
            scale = getattr(self, f'_scale_{i}', None)
            inputs = x if scale is None else \
                x * scale + getattr(self, f'_shift_{i}')
            if self._pool[name] == 'image_features':
                out = model.get_image_features(pixel_values=inputs)
                if not torch.is_tensor(out):
                    # transformers>=5 returns the vision model output
                    out = out.pooler_output
                    if hasattr(model, 'visual_projection'):
                        # the CLIP image embedding is the projected output
                        out = model.visual_projection(out)
                feats[name] = out
            else:
                hidden = model(pixel_values=inputs).last_hidden_state
                if self._pool[name] == 'mean':
                    feats[name] = hidden.mean(dim=1)
                else:
                    # contiguous() materializes the class-token slice;
                    # keeping the view would retain every full hidden-state
                    # tensor downstream
                    feats[name] = hidden[:, 0, :].contiguous()
        return feats

    def forward(self, x, return_feature=False):
        feats = torch.cat(list(self.get_features(x).values()), dim=1)
        if return_feature:
            return feats, feats
        return feats
