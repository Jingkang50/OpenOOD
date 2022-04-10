import torch

from openood.networks.resnet18 import Bottleneck, ResNet18


class ResNet50(ResNet18):
    def __init__(self,
                 block=Bottleneck,
                 num_blocks=[3, 4, 6, 3],
                 num_classes=10,
                 image_size=32,
                 pooling_size=7):
        super(ResNet50, self).__init__(block, num_blocks, num_classes,
                                       image_size, pooling_size)
        self.load_state_dict()

    def load_state_dict(
            self,
            state_dict=torch.load('./checkpoints/resnet50-19c8e357.pth'),
            strict=True):

        missing_keys = []
        unexpected_keys = []
        error_msgs = []

        # copy state_dict so _load_from_state_dict can modify it
        metadata = getattr(state_dict, '_metadata', None)
        state_dict = state_dict.copy()
        if metadata is not None:
            state_dict._metadata = metadata

        def load(module, prefix=''):
            local_metadata = {} if metadata is None else metadata.get(
                prefix[:-1], {})
            module._load_from_state_dict(state_dict, prefix, local_metadata,
                                         strict, missing_keys, unexpected_keys,
                                         error_msgs)
            for name, child in module._modules.items():
                if child is not None:
                    load(child, prefix + name + '.')

        load(self)

        if strict:
            if len(unexpected_keys) > 0:
                error_msgs.insert(
                    0, 'Unexpected key(s) in state_dict: {}. '.format(
                        ', '.join('"{}"'.format(k) for k in unexpected_keys)))
            if len(missing_keys) > 0:
                error_msgs.insert(
                    0, 'Missing key(s) in state_dict: {}. '.format(', '.join(
                        '"{}"'.format(k) for k in missing_keys)))

        if len(error_msgs) > 0:
            print('Warning(s) in loading state_dict for {}:\n\t{}'.format(
                self.__class__.__name__, '\n\t'.join(error_msgs)))
