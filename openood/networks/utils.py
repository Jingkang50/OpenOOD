import torch
import torch.backends.cudnn as cudnn

from .densenet import DenseNet3
from .lenet import LeNet
from .resnet18 import ResNet18
from .resnet18L import ResNet18L
from .wrn import WideResNet


def get_network(config):

    if config['name'] == 'res18':
        net = ResNet18(num_classes=config['num_classes'])

    elif config['name'] == 'res18L':
        net = ResNet18L(num_classes=config['num_classes'])

    elif config['name'] == 'lenet_rgb':
        net = LeNet(num_classes=config['num_classes'], num_channel=3)

    elif config['name'] == 'lenet_bw':
        net = LeNet(num_classes=config['num_classes'], num_channel=1)

    elif config['name'] == 'wrn':
        net = WideResNet(depth=28,
                         widen_factor=10,
                         dropRate=0.0,
                         num_classes=config['num_classes'])

    elif config['name'] == 'densenet':
        net = DenseNet3(
            depth=100,
            growth_rate=12,
            reduction=0.5,
            bottleneck=True,
            dropRate=0.0,
            num_classes=config['num_classes'],
        )

    else:
        raise Exception('Unexpected Network Architecture!')

    if config['checkpoint']:
        net.load_state_dict(torch.load(config['checkpoint']), strict=False)
        print('Model Loading Completed!')

    if config['ngpu'] > 1:
        net = torch.nn.DataParallel(net,
                                    device_ids=list(range(config['ngpu'])))

    if config['ngpu'] > 0:
        net.cuda()
        torch.cuda.manual_seed(1)

    cudnn.benchmark = True

    return net
