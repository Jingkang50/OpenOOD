import argparse
import os
from functools import partial

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from fsood.data import get_dataloader
from fsood.networks import get_network
from fsood.utils import load_yaml


def get_torch_feature_stat(feature_list):
    feature_mean = torch.mean(feature_list.reshape(
        [feature_list.size(0), feature_list.size(1), -1]),
                              dim=-1)
    feature_var = torch.var(feature_list.reshape(
        [feature_list.size(0), feature_list.size(1), -1]),
                            dim=-1)
    if feature_list.size(-2) == 1 and feature_list.size(-1) == 1:
        feature_stat = torch.cat((feature_mean, feature_mean), 1)
    else:
        feature_stat = torch.cat((feature_mean, feature_var), 1)
    return feature_stat


def main(args, config):
    benchmark = config['benchmark']
    if benchmark == 'DIGITS':
        num_classes = 10
    elif benchmark == 'OBJECTS':
        num_classes = 10
    elif benchmark == 'COVID':
        num_classes = 2
    else:
        raise ValueError('Unknown Benchmark!')

    # Init Datasets ############################################################
    print('Initializing Datasets...')
    get_dataloader_default = partial(
        get_dataloader,
        root_dir=args.data_dir,
        benchmark=benchmark,
        num_classes=num_classes,
        stage='test',
        interpolation=config['interpolation'],
        image_size=config['image_size'],
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=args.prefetch,
    )

    test_loader_list = []
    for name in config['datasets']:
        test_ood_loader = get_dataloader_default(name=name)
        test_loader_list.append(test_ood_loader)

    # Init Network #############################################################
    print('Initializing Network...')
    net = get_network(
        config['network'],
        num_classes,
        checkpoint=args.checkpoint,
    )
    net.eval()

    if args.ngpu > 1:
        net = torch.nn.DataParallel(net, device_ids=list(range(args.ngpu)))

    if args.ngpu > 0:
        net.cuda()
        torch.cuda.manual_seed(1)

    cudnn.benchmark = True  # fire on all cylinders

    ########################################################################################################################
    # Inference Stage to get scores
    ########################################################################################################################
    to_np = lambda x: x.data.cpu().numpy()

    for name, test_loader in zip(config['datasets'], test_loader_list):
        highfeat_list, featstat_list, label_list = [], [], []
        with torch.no_grad():
            for batch in test_loader:
                data = batch['data'].cuda()
                label = batch['label']
                _, feat = net(data, return_feature=True)
                high_feat = feat[0]
                feat_stat = get_torch_feature_stat(feat[1])
                highfeat_list.extend(to_np(high_feat))
                featstat_list.extend(to_np(feat_stat))
                label_list.extend(to_np(label))
        highfeat_list = np.array(highfeat_list)
        featstat_list = np.array(featstat_list)
        label_list = np.array(label_list)
        save_dir = args.csv_path[:-4]
        os.makedirs(save_dir, exist_ok=True)
        np.savez(os.path.join(save_dir, name),
                 highfeat_list=highfeat_list,
                 featstat_list=featstat_list,
                 label_list=label_list)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config',
        help='path to config file',
        default='configs/test/digits_msp.yml',
    )
    parser.add_argument(
        '--checkpoint',
        help='path to model checkpoint',
        default='output/net.ckpt',
    )
    parser.add_argument(
        '--data_dir',
        help='directory to dataset',
        default='data',
    )
    parser.add_argument(
        '--csv_path',
        help='path to save evaluation results',
        default='results.csv',
    )
    parser.add_argument('--ngpu',
                        type=int,
                        default=1,
                        help='number of GPUs to use')
    parser.add_argument('--prefetch',
                        type=int,
                        default=4,
                        help='pre-fetching threads.')

    args = parser.parse_args()

    # Load config file
    config = load_yaml(args.config)

    main(args, config)
