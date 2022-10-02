import argparse
import os
import zipfile

import gdown

benchmarks_dict = {
    'bimcv': [
        'bimcv', 'ct', 'xraybone', 'actmed', 'mnist', 'cifar10', 'texture',
        'tin'
    ],
    'mnist': [
        'mnist', 'notmnist', 'fashionmnist', 'texture', 'cifar10', 'tin',
        'places365', 'cinic10'
    ],
    'cifar-10': [
        'cifar10', 'cifar100', 'tin', 'mnist', 'svhn', 'texture', 'places365',
        'cinic10'
    ],
    'cifar-100': [
        'cifar100', 'cifar10', 'tin', 'cifar100', 'svhn', 'texture',
        'places365', 'cifar100c'
    ],
    'imagenet-1k': [
        'imagenet_1k', 'species_sub', 'inaturalist', 'imagenet_o',
        'openimage_o', 'texture', 'mnist', 'imagenet_v2'
    ],
    'imagenet_1k': [
        'imagenet_1k', 'species_sub', 'inaturalist', 'imagenet_o',
        'openimage_o', 'texture', 'mnist', 'imagenet_v2'
    ],
    'misc':
    ['cifar10c', 'fractals_and_fvis', 'usps', 'imagenet10', 'hannover'],
}

dir_dict = {
    'images_classic/': [
        'cifar100', 'tin', 'svhn', 'cinic10', 'imagenet10', 'mnist',
        'fashionmnist', 'cifar10', 'cifar100c', 'places365', 'cifar10c',
        'fractals_and_fvis', 'usps', 'texture', 'notmnist'
    ],
    'images_largescale/': [
        'imagenet_1k', 'species_sub', 'inaturalist', 'imagenet_o',
        'openimage_o', 'imagenet_v2'
    ],
    'images_medical/': ['actmed', 'bimcv', 'ct', 'hannover', 'xraybone'],
}

download_id_dict = {
    'osr': '1L9MpK9QZq-o-JrFHrfo5lM4-FsFPk0e9',
    'mnist_lenet': '13mEvYF9rVIuch8u0RVDLf_JMOk3PAYCj',
    'cifar10_res18': '1rPEScK7TFjBn_W_frO-8RSPwIG6_x0fJ',
    'cifar100_res18': '1OOf88A48yXFw4fSU02XQT-3OQKf31Csy',
    'imagenet_res50': '1tgY_PsfkazLDyI1pniDMDEehntBhFyF3',
    'benchmark_imglist': '1t6-8u7LVsoTKvU8zhK31oOl00udim7wC',
    'usps': '1KhbWhlFlpFjEIb4wpvW0s9jmXXsHonVl',
    'cifar100': '1PGKheHUsf29leJPPGuXqzLBMwl8qMF8_',
    'cifar10': '1Co32RiiWe16lTaiOU6JMMnyUYS41IlO1',
    'cifar10c': '170DU_ficWWmbh6O2wqELxK9jxRiGhlJH',
    'cinic10': '190gdcfbvSGbrRK6ZVlJgg5BqqED6H_nn',
    'svhn': '1DQfc11HOtB1nEwqS4pWUFp8vtQ3DczvI',
    'fashionmnist': '1nVObxjUBmVpZ6M0PPlcspsMMYHidUMfa',
    'cifar100c': '1MnETiQh9RTxJin2EHeSoIAJA28FRonHx',
    'mnist': '1CCHAGWqA1KJTFFswuF9cbhmB-j98Y1Sb',
    'fractals_and_fvis': '1EZP8RGOP-XbMsKex3r-BGI5F1WAP_PJ3',
    'tin': '1PZ-ixyx52U989IKsMA2OT-24fToTrelC',
    'texture': '1OSz1m3hHfVWbRdmMwKbUzoU8Hg9UKcam',
    'imagenet10': '1qRKp-HCLkmfiWwR-PXthN7-2dxIQVKxP',
    'notmnist': '16ueghlyzunbksnc_ccPgEAloRW9pKO-K',
    'places365': '1Ec-LRSTf6u5vEctKX9vRp9OA6tqnJ0Ay',
    'imagenet_v2': '1WYWlRgfbl4waD8KZolfxPzOoemfsWyIJ',
    'species_sub': '1-JCxDx__iFMExkYRMylnGJYTPvyuX6aq',
    'imagenet_o': '1S9cFV7fGvJCcka220-pIO9JPZL1p1V8w',
    'openimage_o': '1VUFXnB_z70uHfdgJG2E_pjYOcEgqM7tE',
    'inaturalist': '1zfLfMvoUD0CUlKNnkk7LgxZZBnTBipdj',
    'imagenet_1k': '11hbDGjQrR25FiqXm9ZNO5tJnvPb8QmkQ',
    'actmed': '1tibxL_wt6b3BjliPaQ2qjH54Wo4ZXWYb',
    'ct': '1k5OYN4inaGgivJBJ5L8pHlopQSVnhQ36',
    'hannover': '1NmqBDlcA1dZQKOvgcILG0U1Tm6RP0s2N',
    'xraybone': '1ZzO3y1-V_IeksJXEvEfBYKRoQLLvPYe9',
    'bimcv': '1nAA45V6e0s5FAq2BJsj9QH5omoihb7MZ',
}


def require_download(filename, path):

    for item in os.listdir(path):
        if item.startswith(filename) or filename.startswith(
                item) or path.endswith(filename):
            return False

    else:
        print(filename + ' needs download:')
        return True


def download_dataset(dataset, args):
    for key in dir_dict.keys():
        if dataset in dir_dict[key]:
            store_path = os.path.join(args.save_dir[0], key, dataset)
            if not os.path.exists(store_path):
                os.makedirs(store_path)
            break
    else:
        print('Invalid dataset detected {}'.format(dataset))
        return

    if require_download(dataset, store_path):
        print(store_path)
        if not store_path.endswith('/'):
            store_path = store_path + '/'
        gdown.download(id=download_id_dict[dataset], output=store_path)

        file_path = os.path.join(store_path, dataset + '.zip')
        with zipfile.ZipFile(file_path, 'r') as zip_file:
            zip_file.extractall(store_path)
        os.remove(file_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Download datasets and checkpoints')
    parser.add_argument('--contents',
                        nargs='+',
                        default=['datasets', 'checkpoints'])
    parser.add_argument('--datasets', nargs='+', default=['default'])
    parser.add_argument('--checkpoints', nargs='+', default=['all'])
    parser.add_argument('--save_dir',
                        nargs='+',
                        default=['./data', './results'])
    parser.add_argument('--dataset_mode', default='default')
    args = parser.parse_args()

    if args.datasets[0] == 'default':
        args.datasets = ['mnist', 'cifar-10', 'cifar-100']
    elif args.datasets[0] == 'all':
        args.datasets = list(benchmarks_dict.keys())

    if args.checkpoints[0] == 'ood':
        args.checkpoints = [
            'mnist_lenet', 'cifar10_res18', 'cifar100_res18', 'imagenet_res50'
        ]
    elif args.checkpoints[0] == 'all':
        args.checkpoints = [
            'mnist_lenet', 'cifar10_res18', 'cifar100_res18', 'imagenet_res50',
            'osr'
        ]

    for content in args.contents:
        if content == 'datasets':

            store_path = args.save_dir[0]
            if not store_path.endswith('/'):
                store_path = store_path + '/'
            if not os.path.exists(os.path.join(store_path,
                                               'benchmark_imglist')):
                gdown.download(id=download_id_dict['benchmark_imglist'],
                               output=store_path)
                file_path = os.path.join(args.save_dir[0],
                                         'benchmark_imglist.zip')
                with zipfile.ZipFile(file_path, 'r') as zip_file:
                    zip_file.extractall(store_path)
                os.remove(file_path)
            dataset_mode = args.dataset_mode
            if dataset_mode == 'default' or dataset_mode == 'benchmark':
                for benchmark in args.datasets:
                    for dataset in benchmarks_dict[benchmark]:
                        download_dataset(dataset, args)

            if dataset_mode == 'dataset':
                for dataset in args.datasets:
                    download_dataset(dataset, args)

        elif content == 'checkpoints':

            store_path = os.path.join(args.save_dir[1], 'checkpoints/')
            if not os.path.exists(store_path):
                os.makedirs(store_path)

            for checkpoint in args.checkpoints:
                if require_download(checkpoint, store_path):
                    gdown.download(id=download_id_dict[checkpoint],
                                   output=store_path)
                    if checkpoint == 'osr':
                        file_path = os.path.join(store_path,
                                                 checkpoint + '.zip')
                        with zipfile.ZipFile(file_path, 'r') as zip_file:
                            zip_file.extractall(store_path)
                        os.remove(file_path)
