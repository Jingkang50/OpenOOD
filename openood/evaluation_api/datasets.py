import os

from torch.utils.data import DataLoader
import torchvision as tvs
if tvs.__version__ >= '0.13':
    tvs_new = True
else:
    tvs_new = False

from openood.datasets.imglist_dataset import ImglistDataset
from openood.preprocessors.test_preprocessor import TestStandardPreProcessor
from openood.preprocessors.utils import get_preprocessor
from openood.utils.config import Config

from .preprocessor import get_default_preprocessor, ImageNetCPreProcessor

DATA_INFO = {
    'cifar10': {
        'num_classes': 10,
        'id': {
            'train': {
                'data_dir': 'images_classic/',
                'imglist_path': 'benchmark_imglist/cifar10/train_cifar10.txt'
            },
            'val': {
                'data_dir': 'images_classic/',
                'imglist_path': 'benchmark_imglist/cifar10/val_cifar10.txt'
            },
            'test': {
                'data_dir': 'images_classic/',
                'imglist_path': 'benchmark_imglist/cifar10/test_cifar10.txt'
            }
        },
        'csid': {
            'datasets': ['cifar10c'],
            'cinic10': {
                'data_dir': 'images_classic/',
                'imglist_path': 'benchmark_imglist/cifar10/val_cinic10.txt'
            },
            'cifar10c': {
                'data_dir': 'images_classic/',
                'imglist_path': 'benchmark_imglist/cifar10/test_cifar10c.txt'
            }
        },
        'ood': {
            'val': {
                'data_dir': 'images_classic/',
                'imglist_path': 'benchmark_imglist/cifar10/val_cifar100.txt'
            },
            'near': {
                'datasets': ['cifar100', 'tin'],
                'cifar100': {
                    'data_dir': 'images_classic/',
                    'imglist_path':
                    'benchmark_imglist/cifar10/test_cifar100.txt'
                },
                'tin': {
                    'data_dir': 'images_classic/',
                    'imglist_path': 'benchmark_imglist/cifar10/test_tin.txt'
                }
            },
            'far': {
                'datasets': ['mnist', 'svhn', 'texture', 'places365'],
                'mnist': {
                    'data_dir': 'images_classic/',
                    'imglist_path': 'benchmark_imglist/cifar10/test_mnist.txt'
                },
                'svhn': {
                    'data_dir': 'images_classic/',
                    'imglist_path': 'benchmark_imglist/cifar10/test_svhn.txt'
                },
                'texture': {
                    'data_dir': 'images_classic/',
                    'imglist_path':
                    'benchmark_imglist/cifar10/test_texture.txt'
                },
                'places365': {
                    'data_dir': 'images_classic/',
                    'imglist_path':
                    'benchmark_imglist/cifar10/test_places365.txt'
                },
            }
        }
    },
    'cifar100': {
        'num_classes': 100,
        'id': {
            'train': {
                'data_dir': 'images_classic/',
                'imglist_path': 'benchmark_imglist/cifar100/train_cifar100.txt'
            },
            'val': {
                'data_dir': 'images_classic/',
                'imglist_path': 'benchmark_imglist/cifar100/val_cifar100.txt'
            },
            'test': {
                'data_dir': 'images_classic/',
                'imglist_path': 'benchmark_imglist/cifar100/test_cifar100.txt'
            }
        },
        'csid': {
            'datasets': [],
        },
        'ood': {
            'val': {
                'data_dir': 'images_classic/',
                'imglist_path': 'benchmark_imglist/cifar100/val_cifar10.txt'
            },
            'near': {
                'datasets': ['cifar10', 'tin'],
                'cifar10': {
                    'data_dir': 'images_classic/',
                    'imglist_path':
                    'benchmark_imglist/cifar100/test_cifar10.txt'
                },
                'tin': {
                    'data_dir': 'images_classic/',
                    'imglist_path': 'benchmark_imglist/cifar100/test_tin.txt'
                }
            },
            'far': {
                'datasets': ['mnist', 'svhn', 'texture', 'places365'],
                'mnist': {
                    'data_dir': 'images_classic/',
                    'imglist_path': 'benchmark_imglist/cifar100/test_mnist.txt'
                },
                'svhn': {
                    'data_dir': 'images_classic/',
                    'imglist_path': 'benchmark_imglist/cifar100/test_svhn.txt'
                },
                'texture': {
                    'data_dir': 'images_classic/',
                    'imglist_path':
                    'benchmark_imglist/cifar100/test_texture.txt'
                },
                'places365': {
                    'data_dir': 'images_classic/',
                    'imglist_path':
                    'benchmark_imglist/cifar100/test_places365.txt'
                }
            },
        }
    },
    'imagenet': {
        'num_classes': 1000,
        'id': {
            'train': {
                'data_dir': 'images_largescale/',
                'imglist_path': 'benchmark_imglist/imagenet/train_imagenet.txt'
            },
            'val': {
                'data_dir': 'images_largescale/',
                'imglist_path': 'benchmark_imglist/imagenet/val_imagenet.txt'
            },
            'test': {
                'data_dir': 'images_largescale/',
                'imglist_path': 'benchmark_imglist/imagenet/test_imagenet.txt'
            }
        },
        'csid': {
            'datasets': ['imagenetv2'],
            'imagenetv2': {
                'data_dir': 'images_largescale/',
                'imglist_path':
                'benchmark_imglist/imagenet/test_imagenetv2.txt'
            }
        },
        'ood': {
            'val': {
                'data_dir': 'images_largescale/',
                'imglist_path':
                'benchmark_imglist/imagenet/val_openimage_o.txt'
            },
            'near': {
                'datasets': ['species', 'imagenet_21k'],
                'species': {
                    'data_dir': 'images_largescale/',
                    'imglist_path':
                    'benchmark_imglist/imagenet/test_species.txt'
                },
                'imagenet_21k': {
                    'data_dir':
                    'images_largescale/',
                    'imglist_path':
                    'benchmark_imglist/imagenet/test_imagenet_21k.txt'
                },
            },
            'far': {
                'datasets':
                ['inaturalist', 'texture', 'places', 'sun', 'openimage_o'],
                'inaturalist': {
                    'data_dir':
                    'images_largescale/',
                    'imglist_path':
                    'benchmark_imglist/imagenet/test_inaturalist.txt'
                },
                'texture': {
                    'data_dir': 'images_classic/',
                    'imglist_path':
                    'benchmark_imglist/imagenet/test_texture.txt'
                },
                'places': {
                    'data_dir': 'images_largescale/',
                    'imglist_path':
                    'benchmark_imglist/imagenet/test_places.txt'
                },
                'sun': {
                    'data_dir': 'images_largescale/',
                    'imglist_path': 'benchmark_imglist/imagenet/test_sun.txt'
                },
                'openimage_o': {
                    'data_dir':
                    'images_largescale/',
                    'imglist_path':
                    'benchmark_imglist/imagenet/test_openimage_o.txt'
                },
            },
        }
    },
}


def get_id_ood_dataloader(id_name, data_root, preprocessor, **loader_kwargs):
    if id_name == 'imagenet':
        if tvs_new:
            if isinstance(preprocessor,
                          tvs.transforms._presets.ImageClassification):
                mean, std = preprocessor.mean, preprocessor.std
            elif isinstance(preprocessor, tvs.transforms.Compose):
                temp = preprocessor.transforms[-1]
                mean, std = temp.mean, temp.std
            else:
                raise TypeError
        else:
            if isinstance(preprocessor, tvs.transforms.Compose):
                temp = preprocessor.transforms[-1]
                mean, std = temp.mean, temp.std
            else:
                raise TypeError
        imagenet_c_preprocessor = ImageNetCPreProcessor(mean, std)

    # weak augmentation for data_aux
    test_standard_preprocessor = get_default_preprocessor(id_name)

    dataloader_dict = {}
    data_info = DATA_INFO[id_name]

    # id
    sub_dataloader_dict = {}
    for split in data_info['id'].keys():
        dataset = ImglistDataset(
            name='_'.join((id_name, split)),
            imglist_pth=os.path.join(data_root,
                                     data_info['id'][split]['imglist_path']),
            data_dir=os.path.join(data_root,
                                  data_info['id'][split]['data_dir']),
            num_classes=data_info['num_classes'],
            preprocessor=preprocessor,
            data_aux_preprocessor=test_standard_preprocessor)
        dataloader = DataLoader(dataset, **loader_kwargs)
        sub_dataloader_dict[split] = dataloader
    dataloader_dict['id'] = sub_dataloader_dict

    # csid
    sub_dataloader_dict = {}
    for dataset_name in data_info['csid']['datasets']:
        dataset = ImglistDataset(
            name='_'.join((id_name, 'csid', dataset_name)),
            imglist_pth=os.path.join(
                data_root, data_info['csid'][dataset_name]['imglist_path']),
            data_dir=os.path.join(data_root,
                                  data_info['csid'][dataset_name]['data_dir']),
            num_classes=data_info['num_classes'],
            preprocessor=preprocessor
            if dataset_name != 'imagenet_c' else imagenet_c_preprocessor,
            data_aux_preprocessor=test_standard_preprocessor)
        dataloader = DataLoader(dataset, **loader_kwargs)
        sub_dataloader_dict[dataset_name] = dataloader
    dataloader_dict['csid'] = sub_dataloader_dict

    # ood
    dataloader_dict['ood'] = {}
    for split in data_info['ood'].keys():
        split_config = data_info['ood'][split]

        if split == 'val':
            # validation set
            dataset = ImglistDataset(
                name='_'.join((id_name, 'ood', split)),
                imglist_pth=os.path.join(data_root,
                                         split_config['imglist_path']),
                data_dir=os.path.join(data_root, split_config['data_dir']),
                num_classes=data_info['num_classes'],
                preprocessor=preprocessor,
                data_aux_preprocessor=test_standard_preprocessor)
            dataloader = DataLoader(dataset, **loader_kwargs)
            dataloader_dict['ood'][split] = dataloader
        else:
            # dataloaders for nearood, farood
            sub_dataloader_dict = {}
            for dataset_name in split_config['datasets']:
                dataset_config = split_config[dataset_name]
                dataset = ImglistDataset(
                    name='_'.join((id_name, 'ood', dataset_name)),
                    imglist_pth=os.path.join(data_root,
                                             dataset_config['imglist_path']),
                    data_dir=os.path.join(data_root,
                                          dataset_config['data_dir']),
                    num_classes=data_info['num_classes'],
                    preprocessor=preprocessor,
                    data_aux_preprocessor=test_standard_preprocessor)
                dataloader = DataLoader(dataset, **loader_kwargs)
                sub_dataloader_dict[dataset_name] = dataloader
            dataloader_dict['ood'][split] = sub_dataloader_dict

    return dataloader_dict


def get_ood_dataloader(config: Config):
    # specify custom dataset class
    ood_config = config.ood_dataset
    CustomDataset = eval(ood_config.dataset_class)
    dataloader_dict = {}
    for split in ood_config.split_names:
        split_config = ood_config[split]
        preprocessor = get_preprocessor(config, split)
        data_aux_preprocessor = TestStandardPreProcessor(config)
        if split == 'val':
            # validation set
            dataset = CustomDataset(
                name=ood_config.name + '_' + split,
                imglist_pth=split_config.imglist_pth,
                data_dir=split_config.data_dir,
                num_classes=ood_config.num_classes,
                preprocessor=preprocessor,
                data_aux_preprocessor=data_aux_preprocessor)
            dataloader = DataLoader(dataset,
                                    batch_size=ood_config.batch_size,
                                    shuffle=ood_config.shuffle,
                                    num_workers=ood_config.num_workers)
            dataloader_dict[split] = dataloader
        else:
            # dataloaders for csid, nearood, farood
            sub_dataloader_dict = {}
            for dataset_name in split_config.datasets:
                dataset_config = split_config[dataset_name]
                dataset = CustomDataset(
                    name=ood_config.name + '_' + split,
                    imglist_pth=dataset_config.imglist_pth,
                    data_dir=dataset_config.data_dir,
                    num_classes=ood_config.num_classes,
                    preprocessor=preprocessor,
                    data_aux_preprocessor=data_aux_preprocessor)
                dataloader = DataLoader(dataset,
                                        batch_size=ood_config.batch_size,
                                        shuffle=ood_config.shuffle,
                                        num_workers=ood_config.num_workers)
                sub_dataloader_dict[dataset_name] = dataloader
            dataloader_dict[split] = sub_dataloader_dict

    return dataloader_dict
