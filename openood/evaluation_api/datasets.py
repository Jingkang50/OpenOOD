import os

from torch.utils.data import DataLoader
import torchvision as tvs
if tvs.__version__ >= '0.13':
    tvs_new = True
else:
    tvs_new = False

from openood.datasets.imglist_dataset import ImglistDataset
from openood.preprocessors import BasePreprocessor

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
                'imglist_path': 'benchmark_imglist/cifar10/val_tin.txt'
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
                'imglist_path': 'benchmark_imglist/cifar100/val_tin.txt'
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
    'imagenet200': {
        'num_classes': 200,
        'id': {
            'train': {
                'data_dir':
                'images_largescale/',
                'imglist_path':
                'benchmark_imglist/imagenet200/train_imagenet200.txt'
            },
            'val': {
                'data_dir': 'images_largescale/',
                'imglist_path':
                'benchmark_imglist/imagenet200/val_imagenet200.txt'
            },
            'test': {
                'data_dir':
                'images_largescale/',
                'imglist_path':
                'benchmark_imglist/imagenet200/test_imagenet200.txt'
            }
        },
        'csid': {
            'datasets': ['imagenet_v2', 'imagenet_c', 'imagenet_r'],
            'imagenet_v2': {
                'data_dir':
                'images_largescale/',
                'imglist_path':
                'benchmark_imglist/imagenet200/test_imagenet200_v2.txt'
            },
            'imagenet_c': {
                'data_dir':
                'images_largescale/',
                'imglist_path':
                'benchmark_imglist/imagenet200/test_imagenet200_c.txt'
            },
            'imagenet_r': {
                'data_dir':
                'images_largescale/',
                'imglist_path':
                'benchmark_imglist/imagenet200/test_imagenet200_r.txt'
            },
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
            'datasets': ['imagenet_v2', 'imagenet_c', 'imagenet_r'],
            'imagenet_v2': {
                'data_dir': 'images_largescale/',
                'imglist_path':
                'benchmark_imglist/imagenet/test_imagenet_v2.txt'
            },
            'imagenet_c': {
                'data_dir': 'images_largescale/',
                'imglist_path':
                'benchmark_imglist/imagenet/test_imagenet_c.txt'
            },
            'imagenet_r': {
                'data_dir': 'images_largescale/',
                'imglist_path':
                'benchmark_imglist/imagenet/test_imagenet_r.txt'
            },
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
    'aircraft': {
        'num_classes': 50,
        'id': {
            'train': {
                'data_dir': 'images_largescale/',
                'imglist_path': 'benchmark_imglist/aircraft/train_id.txt'
            },
            'val': {
                'data_dir': 'images_largescale/',
                'imglist_path': 'benchmark_imglist/aircraft/val_id.txt'
            },
            'test': {
                'data_dir': 'images_largescale/',
                'imglist_path': 'benchmark_imglist/aircraft/test_id.txt'
            }
        },
        'csid': {
            'datasets': [],
        },
        'ood': {
            'val': {
                'data_dir': 'images_largescale/',
                'imglist_path': 'benchmark_imglist/aircraft/val_ood.txt'
            },
            'near': {
                'datasets': ['hardood'],
                'hardood': {
                    'data_dir': 'images_largescale/',
                    'imglist_path':
                    'benchmark_imglist/aircraft/test_ood_hard.txt'
                },
            },
            'far': {
                'datasets': ['easyood'],
                'easyood': {
                    'data_dir': 'images_largescale/',
                    'imglist_path':
                    'benchmark_imglist/aircraft/test_ood_easy.txt'
                },
            },
        }
    },
    'cub': {
        'num_classes': 100,
        'id': {
            'train': {
                'data_dir': 'images_largescale/',
                'imglist_path': 'benchmark_imglist/cub/train_id.txt'
            },
            'val': {
                'data_dir': 'images_largescale/',
                'imglist_path': 'benchmark_imglist/cub/val_id.txt'
            },
            'test': {
                'data_dir': 'images_largescale/',
                'imglist_path': 'benchmark_imglist/cub/test_id.txt'
            }
        },
        'csid': {
            'datasets': [],
        },
        'ood': {
            'val': {
                'data_dir': 'images_largescale/',
                'imglist_path': 'benchmark_imglist/cub/val_ood.txt'
            },
            'near': {
                'datasets': ['hardood'],
                'hardood': {
                    'data_dir': 'images_largescale/',
                    'imglist_path': 'benchmark_imglist/cub/test_ood_hard.txt'
                },
            },
            'far': {
                'datasets': ['easyood'],
                'easyood': {
                    'data_dir': 'images_largescale/',
                    'imglist_path': 'benchmark_imglist/cub/test_ood_easy.txt'
                },
            },
        }
    }
}


def get_id_ood_dataloader(id_name, data_root, preprocessor, **loader_kwargs):
    if 'imagenet' in id_name:
        if tvs_new:
            if isinstance(preprocessor,
                          tvs.transforms._presets.ImageClassification):
                mean, std = preprocessor.mean, preprocessor.std
            elif isinstance(preprocessor, tvs.transforms.Compose):
                temp = preprocessor.transforms[-1]
                mean, std = temp.mean, temp.std
            elif isinstance(preprocessor, BasePreprocessor):
                temp = preprocessor.transform.transforms[-1]
                mean, std = temp.mean, temp.std
            else:
                raise TypeError
        else:
            if isinstance(preprocessor, tvs.transforms.Compose):
                temp = preprocessor.transforms[-1]
                mean, std = temp.mean, temp.std
            elif isinstance(preprocessor, BasePreprocessor):
                temp = preprocessor.transform.transforms[-1]
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
