from re import S
import torch
from numpy import load
from torch.utils.data import DataLoader

from openood.utils.config import Config
from openood.preprocessors.utils import get_preprocessor

from .feature_dataset import FeatDataset
from .imglist_dataset import ImglistDataset


def get_dataloader(config: Config):
    # prepare a dataloader dictionary
    dataset_config = config.dataset
    dataloader_dict = {}
    for split in dataset_config.split_names:
        split_config = dataset_config[split]
        # currently we only support ImglistDataset
        preprocessor = get_preprocessor(split, config)    # all script file need to pass in train_preprocessor config file
        CustomDataset = eval(split_config.dataset_class)
        dataset = CustomDataset(name=dataset_config.name + '_' + split,
                                split=split,
                                interpolation=split_config.interpolation,
                                image_size=dataset_config.image_size,
                                imglist_pth=split_config.imglist_pth,
                                data_dir=split_config.data_dir,
                                num_classes=dataset_config.num_classes,
                                preprocessor=preprocessor)
        dataloader = DataLoader(dataset,
                                batch_size=split_config.batch_size,
                                shuffle=split_config.shuffle,
                                num_workers=dataset_config.num_workers)

        dataloader_dict[split] = dataloader
    return dataloader_dict


def get_ood_dataloader(config: Config):
    # specify custom dataset class
    ood_config = config.ood_dataset
    CustomDataset = eval(ood_config.dataset_class)
    dataloader_dict = {}
    for split in ood_config.split_names:
        split_config = ood_config[split]
        preprocessor = get_preprocessor(split, config)
        if split == 'val':
            # validation set
            dataset = CustomDataset(name=ood_config.name + '_' + split,
                                    split=split,
                                    interpolation=ood_config.interpolation,
                                    image_size=ood_config.image_size,
                                    imglist_pth=split_config.imglist_pth,
                                    data_dir=split_config.data_dir,
                                    num_classes=ood_config.num_classes,
                                    preprocessor=preprocessor)
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
                dataset = CustomDataset(name=ood_config.name + '_' + split,
                                        split=split,
                                        interpolation=ood_config.interpolation,
                                        image_size=ood_config.image_size,
                                        imglist_pth=dataset_config.imglist_pth,
                                        data_dir=dataset_config.data_dir,
                                        num_classes=ood_config.num_classes,
                                        preprocessor=preprocessor)
                dataloader = DataLoader(dataset,
                                        batch_size=ood_config.batch_size,
                                        shuffle=ood_config.shuffle,
                                        num_workers=ood_config.num_workers)
                sub_dataloader_dict[dataset_name] = dataloader
            dataloader_dict[split] = sub_dataloader_dict

    return dataloader_dict


def get_feature_dataloader(dataset_config: Config):
    # load in the cached feature
    loaded_data = load(dataset_config.feat_path, allow_pickle=True)
    total_feat = torch.from_numpy(loaded_data['feat_list'])
    del loaded_data
    # reshape the vector to fit in to the network
    total_feat.unsqueeze_(-1).unsqueeze_(-1)
    # let's see what we got here should be something like:
    # torch.Size([total_num, channel_size, 1, 1])
    print(total_feat.shape)

    split_config = dataset_config['train']

    dataset = FeatDataset(feat=total_feat)
    dataloader = DataLoader(dataset,
                            batch_size=split_config.batch_size,
                            shuffle=split_config.shuffle,
                            num_workers=dataset_config.num_workers)

    return dataloader
