from torch.utils.data import DataLoader

from openood.utils.config import Config

from .imglist_dataset import ImglistDataset


def get_dataloader(dataset_config: Config):
    # prepare a dataloader dictionary
    dataloader_dict = {}

    for split in dataset_config.split_names:
        split_config = dataset_config[split]
        CustomDataset = eval(split_config.dataset_class)

        dataset = CustomDataset(name=dataset_config.name + '_' + split,
                                split=split,
                                interpolation=split_config.interpolation,
                                image_size=dataset_config.image_size,
                                imglist_pth=split_config.imglist_pth,
                                data_dir=split_config.data_dir,
                                num_classes=dataset_config.num_classes)

        dataloader = DataLoader(dataset,
                                batch_size=split_config.batch_size,
                                shuffle=split_config.shuffle,
                                num_workers=dataset_config.num_workers)

        dataloader_dict[split] = dataloader

    return dataloader_dict


def get_ood_dataloader(ood_config: Config):
    # specify custom dataset class
    CustomDataset = eval(ood_config.dataset_class)
    dataloader_dict = {}
    for split in ood_config.split_names:
        split_config = ood_config[split]
        if split == 'val':
            # validation set
            dataset = CustomDataset(name=ood_config.name + '_' + split,
                                    split=split,
                                    interpolation=ood_config.interpolation,
                                    image_size=ood_config.image_size,
                                    imglist_pth=split_config.imglist_pth,
                                    data_dir=split_config.data_dir,
                                    num_classes=ood_config.num_classes)
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
                                        num_classes=ood_config.num_classes)
                dataloader = DataLoader(dataset,
                                        batch_size=ood_config.batch_size,
                                        shuffle=ood_config.shuffle,
                                        num_workers=ood_config.num_workers)
                sub_dataloader_dict[dataset_name] = dataloader
            dataloader_dict[split] = sub_dataloader_dict

    return dataloader_dict
