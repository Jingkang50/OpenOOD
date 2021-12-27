from pathlib import Path

from torch.utils.data import DataLoader, dataloader

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
