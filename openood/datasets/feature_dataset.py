from torch.utils.data import Dataset


class FeatDataset(Dataset):
    def __init__(self, feat):
        self.data = feat
        self.len = feat.shape[0]

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        data = self.data[idx]
        return data
