from torch.utils.data import Dataset


class FeatDataset(Dataset):
    def __init__(self, feat, labels):
        self.data = feat
        self.labels = labels
        self.len = feat.shape[0]
        assert self.len == len(labels)

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        data = self.data[idx]
        label = self.labels[idx]
        return {'data': data, 'label': label}
