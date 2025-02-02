import torch
from torch.utils.data import Dataset


class TextClassificationDataset(Dataset):
    def __init__(self, encodings, labels, indices):
        self.encodings = encodings
        self.labels = labels
        self.indices = indices

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        item["idx"] = torch.tensor(self.indices[idx])
        return item

    def __len__(self):
        return len(self.labels)
