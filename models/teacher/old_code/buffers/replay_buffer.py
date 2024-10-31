import random
from collections import deque
import torch
from torch.utils.data import Dataset


class ReplayBuffer(Dataset):

    
    def __init__(self, maxlen=None, transform=None):
        super().__init__()
        self.container = deque(maxlen=maxlen)
        self.transform = transform

    def __len__(self):
        return len(self.container)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        item = self.container[idx]
        return item
    
    def append(self, item):
        if self.transform:
            item = self.transform(item)
        
        self.container.append(item)
    
    def clear(self):
        self.container.clear()
    
    def zip(self):
        items = {}
        keys = self.container[0].keys()
        for key in keys:
            values = torch.stack([item[key] for item in self.container])
            items[key] = values
        return items
