from torch.utils.data import Dataset
import numpy as np

class DoubleXDataset(Dataset):
  def __init__(self, dataset):
    self.dataset = dataset

  def __len__(self):
    return len(self.dataset)

  def __getitem__(self, idx):
    data, _ = self.dataset[idx]
    return (data, data)

class Subset(Dataset):
  def __init__(self, dataset, num, random=True):
    self.dataset = dataset
    self.num = num
    if random:
      self.indices = np.random.choice(len(dataset), num, replace=False)
    else:
      self.indices = np.arange(num)

  def __len__(self):
    return self.num

  def __getitem__(self, idx):
    return self.dataset[self.indices[idx]]