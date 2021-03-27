import numpy as np
import torch
import h5py
from torch.utils.data import Dataset, DataLoader


class ModelNetDataset(Dataset):

    def __init__(self, point_clouds, class_ids):
        self.point_clouds = torch.from_numpy(point_clouds).float()
        self.class_ids = torch.from_numpy(class_ids).long()

    def __len__(self):
        return np.shape(self.point_clouds)[0]

    def __getitem__(self, idx):
        return self.point_clouds[idx], self.class_ids[idx]


def get_train_and_test_loaders(batch_size, num_workers, train=True):
    f = h5py.File('data/ModelNet/modelnet_classification.h5', 'r')

    train_data = ModelNetDataset(f['train_point_clouds'][:], f['train_class_ids'][:])
    test_data = ModelNetDataset(f['test_point_clouds'][:], f['test_class_ids'][:])

    n_classes = np.amax(f['train_class_ids']) + 1

    train_dataloader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=train,
        num_workers=int(num_workers))

    test_dataloader = DataLoader(
        test_data,
        batch_size=batch_size,
        shuffle=train,
        num_workers=int(num_workers))

    return train_data, train_dataloader, test_data, test_dataloader, n_classes