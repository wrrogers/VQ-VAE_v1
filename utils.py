import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import time
import os
import numpy as np
from vqvae import VQVAE


def readable_timestamp():
    return time.ctime().replace('  ', ' ').replace(
        ' ', '_').replace(':', '_').lower()


def save_model_and_results(model, results, hyperparameters, saved_name):

    results_to_save = {
        'model': model.state_dict(),
        'results': results,
        'hyperparameters': hyperparameters
    }
    torch.save(results_to_save, saved_name)


class ImageDataset(Dataset):
    """
    Creates image dataset of NXN images with 3 channels
    requires numpy and cv2 to work
    """

    def __init__(self, file_path, transform=None, path_length=50):
        print('Loading data')
        data = np.load(file_path, allow_pickle=True)
        print('Done loading data')
        self.actions = np.array(data.item().get('actions'))
        self.observations = np.array(data.item().get('observations'))
        self.data = np.array(data.item().get('images'))

        self.transform = transform
        self.path_length = path_length

    def __getitem__(self, index):
        img = self.data[index]

        if self.transform is not None:
            img = self.transform(img)

        label = 0
        return img, label

    def __len__(self):
        return len(self.data)