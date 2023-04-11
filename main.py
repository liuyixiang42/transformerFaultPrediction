import os
import pandas as pd
import constants
import numpy as np
from torch.utils.data import Dataset, DataLoader, TensorDataset

from utils import cut_array


def load_dataset(dataset):
    folder = os.path.join(constants.output_folder, dataset)
    if not os.path.exists(folder):
        raise Exception('Processed Data not found.')
    loader = []
    for file in ['train', 'test', 'labels']:
        if dataset == 'SMAP': file = 'P-1_' + file
        if dataset == 'MSL': file = 'C-1_' + file
        loader.append(np.load(os.path.join(folder, f'{file}.npy')))
    if True:
        loader[0] = cut_array(0.2, loader[0])
    train_loader = DataLoader(loader[0], batch_size=loader[0].shape[0])
    test_loader = DataLoader(loader[1], batch_size=loader[1].shape[0])
    labels = loader[2]
    return train_loader, test_loader, labels


if __name__ == '__main__':
    train_loader, test_loader, labels = load_dataset('SMAP')
    trainD, testD = next(iter(train_loader)), next(iter(test_loader))
    trainO, testO = trainD, testD
    trainD, testD = convert_to_windows(trainD, model), convert_to_windows(testD, model)
