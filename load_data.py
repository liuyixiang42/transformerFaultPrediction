import numpy as np
import pandas as pd
from data_enhance_tranditional import *
from preprocess import *
import torch


def load_fault_data():
    fault_data = pd.read_excel('data/故障样本.xlsx', usecols=column)

    fault_data.dropna(inplace=True)

    fault_data = fault_data.values

    fault_data = fault_data[:191, :]

    rows_to_delete = [159, 127, 95, 63, 31]
    fault_data = np.delete(fault_data, rows_to_delete, axis=0)
    fault_data = fault_data.reshape(6, 31, 9)
    fault_data = fault_data[:, 3:28, :]
    fault_data = fault_data.astype(np.float64)
    fault_data = draw_data(fault_data, 50)
    fault_data = fault_data.reshape(300, 9)
    fault_data, _, _ = normalize3(fault_data)
    fault_data = torch.from_numpy(fault_data)
    label = np.concatenate([np.zeros((44, 9)), np.ones((6, 9))], axis=0)
    label = np.concatenate([label] * 6, axis=0)

    return fault_data, label



