import constants
import os
import pandas as pd
import numpy as np
import preprocess


def load_data(dataset):
    folder = os.path.join(constants.output_folder, dataset)
    os.makedirs(folder, exist_ok=True)
    dataset_folder = 'data/SMAP_MSL'
    file = os.path.join(dataset_folder, 'labeled_anomalies.csv')
    values = pd.read_csv(file)
    values = values[values['spacecraft'] == dataset]
    filenames = values['chan_id'].values.tolist()
    for fn in filenames:
        train = np.load(f'{dataset_folder}/train/{fn}.npy')
        test = np.load(f'{dataset_folder}/test/{fn}.npy')
        train, min_a, max_a = preprocess.normalize(train)
        test, _, _ = preprocess.normalize(test, min_a, max_a)
        np.save(f'{folder}/{fn}_train.npy', train)
        np.save(f'{folder}/{fn}_test.npy', test)
        labels = np.zeros(test.shape)
        indices = values[values['chan_id'] == fn]['anomaly_sequences'].values[0]
        indices = indices.replace(']', '').replace('[', '').split(', ')
        indices = [int(i) for i in indices]
        for i in range(0, len(indices), 2):
            labels[indices[i]:indices[i + 1], :] = 1
        np.save(f'{folder}/{fn}_labels.npy', labels)


if __name__ == '__main__':
    load_data('SMAP')
    load_data('MSL')
