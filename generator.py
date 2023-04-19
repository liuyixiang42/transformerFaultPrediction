import torch
import random
import numpy as np
from preprocess import *

DAY_SECONDS = 86400


def config():
    config_dict = {
        'pt_num': 4,
        'interval': 10,
        'start_time': 21600,

        'base': {
            'env_temp': 10.5,
            'env_humid': 0.42,
            'pt_temp': 15.0
        },
        'fluc': {
            'env_temp': 2,
            'env_humid': 0.35,
            'env_freq': 180,
            'pt_temp': 0.5,
            'pt_freq': 30
        },
        'period': {
            'long': 100000
        },

        'label': {
            'point_anomal': {
                'point_cnt': [2, 8],
                'delta_max': 50,
                'fluc': 2
            },
            'linear_anomal': {
                'point_cnt': [8, 12],
                'delta_max': 70,
                'fluc': 2,
            },
            'square_anomal': {
                'point_cnt': [9, 14],
                'delta_max': 7,
                'fluc': 2,
                'prop': 0.5
            },
        }
    }
    return config_dict


def generate_normal(config, start_time):
    start_time = start_time % DAY_SECONDS
    normal_dict = {}
    for period, seconds in config['period'].items():
        point_cnt = int(seconds / config['interval'])
        normal_array = [[start_time + i * 10 for i in range(point_cnt)]] + \
                       [[config['base']['pt_temp']] * point_cnt for _ in range(config['pt_num'])] + \
                       [[config['base']['env_temp']] * point_cnt] + [[config['base']['env_humid']] * point_cnt]
        start_time = normal_array[0][-1]
        env_temp_delta = config['fluc']['env_temp'] / config['fluc']['env_freq']
        env_humid_delta = config['fluc']['env_humid'] / config['fluc']['env_freq']
        pt_temp_delta = config['fluc']['pt_temp'] / config['fluc']['pt_freq']
        for i in range(point_cnt):
            for j in range(1, 1 + config['pt_num']):
                normal_array[j][i] += round(pt_temp_delta * random.uniform(-1, 1), 3)
            normal_array[5][i] += round(env_temp_delta * random.uniform(-1, 1), 3)
            normal_array[6][i] += round(env_humid_delta * random.uniform(-1, 1), 4)
        normal_dict[period] = normal_array
    return normal_dict


def generate_anomal(array_list, config_dict):
    n = len(array_list[0])
    labels = np.zeros((6, n))
    anomal_num = 20
    for i in range(anomal_num):
        a = random.randint(0, 2)
        if a == 0:
            generate_point_anomal(array_list, labels, i * 100, config_dict)
        elif a == 1:
            generate_linear_anomal(array_list, labels, i * 100, config_dict)
        elif a == 2:
            generate_square_anomal(array_list, labels, i * 100, config_dict)

    return labels.T


def generate_point_anomal(array_list, labels, start_index, config_dict):
    n = len(array_list[0])
    args = config_dict['label']['point_anomal']
    point_cnts = random.randint(args['point_cnt'][0], args['point_cnt'][1])
    for j in range(1, 1 + config_dict['pt_num']):
        for i in range(start_index, start_index + point_cnts):
            if i >= n:
                break
            array_list[j][i] += args['delta_max'] + round(random.uniform(-1, 1) * args['fluc'], 3)
    for j in range(0, 6):
        for i in range(start_index, start_index + point_cnts):
            if i >= n:
                break
            labels[j][i] = 1


def generate_linear_anomal(array_list, labels, start_index, config_dict):
    n = len(array_list[0])
    args = config_dict['label']['linear_anomal']
    point_cnts = random.randint(args['point_cnt'][0], args['point_cnt'][1])
    for j in range(1, 1 + config_dict['pt_num']):
        half_point_cnt = int(point_cnts / 2)
        delta = args['delta_max'] / half_point_cnt
        for i in range(start_index, start_index + half_point_cnt + 1):
            if i >= n:
                break
            array_list[j][i] = array_list[j][i - 1] + delta + round(random.uniform(-1, 1) * args['fluc'], 3)
        for i in range(start_index + half_point_cnt + 1, start_index + point_cnts):
            if i >= n:
                break
            array_list[j][i] = max(array_list[j][i],
                                   array_list[j][i - 1] - delta + round(random.uniform(-1, 1) * args['fluc'], 3))
    for j in range(0, 6):
        for i in range(start_index, start_index + point_cnts):
            if i >= n:
                break
            labels[j][i] = 1


def generate_square_anomal(array_list, labels, start_index, config_dict):
    n = len(array_list[0])
    args = config_dict['label']['square_anomal']
    point_cnts = random.randint(args['point_cnt'][0], args['point_cnt'][1])
    for j in range(1, 1 + config_dict['pt_num']):
        i = start_index
        while i < start_index + point_cnts:
            if i >= n:
                break
            delta = args['prop'] * pow(i - start_index + 1, 2)
            if delta > args['delta_max']:
                break
            array_list[j][i] = array_list[j][i - 1] + delta + round(random.uniform(-1, 1) * args['fluc'], 3)
            i += 1
        delta = args['delta_max'] / (start_index + point_cnts - i)
        while i < start_index + point_cnts:
            if i >= n:
                break
            array_list[j][i] = max(array_list[j][i],
                                   array_list[j][i - 1] - delta + round(random.uniform(-1, 1) * args['fluc'], 3))
            i += 1
    for j in range(0, 6):
        for i in range(start_index, start_index + point_cnts):
            if i >= n:
                break
            labels[j][i] = 1


def generator():
    config_dict = config()
    start_time = config_dict['start_time']
    normal_dict = generate_normal(config_dict, start_time)
    array_list = normal_dict['long']
    labels = generate_anomal(array_list, config_dict)
    array_list = np.array(array_list).T
    array_list = array_list[:, 1:]
    array_list, _, _ = normalize3(array_list)
    array_list = torch.from_numpy(array_list)
    return array_list, labels
