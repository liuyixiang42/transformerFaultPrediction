import numpy as np
from scipy.interpolate import CubicSpline


column = ['氢气', '一氧化碳', '甲烷', '乙烯', '乙炔', '乙烷', '总烃', '总可燃气体', '油温']
labels = ['Hydrogen', 'Carbon Monoxide', 'Methane', 'Ethylene', 'Acetylene', 'Ethane', 'Total Hydrocarbons',
          'Total Combustible Gases', 'Oil Temperature']
fault_category = ['insulation aging', 'winding fault', 'insulation breakdown', 'overload fault', 'gas leakage',
                  'cooling fault']


add_noise_level = [1.7, 1.7, 1.4, 0.2, 0, 1, 2, 2.1, 2]
frequency_noise_levels = [7, 7, 5, 2, 0, 3, 17, 25, 20]

noise_factor = [8, 7, 6, 1, 1, 4, 10, 25, 60]

time_scale_typy = {'forward': 0, 'backward': 1}


def smooth_and_extend(seq, length):
    # 定义原始序列
    x = seq
    # 定义插值的x坐标轴
    x_new = np.linspace(0, len(x) - 1, num=length, endpoint=True)
    # 进行样条插值
    cs = CubicSpline(np.arange(len(x)), x)
    smooth = cs(x_new)
    # 确保起始值和终值不变
    smooth[0] = x[0]
    smooth[-1] = x[-1]
    return smooth


def draw_data(fault_data, length):
    new_fault_data = []
    for i in range(5):
        new_data = []
        data = fault_data[i]
        for j in range(9):
            seq = data[:, j]
            new_seq = smooth_and_extend(seq, length)
            new_data.append(new_seq)
        new_fault_data.append(new_data)
    return np.transpose(new_fault_data, (0, 2, 1))


def draw_single_data(fault_data, length):
    new_data = []
    data = fault_data
    for j in range(9):
        seq = data[:, j]
        new_seq = smooth_and_extend(seq, length)
        new_data.append(new_seq)
    return np.array(new_data).T


# 噪声注入
def add_noise(data):
    noisy_data = np.zeros_like(data)
    # 根据不同油气和油温的变化幅度采用不同的noise_level
    for i in range(9):
        noise = np.random.normal(0, add_noise_level[i], len(data))
        factor = data[:, i] / noise_factor[i]
        for j in range(len(factor)):
            if factor[j] > 1.7:
                factor[j] = 1.7
        noise = factor * noise
        noisy_data[:, i] = data[:, i] + noise

    # 将负值替换成0
    noisy_data[noisy_data < 0] = 0
    return noisy_data


def add_frequency_noise(data):
    # 将数据转换到频域
    freq_data = np.fft.fft(data, axis=0)
    # 增加噪声
    noise = np.zeros_like(freq_data)
    for i in range(data.shape[1]):
        noise[:, i] = np.random.normal(scale=frequency_noise_levels[i], size=freq_data.shape[0])
        factor = data[:, i] / noise_factor[i]
        for j in range(len(factor)):
            if factor[j] > 1.5:
                factor[j] = 1.5
        noise[:, i] = noise[:, i] * factor

    freq_data += noise
    # 将数据转换回时域
    noisy_data = np.fft.ifft(freq_data, axis=0).real

    # 将负值替换成0
    noisy_data[noisy_data < 0] = 0
    return noisy_data

def time_scale(data):
    type = np.random.randint(0, 2)
    if type == time_scale_typy['forward']:
        f_data = data[:30, :]
        b_data = data[30:, :]
        f_data = f_data[0::3, :]
        b_data = draw_single_data(b_data, 40)
        return np.concatenate((f_data, b_data), axis=0)
    elif type == time_scale_typy['backward']:
        f_data = data[:20, :]
        b_data = data[20:, :]
        f_data = draw_single_data(f_data, 40)
        b_data = b_data[0::3, :]
        return np.concatenate((f_data, b_data), axis=0)



