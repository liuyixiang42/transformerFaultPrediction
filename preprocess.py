import numpy as np
import torch

def timestamp_encoding(data, max_sequence_length):
    """
    对数据的时间戳进行编码
    在时间序列数据中，时间戳是非常重要的信息。
    在数据预处理中，对时间戳进行编码可以使模型更好地理解时间序列数据，并更好地捕捉序列中的时间模式。
    具体来说，时间戳编码是将时间戳转换为数值表示，以便模型能够对其进行计算。这样可以使时间戳信息更容易被整合到模型中，进而提高模型的性能。在实现时，通常会使用不同的编码方法，如时间差编码、季节编码、周期编码等。
    Args:
        data: 原始数据，numpy array，形状为 [n_samples, n_features]
        max_sequence_length: 时间序列的最大长度，int

    Returns:
        encoded_data: 编码后的数据，numpy array，形状为 [n_samples, max_sequence_length, n_features + 1]
    """
    n_samples, n_features = data.shape
    encoded_data = np.zeros((n_samples, max_sequence_length, n_features + 1))

    for i in range(n_samples):
        time_stamps = np.arange(1, n_features + 1)
        time_stamps = time_stamps.reshape((1, n_features))

        # 扩展时间戳序列，使其长度等于 max_sequence_length
        expanded_time_stamps = np.repeat(time_stamps, max_sequence_length, axis=0)
        mask = np.less(expanded_time_stamps, max_sequence_length + 1)
        expanded_time_stamps = expanded_time_stamps * mask

        # 将时间戳序列添加到数据中
        data_with_time = np.concatenate((data[i].reshape((1, n_features)), expanded_time_stamps), axis=1)

        # 对时间戳进行归一化
        encoded_data[i] = data_with_time / np.max(data_with_time)

    return encoded_data


def normalize(data, feature_min = None, feature_max = None):
    """
    对数据进行归一化处理
    Args:
        data: 原始数据，numpy array，形状为 [n_samples, n_features]

    Returns:
        normalized_data: 归一化后的数据，numpy array，形状为 [n_samples, n_features]
        feature_min: 每个特征的最小值，numpy array，形状为 [n_features]
        feature_max: 每个特征的最大值，numpy array，形状为 [n_features]
    """
    if feature_min is None:
        feature_min = np.min(data, axis=0)
        feature_max = np.max(data, axis=0)
    normalized_data = (data - feature_min) / (feature_max - feature_min + 1e-4)
    return normalized_data, feature_min, feature_max


def impute_missing_values(x):
    """
    基于自注意力机制的缺失值填充方法
    :param x: 输入数据，shape为(batch_size, num_time_steps, num_features)
    :return: 填充后的数据
    """
    # 获取输入数据的一些参数
    batch_size, num_time_steps, num_features = x.shape

    # 构造掩码矩阵
    mask = np.ones((batch_size, num_time_steps, num_features))
    mask[x == 0] = 0

    # 构造输入序列的掩码矩阵
    input_mask = np.ones((batch_size, num_time_steps))
    input_mask[np.sum(x, axis=-1) == 0] = 0

    # 将numpy数组转化为PyTorch张量
    x = torch.from_numpy(x).float()
    mask = torch.from_numpy(mask).float()
    input_mask = torch.from_numpy(input_mask).float()

    # 将掩码矩阵扩展到三维，方便与输入数据进行运算
    mask = mask.unsqueeze(-1)

    # 构造Transformer模型
    num_layers = 6
    num_heads = 8
    hidden_size = 512

    encoder_layer = torch.nn.TransformerEncoderLayer(d_model=num_features, nhead=num_heads)
    encoder = torch.nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    # 运行Transformer模型
    x = encoder(x * mask)

    # 计算缺失值的加权平均值
    x_sum = torch.sum(x * mask, dim=1)
    mask_sum = torch.sum(mask, dim=1)
    x_mean = x_sum / mask_sum
    x_mean = x_mean.unsqueeze(1).repeat(1, num_time_steps, 1)

    # 将填充后的数据与输入数据中的非缺失值进行拼接
    x = torch.where(input_mask.unsqueeze(-1) == 1, x, x_mean)

    # 将张量转化为numpy数组
    x = x.detach().numpy()

    return x


def convert_to_windows(data, model):
	windows = []; w_size = model.n_window
	for i, g in enumerate(data):
		if i >= w_size: w = data[i-w_size:i]
		else: w = torch.cat([data[0].repeat(w_size-i, 1), data[0:i]])
		windows.append(w if 'TranAD' in args.model or 'Attention' in args.model else w.view(-1))
	return torch.stack(windows)


