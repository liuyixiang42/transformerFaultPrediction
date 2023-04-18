import torch.nn as nn
import torch
from torch.autograd import Variable
import math
import numpy as np

import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.normalization import LayerNorm


class LayerDrop(nn.Module):
    def __init__(self, drop_prob=0.1):
        super(LayerDrop, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x, residual_fn):
        if not self.training or torch.rand(1) > self.drop_prob:
            return residual_fn(x)
        else:
            return x
class TransformerEncoderLayer2(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=16, dropout=0.1, layerdrop=0.1):
        super(TransformerEncoderLayer2, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.self_attn_drop = LayerDrop(layerdrop)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.linear_drop = LayerDrop(layerdrop)

        self.activation = nn.GELU()

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        src2 = self.self_attn(src, src, src, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)[0]
        src = self.self_attn_drop(src2, lambda x: src + x)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = self.linear_drop(src2, lambda x: src + x)
        return src

class DropoutLayer(nn.Module):
    def __init__(self, p):
        super(DropoutLayer, self).__init__()
        self.dropout = nn.Dropout(p)

    def forward(self, x):
        return x * self.dropout(x)

class TransformerDecoderLayer2(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=16, dropout=0.1):
        super(TransformerDecoderLayer2, self).__init__()

        # 增加层数
        self.self_attn_layers = nn.ModuleList([
            nn.MultiheadAttention(d_model, nhead, dropout=dropout) for _ in range(3)
        ])
        self.multihead_attn_layers = nn.ModuleList([
            nn.MultiheadAttention(d_model, nhead, dropout=dropout) for _ in range(3)
        ])
        self.linear_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, dim_feedforward),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(dim_feedforward, d_model),
                nn.Dropout(dropout),
            ) for _ in range(3)
        ])

        # Layer Normalization
        self.self_attn_norm = nn.ModuleList([LayerNorm(d_model) for _ in range(3)])
        self.multihead_attn_norm = nn.ModuleList([LayerNorm(d_model) for _ in range(3)])
        self.linear_norm = nn.ModuleList([LayerNorm(d_model) for _ in range(3)])

        # LayerDrop
        self.self_attn_drop = nn.ModuleList([
            DropoutLayer(dropout) for _ in range(3)
        ])
        self.multihead_attn_drop = nn.ModuleList([
            DropoutLayer(dropout) for _ in range(3)
        ])
        self.linear_drop = nn.ModuleList([
            DropoutLayer(dropout) for _ in range(3)
        ])

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None, tgt_key_padding_mask=None,
                memory_key_padding_mask=None):
        # Self-attention layer
        attn_tgt = tgt
        for i in range(3):
            attn_tgt2 = self.self_attn_layers[i](attn_tgt, attn_tgt, attn_tgt, attn_mask=tgt_mask)[0]
            attn_tgt = attn_tgt + self.self_attn_drop[i](attn_tgt2)
            attn_tgt = self.self_attn_norm[i](attn_tgt)

        # Multi-head attention layer
        attn_mem = memory
        for i in range(3):
            attn_tgt2 = self.multihead_attn_layers[i](attn_tgt, attn_mem, attn_mem, key_padding_mask=memory_key_padding_mask)[0]
            attn_tgt = attn_tgt + self.multihead_attn_drop[i](attn_tgt2)
            attn_tgt = self.multihead_attn_norm[i](attn_tgt)

        # Feedforward layer
        ff_tgt = attn_tgt
        for i in range(3):
            ff_tgt2 = self.linear_layers[i](ff_tgt)
            ff_tgt = ff_tgt + self.linear_drop[i](ff_tgt2)
            ff_tgt = self.linear_norm[i](ff_tgt)

        return ff_tgt


class PositionalEncoding2(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding2, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        self.max_len = max_len
        self.d_model = d_model

        # Learnable parameters for generating position encoding
        self.alpha = nn.Parameter(torch.randn(1, 1, d_model))
        self.beta = nn.Parameter(torch.randn(1, 1, d_model))

        # Adaptive learning rate scheduler for alpha and beta
        self.lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer=torch.optim.Adam([self.alpha, self.beta], lr=1e-3),
            mode='min',
            factor=0.5,
            patience=2,
            threshold=0.01,
            verbose=True,
        )

    def generate_positional_encoding(self, pos):
        """
        Generate positional encoding given a sequence position.
        """
        freq_divider = torch.exp(torch.arange(0, self.d_model, 2).float() * (-math.log(10000.0) / self.d_model))
        pos = torch.tensor([pos]).float()
        sin_part = torch.sin(pos * freq_divider)
        cos_part = torch.cos(pos * freq_divider)
        encoding = torch.zeros(pos.size(0), self.d_model, device=pos.device)
        encoding[:, 0::2] = sin_part
        encoding[:, 1::2] = cos_part
        encoding = encoding.unsqueeze(0)
        return encoding

    def forward(self, x, pos=0):
        # Calculate learnable parameters for position encoding
        alpha = self.alpha.repeat(x.size(0), 1, 1)
        beta = self.beta.repeat(x.size(0), 1, 1)

        # Generate position encoding
        pos_encoding = alpha * self.generate_positional_encoding(pos) + beta

        # Pad position encoding if sequence length is less than max_len
        if x.size(0) < self.max_len:
            pad_len = self.max_len - x.size(0)
            pad_encoding = torch.zeros(pad_len, self.d_model, device=x.device)
            pos_encoding = torch.cat((pos_encoding, pad_encoding), dim=0)

        # Add position encoding to input tensor
        x = x + pos_encoding[:x.size(0), :, :]

        # Update learning rate for alpha and beta
        self.lr_scheduler.step(alpha.abs().mean())

        return self.dropout(x)

















# 位置编码（Positional Encoding）类。它的作用是为输入的序列添加位置信息，以便在Transformer等模型中更好地处理序列信息。
#
# d_model表示模型的维度大小，dropout表示dropout的比例，max_len表示最大的序列长度。
# 在初始化函数中，首先通过调用nn.Dropout函数创建了一个dropout层，然后创建一个形状为(max_len, d_model)的全零张量pe，代表了最大长度为max_len的位置编码矩阵。
# 接着，通过torch.arange函数生成了一个长度为max_len的序列position，然后用一个div_term变量来计算每个位置上每个维度的编码值。
# 最后，将计算出的sin和cos函数的结果相加得到位置编码矩阵pe，将其转换为(batch_size, max_len, d_model)的形状，然后通过self.register_buffer将pe注册为一个固定的缓存区。
#
# 类的前向传播函数forward中有两个参数：x表示输入的序列张量，pos表示起始位置。首先将输入张量x与位置编码矩阵pe的一部分相加，用以为输入序列添加位置信息，然后再通过dropout层进行dropout操作。最终返回经过位置编码和dropout后的序列张量。
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model).float() * (-math.log(10000.0) / d_model))
        pe += torch.sin(position * div_term)
        pe += torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x, pos=0):
        x = x + self.pe[pos:pos + x.size(0), :]
        return self.dropout(x)


# 这段代码实现了Transformer的编码器层。其作用是将输入序列进行特征提取和编码，并生成一系列高维度的特征表示。
# 编码器层使用了自注意力机制，通过对输入序列中的每个位置进行自注意力计算，从而实现了全局信息的聚合。同时，编码器层还使用了前向神经网络来进一步提高特征的表达能力。整个编码器层是由多个编码器单元组成的，每个编码器单元都包含了自注意力机制和前向神经网络，用于对输入序列进行特征提取和编码。
#
# 类的初始化函数中有四个参数：d_model表示输入的特征维度，nhead表示头的数量，dim_feedforward表示前向神经网络中间层的维度，dropout表示dropout的比例。在初始化函数中，首先创建了一个多头自注意力层，一个线性层和一些dropout层。同时，使用LeakyReLU激活函数来激活中间层的输出。
#
# 类的前向传播函数forward中有三个参数：src表示输入张量，src_mask表示掩码张量，src_key_padding_mask表示输入张量中要忽略的掩码。
# 在前向传播函数中，首先使用多头自注意力层进行自注意力计算，得到一个新的张量src2。然后将原始输入张量src与新计算的张量src2相加，使用dropout1层进行dropout操作。
# 接着，将src2通过线性层和激活函数进行计算，并使用dropout2层进行dropout操作。最终将两个计算结果相加，并返回得到的张量src。
# 其中src_mask和src_key_padding_mask参数用于在注意力计算中控制计算哪些位置的注意力权重和忽略哪些位置的输入。
class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=16, dropout=0):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = nn.LeakyReLU(True)

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        src2 = self.self_attn(src, src, src)[0]
        src = src + self.dropout1(src2)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        return src


# 以上代码实现了Transformer的解码器层，其作用是在给定编码器输出的基础上，生成目标序列的高维度特征表示。
# 具体来说，解码器层通过自注意力机制和跨注意力机制对输入序列和目标序列之间进行信息交互和整合，实现了对目标序列的特征提取和编码。
# 和编码器层类似，解码器层也使用了前向神经网络来进一步提高特征的表达能力。整个解码器层是由多个解码器单元组成的，每个解码器单元都包含了自注意力机制、跨注意力机制和前向神经网络，用于对目标序列进行特征提取和编码。
class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=16, dropout=0):
        super(TransformerDecoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = nn.LeakyReLU(True)

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None, tgt_key_padding_mask=None,
                memory_key_padding_mask=None):
        tgt2 = self.self_attn(tgt, tgt, tgt)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt2 = self.multihead_attn(tgt, memory, memory)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt


# 定义了一个计算DAGMM（Deep Autoencoding Gaussian Mixture Model）模型的损失函数和参数计算方法的类
# 实例化参数包括model、lambda_energy、lambda_cov、device和n_gmm。lambda_energy和lambda_cov是损失函数的两个超参数，n_gmm是高斯混合模型中的高斯分量数目。
#
# forward方法计算DAGMM的总损失函数，包括重构误差、样本能量和协方差对角线项。其中，reconst_loss表示重构误差，sample_energy和cov_diag分别表示样本能量和协方差对角线项。
#
# compute_energy方法计算样本能量函数。其中，phi、mu和cov分别表示高斯混合模型的权重、均值和协方差矩阵。这些参数通过compute_params方法计算得出。然后，计算样本能量函数并返回结果。
#
# compute_params方法计算高斯混合模型的参数phi、mu和cov。具体地，phi表示每个高斯分量的权重，mu表示每个高斯分量的均值，cov表示每个高斯分量的协方差矩阵。
#
# 通过以上三个函数的组合，ComputeLoss可以计算DAGMM算法的损失函数，从而实现对数据的聚类
class ComputeLoss:
    def __init__(self, model, lambda_energy, lambda_cov, device, n_gmm):
        self.model = model
        self.lambda_energy = lambda_energy
        self.lambda_cov = lambda_cov
        self.device = device
        self.n_gmm = n_gmm

    def forward(self, x, x_hat, z, gamma):
        """Computing the loss function for DAGMM."""
        reconst_loss = torch.mean((x - x_hat).pow(2))

        sample_energy, cov_diag = self.compute_energy(z, gamma)

        loss = reconst_loss + self.lambda_energy * sample_energy + self.lambda_cov * cov_diag
        return Variable(loss, requires_grad=True)

    def compute_energy(self, z, gamma, phi=None, mu=None, cov=None, sample_mean=True):
        """Computing the sample energy function"""
        if (phi is None) or (mu is None) or (cov is None):
            phi, mu, cov = self.compute_params(z, gamma)

        z_mu = (z.unsqueeze(1) - mu.unsqueeze(0))

        eps = 1e-12
        cov_inverse = []
        det_cov = []
        cov_diag = 0
        for k in range(self.n_gmm):
            cov_k = cov[k] + (torch.eye(cov[k].size(-1)) * eps).to(self.device)
            cov_inverse.append(torch.inverse(cov_k).unsqueeze(0))
            det_cov.append((Cholesky.apply(cov_k.cpu() * (2 * np.pi)).diag().prod()).unsqueeze(0))
            cov_diag += torch.sum(1 / cov_k.diag())

        cov_inverse = torch.cat(cov_inverse, dim=0)
        det_cov = torch.cat(det_cov).to(self.device)

        E_z = -0.5 * torch.sum(torch.sum(z_mu.unsqueeze(-1) * cov_inverse.unsqueeze(0), dim=-2) * z_mu, dim=-1)
        E_z = torch.exp(E_z)
        E_z = -torch.log(torch.sum(phi.unsqueeze(0) * E_z / (torch.sqrt(det_cov)).unsqueeze(0), dim=1) + eps)
        if sample_mean == True:
            E_z = torch.mean(E_z)
        return E_z, cov_diag

    def compute_params(self, z, gamma):
        """Computing the parameters phi, mu and gamma for sample energy function """
        # K: number of Gaussian mixture components
        # N: Number of samples
        # D: Latent dimension
        # z = NxD
        # gamma = NxK

        # phi = D
        phi = torch.sum(gamma, dim=0) / gamma.size(0)

        # mu = KxD
        mu = torch.sum(z.unsqueeze(1) * gamma.unsqueeze(-1), dim=0)
        mu /= torch.sum(gamma, dim=0).unsqueeze(-1)

        z_mu = (z.unsqueeze(1) - mu.unsqueeze(0))
        z_mu_z_mu_t = z_mu.unsqueeze(-1) * z_mu.unsqueeze(-2)

        # cov = K x D x D
        cov = torch.sum(gamma.unsqueeze(-1).unsqueeze(-1) * z_mu_z_mu_t, dim=0)
        cov /= torch.sum(gamma, dim=0).unsqueeze(-1).unsqueeze(-1)

        return phi, mu, cov

# 实现了一个自定义的Cholesky分解计算方法，用于计算对称正定矩阵的分解。
class Cholesky(torch.autograd.Function):
    def forward(ctx, a):
        l = torch.cholesky(a, False)
        ctx.save_for_backward(l)
        return l

    def backward(ctx, grad_output):
        l, = ctx.saved_variables
        linv = l.inverse()
        inner = torch.tril(torch.mm(l.t(), grad_output)) * torch.tril(
            1.0 - Variable(l.data.new(l.size(1)).fill_(0.5).diag()))
        s = torch.mm(linv.t(), torch.mm(inner, linv))
        return s
