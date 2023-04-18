from torch.nn import TransformerEncoder
from torch.nn import TransformerDecoder
import torch.nn.functional as F
from src.dlutils import *
from src.constants import *

torch.manual_seed(1)


class Transformer(nn.Module):
    def __init__(self, feats):
        super(Transformer, self).__init__()
        self.name = 'Transformer'
        self.lr = lr  # 学习率
        self.batch = 128
        self.n_feats = feats  # 特征数
        self.n_window = 10  # 滑动窗口
        self.n = self.n_feats * self.n_window
        self.pos_encoder = PositionalEncoding(2 * feats, 0.1, self.n_window)
        encoder_layers = TransformerEncoderLayer(d_model=2 * feats, nhead=feats, dim_feedforward=16, dropout=0.1)
        self.transformer_encoder = TransformerEncoder(encoder_layers, 1)
        decoder_layers1 = TransformerDecoderLayer(d_model=2 * feats, nhead=feats, dim_feedforward=16, dropout=0.1)
        self.transformer_decoder1 = TransformerDecoder(decoder_layers1, 1)
        decoder_layers2 = TransformerDecoderLayer(d_model=2 * feats, nhead=feats, dim_feedforward=16, dropout=0.1)
        self.transformer_decoder2 = TransformerDecoder(decoder_layers2, 1)
        self.fcn = nn.Sequential(nn.Linear(2 * feats, feats), nn.Sigmoid())

    # encode函数将输入src和控制信号c拼接在一起，并进行位置编码。
    # 然后将编码后的src输入到变换器编码器中，生成memory。最后将tgt重复两次，并返回tgt和memory。
    def encode(self, src, c, tgt):
        src = torch.cat((src, c), dim=2)
        src = src * math.sqrt(self.n_feats)
        src = self.pos_encoder(src)
        memory = self.transformer_encoder(src)
        tgt = tgt.repeat(1, 1, 2)
        return tgt, memory

    # forward函数是前向传播函数，接收src和tgt作为输入。
    # 首先将控制信号c初始化为与src相同的形状，并将src和c一起输入到变换器解码器1中，经过全连接层得到x1。
    # 然后将控制信号更新为(x1 - src)的平方，并将其与src一起输入到变换器解码器2中，再次经过全连接层得到x2。最后返回x1和x2。
    def forward(self, src, tgt):
        # Phase 1 - Without anomaly scores
        c = torch.zeros_like(src)
        x1 = self.fcn(self.transformer_decoder1(*self.encode(src, c, tgt)))
        # Phase 2 - With anomaly scores
        c = (x1 - src) ** 2
        x2 = self.fcn(self.transformer_decoder2(*self.encode(src, c, tgt)))
        return x1, x2




