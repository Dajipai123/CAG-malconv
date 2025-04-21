import torch
from torch import nn
import torch.nn.functional as F
from torchviz import make_dot


class AttentionGRU(nn.Module):
    def __init__(self, input_dim, hidden_dim, bidirectional):
        super().__init__()

        # GRU层
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True, bidirectional=bidirectional)

        # 注意力层
        hidden_dim = hidden_dim * 2 if bidirectional else hidden_dim
        self.attention = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x):
        # GRU层
        out = self.gru(x)[0]

        # 注意力层
        attention_weights = torch.softmax(self.attention(out), dim=1)
        return torch.sum(attention_weights * out, dim=1)

class DepthwiseSeparableConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, stride, kernel_size, bias=True):
        super().__init__()

        # 减小步长和卷积核大小
        new_stride = stride // 2 if stride > 1 else stride
        new_kernel_size = kernel_size // 2 if kernel_size > 1 else kernel_size

        self.depthwise = nn.Conv1d(in_channels, in_channels, new_kernel_size, new_stride, groups=in_channels, bias=bias)
        self.pointwise = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=bias)

    def forward(self, x):
        return self.pointwise(self.depthwise(x))


class CBAMLayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CBAMLayer, self).__init__()
 
        # channel attention 压缩H,W为1
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
 
        # shared MLP
        self.mlp = nn.Sequential(
            nn.Conv1d(channel, channel // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv1d(channel // reduction, channel, 1, bias=False)
        )
 
        # spatial attention
        self.conv = nn.Conv1d(2, 1, kernel_size=1, padding=0, bias=False)
        self.sigmoid = nn.Sigmoid()
 
    def forward(self, x):
        max_out = self.mlp(self.max_pool(x))
        avg_out = self.mlp(self.avg_pool(x))
        channel_out = self.sigmoid(max_out + avg_out)
        x = channel_out * x
 
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        avg_out = torch.mean(x, dim=1, keepdim=True)
        spatial_out = self.sigmoid(self.conv(torch.cat([max_out, avg_out], dim=1)))
        x = spatial_out * x
        return x


class MalEncoder(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.embedding_dim = 8
        self.conv_dim = 128
        self.embedding = nn.Embedding(257, self.embedding_dim)

        self.conv = nn.Sequential(
            nn.Conv1d(in_channels=self.embedding_dim, out_channels=self.conv_dim, kernel_size=512, stride=512),
            nn.BatchNorm1d(self.conv_dim),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool1d(kernel_size=4, stride=4),
            # SpatialDropout1D(0.1),
        )

        self.depthwise_separable_conv = nn.Sequential(
            DepthwiseSeparableConv1d(in_channels=self.embedding_dim, out_channels=self.conv_dim, kernel_size=512, stride=512),
            nn.BatchNorm1d(self.conv_dim),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool1d(kernel_size=4, stride=4),
        )

        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels=self.conv_dim, out_channels=1, kernel_size=1),
            nn.LeakyReLU(inplace=True),
        )

        self.pool_size = 64
        self.pool = nn.AdaptiveMaxPool1d(self.pool_size)

        self.feed_forward0 = nn.Sequential(
            nn.Linear(self.conv_dim, self.conv_dim*2),
            nn.BatchNorm1d(self.pool_size),
            nn.LeakyReLU(),
            nn.Linear(self.conv_dim*2, self.conv_dim),
            nn.BatchNorm1d(self.pool_size),
            nn.LeakyReLU()
        )

        self.cbam = CBAMLayer(self.conv_dim)

        self.gru = AttentionGRU(self.conv_dim, self.conv_dim, bidirectional=True)
        self.feed_forward = nn.Sequential(
            nn.Linear(self.conv_dim*2, self.conv_dim),
            nn.BatchNorm1d(self.conv_dim),
            nn.LeakyReLU(),
        )

        self.classifier = nn.Sequential(
            nn.Linear(self.conv_dim, self.conv_dim // 2),
            nn.BatchNorm1d(self.conv_dim // 2),
            nn.LeakyReLU(),
            nn.Linear(self.conv_dim // 2, 2),
        )

    def forward(self, x):
        x = self.embedding(x).transpose(1, 2)
        x = self.conv(x)
        x = self.pool(x)
        # x = self.feed_forward0(x)
        x = self.cbam(x).transpose(1, 2)
        x= self.gru(x)     # # accpet shape: (batch_size, seq_len, hidden_size)

        x = self.feed_forward(x)
        
        # return x
        return self.classifier(x)
    