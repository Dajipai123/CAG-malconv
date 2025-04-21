import torch
from torch import nn
import torch.nn.functional as F
from torchviz import make_dot

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)

        self.fc1 = nn.Conv1d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv1d(in_planes // ratio, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv1d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class CBAM(nn.Module):
    def __init__(self, in_planes, ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.ca = ChannelAttention(in_planes, ratio)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x):
        x = self.ca(x) * x
        x = self.sa(x) * x
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

        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels=self.conv_dim, out_channels=1, kernel_size=1),
            nn.LeakyReLU(inplace=True),
        )

        self.cbam = CBAM(self.conv_dim)

        self.pool_size = 32
        self.pool = nn.AdaptiveMaxPool1d(self.pool_size)

        self.feed_forward0 = nn.Sequential(
            nn.Linear(self.conv_dim, self.conv_dim*2),
            nn.BatchNorm1d(self.pool_size),
            nn.LeakyReLU(),
            nn.Linear(self.conv_dim*2, self.conv_dim),
            # nn.BatchNorm1d(self.pool_size),
            nn.LeakyReLU()
        )

        self.feed_forward = nn.Sequential(
            nn.Linear(self.conv_dim*2, self.conv_dim),
            nn.BatchNorm1d(self.conv_dim),
            nn.LeakyReLU(),
        )

        self.classifier = nn.Sequential(
            nn.Linear(self.conv_dim*32, self.conv_dim // 2),
            nn.BatchNorm1d(self.conv_dim // 2),
            nn.LeakyReLU(),
            nn.Linear(self.conv_dim // 2, 2),
        )

    def forward(self, x):
        x = self.embedding(x).transpose(1, 2)
        
        x = self.conv(x)
        x = self.pool(x)
        x = self.cbam(x)
        # x = self.feed_forward0(x)
        x = torch.flatten(x, start_dim=1)
        return self.classifier(x)
    