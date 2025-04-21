import torch
from torch import nn
import torch.nn.functional as F
from torchviz import make_dot


class AttentionLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, bidirectional):
        super().__init__()

        # LSTM层
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True, bidirectional=bidirectional)

        # 注意力层
        hidden_dim = hidden_dim * 2 if bidirectional else hidden_dim
        self.attention = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x):
        # LSTM层
        out = self.lstm(x)[0]

        # 注意力层
        attention_weights = torch.softmax(self.attention(out), dim=1)
        return torch.sum(attention_weights * out, dim=1)


class MalEncoder(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.embedding_dim = 8
        self.conv_dim = 128
        self.embedding = nn.Embedding(257, self.embedding_dim)

        self.conv = nn.Sequential(
            nn.Conv1d(in_channels=self.embedding_dim, out_channels=self.conv_dim, kernel_size=512, stride=512),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool1d(kernel_size=4, stride=4),
            # SpatialDropout1D(0.1),
        )

        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels=self.conv_dim, out_channels=1, kernel_size=1),
            nn.LeakyReLU(inplace=True),
        )

        self.pool_size = 32
        self.pool = nn.AdaptiveMaxPool1d(self.pool_size)

        # self.feed_forward = nn.Sequential(
        #     nn.Linear(self.pool_size, self.conv_dim),
        #     nn.BatchNorm1d(self.conv_dim),
        #     nn.LeakyReLU(),
        # )

        # self.classifier = nn.Sequential(
        #     nn.Linear(self.conv_dim, self.conv_dim // 2),
        #     nn.BatchNorm1d(self.conv_dim // 2),
        #     nn.LeakyReLU(),
        #     nn.Linear(self.conv_dim // 2, 2),
        # )

        self.lstm = AttentionLSTM(self.conv_dim, self.conv_dim, bidirectional=True)
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
        x = self.pool(x).transpose(1, 2)

        x= self.lstm(x)

        # print(x.size())
        # exit()

        # x = self.conv1(x).squeeze(1)

        # x = self.pool(x)

        x = self.feed_forward(x)
        return self.classifier(x)
    