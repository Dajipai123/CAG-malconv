import torch
from torch import nn
import torch.nn.functional as F
from torchviz import make_dot
import math

# class PositionalEncoding(nn.Module):
#     def __init__(self, d_model, dropout=0.1, max_len=128):
#         super(PositionalEncoding, self).__init__()
#         self.dropout = nn.Dropout(p=dropout)
        
#         pe = torch.zeros(max_len, d_model)
#         position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # (max_len, 1)
#         div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))  # (d_model / 2)
#         pe[:, 0::2] = torch.sin(position * div_term)  # (max_len, d_model / 2)
#         pe[:, 1::2] = torch.cos(position * div_term)  # (max_len, d_model / 2)
#         pe = pe.unsqueeze(0)  # (1, max_len, d_model)
#         self.register_buffer('pe', pe)
        
#     def forward(self, x):
#         x = x + self.pe[:, :x.size(1)]
#         return self.dropout(x)

class LearnedPositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=128):
        super(LearnedPositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        self.position_embedding = nn.Embedding(max_len, d_model)
        
    def forward(self, x):
        position_ids = torch.arange(x.size(1), dtype=torch.long, device=x.device)
        position_ids = position_ids.unsqueeze(0).repeat(x.size(0), 1)  # Repeat for each batch
        position_embeddings = self.position_embedding(position_ids)
        x = x + position_embeddings
        return self.dropout(x)
    

class MalEncoder(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.embedding_dim = 8
        self.conv_dim = 128
        self.embedding = nn.Embedding(257, self.embedding_dim)
        self.nhead = 4
        self.num_layers = 6
        self.pos_encoder = LearnedPositionalEncoding(self.conv_dim)

        self.conv = nn.Sequential(
            nn.Conv1d(in_channels=self.embedding_dim, out_channels=self.conv_dim, kernel_size=512, stride=512),  # Adjusted kernel size and stride
            nn.LeakyReLU(inplace=True),
        )

        self.pool = nn.AdaptiveMaxPool1d(128)

        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=self.conv_dim, nhead=self.nhead),
            num_layers=self.num_layers
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
        x = self.pool(x).transpose(1, 2)  # Transformer需要(batch_size, seq_length, feature)的输入格式

        x = self.pos_encoder(x)

        x = self.transformer(x)

        x = F.max_pool1d(x, kernel_size=x.size(2)).squeeze(2)
        x = self.classifier(x)
        return x
    
    