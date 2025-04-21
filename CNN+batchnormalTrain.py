import os
from collections import deque

import random
import numpy as np

# from tqdm import tqdm_notebook as tqdm
from tqdm import tqdm, trange
import multiprocessing

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

import torch.optim as optim

from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler

from binaryLoader import BinaryDataset, RandomChunkSampler, pad_collate_func
from sklearn.metrics import roc_auc_score,f1_score,recall_score

import argparse
import time


from model import MalEncoder


# Check if the input is a valid directory
def dir_path(string):
    if os.path.isdir(string):
        return string
    else:
        raise NotADirectoryError(string)


parser = argparse.ArgumentParser(description='Train a MalConv model')

parser.add_argument('--filter_size', type=int, default=256, help='How wide should the filter be')
parser.add_argument('--filter_stride', type=int, default=64, help='Filter Stride')
parser.add_argument('--embd_size', type=int, default=8, help='Size of embedding layer')
parser.add_argument('--num_channels', type=int, default=128, help='Total number of channels in output')
parser.add_argument('--epochs', type=int, default=30, help='How many training epochs to perform')
parser.add_argument('--non-neg', type=bool, default=False, help='Should non-negative training be used')
parser.add_argument('--batch_size', type=int, default=32, help='Batch size during training')
# Default is set ot 16 MB!
parser.add_argument('--max_len', type=int, default=16000000,
                    help='Maximum length of input file in bytes, at which point files will be truncated')

parser.add_argument('--gpus', nargs='+', type=int)


parser.add_argument('mal_train', type=dir_path, help='Path to directory containing malware files for training')
parser.add_argument('ben_train', type=dir_path, help='Path to directory containing benign files for training')
parser.add_argument('mal_test', type=dir_path, help='Path to directory containing malware files for testing')
parser.add_argument('ben_test', type=dir_path, help='Path to directory containing benign files for testing')

# 设置随机数种子
random_seed = 15
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
torch.cuda.manual_seed_all(random_seed)
np.random.seed(random_seed)
random.seed(random_seed)
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False


def z_score_normalize(data):
    # dtype = data.dtype
    data = data.cpu().numpy()
    std = np.nanstd(data)
    mean = np.nanmean(data)
    data = (data - mean) / std
    return torch.as_tensor(data, dtype=torch.float16)


if __name__ == '__main__':
    args = parser.parse_args()

    GPUS = args.gpus

    NON_NEG = args.non_neg
    EMBD_SIZE = args.embd_size
    FILTER_SIZE = args.filter_size
    FILTER_STRIDE = args.filter_stride
    NUM_CHANNELS = args.num_channels
    EPOCHS = args.epochs
    MAX_FILE_LEN = args.max_len

    BATCH_SIZE = args.batch_size

    whole_dataset = BinaryDataset(args.ben_train, args.mal_train, sort_by_size=True, max_len=MAX_FILE_LEN)
    test_dataset = BinaryDataset(args.ben_test, args.mal_test, sort_by_size=True, max_len=MAX_FILE_LEN)

    loader_threads = max(multiprocessing.cpu_count() - 4, multiprocessing.cpu_count() // 2 + 1)

    # collate_fn 将一个list的sample组成一个mini-batch的函数，这里因为数据的长度不同因此自定义了方法
    # sampler 参数用于指定从数据集中抽取样本的策略。
    train_loader = DataLoader(whole_dataset, batch_size=BATCH_SIZE, num_workers=0, collate_fn=pad_collate_func,
                            sampler=RandomChunkSampler(whole_dataset, BATCH_SIZE))

    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, num_workers=0, collate_fn=pad_collate_func,
                            sampler=RandomChunkSampler(test_dataset, BATCH_SIZE))
    
    device = torch.device('cuda:0')
    model = MalEncoder().to(device)
    optimizer = optim.Adam(params=model.parameters(),lr=3e-4, weight_decay=3e-5)
    criterion = nn.CrossEntropyLoss().to(device)
    scaler = GradScaler()
    
    for i in trange(EPOCHS, desc='Train'):
        model.train()
        
        with tqdm(total=len(train_loader), desc='Train') as epoch_progress:
            for train_data, train_label in train_loader:
                optimizer.zero_grad()

                with autocast():
                    # train_data = z_score_normalize(train_data).to(device)
                    train_data = train_data.to(dtype=torch.int32, device=device)
                    # train_data = train_data.unsqueeze(1).to(dtype=torch.float32)
                    out = model(train_data)
                    loss = criterion(out, train_label.to(device))

                train_loss = loss.item()
                scaler.scale(loss).backward()
                # nn.utils.clip_grad_norm_(model.parameters(), max_norm=20, norm_type=2)
                scaler.step(optimizer)
                scaler.update()
                # loss.backward()
                # optimizer.step()
                epoch_progress.set_postfix({'loss': train_loss})
                epoch_progress.update(1)
    
    model.eval()
    eval_train_correct = 0
    eval_train_total = 0

    preds = []
    preds_labels = []
    truths = []
    epoch_stats = {}
    test_loss = 0.0
    with autocast():
        with torch.no_grad():
            for inputs, labels in tqdm(test_loader):
                inputs = inputs.to(dtype=torch.int32, device=device)
                # inputs = z_score_normalize(inputs).to(device)

                outputs = model(inputs)
                labels = labels.to(device)

                test_loss += criterion(outputs, labels).item()
                probs = F.softmax(outputs, dim=1)
                y_pred = probs.argmax(1).detach()

                probs = torch.max(probs, 1)[1].detach().cpu().numpy().ravel()

                # 预测概率
                preds.extend(probs)
                # 预测标签
                preds_labels.extend(y_pred.cpu().numpy().ravel())
                # 真实标签
                truths.extend(labels.detach().cpu().numpy().ravel())

                eval_train_total += labels.size(0)
                eval_train_correct += (y_pred == labels).sum().item()
    epoch_stats['test_loss'] = test_loss / len(test_loader)
    epoch_stats['test_acc'] = eval_train_correct * 1.0 / eval_train_total
    epoch_stats['test_auc'] = roc_auc_score(truths, preds)
    epoch_stats['test_f1_score'] = f1_score(truths, preds_labels)
    epoch_stats['test_recall'] = recall_score(truths, preds_labels)
    from beeprint import pp
    pp(epoch_stats)

