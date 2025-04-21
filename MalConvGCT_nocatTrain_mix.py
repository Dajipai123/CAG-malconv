import os
from collections import deque

import random
import numpy as np

# from tqdm import tqdm_notebook as tqdm
from tqdm import tqdm
import multiprocessing

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

import torch.optim as optim

from torch.utils import data

from torch.utils.data import Dataset, DataLoader, Subset

# from MalConv import MalConv
from MalConvGCT_nocat import MalConvGCT

from binaryLoader import BinaryDataset, RandomChunkSampler, pad_collate_func
from sklearn.metrics import roc_auc_score,f1_score,recall_score

import argparse
import lightgbm as lgb

import time

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
parser.add_argument('--batch_size', type=int, default=128, help='Batch size during training')
# Default is set ot 16 MB!
parser.add_argument('--max_len', type=int, default=16000000,
                    help='Maximum length of input file in bytes, at which point files will be truncated')

parser.add_argument('--gpus', nargs='+', type=int)


parser.add_argument('mal_train', type=dir_path, help='Path to directory containing malware files for training')
parser.add_argument('ben_train', type=dir_path, help='Path to directory containing benign files for training')
parser.add_argument('mal_test', type=dir_path, help='Path to directory containing malware files for testing')
parser.add_argument('ben_test', type=dir_path, help='Path to directory containing benign files for testing')

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

# 不传入参数则使用cuda:0，如果传入list，则使用list中的第一个GPU，如果传入-1，则使用CPU
if GPUS is None:  # use ALL of them! (Default)
    device_str = "cuda:0"
else:
    if GPUS[0] < 0:
        device_str = "cpu"
    else:
        device_str = "cuda:{}".format(GPUS[0])

device = torch.device(device_str if torch.cuda.is_available() else "cpu")
print(device)
model = MalConvGCT(channels=NUM_CHANNELS, window_size=FILTER_SIZE, stride=FILTER_STRIDE, embd_size=EMBD_SIZE,
                   low_mem=False, device=device).to(device)

base_name = "nocat_{}_channels_{}_filterSize_{}_stride_{}_embdSize_{}".format(
    type(model).__name__,
    NUM_CHANNELS,
    FILTER_SIZE,
    FILTER_STRIDE,
    EMBD_SIZE,
)

if NON_NEG:
    base_name = "NonNeg_" + base_name

# if GPUS is None or len(GPUS) > 1:
#     model = nn.DataParallel(model, device_ids=GPUS)

if not os.path.exists(base_name):
    os.makedirs(base_name)
file_name = os.path.join(base_name, base_name)

headers = ['epoch', 'train_acc', 'train_auc', 'test_acc', 'test_auc', 'test_f1_score', 'test_recall']

csv_log_out = open(file_name + ".csv", 'w')
csv_log_out.write(",".join(headers) + "\n")

criterion = nn.CrossEntropyLoss()
# optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
optimizer = optim.Adam(model.parameters())

def get_intermediate_activations(model, data_loader):
    model.eval()
    penultimate_activations = []
    conv_activations = []
    labels = []
    with torch.no_grad():
        for inputs, label in tqdm(data_loader):
            # inputs, labels = inputs.to(device), labels.to(device)
            # Keep inputs on CPU, the model will load chunks of input onto device as needed
            labels.append(label.cpu())
            _, penultimate_activ, conv_active = model(inputs)
            penultimate_activations.append(penultimate_activ.cpu())
            conv_activations.append(conv_active.cpu())
    penultimate_activations = torch.cat(penultimate_activations).numpy()
    conv_activations = torch.cat(conv_activations).numpy()
    labels = torch.cat(labels).numpy()
    return penultimate_activations, conv_activations, labels

# params = {
#     'device': 'cpu',
#     'boosting_type': 'gbdt',
#     'objective': 'binary',
#     'metric': 'auc',
#     'num_leaves': 31,
#     'learning_rate': 0.05,
#     'feature_fraction': 0.9,
#     'bagging_fraction': 0.8,
#     'bagging_freq': 5,
#     'verbose': 1
# }

params = {
    'device': 'cpu',
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metric': 'auc',
    'num_leaves': 31,
    'learning_rate': 0.1,
    'feature_fraction': 1.0,
    'bagging_fraction': 0.7,
    'bagging_freq': 5,
    'verbose': 1
}

train_acc_list = []
bst = None  # Initialize the LightGBM model


for epoch in tqdm(range(EPOCHS)):
    start_time = time.time()

    train_data, train_conv, train_label = get_intermediate_activations(model, train_loader)

    from sklearnex import patch_sklearn
    patch_sklearn()
    from sklearn.ensemble import RandomForestClassifier

    train_time = time.time()
    print(f'RandomForestClassifier Train...', end='\r')
    rfc = RandomForestClassifier(n_estimators=100)
    rfc.fit(train_data, train_label)
    print(f'RandomForestClassifier Train costs {time.time()-train_time:.1f} s.')

    train_time = time.time()
    print(f'LightGBMClassifier Train...', end='\r')
    if bst is None:
        bst = lgb.train(params, lgb.Dataset(train_conv, train_label), 100)
    else:
        bst = lgb.train(params, lgb.Dataset(train_conv, train_label), 100, init_model=bst)
    print(f'LightGBMClassifier Train costs {time.time()-train_time:.1f} s.')

    test_data, test_conv, test_label = get_intermediate_activations(model, test_loader)

    test_time = time.time()
    print(f'RandomForestClassifier Predict...', end='\r')
    y_pred = rfc.predict_log_proba(test_data)
    print(f'RandomForestClassifier Predict costs {time.time()-test_time:.1f} s.')
    ypred_fin = np.argmax(y_pred, axis=1)
    rfc_acc = np.mean(ypred_fin == test_label)
    rfc_auc = roc_auc_score(test_label, y_pred)
    rfc_f1 = f1_score(test_label, ypred_fin)
    rfc_recall = recall_score(test_label, ypred_fin)

    # model prediction, predict the probability of each sample
    test_time = time.time()
    print(f'LightGBMClassifier Predict...', end='\r')
    y_pred = bst.predict(test_conv)
    print(f'LightGBMClassifier Predict costs {time.time()-test_time:.1f} s.')

    # Calculate test_acc
    ypred_fin = np.where(y_pred > 0.5, 1, 0)
    bst_acc = np.mean(ypred_fin == test_label)
    bst_auc = roc_auc_score(test_label, y_pred)
    bst_f1 = f1_score(test_label, ypred_fin)
    bst_recall = recall_score(test_label, ypred_fin)

    csv_log_out.write("{},{},{},{},{},{},{},{},{}\n".format(epoch, rfc_acc, rfc_auc, rfc_f1, rfc_recall, bst_acc, bst_auc, bst_f1, bst_recall))
    csv_log_out.flush()
    print("Epoch {} took {} seconds".format(epoch, time.time() - start_time))

csv_log_out.close()











        

        
        




