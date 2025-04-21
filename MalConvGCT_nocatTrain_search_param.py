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
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer
from sklearn.metrics import accuracy_score


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

pretrained_model_path = "/home/fhh/ember-master/MalConv2-main/best_trick.checkpoint"
checkpoint = torch.load(pretrained_model_path)
model.load_state_dict(checkpoint['model_state_dict'])

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
            labels.append(label)
            outputs, penultimate_activ, conv_active = model(inputs)
            penultimate_activations.append(penultimate_activ)
            conv_activations.append(conv_active)
    return penultimate_activations, conv_activations, labels

params = {
    'device': 'cpu',
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metric': 'auc',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': 1
}

train_acc_list = []
bst = None  # Initialize the LightGBM model

penultimate_activations, conv_activations, labels = get_intermediate_activations(model, test_loader)

# Move the data to CPU
penultimate_activations = [activations.cpu() for activations in penultimate_activations]
conv_activations = [activations.cpu() for activations in conv_activations]
labels = [label.cpu() for label in labels]

test_data = np.concatenate(penultimate_activations, axis=0)
test_label = np.concatenate(labels, axis=0)
test_conv = np.concatenate(conv_activations, axis=0)

# Define the LightGBM classifier
lgb_model = lgb.LGBMClassifier()

# Define the parameter grid for grid search
param_grid = {
    'num_leaves': [15, 31, 63],
    'learning_rate': [0.01, 0.05, 0.1],
    'feature_fraction': [0.8, 0.9, 1.0],
    'bagging_fraction': [0.7, 0.8, 0.9],
    'bagging_freq': [3, 5, 7],
    # 'max_depth': [5, 10, 15],
}

# Perform cross-validation and grid search
grid_search = GridSearchCV(estimator=lgb_model, param_grid=param_grid, scoring='accuracy', cv=5, verbose=1)
grid_search.fit(test_data, test_label)


# Print the best parameters and the corresponding ROC AUC score
print("Best parameters: ", grid_search.best_params_)
print("Best ROC AUC score: ", grid_search.best_score_)


 








        

        
        




