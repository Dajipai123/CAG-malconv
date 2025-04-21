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
from sklearn.metrics import roc_auc_score,f1_score,recall_score,accuracy_score

import argparse
import time

from beeprint import pp
import datetime

from sklearnex.manifold import TSNE
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

# Check if the input is a valid directory
def dir_path(string):
    if os.path.isdir(string):
        return string
    else:
        raise NotADirectoryError(string)

def plt_scatter_output(data , label , name , epoch):

    # # 降维
    # data = TSNE(n_components=2, random_state=15).fit_transform(data)

    # # 缩放到0-1
    # data = (data - data.min()) / (data.max() - data.min())

    # # 绘制散点图
    # plt.figure(figsize=(10, 10))
    # plt.scatter(data[:, 0], data[:, 1], c=label, cmap=plt.colormaps.get_cmap("jet"),s=5)
    # plt.colorbar(ticks=range(2))
    # plt.clim(-0.5, 1.5)
    # # plt.title(name)
    # plt.savefig('/home/fhh/ember-master/MalConv2-main/pictures/Cluster-3-13-malconv2/' + 'epoch' + str(epoch) + '.png')

    # 创建TSNE-3D图
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # 降维
    data = TSNE(n_components=3, random_state=15).fit_transform(data)

    # 缩放到0-1
    data = (data - data.min()) / (data.max() - data.min())

    # # 根据标签进行颜色区分
    # ben_data = data[label == 0]
    # mal_data = data[label == 1]

    # # 使用不同的颜色和标记来表示不同的类别，并添加标签
    # ax.scatter(ben_data[:, 0], ben_data[:, 1], ben_data[:, 2], c='dodgerblue', marker='o', label='Benign')
    # ax.scatter(mal_data[:, 0], mal_data[:, 1], mal_data[:, 2], c='salmon', marker='^', label='Malicious')


    # 混合绘制散点图
    scatter = ax.scatter(data[:, 0], data[:, 1], data[:, 2], c=label, cmap=plt.colormaps.get_cmap("jet"), vmin=-0.5, vmax=1.5, s=5)
    plt.colorbar(scatter, ticks=range(2))

    # 确保label中包含所有想要在图例中显示的值
    label = np.append(label, [0, 1])

    # 获取散点图的颜色
    colors = scatter.to_rgba(np.unique(label))

    # 创建一个代表每种颜色的标签
    legend_elements = [Line2D([0], [0], marker='o', color='w', markerfacecolor=clr, markersize=10) for clr in colors]

    # 加上图例
    ax.legend(handles=legend_elements, loc='best', labels=["Benign", "Malicious"])


    # 调整视角
    ax.view_init(elev=20, azim=45)

    # 设置三个轴的坐标范围为0-1
    # ax.set_xlim([0, 1])
    # ax.set_ylim([0, 1])
    # ax.set_zlim([0, 1])

    plt.savefig('/home/fhh/ember-master/MalConv2-main/pictures/Cluster-3-18-test/' + 'epoch' + str(epoch) + '.pdf')


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

# headers = ['epoch', 'train_acc', 'train_auc', 'test_acc', 'test_auc', 'test_f1_score', 'test_recall']

# csv_log_out = open(file_name + ".csv", 'w')
# csv_log_out.write(",".join(headers) + "\n")

criterion = nn.CrossEntropyLoss()
# optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
optimizer = optim.Adam(model.parameters())

for epoch in tqdm(range(EPOCHS)):
    start_time = time.time()

    train_preds = []
    train_truths = []
    running_loss = 0.0

    train_correct = 0
    train_total = 0

    epoch_stats = {}

    ## 用来画散点图的
    train_out_list = []
    train_label_list = []

    model.train()
    for inputs, labels in tqdm(train_loader):

        # inputs, labels = inputs.to(device), labels.to(device)
        # Keep inputs on CPU, the model will load chunks of input onto device as needed
        labels = labels.to(device)

        optimizer.zero_grad()

        #     outputs, penultimate_activ, conv_active = model.forward_extra(inputs)
        outputs, penultimate_activ, conv_active = model(inputs.to(device))

        # 收集数据，用来画散点图
        train_out_list.append(penultimate_activ.detach().cpu().numpy())
        train_label_list.append(labels.detach().cpu().numpy())

        loss = criterion(outputs, labels)
        # loss = loss   + decov_lambda*(decov_penalty(penultimate_activ) + decov_penalty(conv_active))
        #     loss = loss + decov_lambda*(decov_penalty(conv_active))
        loss.backward()
        optimizer.step()
        if NON_NEG:
            for p in model.parameters():
                p.data.clamp_(0)

        # running_loss += loss.item()
        # # predicted = outputs.data.argmax(1)
        # _, predicted = torch.max(outputs.data, 1)

        # with torch.no_grad():
        #     train_preds.extend(F.softmax(outputs, dim=-1).data[:, 1].detach().cpu().numpy().ravel())
        #     train_truths.extend(labels.detach().cpu().numpy().ravel())

        # train_total += labels.size(0)
        # train_correct += (predicted == labels).sum().item()
    train_out_list = np.concatenate(train_out_list, axis=0)
    train_label_list = np.concatenate(train_label_list, axis=0)

    # 画散点图
    plt_scatter_output(train_out_list, train_label_list, 'train', epoch)
    # print("Training Accuracy: {}".format(train_correct*100.0/train_total))

    # epoch_stats['train_acc'] = train_correct * 1.0 / train_total
    # epoch_stats['train_auc'] = roc_auc_score(train_truths, train_preds)  
    # epoch_stats['f1_score'] = f1_score(train_truths, train_preds)
    # epoch_stats['recall'] = recall_score(train_truths, train_preds)
    # epoch_stats['train_loss'] = roc_auc_score(train_truths, train_preds)

    # # Save the model and current state!
    # model_path = os.path.join(base_name, "epoch_{}.checkpoint".format(epoch))

    # # Have to handle model state special if multi-gpu was used
    # if type(model).__name__ is "DataParallel":
    #     mstd = model.module.state_dict()
    # else:
    #     mstd = model.state_dict()

    # torch.save({
    #     'epoch': epoch,
    #     'model_state_dict': mstd,
    #     'optimizer_state_dict': optimizer.state_dict(),
    #     'channels': NUM_CHANNELS,
    #     'filter_size': FILTER_SIZE,
    #     'stride': FILTER_STRIDE,
    #     'embd_dim': EMBD_SIZE,
    #     'non_neg': NON_NEG,
    # }, model_path)

    # # Test Set Eval
    # model.eval()
    # eval_train_correct = 0
    # eval_train_total = 0

    # test_preds = []
    # test_preds_labels = []
    # truths = []

    # with torch.no_grad():
    #     for inputs, labels in tqdm(test_loader):
    #         inputs, labels = inputs.to(device), labels.to(device)

    #         outputs, penultimate_activ, conv_active = model(inputs)

    #         _, predicted = torch.max(outputs.data, 1)
            
    #         # test_probs = F.softmax(outputs, dim=1)
    #         # test_probs = torch.max(test_probs, 1)[1].detach().cpu().numpy().ravel()

    #         # 预测概率
    #         test_preds.extend(F.softmax(outputs, dim=-1).data[:, 1].detach().cpu().numpy().ravel())
    #         # test_preds.extend(test_probs)
    #         # 预测标签
    #         test_preds_labels.extend(predicted.detach().cpu().numpy().ravel())
    #         # 真实标签
    #         truths.extend(labels.detach().cpu().numpy().ravel())

    #         eval_train_total += labels.size(0)
    #         eval_train_correct += (predicted == labels).sum().item()

    #         # 保存模型
    #         # torch.save(model.state_dict(), f'/home/fhh/ember-master/MalConv2-main/models/raw_model_check_{epoch}.pth')

    # epoch_stats['test_acc'] = accuracy_score(truths, test_preds_labels)
    # epoch_stats['test_auc'] = roc_auc_score(truths, test_preds)
    # epoch_stats['test_f1_score'] = f1_score(truths, test_preds_labels)
    # epoch_stats['test_recall'] = recall_score(truths, test_preds_labels)
    
    # pp(epoch_stats)
    
    # # 将输出保存到文件
    # with open('/home/fhh/ember-master/MalConv2-main/csv_result/randsomeware_rawmodel.csv', 'a') as file:
    #     file.write(f"epoch:{epoch} , cost_time:{time.time() - start_time} , " +str(epoch_stats) + " , " + str(datetime.datetime.now()) + "\n")

    # # csv_log_out.write(",".join([str(epoch_stats[h]) for h in headers]) + "\n")
    # # csv_log_out.flush()
    # print("Epoch: {} took {} seconds".format(epoch, time.time() - start_time))

# csv_log_out.close()




