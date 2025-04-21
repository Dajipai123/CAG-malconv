import os

import random
import numpy as np

# from tqdm import tqdm_notebook as tqdm
from sympy import Line2D
from tqdm import tqdm, trange

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

import torch.optim as optim

from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler

from binaryLoader import BinaryDataset, RandomChunkSampler, pad_collate_func
from sklearn.metrics import roc_auc_score,f1_score,recall_score,roc_curve

import argparse
import time 


# from model import MalEncoder
# from model_transformer import MalEncoder
from model_all import MalEncoder
# from model_baseline_cbam import MalEncoder
# from model_baseline_bigruattention import MalEncoder
# from model_baseline import MalEncoder

from beeprint import pp

from matplotlib import pyplot as plt
from sklearnex import patch_sklearn
patch_sklearn()
from sklearnex.manifold import TSNE
import datetime
import seaborn as sns

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
# parser.add_argument('--randseeds', type=int, default=11 , help='Number of random seeds')
parser.add_argument('--randseeds', type=int, default=29 , help='Number of random seeds')

# Default is set ot 16 MB!
parser.add_argument('--max_len', type=int, default=16000000,
                    help='Maximum length of input file in bytes, at which point files will be truncated')

parser.add_argument('--gpus', nargs='+', type=int)


parser.add_argument('--mal_train', default='./dataset/mal_train_10000', type=dir_path, help='Path to directory containing malware files for training')
parser.add_argument('--ben_train', default='./dataset/ben_train_10000', type=dir_path, help='Path to directory containing benign files for training')
parser.add_argument('--mal_test', default='./dataset/mal_test_2000', type=dir_path, help='Path to directory containing malware files for testing')
parser.add_argument('--ben_test', default='./dataset/ben_test_2000', type=dir_path, help='Path to directory containing benign files for testing')



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
    scatter = ax.scatter(data[:, 0], data[:, 1], data[:, 2], c=label, cmap=plt.colormaps.get_cmap("jet"), vmin=-0.5, vmax=1.5, s=4)
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


def plt_roc_curve(fpr,tpr,auc):
    
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % auc)

    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve on PE Machine learning dataset')
    plt.legend(loc="lower right")

    plt.savefig('/home/fhh/ember-master/MalConv2-main/pictures' +  'roc_curve' + '_epoch_' + str(epoch) + '.png')

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=["Predicted Negative", "Predicted Positive"],
                yticklabels=["Actual Negative", "Actual Positive"])
    plt.title("Confusion Matrix Heatmap")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()

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

    # 设置随机数种子
    RAND_SEEDS = args.randseeds
    torch.manual_seed(RAND_SEEDS)
    torch.cuda.manual_seed(RAND_SEEDS)
    torch.cuda.manual_seed_all(RAND_SEEDS)
    np.random.seed(RAND_SEEDS)
    random.seed(RAND_SEEDS)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False


    ben_valid = '/home/fhh/ember-master/MalConv2-main/dataset/ben_test_2000'
    mal_valid = '/home/fhh/ember-master/MalConv2-main/dataset/mal_test_2000'

    whole_dataset = BinaryDataset(args.ben_train, args.mal_train, sort_by_size=True, max_len=MAX_FILE_LEN)
    test_dataset = BinaryDataset(args.ben_test, args.mal_test, sort_by_size=True, max_len=MAX_FILE_LEN)
    valid_dataset = BinaryDataset(ben_valid, mal_valid, sort_by_size=True, max_len=MAX_FILE_LEN)

    # collate_fn 将一个list的sample组成一个mini-batch的函数，这里因为数据的长度不同因此自定义了方法
    # sampler 参数用于指定从数据集中抽取样本的策略。
    train_loader = DataLoader(whole_dataset, batch_size=BATCH_SIZE, num_workers=0, collate_fn=pad_collate_func,
                            sampler=RandomChunkSampler(whole_dataset, BATCH_SIZE))

    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, num_workers=0, collate_fn=pad_collate_func,
                            sampler=RandomChunkSampler(test_dataset, BATCH_SIZE))
    
    valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, num_workers=0, collate_fn=pad_collate_func,
                            sampler=RandomChunkSampler(valid_dataset, BATCH_SIZE))
    
    
    device = torch.device('cuda:1')
    model = MalEncoder().to(device)
    optimizer = optim.Adam(params=model.parameters(),lr=3e-4, weight_decay=3e-5)
    criterion = nn.CrossEntropyLoss().to(device)
    scaler = GradScaler()   # 混合精度训练，自动缩放损失和反向传播的梯度

    plt_train_loss = []
    plt_test_loss = []
    plt_train_acc = []
    plt_test_acc = []
    
    best_val_acc = 0.0
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=1, factor=0.2, verbose=True)

    for epoch in trange(EPOCHS):
        start_time = time.time()
        epoch_stats = {}
        train_correct = 0
        train_total = 0
        train_loss_num = 0.0
        train_total = 0
        train_preds = []
        train_truths = []
        
        with tqdm(total=len(train_loader), desc='Train') as epoch_progress:
            # train ###
            # 用来画散点图的
            train_out_list = []
            train_label_list = []
            
            model.train()

            for train_data, train_label in train_loader:
                optimizer.zero_grad()

                with autocast():
                    train_data = train_data.to(dtype=torch.int32, device=device)
                    out = model(train_data)

                    train_out_list.append(out.detach().cpu().numpy())
                    train_label_list.append(train_label.detach().cpu().numpy())
                    
                    # # 生成模型图
                    # dot = make_dot(out, params=dict(model.named_parameters()))
                    # dot.render(filename='model_graph', format='png' ,directory='/home/fhh/ember-master/MalConv2-main')
                    # exit()
                    
                    loss = criterion(out, train_label.to(device))
                # 用来计算loss，更新模型参数
                train_loss = loss.item()
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                epoch_progress.set_postfix({'loss': train_loss})
                epoch_progress.update(1)

                with torch.no_grad():
                    train_preds.extend(F.softmax(out, dim=-1).data[:, 1].detach().cpu().numpy().ravel())
                    train_total += train_label.size(0)
                    train_correct += (out.argmax(1) == train_label.to(device)).sum().item()
                    train_truths.extend(train_label.detach().cpu().numpy().ravel())
                
            epoch_stats['train_acc'] = train_correct * 1.0 / train_total
            epoch_stats['train_auc'] = roc_auc_score(train_truths, train_preds)
            # epoch_stats['train_loss'] = train_loss_num / len(train_loader)

            train_time = time.time() - start_time
            
            train_out_list = np.concatenate(train_out_list, axis=0)
            train_label_list = np.concatenate(train_label_list, axis=0)

            # # 降维并绘制散点图
            plt_scatter_output(train_out_list, train_label_list, 'train', epoch+1)
            

            ### valid ###
            model.eval()
            eval_valid_correct = 0
            eval_valid_total = 0

            preds = []
            preds_labels = []
            truths = []

            valid_loss = 0.0

            valid_outputs_list = []
            valid_labels_list = []

            with autocast():
                with torch.no_grad():
                    for inputs, labels in tqdm(valid_loader, desc='Valid'):
                        inputs = inputs.to(dtype=torch.int32, device=device)
                        
                        outputs = model(inputs)
                        labels = labels.to(device)

                        valid_loss += criterion(outputs, labels).item()
                        probs = F.softmax(outputs, dim=1)
                        y_pred = probs.argmax(1).detach()

                        probs = torch.max(probs, 1)[1].detach().cpu().numpy().ravel()

                        # 预测概率
                        preds.extend(probs)
                        # 预测标签
                        preds_labels.extend(y_pred.cpu().numpy().ravel())
                        # 真实标签
                        truths.extend(labels.detach().cpu().numpy().ravel())

                        eval_valid_total += labels.size(0)
                        eval_valid_correct += (y_pred == labels).sum().item()

            valid_acc = eval_valid_correct * 1.0 / eval_valid_total
            valid_f1 = f1_score(truths, preds_labels)

            lr_scheduler.step(valid_f1)

            # if valid_acc > best_val_acc:
            #     best_val_acc = valid_acc
            #     torch.save(model.state_dict(), '/home/fhh/ember-master/MalConv2-main/model_best.pth')


            ### Test ###
            model.eval()
            eval_test_correct = 0
            eval_test_total = 0

            test_preds = []
            preds_labels = []
            test_truths_labels = []

            test_loss = 0.0
    
            # test_outputs_list = []
            # test_labels_list = []
            with autocast():
                with torch.no_grad():
                    for inputs, labels in tqdm(test_loader, desc='Test'):
                        test_start = time.time()
                        inputs = inputs.to(dtype=torch.int32, device=device)

                        outputs = model(inputs)
                        labels = labels.to(device)

                        # test_outputs_list.append(outputs.detach().cpu().numpy())
                        # test_labels_list.append(labels.detach().cpu().numpy())

                        # test_porbs = F.softmax(outputs, dim=1)
                        # test_porbs = torch.max(test_porbs, 1)[1].detach().cpu().numpy().ravel()

                        test_loss += criterion(outputs, labels).item()
                        
                        test_pred = outputs.argmax(1).detach()


                        # 预测概率
                        test_preds.extend(F.softmax(outputs, dim=-1).data[:, 1].detach().cpu().numpy().ravel())
                        # test_preds.extend(test_porbs)
                        # 预测标签
                        preds_labels.extend(test_pred.cpu().numpy().ravel())
                        # 真实标签
                        test_truths_labels.extend(labels.detach().cpu().numpy().ravel())

                        eval_test_total += labels.size(0)
                        eval_test_correct += (test_pred == labels).sum().item()

                        # save the model
                        # torch.save(model.state_dict(), f'/home/fhh/ember-master/MalConv2-main/models/ransomware_models/bigdataset/model_{epoch}.pth')


            # epoch_stats['test_loss'] = test_loss / len(test_loader)
            epoch_stats['test_acc'] = eval_test_correct * 1.0 / eval_test_total
            epoch_stats['test_auc'] = roc_auc_score(test_truths_labels, test_preds)
            fpr ,tpr , thresholds = roc_curve(test_truths_labels, test_preds)
            epoch_stats['test_f1_score'] = f1_score(test_truths_labels, preds_labels)
            epoch_stats['test_recall'] = recall_score(test_truths_labels, preds_labels)

            pp(epoch_stats)

            cost_time = time.time() - test_start + train_time 

    #         # 将评估值输出保存到文件
    #         with open('/home/fhh/ember-master/MalConv2-main/csv_result/randsomeware_mymodels.csv', 'a') as file:
    #             file.write(f"epoch:{epoch} , cost_time:{cost_time} , " + str(epoch_stats) + " , " + str(datetime.datetime.now()) + "\n")

    #         print("Epoch: {} took {} seconds".format(epoch, cost_time))

    #         # with open('/home/fhh/ember-master/MalConv2-main/fpr_tpr_auc.csv', 'a') as file:
    #         #     file.write(f"epoch:{epoch} , fpr:{fpr} , tpr:{tpr} , auc:{epoch_stats['test_auc']} , " + str(datetime.datetime.now()) + "\n")


            
    #         # plt_train_loss.append(epoch_stats['train_loss'])
    #         # plt_test_loss.append(epoch_stats['test_loss'])
    #         # plt_train_acc.append(epoch_stats['train_acc'])
    #         # plt_test_acc.append(epoch_stats['test_acc'])

    #         # 降维并绘制散点图
    #         # test_outputs_list = np.concatenate(test_outputs_list, axis=0)
    #         # test_labels_list = np.concatenate(test_labels_list, axis=0)
    #         # plt_scatter_output(test_outputs_list, test_labels_list, 'test', epoch+1)



    # # 绘制损失和准确率曲线
    # # plt.figure(figsize=(10, 5))

    # # plt.plot(range( 1, EPOCHS+1), plt_train_loss, label='train_loss', color='blue')
    # # plt.plot(range( 1, EPOCHS+1), plt_test_loss, label='test_loss', color='red')
    # # plt.plot(range( 1, EPOCHS+1), plt_train_acc, label='train_acc', color='green')
    # # plt.plot(range( 1, EPOCHS+1), plt_test_acc, label='test_acc', color='purple')
    # # plt.legend()
    # # plt.xlabel('epoch')
    # # plt.ylabel('loss/Accuracy')
    # # plt.title('Loss and Accuracy')
    # # plt.savefig(f'/home/fhh/ember-master/MalConv2-main/loss_acc_{RAND_SEEDS}.png')
    # # plt.close()


    

