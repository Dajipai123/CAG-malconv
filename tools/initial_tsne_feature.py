import os
import random
import numpy as np
from tqdm import tqdm, trange
import torch
from binaryLoader import BinaryDataset, RandomChunkSampler, pad_collate_func
from sklearnex import patch_sklearn
from sklearnex.manifold import TSNE
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader

patch_sklearn()

# Set the paths to the directories containing the malware and benign files
mal_train = '/home/fhh/ember-master/MalConv2-main/dataset/mal_test_2000'
ben_train = '/home/fhh/ember-master/MalConv2-main/dataset/ben_test_2000'

# Create the BinaryDataset object
dataset = BinaryDataset(ben_train, mal_train, sort_by_size=True)

# Create the DataLoader object
loader = DataLoader(dataset, batch_size=32, num_workers=0, collate_fn=pad_collate_func,
                    sampler=RandomChunkSampler(dataset, 32))

# 获取所有数据
data = []
labels = []
for data_,labels_ in tqdm(loader):
    data.append(data_)
    labels.append(labels_)

# # 获取最小的维度
min_dim = min([x.shape[1] for x in data])

print(min_dim)

# 将所有数据截断到最小的维度
for i in range(len(data)):
    data[i] = data[i][:,:min_dim]

# # 将所有数据转换为numpy数组
data = [x.numpy() for x in data]
labels = [x.numpy() for x in labels]

# 堆叠
data = np.vstack(data)
labels = np.hstack(labels)

# Perform t-SNE dimensionality reduction
tsne = TSNE(n_components=2, random_state=15)
data_tsne = tsne.fit_transform(data)

# 缩放到0-1
data_tsne = (data_tsne - data_tsne.min()) / (data_tsne.max() - data_tsne.min())

# Plot the scatter plot
plt.figure(figsize=(10, 10))
plt.scatter(data_tsne[:, 0], data_tsne[:, 1], c=labels, cmap=plt.colormaps.get_cmap("jet"))
plt.colorbar(ticks=range(2))
plt.clim(-0.5, 1.5)
# plt.title('t-SNE Scatter Plot')
plt.savefig('/home/fhh/ember-master/MalConv2-main/pictures/t-SNE_Plot.png')
print('t-SNE Scatter Plot.png saved')

