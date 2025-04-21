import numpy as np
import torch
from model_all import MalEncoder
import os
import csv
from tqdm import tqdm
import matplotlib.pyplot as plt

model_path = '/home/fhh/ember-master/MalConv2-main/models/model_10.pth'
malware_sample = '/home/fhh/ember-master/MalConv2-main/dataset/malconv2_mal_train_10000/12011'
ben_sample = '/home/fhh/ember-master/MalConv2-main/dataset/ben_test_2000/192804'
test_sample = '/home/fhh/ember-master/MalConv2-main/dataset/silver_fox_182_mixed_to_200/VirusShare_dac868a6afd948df8a5720205efeb132'

def process_binary_file(file_path, max_len=16000000):
    # 读取文件的原始字节
    with open(file_path, 'rb') as f:
        bytes_data = f.read(max_len)
        bytes_array = np.frombuffer(bytes_data, dtype=np.uint8).astype(np.int16) + 1  # index 0 will be special padding index
            
    # 转换为 torch 张量
    tensor_data = torch.tensor(bytes_array)
    
    return tensor_data

def predict(model,tensor_data,csv_path = None):
    # 预测
    with torch.no_grad():
        output = model(tensor_data)
        # 获取激活
        # activation = model.conv.activation
        activation = model.gru.activation
        pred = torch.sigmoid(output).tolist()[0]
        # if pred >= 0.5:
        #     tag = 1
        # else:
        #     tag = 0
        print(pred[1])
        return activation


def hook_fn(module, input, output):
    module.activation = output


# 对激活进行上采样 
def upsample_activation(activation, size):
    # 从激活中获取每个字节的激活
    activation = activation.mean(0).cpu().numpy()

    # 通过线性插值将激活上采样到原始字节大小
    upsampled_activation = np.interp(np.linspace(0, len(activation), size), np.arange(len(activation)), activation)

    return upsampled_activation

# 可视化激活，横轴是字节，纵轴是激活
def visualize_activation(activation):
    plt.figure(figsize=(20, 5))
    plt.plot(activation)
    plt.savefig('/home/fhh/ember-master/MalConv2-main/pictures/activation.png')



if __name__ == '__main__':
    # 加载模型
    model = MalEncoder()
    model.eval()
    model.to(torch.device('cuda:1'))
    checkpoint = torch.load(model_path, map_location=torch.device('cuda:1'))
    model.load_state_dict(checkpoint)
    
    # 注册钩子
    # model.conv.register_forward_hook(hook_fn)
    model.gru.register_forward_hook(hook_fn)

    # 处理文件
    tensor_data = process_binary_file(malware_sample).unsqueeze(0).to(torch.long)
    tensor_data = tensor_data.to(torch.device('cuda:1'))
    print(tensor_data.shape)

    # 获取激活
    activation = predict(model, tensor_data)
    print(activation[0].shape)
    # exit()

    # 激活进行上采样
    upsampled_activation = upsample_activation(activation, tensor_data.shape[-1])
    # 数值缩放到0-1
    upsampled_activation = (upsampled_activation - upsampled_activation.min()) / (upsampled_activation.max() - upsampled_activation.min())
    print(upsampled_activation.shape)

    # 可视化
    visualize_activation(upsampled_activation)

