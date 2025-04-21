import numpy as np
import torch
from model_all import MalEncoder
from MalConvGCT_nocat import MalConvGCT
import os
import csv
from tqdm import tqdm
import matplotlib.pyplot as plt

model_path = r'/home/fhh/ember-master/MalConv2-main/models/ransomware_models/model_29.pth'
malware_sample = r'/home/fhh/ember-master/MalConv2-main/dataset/silver_fox_182_mixed_to_200/3b89a5f34763a2d7e42c44dc67b0c60f459965d7463af7b3341c2f2a1e75d0de'
malware_path = r'/home/fhh/ember-master/MalConv2-main/dataset/test/'
ben_path = r'/home/fhh/ember-master/mytest/ben_200'
ben_result_csv = r'/home/fhh/ember-master/MalConv2-main/csv_result/rawmodel_ben_result.csv'
mal_result_csv = r'/home/fhh/ember-master/MalConv2-main/csv_result/rawmodel_silver_result.csv'


def process_binary_file(file_path, max_len=16000000):
    # 读取文件的原始字节
    with open(file_path, 'rb') as f:
        bytes_data = f.read(max_len)
        bytes_array = np.frombuffer(bytes_data, dtype=np.uint8).astype(np.int16) + 1  # index 0 will be special padding index
            
    # 转换为 torch 张量
    tensor_data = torch.tensor(bytes_array)
    
    return tensor_data


def predict(malware_sample,csv_path):

    # 读取样本
    tensor_data = process_binary_file(malware_sample).unsqueeze(0).to(torch.long).to(torch.device('cuda:1'))
    # model.gru.gru.register_forward_hook(hook_fn)
    
    # 预测
    with torch.no_grad():
        output = model(tensor_data)
        # activation = model.gru.gru.activation
        pred = torch.sigmoid(output).tolist()[0][1]
        if pred >= 0.5:
            tag = 1
        else:
            tag = 0
        sample_name = os.path.basename(malware_sample)
        print(f"Sample: {sample_name}, Prediction: {pred}, Tag: {tag}")
        # with open(csv_path, 'a', newline='') as f:
        #     writer = csv.writer(f)
        #     writer.writerow([sample_name,tag,pred])

        # return activation

def hook_fn(module, input, output):
    module.activation = output[0].detach()


if __name__ == '__main__':
    # 加载模型
    # model = MalConvGCT()
    model = MalEncoder()
    # model.conv1.register_forward_hook(hook_fn)
    checkpoint = torch.load(model_path, map_location=torch.device('cuda:1'))
    model.load_state_dict(checkpoint)
    model.eval()
    model.to(torch.device('cuda:1'))

    # # 写文件头
    # with open(mal_result_csv, 'a', newline='') as f:
    #     writer = csv.writer(f)
    #     writer.writerow(['sample','tag','score'])

    # 遍历恶意样本
    for root, dirs, files in os.walk(malware_path):
        for file in files:
            predict(os.path.join(root, file),mal_result_csv)

    # 写文件头
    with open(ben_result_csv, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['sample','tag','score'])
    
    # # 遍历良性样本
    # for root, dirs, files in os.walk(ben_path):
    #     for file in files:
    #         predict(os.path.join(root, file),ben_result_csv)

    print('Done!')

    # pre_activation = predict(malware_sample)
    # print(pre_activation.shape)
    # plt.plot(pre_activation[0].cpu().numpy())
    # plt.savefig('/home/fhh/ember-master/MalConv2-main/csv_result/activation.png')



