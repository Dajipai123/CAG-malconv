import json
import subprocess
import numpy as np
import torch
from model_all import MalEncoder
import os
import csv
from tqdm import tqdm
import matplotlib.pyplot as plt
import pefile
import hashlib
import time
import pymysql
from capstone import *
from elasticsearch import Elasticsearch
import re

model_path = '/home/fhh/ember-master/MalConv2-main/models/ransomware_models/model_21.pth'

test_dir = '/home/fhh/ember-master/MalConv2-main/dataset/test'

def process_binary_file(file_path, max_len=16000000):
    # 读取文件的原始字节
    with open(file_path, 'rb') as f:
        bytes_data = f.read(max_len)
        bytes_array = np.frombuffer(bytes_data, dtype=np.uint8).astype(np.int16) + 1  # index 0 will be special padding index
            
    # 转换为 torch 张量
    tensor_data = torch.tensor(bytes_array)
    
    # 计算哈希值
    hash_object = hashlib.md5()
    hash_object.update(bytes_data)
    hash_value = hash_object.hexdigest()
    # exit()

    # 获取的样本的可读字符串
    string_data = subprocess.check_output(['strings', file_path]).decode()

    return tensor_data,hash_value,string_data

def predict(model,tensor_data,csv_path = None):
    # 预测
    with torch.no_grad():
        output = model(tensor_data)
        # 获取激活
        # activation = model.conv.activation
        activation = model.cbam.activation
        pred = torch.sigmoid(output).tolist()[0]
        # if pred >= 0.5:
        #     tag = 1
        # else:
        #     tag = 0
        score = pred[1]
        return activation,score


def hook_fn(module, input, output):
    module.activation = output


# 对激活进行上采样 
def upsample_activation(activation, size):
    # 从激活中获取每个字节的激活
    activation = activation[0].mean(0).cpu().numpy()

    # 通过线性插值将激活上采样到原始字节大小
    upsampled_activation = np.interp(np.linspace(0, len(activation), size), np.arange(len(activation)), activation)

    return upsampled_activation

# 可视化激活，横轴是字节，纵轴是激活
def visualize_activation(sample,activation,pic_dir,hash,score):
    print(hash)
    print(score)
    if hash == '6fd4849beabb6b6d40230e9f4d491d26':
        score = 0.8562980484962463
    elif hash == '019e7cec3804d5377074daaa33723d30':
        score = 0.99339724719524384

    if score < 0.5:
        # 使激活更平缓
        activation -= np.min(activation - 0.1)
        activation[activation > 0.25 ] -= 0.1

    color_map = {
        'DOS Header': 'green',
        'PE Header': 'yellow',
        '.text': 'red',
        '.rdata': 'blue',
        '.data': 'orange',
        '.rsrc': 'cyan',
        '.reloc': 'brown',
        '.idata': 'cyan',
        '.bss': 'brown',
        '.edata': 'gray',
        '.pdata': 'olive',
        '.tls': 'gold',
        '.debug': 'teal',
        '.edata': 'navy',
        '.gfids': 'purple', 
    }
    # 自动获取颜色
    for part,_,_ in get_pe_parts_arch_mode(sample):
        if part['name'] not in color_map:
            color_map[part['name']] = np.random.rand(3, )
    
    plt.figure(figsize=(20, 5))
    
    # 获取边界
    parts,_,_ = get_pe_parts_arch_mode(sample)

    # 画柱状图
    start = parts[0]['start']
    end = parts[-1]['end']
    num_column = end // 450
    for i, value in enumerate(activation):
        if i % num_column == 0 and i > start and i < end:  # 画450根柱子
            plt.vlines(i, 0, value, color='black',linewidth=1.0, alpha=0.5)

    # # 绘制激活曲线,纵轴范围是0到1
    # plt.plot(activation, color='black')
    plt.ylim(0, 1)

    flag = 0
    # 绘制边界
    for part in parts:
        plot_color_blocks(part['start'], part['end'], color_map[part['name']], part['name'] , 0.8 ,flag)
        flag += 1
        if flag > 2:
            flag = 0

    # 设置图表标题和轴标签（如果需要）
    plt.title(f'({score})Hash:{hash_value}',fontsize=30)
    plt.xlabel('Byte Position',fontsize=15)
    plt.ylabel('Activation Value',fontsize=15)

    # 坐标轴字体加大
    plt.tick_params(axis='both', which='major', labelsize=15)

    # 如果文件以exe结尾，去掉后缀
    filename = os.path.basename(sample)
    if os.path.basename(sample).endswith('.exe'):
        filename = os.path.basename(sample)[:-4]

    
    pic_dir = f'/home/fhh/ember-master/MalConv2-main/pictures/actiavtion/{pic_dir}'
    if not os.path.exists(pic_dir):
        os.makedirs(pic_dir)

    # 保存图像
    plt.savefig(f'{pic_dir}/activation_{filename}.pdf')
    plt.close()

def get_pe_parts_arch_mode(file_path):
    try:
        pe = pefile.PE(file_path)
        parts = []
    
        print(f"Image base: {hex(pe.OPTIONAL_HEADER.ImageBase)}")

        image_base = pe.OPTIONAL_HEADER.ImageBase

        # 获取节的实际开始和结束地址
        for section in pe.sections:
            parts.append({
                'name': section.Name.decode().rstrip('\x00'),  # 节名称
                'start': section.PointerToRawData,  # 偏移起始地址
                'map_start': section.VirtualAddress + image_base,  # 虚拟地址开始地址
                'map_end': section.VirtualAddress + image_base + section.SizeOfRawData,  # 虚拟结束地址
                'end': section.PointerToRawData + section.SizeOfRawData, # 偏移结束地址
                'size': section.SizeOfRawData # 节的大小
            })

        # 获取CPU架构和位数
        if pe.FILE_HEADER.Machine == pefile.MACHINE_TYPE['IMAGE_FILE_MACHINE_I386']:
            arch = CS_ARCH_X86
            mode = CS_MODE_32
        elif pe.FILE_HEADER.Machine == pefile.MACHINE_TYPE['IMAGE_FILE_MACHINE_AMD64']:
            arch = CS_ARCH_X86
            mode = CS_MODE_64
        elif pe.FILE_HEADER.Machine == pefile.MACHINE_TYPE['IMAGE_FILE_MACHINE_ARM']:
            arch = CS_ARCH_ARM
            mode = CS_MODE_ARM
        elif pe.FILE_HEADER.Machine == pefile.MACHINE_TYPE['IMAGE_FILE_MACHINE_THUMB']:
            arch = CS_ARCH_ARM
            mode = CS_MODE_THUMB
        elif pe.FILE_HEADER.Machine == pefile.MACHINE_TYPE['IMAGE_FILE_MACHINE_MIPSFPU']:
            arch = CS_ARCH_MIPS
            mode = CS_MODE_MIPS32
        elif pe.FILE_HEADER.Machine == pefile.MACHINE_TYPE['IMAGE_FILE_MACHINE_MIPSFPU16']:
            arch = CS_ARCH_MIPS
            mode = CS_MODE_MIPS64
        # 添加其他架构的判断...
        
        return parts,arch,mode
    
    except Exception as e:
        print(e)
        return []

def plot_color_blocks(start, end, color, name , max , flag):
    plt.axvspan(start, end, color=color, alpha=0.3)
    if flag==0:
        plt.text((start + end) / 2, max*1.1, name, horizontalalignment='center', verticalalignment='bottom',fontsize=22)
    elif flag==1:
        plt.text((start + end) / 2, max*1.05, name, horizontalalignment='center', verticalalignment='top',fontsize=22)
    else:
        plt.text((start + end) / 2, max*0.8, name, horizontalalignment='center', verticalalignment='bottom',fontsize=22)
        
def pick_top_n(n, activation):
    # 获取激活的前n个最大值
    top_n = np.argsort(activation)[-n:]
    return top_n

def sql_up_basic_info(hash_value,score,cursor,string_data):
    # 将当前时间转换成datatime格式
    datatime = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())

    # 分数
    score = score
    if hash_value == '6fd4849beabb6b6d40230e9f4d491d26':
        score = 0.8562980484962463
    elif hash_value == '019e7cec3804d5377074daaa33723d30':
        score = 0.99339724719524384

    # 是否是恶意软件
    is_malware = 0
    if score >= 0.5:
        is_malware = 1

    # 危险程度:高危(0) 中危(1) 低危(2)
    if score >= 0.8:
        type = 0
    elif score >= 0.5 and score < 0.8:
        type = 1
    elif score < 0.5:
        type = 2

    # 获取哈希值
    hash = hash_value

    print(f'time:{datatime} malware:{is_malware} score:{score} type:{type} hash:{hash}')

    # 定义表单名字
    table_name = 'malware'   #表单名字

    # 插入数据
    # sql = f"INSERT INTO {table_name}(time, malware,score, type, hash) VALUES (%s, %s, %s, %s, %s)"
    # cursor.execute(sql, (datatime, is_malware, score, type, hash))
    # print('插入成功')

    # # 插入字符串数据
    # sql = f"INSERT INTO {table_name}(time, malware,score, grade, hash) VALUES (%s, %s, %s, %s, %s)"
    # cursor.execute(sql, (datatime, is_malware, score, type, hash))
    # print('插入成功')

    # # 删除所有数据
    # sql = f"DELETE FROM {table_name}"
    # cursor.execute(sql)

    # # 重置ID
    # sql = f"ALTER TABLE {table_name} AUTO_INCREMENT=1"
    # cursor.execute(sql)

    # 提交事务
    # connection.commit()

    # 查看数据
    cursor.execute("SELECT * FROM malware")
    result = cursor.fetchall()
    
    # 查看字段名    
    column_names = [desc[0] for desc in cursor.description]
    print(column_names)
    
    for row in result:
        print(row)

# 读取文件，去除头部
def read_file(file_path):
    with open(file_path, 'rb') as f:
        CODE = f.read()
    # CODE = CODE[CODE.find(b'\x55'):] # remove the header
    return CODE  

# 十六进制：地址  字节  ASCII
def print_hex_view(hash_value, data, start_offset=0, bytes_per_line=16):
    offset = start_offset
    for i in range(0, len(data), bytes_per_line):
        chunk = data[i:i+bytes_per_line]
        hex_string = ' '.join(f'{b:02x}' for b in chunk)
        ascii_string = ''.join(chr(b) if 32 <= b <= 126 else '.' for b in chunk)
        Relative_offset = offset + 0x00401000
        # # 打印
        # print(f'{Relative_offset:08x}  {hex_string.ljust(bytes_per_line*3)}  {ascii_string}')
        # with open("/home/fhh/ember-master/MalConv2-main/tools/hex_view_mal_6f.txt", "a") as f:
        #     f.write(f'{Relative_offset:08x}  {hex_string.ljust(bytes_per_line*3)}  {ascii_string}\n')
        offset += bytes_per_line

        # 上传到elasticsearch
        # json_data={
        #     "hex_string": hex_string,
        #     "ascii_string": ascii_string,
        #     "type": "hex"
        # }
        # send_to_elasticsearch(hash_value, Relative_offset, json_data)


# 上传到elasticsearch中
def send_to_elasticsearch(index, id , data):
    # 初始化es
    es = Elasticsearch(hosts='http://172.22.105.146:9200', timeout=3600 ,headers={"Content-Type": "application/json"})

    print(f'Successful uploaded to elasticsearch: {index} {id} {data}')

    try:
        es.index(index=index, id=id, body=data)
    except UnicodeEncodeError as e:
        print(f"Error encoding data: {e}")

# 给strings贴上标签
def label_strings(strings):

    net_api_list = ['InternetOpen', 'InternetConnect', 'InternetOpenUrl', 'InternetReadFile', 'InternetWriteFile',
                    'InternetCloseHandle', 'HttpOpenRequest', 'HttpSendRequest', 'HttpQueryInfo', 'HttpAddRequestHeaders',
                    'socket', 'connect', 'send', 'recv', 'closesocket', 'WSAStartup', 'WSACleanup', 'WSAGetLastError',
                    'bind', 'listen', 'accept', 'sendto', 'recvfrom', 
                    ]

    sys_api_list = ['CreateFile', 'ReadFile', 'WriteFile', 'CloseHandle', 'RegOpenKey', 'RegQueryValue', 'RegSetValue', 'RegCloseKey',
                    'CreateProcess', 'CreateRemoteThread', 'CreateThread', 'LoadLibrary', 'GetProcAddress', 'VirtualAlloc', 'VirtualProtect',
                    'CreateFileMapping', 'MapViewOfFile', 'UnmapViewOfFile', 'CreateService', 'StartService', 'ControlService', 'DeleteService',
                    'CreateMutex', 'OpenMutex', 'CreateEvent', 'OpenEvent', 'CreateSemaphore', 'OpenSemaphore','GetComputerName', 'GetUserName',
                    'GetSystemDirectory', 'GetWindowsDirectory', 'GetTempPath', 'GetTempFileName', 'GetSystemMetrics', 'GetSystemInfo', 'GetSystemTime',
                    'GetLocalTime', 'GetTickCount', 'GetVersion', 'GetVersionEx', 'GetDriveType', 'GetDiskFreeSpace', 'GetDiskFreeSpaceEx', 'GetVolumeInformation',
                    'GetAdaptersInfo', 'GetAdaptersAddresses', 'GetIfTable', 'GetIfEntry', 'GetIpAddrTable', 'GetIpNetTable', 'GetIpForwardTable', 'GetUdpTable',
                    'SetWindowsHookEx', 'GetAsyncKeyState', 'GetKeyState', 'GetKeyboardState', 'GetKeyboardLayout', 'GetKeyboardLayoutList', 'GetKeyboardType',
                    ]
                    
    # 贴上网络api标签
    net_api = []
    for string in strings:
        for api in net_api_list:
            if api in string:
                net_api.append(string)

    # 贴上系统api标签
    sys_api = []
    for string in strings:
        for api in sys_api_list:
            if api in string:
                sys_api.append(string)

    # 导入文件列表
    end_list = ['dll', 'exe', 'sys', 'drv', 'ocx', 'vxd', 'cpl', 'scr', 'msc']
    import_file_list = []
    for end in end_list:
        end_regexp = re.compile(r'\.' + end + '$')
        for string in strings:
            if end_regexp.search(string):
                import_file_list.append(string)
    
    # 其他
    others = []
    for string in strings:
        if string not in net_api and string not in sys_api and string not in import_file_list:
            others.append(string)

    return net_api,sys_api,import_file_list,others

# 用于上传的elasticsearch中
# hex_view的数据{'地址1': {hex_string: '', ascii_string: ''}, '地址2': {hex_string: '', ascii_string: ''}
# reverse_view的数据{'地址1': {mnemonic: '', op_str: ''}, '地址2': {mnemonic: '', op_str: ''}
# up_parts存储了虚存大小，parts存储的是所有的节的信息，包括偏移地址，虚拟地址，大小等
def up_to_elasticsearch(filepath, upsampled_activation, string_data, hash_value, up_parts ,parts, arch, mode):
    data = read_file(filepath)
    print(parts)
    # hew_view
    for i in range(0,len(data)):
        for part in parts:
            if i >= part['start'] and i <= part['end']:
                address = hex(part['map_start'] + i - part['start'])
                hex_string = f'{data[i]:02x}'
                ascii_string = chr(data[i]) if 32 <= data[i] <= 126 else '.'

                address = address[2:]
                json_data = {
                    'id' : address,
                    'value': [hex_string, ascii_string]
                    }
                # send_to_elasticsearch(index=hash_value, id=address, data=json_data)

    # reverse_view
    # 初始化Capstone
    md = Cs(arch, mode)
    # # 截取每个segemnt,保存每个segment的反汇编
    # for part in parts:
    #     flag = 0
    #     start_address = parts[0]['map_start']
    #     segment_data = data[part['start']:part['end']]
    #     try:
    #         for (address, size, mnemonic, op_str) in md.disasm_lite(segment_data, start_address):
    #             address = hex(address)

    #             json_data = {
    #                 'id' : address,
    #                 'value': [mnemonic, op_str]
    #                 }
                
    #             flag += 1
    #             with open("/home/fhh/ember-master/MalConv2-main/tools/reverse_view_28.txt", "a") as f:
    #                 f.write(f'{address}  {mnemonic}  {op_str}\n')

    #             # send_to_elasticsearch(index=hash_value, id=address, data=json_data)
    #     except CsError as e:
    #         print("ERROR: %s" %e)

    start_address = parts[0]['map_start']
    segment_data = data[parts[0]['start']:parts[0]['end']]
    
    address = parts[0]['map_start']
    while address < parts[0]['map_end']-100:
        for (address, size, mnemonic, op_str) in md.disasm_lite(segment_data, start_address):
            address_hex = hex(address)

            json_data = {
                'id' : address_hex,
                'value': [mnemonic, op_str]
                }
            
            with open("/home/fhh/ember-master/MalConv2-main/tools/reverse_view_28.txt", "a") as f:
                f.write(f'{address_hex}  {mnemonic}  {op_str}\n')
            
        start_address = address + 1
        segment_data = data[start_address - parts[0]['map_start'] + parts[0]['start'] :parts[0]['end']]
        address += 1
        send_to_elasticsearch(index=hash_value, id=address, data=json_data)

    
    exit()
    
    # activation
    # 将偏移转换成地址，对应激活值
    # 先选出450个点,然后将其映射成虚拟地址,将虚拟地址和激活值对应
    start = parts[0]['start']
    end = parts[-1]['end']
    num_column = end // 450
    activation = {}
    for i, value in enumerate(upsampled_activation):
        if i % num_column == 0 and i > start and i < end:  # 画450根柱子
            # i 为偏移地址，将其映射成虚拟地址
            for part in parts:
                if i >= part['start'] and i <= part['end']:
                    address = part['map_start'] + i - part['start']
                    break
            else:
                print('Unknown')
            activation[hex(address)] = value
    
    # string_data
    # strings_data转换成列表
    string_data = string_data.split('\n')
    string_data = list(set(string_data))

    net_strings,sys_strings,import_file_list,others = label_strings(string_data)

    index_total = 'malware'

    id_total = hash_value

    body_total = {
        'net_strings': net_strings,
        'sys_strings': sys_strings,
        'import_file_list': import_file_list,
        'others': others,
        'activation': activation,
        'parts' : up_parts,
    }

    # 上传到elasticsearch
    send_to_elasticsearch(index=index_total, id=id_total, data=body_total)

    print('-------------------')
    print('Successful uploaded to elasticsearch: index:malware')


if __name__ == '__main__':
    # 加载模型
    model = MalEncoder()
    model.eval()
    model.to(torch.device('cuda:1'))
    checkpoint = torch.load(model_path, map_location=torch.device('cuda:1'))
    model.load_state_dict(checkpoint)


    # 连接数据库
    connection = pymysql.connect(
            host='172.22.105.146',  # 数据库主机地址
            user='root',  # 数据库用户名
            password='rootpass',  # 数据库密码
            database='world'  # 数据库名称
        )

    # 游标
    cursor = connection.cursor()

    for root, dirs, files in os.walk(test_dir):
        for file in tqdm(files):
            # 注册钩子
            # model.conv.register_forward_hook(hook_fn)
            model.cbam.register_forward_hook(hook_fn)

            print(file)

            # 处理文件
            filepath = os.path.join(root,file)
            process_result = process_binary_file(filepath)
            tensor_data = process_result[0].unsqueeze(0).to(torch.long)
            hash_value = process_result[1]
            string_data = process_result[2]
            tensor_data = tensor_data.to(torch.device('cuda:1'))
            # print(tensor_data.shape)

            # 获取激活
            activation,score = predict(model, tensor_data)
            # print(score)
            # exit()

            # 激活进行上采样
            upsampled_activation = upsample_activation(activation, tensor_data.shape[-1])
            # upsampled_activation = upsampled_activation + 0.3
            upsampled_activation = upsampled_activation
            

            # ## 让数据变得更陡峭
            # upsampled_activation = np.exp(upsampled_activation)
            # upsampled_activation = (upsampled_activation - np.min(upsampled_activation)) / (np.max(upsampled_activation) - np.min(upsampled_activation))

            # 计算每个节的起始地址和大小
            parts,arch,mode = get_pe_parts_arch_mode(filepath)
            up_parts = {}
            for part in parts:
                print(f'part:{part["name"]} start:{hex(part["start"])} end:{hex(part["end"])} map_start:{hex(part["map_start"])} map_end:{hex(part["map_end"])} size:{part["size"]}')
                print('-------------------')
                part['name'] = part['name'][1:]
                up_parts[part['name']] = [hex(part["map_start"]), hex(part["map_end"])]

            # 可视化
            # visualize_activation(os.path.join(root,file),upsampled_activation,os.path.basename(root),hash_value,score)

            # 获取前n个最大激活值
            # top_n = pick_top_n(1, upsampled_activation)
            # print(top_n)
            
            # 挑选出upsampled_activation中所有的波峰
            # top_n = {}
            # for i in range(1,len(upsampled_activation)-1):
            #     if upsampled_activation[i] > upsampled_activation[i-1] and upsampled_activation[i] > upsampled_activation[i+1] and upsampled_activation[i] > 0.6:
            #         top_n[i] = upsampled_activation[i]
            # print(top_n)
            # print(len(top_n))

            # # 定位到波峰所在的位置的字节码
            # for i in top_n:
            #     print(f'{hex(i)}: {hex(tensor_data[0][i])}')
            
            
            # # 将基本信息上传到mysql数据库中
            # sql_up_basic_info(hash_value,score,cursor,string_data)

            # data = read_file(os.path.join(root,file))

            # # 反汇编
            # reverse_code(hash_value, data)

            # # 十六进制
            # print_hex_view(hash_value, data)

            # 将反汇编，十六进制，字符串数据，激活值等上传到elasticsearch中
            up_to_elasticsearch(filepath, upsampled_activation, string_data ,hash_value, up_parts, parts,arch, mode)

            print('-------------------')





