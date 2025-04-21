import os

folder_path = '/home/fhh/ember-master/MalConv2-main/dataset/10000_VirusShare_CryptoRansom_20160715'
# folder_path = '/home/fhh/ember-master/MalConv2-main/dataset/malconv2_mal_train_10000'
folder_path = '/home/fhh/ember-master/MalConv2-main/dataset/2000_VirusShare_CryptoRansom_20160715'
file_list = os.listdir(folder_path)

# Get file sizes
file_sizes = [(file, os.path.getsize(os.path.join(folder_path, file))) for file in file_list]

# Sort files by size
sorted_files = sorted(file_sizes, key=lambda x: x[1])

# Select the smallest ten files
smallest_files = sorted_files[:10]

# Print the smallest files
for file, size in smallest_files:
    print(file, size)

# # 删除所有size小于100的文件
# for file, size in sorted_files:
#     if size < 1500:
#         os.remove(os.path.join(folder_path, file))
#         print(file, size)
