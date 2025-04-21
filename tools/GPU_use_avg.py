
# 计算数组的平均值
def time_avg(data):
    return sum(data) / len(data)

# 测试用例
# raw = [41,27,13,71,74,33,15,62,43,50,57,18,60,30,60,22,39,47,41,28,52,44,69,35,45,34,19,46]
raw = [32,25,44,21,84,71,38,67,48,42,48,33,25,28,40,37,24,35,58,40,44,38,43,33,60,44,56,97,54]
mymodel_cbam = [69,43,19,83,58,54,25,62,21,33,62,44,73,55,64,19,49,55,68,66,76,69,62,59,66,70,57,66,60,28] 
mymodel_gru = [74,68,47,60,31,23,66,75,68,57,74,62,25,56,14,71,67,64,50,71,61,47,83,27,78,62,25,59]
my_model_all = [37,59,60,47,62,57,90,26,72,70,65,75,25,53,55,63,22,67,76,58,20,74,26,66,70,76,63,56]


raw_mem_use = time_avg(raw)/100 * 81920 - 5979
mymodel_cbam_mem_use = time_avg(mymodel_cbam)/100 * 81920 - 8654
mymodel_gru_mem_use = time_avg(mymodel_gru)/100 * 81920 - 8654
my_allmodel_mem_use = time_avg(my_model_all)/100 * 81920 - 8654

# 转换成GB
raw_mem_use = round(raw_mem_use / 1024,2)
mymodel_cbam_mem_use = round(mymodel_cbam_mem_use / 1024,2)
mymodel_gru_mem_use = round(mymodel_gru_mem_use / 1024,2)
my_allmodel_mem_use = round(my_allmodel_mem_use / 1024,2)

print(f"Raw Model: {raw_mem_use}G")
print(f"My Model_cbam: {mymodel_cbam_mem_use}G")
print(f"My Model_gru: {mymodel_gru_mem_use}G")
print(f"My Model_all: {my_allmodel_mem_use}G")