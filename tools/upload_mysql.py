import pymysql

connection = pymysql.connect(
            host='172.22.105.146',  # 数据库主机地址
            user='root',  # 数据库用户名
            password='rootpass',  # 数据库密码
            database='world'  # 数据库名称
        )

# 创建一个游标对象 cursor
cursor = connection.cursor()

# 定义表单名字
TABLE= 'malware'   #表单名字

# # 查询所有的表单
# sql = "SHOW TABLES"
# cursor.execute(sql)
# result = cursor.fetchall()
# print(result)

# 更新字段的值
sql = "UPDATE {} SET malware = 1 WHERE hash = '019e7cec3804d5377074daaa33723d30'".format(TABLE)
cursor.execute(sql)
connection.commit()

# 查询所有字段名称
# sql = "SHOW COLUMNS FROM {}".format(TABLE)
sql = "SELECT * FROM  {}".format(TABLE) 
cursor.execute(sql)
result = cursor.fetchall()



# 获取所有字段名称
column_names = [desc[0] for desc in cursor.description]
print(column_names)

# 打印所有字段名称
for row in result:
    print(row)



# TABLE= 'malware'   #表单名字

# def send_to_mysql(data):
#     # print(data_list)
#     sql = "INSERT INTO {TABLE}(time, malware,score, type, hash) VALUES (%s, %s, %s, %s, %s)"
#     # 执行插入操作
#     # print("执行插入操作")
#     cursor.executemany(sql, data)

#     # 提交事务
#     connection.commit()
#     # print("提交修改")
#     # 关闭游标和数据库连接

# with connection.cursor() as cursor:
#     data=("1711608382236", 1, 0.53, 2, "一串哈希值")
#     send_to_mysql(data)
#     cursor.close()

# connection.close()