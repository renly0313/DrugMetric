import pandas as pd

# 读取两个CSV文件
df1 = pd.read_csv('/home/dell/wangzhen/RealQED(2.17)/test/data/LARGE/affinity_large_than_8.csv', header=None)
df2 = pd.read_csv('/home/dell/wangzhen/RealQED(2.17)/test/data/LARGE/affinity_large_than_8_invalid.csv', header=None)

# 将两个数据帧合并
combined_df = pd.concat([df1, df2], ignore_index=True)

# 去除重复值
unique_df = combined_df.drop_duplicates(keep='first')

# 保存到新的CSV文件
unique_df.to_csv('/home/dell/wangzhen/RealQED(2.17)/test/data/LARGE/affinity_large_than_8_no_duplicates.csv', index=False, header=None)
