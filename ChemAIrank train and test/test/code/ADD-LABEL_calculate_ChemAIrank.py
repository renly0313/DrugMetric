import pandas as pd

# 定义数据集标签
labels = ['Candidate drug', 'ChEMBL', 'ZINC', 'GDB']

# 读取数据集文件
df_candidate = pd.read_csv('/home/dell/wangzhen/RealQED(2.17)/test/result/perdiciton/candidate_drug_test_rand1_prediction_scores.csv')
df_chembl = pd.read_csv('/home/dell/wangzhen/RealQED(2.17)/test/result/perdiciton/ChEMBL_test_rand1_prediction_scores.csv')
df_zinc = pd.read_csv('/home/dell/wangzhen/RealQED(2.17)/test/result/perdiciton/ZINC_test_rand1_prediction_scores.csv')
df_gdb = pd.read_csv('/home/dell/wangzhen/RealQED(2.17)/test/result/perdiciton/GDB_test_rand1_prediction_scores.csv')

# 添加标签列
df_candidate = df_candidate.assign(datasets=labels[0])
df_chembl = df_chembl.assign(datasets=labels[1])
df_zinc = df_zinc.assign(datasets=labels[2])
df_gdb = df_gdb.assign(datasets=labels[3])

# 合并数据集
df_all = pd.concat([df_candidate, df_chembl, df_zinc, df_gdb], ignore_index=True)

# 保存为新文件
df_all.to_csv('/home/dell/wangzhen/RealQED(2.17)/test/result/perdiciton/all_datasets_test_rand1_prediction_scores.csv', index=False)
