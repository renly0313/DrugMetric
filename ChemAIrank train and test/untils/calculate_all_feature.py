
import pandas as pd

# 重命名df_prediction中的列名
df_prediction = pd.read_csv('/home/dell/wangzhen/RealQED(2.17)/test/data/finetune/toxcast/toxcast_prediction.csv')
df_prediction = df_prediction.rename(columns={'SMILES': 'smiles'})

# 读取df_bace和df_prediction
df_bace = pd.read_csv('/home/dell/wangzhen/RealQED(2.17)/test/data/finetune/toxcast/toxcast.csv')
df_prediction = df_prediction.round(2)

# 在df_prediction中添加df_bace中存在的分子及其特征
df_overlap = pd.merge(df_bace, df_prediction, on='smiles')

# 将结果写入csv文件
df_overlap.to_csv('/home/dell/wangzhen/RealQED(2.17)/test/data/finetune/toxcast/toxcast_prediction_new.csv', index=False)
