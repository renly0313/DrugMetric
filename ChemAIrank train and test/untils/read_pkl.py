import pandas as pd
import pickle

# 读取pickle文件
with open('/home/dell/wangzhen/RealQED(2.17)/test/data/finetune/bace/bace.pkl', 'rb') as f:
    data = pickle.load(f)

# 将pickle数据转换为数据框
df = pd.DataFrame(data)

# 将数据框保存为CSV文件
df.to_csv('../autogluen/AutogluonModels/anticancer.csv', index=False)
