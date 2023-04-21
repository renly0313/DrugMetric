import pandas as pd
import pickle

# 读取pickle文件
with open('path/to/pickle/file.pkl', 'rb') as f:
    data = pickle.load(f)

# 将pickle数据转换为数据框
df = pd.DataFrame(data)

