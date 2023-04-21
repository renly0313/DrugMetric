import pandas as pd
from autogluon.tabular import TabularDataset, TabularPredictor
import numpy as np

# 加载预训练的AutoGluon模型

predictor = TabularPredictor.load('/home/dell/wangzhen/RealQED(2.17)/train/model/autogluen/AutogluonModels/ag-20230308_124517/', require_py_version_match=False)

# 读取数据集
df = pd.read_csv('/home/dell/wangzhen/RealQED(2.17)/preprocess/calculate_distrbution/result/test_distribution.csv')
test_data = pd.DataFrame(df, columns=[str(i) for i in range(0, 64)])
# 创建新的数据集对象
test_data = TabularDataset(test_data)
# 将预测结果添加到原始数据集中
y_pred = round(predictor.predict(test_data), 2)

# 将数据集保存为CSV文件
y_pred.to_csv('/home/dell/wangzhen/RealQED(2.17)/test/result/perdiciton/test_prediction.csv', index=False)

