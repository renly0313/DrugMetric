# 输入通过ADMETLAB2.0计算得到的包含80多个个特征的CSV文件，计算QED和ChemAIra与其他特征的皮尔森相关系数，并绘制热力图。
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 从CSV文件中读取数据集
data = pd.read_csv('/home/dell/wangzhen/RealQED(2.17)/test/data/admetlab2/result/candidate_drug_admet.csv')

# 提取QED和ChemAIra特征
QED = data['QED']
chemAIrank = data['ChemAIrank']

# 筛选数值特征
num_cols = [col for col in data.columns if data[col].dtype != 'O' and col != '[QED]' and col != 'ChemAIrank']

# 计算皮尔森相关系数
corr_qed = data[num_cols].corrwith(QED)
corr_chemAIrank = data[num_cols].corrwith(chemAIrank)

# 绘制热力图
fig, ax = plt.subplots(1, 2, figsize=(20, 10))

# 绘制QED的热力图
sns.heatmap(pd.DataFrame(corr_qed, columns=['correlation']).T, cmap='coolwarm', annot=False, vmin=-1, vmax=1,  ax=ax[0])
ax[0].set_title('Correlation between QED and other numerical features')

# 绘制ChemAIra的热力图
sns.heatmap(pd.DataFrame(corr_chemAIrank, columns=['correlation']).T, cmap='coolwarm', annot=False, vmin=-1, vmax=1, ax=ax[1])
ax[1].set_title('Correlation between ChemAIra and other numerical features')

plt.savefig('../test/result/heatmap/heatmap.png')

plt.show()