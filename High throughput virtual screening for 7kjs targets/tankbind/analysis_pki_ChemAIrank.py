import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

folder_path = "/home/dell/wangzhen/TankBind-main/tankbind/output/merged"
files = [f"{folder_path}/{i}.csv" for i in range(1, 11)]

cdk2_pki_values = []
chemai_rank_values = []

for file in files:
    df = pd.read_csv(file)
    cdk2_pki_values.extend(df['CDK2_pki'].tolist())
    chemai_rank_values.extend(df['QED'].tolist())

correlation = np.corrcoef(cdk2_pki_values, chemai_rank_values)[0, 1]

plt.scatter(cdk2_pki_values, chemai_rank_values, alpha=0.5)
plt.xlabel('CDK2_pki')
plt.ylabel('QED')
plt.title(f'CDK2_pki vs ChemAIrank (Correlation: {correlation:.2f})')
plt.show()
