import os
import pandas as pd
import matplotlib.pyplot as plt

folder_path = "/home/dell/wangzhen/TankBind-main/tankbind/output/merged"
files = [f"{folder_path}/{i}.csv" for i in range(1, 11)]

pki_counts = []

for file in files:
    df = pd.read_csv(file)
    pki_count = len(df[df['CDK2_pki'] > 6])
    pki_counts.append(pki_count)

fig, ax = plt.subplots()
ax.bar(range(1, 11), pki_counts)
ax.set_xlabel('CSV File')
ax.set_ylabel('CDK2_pki > 6 Count')
ax.set_title('Count of Molecules with CDK2_pki > 6 in CSV Files')
plt.xticks(range(1, 11))
plt.show()
