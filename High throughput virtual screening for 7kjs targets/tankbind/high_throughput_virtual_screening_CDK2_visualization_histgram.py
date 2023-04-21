import matplotlib.pyplot as plt
import pandas as pd
# Load the prediction results CSV file
df = pd.read_csv('/home/dell/wangzhen/TankBind-main/examples/HTVS/info/info.csv')

# Set the number of bins
bin_num = 25

# Draw the histogram
plt.hist(df.affinity, bins=bin_num, color='skyblue', edgecolor='black', alpha=0.7)

# Add axis labels
plt.xlabel('Affinity Prediction(kcal/mol)')
plt.ylabel('Frequency')

# Add title
# plt.title('Histogram of Affinity Prediction Values')

# Modify font size
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

# Add grid lines
plt.grid(axis='y', linestyle='--', alpha=0.7)

# 设置保存路径和文件名
save_path = '/home/dell/wangzhen/TankBind-main/examples/HTVS/result/histogram.png'

# 设置图片边缘
plt.subplots_adjust(left=0.15)

# 保存图片，设置dpi
plt.savefig(save_path, dpi=300)

# Show the plot
plt.show()
