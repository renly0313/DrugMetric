import os
import csv

# 设置输入和输出目录
input_dir = '/home/dell/wangzhen/RealQED(2.17)/test/data/anticancer'
output_dir = '/home/dell/wangzhen/RealQED(2.17)/test/data/admetlab2/anticancer'

# 遍历输入目录下的每个文件夹
for folder_name in os.listdir(input_dir):
    folder_path = os.path.join(input_dir, folder_name)
    if not os.path.isdir(folder_path):
        continue

    # 遍历当前文件夹下的CSV文件
    for file_name in os.listdir(folder_path):
        if not file_name.endswith('.csv'):
            continue
        file_path = os.path.join(folder_path, file_name)

        # 打开CSV文件并读取第一列SMILES
        with open(file_path, 'r') as csv_file:
            reader = csv.reader(csv_file)
            smiles_list = [row[0] for row in reader]

        # 将SMILES写入新文件
        output_path = os.path.join(output_dir, file_name)
        with open(output_path, 'w') as output_file:
            for smiles in smiles_list:
                output_file.write(smiles + '\n')
