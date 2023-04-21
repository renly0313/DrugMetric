import os
import random
import shutil
from tqdm import tqdm

def sample_txt_files(input_dir, output_dir, sample_size, num_samples):
    """
    对指定文件夹内的所有txt文件进行多次随机采样，并将采样结果保存到指定文件夹中。

    Args:
        input_dir (str): 需要采样的文件夹路径。
        output_dir (str): 保存采样结果的文件夹路径。
        sample_size (int): 每次采样的行数。
        num_samples (int): 采样次数。

    Returns:
        None: 将采样结果保存到指定的文件夹中。

    Raises:
        FileNotFoundError: 如果输入文件夹不存在，则会抛出FileNotFoundError异常。

    Example:
        >>> input_dir = "path/to/input/directory"
        >>> output_dir = "path/to/output/directory"
        >>> sample_size = 4527
        >>> num_samples = 5
        >>> sample_txt_files(input_dir, output_dir, sample_size, num_samples)
    """
    # 遍历输入文件夹中的所有txt文件，忽略文件夹名中包含"candidate"的文件
    txt_files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f)) and f.endswith('.txt') and 'candidate' not in f]
    for txt_file in tqdm(txt_files, desc="Sampling files", unit="file"):
        # 构建输出文件名
        output_filename = os.path.join(output_dir, f"{os.path.splitext(txt_file)[0]}_{num_samples}samples")
        # 读取txt文件中的内容
        with open(os.path.join(input_dir, txt_file), 'r') as f:
            lines = f.readlines()
        # 对txt文件进行多次随机采样
        for i in tqdm(range(num_samples), desc="Sampling", unit="sample", leave=False):
            # 随机选择采样起始位置
            start_index = random.randint(0, len(lines) - sample_size)
            # 构建采样内容
            sample_lines = lines[start_index : start_index + sample_size]
            # 将采样内容写入输出文件
            with open(f"{output_filename}_seed{i}.txt", 'w') as f:
                f.writelines(sample_lines)
    print("Sampling complete!")

input_dir = "../data/preprocess_orign_data"
output_dir = "../data/preprocess_origin_sample"
sample_size = 4527
num_samples = 5

sample_txt_files(input_dir, output_dir, sample_size, num_samples)