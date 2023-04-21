# 这是一个 Python 脚本，用于执行化合物的预测和自动生成。
# 脚本中使用了多个 Python 库和模块，
# 如 pandas、argparse、numpy、pickle、logging、tqdm、rdkit、torch 等。
# 脚本的主要流程如下：
#
# 将 SMILES 序列转化为 MolTree 对象
# 过滤出能够成功转化的 SMILES 序列和无法转化的 SMILES 序列
# 将能够成功转化的 SMILES 序列转化为潜在表示，并保存到字典中
# 加载预训练的 AutoGluon 模型
# 使用 AutoGluon 模型预测 logP 值
# 将预测结果保存到 CSV 文件中
# 其中，步骤 1 中使用了 rdkit 库中的 Chem 和 MolTree 方法；
# 步骤 2 中使用了 logging 库记录日志，并返回能够成功转化的 SMILES 序列
# 和无法转化的 SMILES 序列；
# 步骤 3 中使用了 fast_jtnn 库中的JTNNVAE、Vocab、MolTreeFolder 和 MolTree 方法，
# 将能够成功转化的 SMILES 序列转化为潜在表示，并将潜在表示保存到字典中；
# 步骤 4 中使用了 autogluon.tabular 库中的 TabularPredictor 方法，
# 加载预训练的 AutoGluon 模型；
# 步骤 5 中使用了预训练的 AutoGluon 模型，对步骤 3 中保存的潜在表示进行预测，
# 并将预测结果保存到列表中；步骤 6 中将预测结果保存到 CSV 文件中，
# 并返回无法转化的 SMILES 序列
import sys
import os
import pandas as pd
import argparse
import numpy as np
import pickle as pickle
import logging
from tqdm import tqdm
from rdkit import Chem
from fast_jtnn import *
from autogluon.tabular import TabularPredictor
import torch
import rdkit

def tensorize(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
    except Exception as e:
        logging.warning(f"Failed to convert SMILES to molecule: {smiles}")
        logging.warning(str(e))
        return None
    if mol is None:
        logging.warning(f"Invalid SMILES: {smiles}")
        return None

    mol_tree = MolTree(smiles)
    mol_tree.recover()

    mol_tree.assemble()
    for node in mol_tree.nodes:
        if node.label not in node.cands:
            node.cands.append(node.label)

    del mol_tree.mol
    for node in mol_tree.nodes:
        del node.mol

    return mol_tree


def filter_valid_molecules(smiles_list):
    valid_smiles_list = []
    invalid_smiles_list = []
    for smiles in smiles_list:
        mol_tree = tensorize(smiles)
        if mol_tree is not None:
            valid_smiles_list.append(smiles)
        else:
            invalid_smiles_list.append(smiles)
    return valid_smiles_list, invalid_smiles_list


def main_vae_inference(input_dir, vocab_path, model_path, batch_size=8, latent_size=64):
    # 加载分子词汇表
    with open(vocab_path) as f:
        vocab = [line.strip() for line in f]
    vocab = Vocab(vocab)

    # 加载预训练的 JT-VAE 模型
    model = JTNNVAE(vocab, hidden_size=450, latent_size=latent_size, depthT=20, depthG=3).cuda()
    model.load_state_dict(torch.load(model_path))

    # 对输入文件夹中的每个 CSV 文件进行处理
    for file_name in os.listdir(input_dir):
        if not file_name.endswith(".csv"):
            continue
        file_path = os.path.join(input_dir, file_name)
        logging.info(f"Processing {file_name}...")

        # 读入文件中的 SMILES 序列
        data = pd.read_csv(file_path)["smiles"].values.tolist()

        # 过滤出能够成功转化的 SMILES 序列和无法转化的 SMILES 序列
        valid_smiles_list, invalid_smiles_list = filter_valid_molecules(data)

        # 将能够成功转化的 SMILES 序列转化为潜在表示，并保存到字典中
        with torch.no_grad():
            valid_processed_dict = {}
            for smiles in tqdm(valid_smiles_list):
                try:
                    mol_tree = tensorize(smiles)
                    if mol_tree is None:
                        continue
                    mol_tree = [mol_tree]
                    loader = MolTreeFolder(mol_tree, vocab, batch_size)
                    for batch in loader:
                        try:
                            mol_tree_mean = model.forward_dis(batch)
                            mol_tree_mean = mol_tree_mean.cpu().detach().numpy()
                            valid_processed_dict[smiles] = mol_tree_mean
                        except Exception as e:
                            logging.warning(f"Error while processing molecule: {smiles}")
                            logging.warning(str(e))
                            continue
                except Exception as e:
                    logging.warning(f"Error while converting SMILES to MolTree: {smiles}")
                    logging.warning(str(e))
                    continue

            # 将无法转化的 SMILES 序列保存到列表中
            invalid_smiles_df = pd.DataFrame(invalid_smiles_list, columns=['Invalid SMILES'])

            # 如果没有能够成功转化的 SMILES 序列，则输出提示信息并跳过该文件
            if len(valid_processed_dict) == 0:
                logging.warning(f"No valid molecules found in the file {file_name}.")
                continue

            # 加载预训练的 AutoGluon 模型
            predictor = TabularPredictor.load(
                '/home/dell/wangzhen/RealQED(2.17)/train/model/autogluen/AutogluonModels/ag-20230308_124517/',
                require_py_version_match=False)

            # 将预测结果保存到列表中
            y_pred_list = []
            for smiles, mol_tree_mean in valid_processed_dict.items():
                mol_tree_mean = np.clip(mol_tree_mean, -20, 20)  # clip latent representations
                y_pred = predictor.predict(pd.DataFrame(mol_tree_mean, columns=[str(i) for i in range(0, 64)]))
                y_pred_list.append((smiles, y_pred[0]))

            # 将预测结果保存到 CSV 文件中
            df = pd.DataFrame(y_pred_list, columns=['SMILES', 'Prediction'])
            df_file_name = file_name.split(".")[0] + "_prediction.csv"
            df_file_path = os.path.join(os.path.dirname(file_path), "result", "prediction", df_file_name)
            os.makedirs(os.path.dirname(df_file_path), exist_ok=True)  # 如果文件夹不存在，创建新文件夹
            df.to_csv(df_file_path, index=False)

            # 将无法转化的 SMILES 序列保存到 CSV 文件中
            invalid_file_name = file_name.split(".")[0] + "_invalid_smiles.csv"
            invalid_file_path = os.path.join(os.path.dirname(file_path), "result", "prediction", invalid_file_name)
            os.makedirs(os.path.dirname(invalid_file_path), exist_ok=True)  # 如果文件夹不存在，创建新文件夹
            invalid_smiles_df.to_csv(invalid_file_path, index=False)

if __name__ == '__main__':
    lg = rdkit.RDLogger.logger()
    lg.setLevel(rdkit.RDLogger.CRITICAL)
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default="/home/dell/wangzhen/RealQED(2.17)/test/data/LARGE",
                        help='Input directory or CSV file path')
    parser.add_argument('--vocab', type=str, default="/home/dell/wangzhen/RealQED(2.17)/data/vocab/all_data_vocab.txt",
                        help='Vocab file path')
    parser.add_argument('--model_path', type=str,
                         default="/home/dell/wangzhen/RealQED(2.17)/data/save_model/pre_zinc250_mix_rand1_processed_model1206/pre_zinc250_mix_rand1_processed_model1206best_model.pkl",
                         help='Model file path')
    parser.add_argument('--latent_size', type=int, default=64)
    args = parser.parse_args()

    # 设置日志格式和级别
    log_format = '%(asctime)s %(levelname)s %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_format)

    main_vae_inference(args.input, args.vocab, args.model_path, args.latent_size)
