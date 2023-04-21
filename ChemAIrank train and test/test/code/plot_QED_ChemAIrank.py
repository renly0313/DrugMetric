import pandas as pd
from rdkit import Chem
from rdkit.Chem import Draw

# 读取CSV文件
file_path = "/home/dell/wangzhen/RealQED(2.17)/test/data/admetlab2/result/4databset_test_admet.csv"
data = pd.read_csv(file_path)

# 按QED排序
data_qed_sorted = data.sort_values(by="QED", ascending=False)

# 按ChemAIrank排序
data_ChemAIrank_sorted = data.sort_values(by="ChemAIrank", ascending=False)

# 提取前25个和后25个分子
top_25_qed = data_qed_sorted.head(25)
bottom_25_qed = data_qed_sorted.tail(25)
top_25_ChemAIrank = data_ChemAIrank_sorted.head(25)
bottom_25_ChemAIrank = data_ChemAIrank_sorted.tail(25)

# 绘制分子
def draw_molecules(smiles_list, file_path, dpi=300):
    molecules = [Chem.MolFromSmiles(smiles) for smiles in smiles_list]
    img = Draw.MolsToGridImage(molecules, molsPerRow=5, subImgSize=(200, 200), legends=[str(i + 1) for i in range(len(molecules))])
    img.save(file_path)

save_path = '/home/dell/wangzhen/RealQED(2.17)/test/result/25molecule_plot'

# 绘制QED排序前25个分子
draw_molecules(top_25_qed['smiles'], f'{save_path}/top_25_qed.png', dpi=300)

# 绘制QED排序后25个分子
draw_molecules(bottom_25_qed['smiles'], f'{save_path}/bottom_25_qed.png', dpi=300)

# 绘制ChemAIrank排序前25个分子
draw_molecules(top_25_ChemAIrank['smiles'], f'{save_path}/top_25_ChemAIrank.png', dpi=300)

# 绘制ChemAIrank排序后25个分子
draw_molecules(bottom_25_ChemAIrank['smiles'], f'{save_path}/bottom_25_ChemAIrank.png', dpi=300)
