from rdkit import Chem
from rdkit.Chem import rdFMCS
from rdkit.Chem.Draw import MolsToGridImage

# 你的分子列表
smiles_list = [
    "FC(c1cc2cnc(nc2n(c1=O)C1CCCC1(C)O)NC1CCN(CC1)S(=O)(=O)C)F",
    "NCCS(N(CC1)CCC1NC2=NC=C3C(C(C4=C(F)C=C(C(NCC5COCC5)=O)C=C4)=CS3)=N2)(=O)=O",
    "O=C(OC(C)C)NC1=CC=C(C(F)=C1)C2=CSC3=CN=C(NC4CCN(S(=O)(CCN)=O)CC4)N=C32",
    "O=C(OCC(F)(F)F)NC1=CC=C(C(F)=C1)C2=CSC3=CN=C(NC4CCN(S(=O)(CCN)=O)CC4)N=C32",
    "O=C(OC)NC1=CC=C(C(F)=C1)C2=CSC3=CN=C(NC4CCN(S(=O)(CCN)=O)CC4)N=C32",
    "O=C(OC1COC1)NC2=CC=C(C(F)=C2)C3=CSC4=CN=C(NC5CCN(S(=O)(C)=O)CC5)N=C43",
    "O=C(OC)NC1=CC=C(C(Cl)=C1)C2=CSC3=CN=C(NC4CCN(S(=O)(CCN)=O)CC4)N=C32",
    "O=C(OC)NC1=CC=C(C(C)=C1)C2=CSC3=CN=C(NC4CCN(S(=O)(CCN)=O)CC4)N=C32",
    "O=C(OC)NC1=CC=C(C(F)=C1)C2=CSC3=CN=C(NC4CCN(S(=O)(C)=O)CC4)N=C32",
    "O=S(N(CC1)CCC1NC2=NC=C3C(C(C4=CC=C(C(NCC5(F)COC5)=O)C=C4)=CS3)=N2)(C)=O",
    "O=C(C1COC1)NC2=CC=C(C=C2)C3=CSC4=CN=C(NC5CCN(S(=O)(C)=O)CC5)N=C43",
]

# 生成分子对象列表
molecules = [Chem.MolFromSmiles(smiles) for smiles in smiles_list]

# 计算目标分子（第一个分子）与其他分子的最大公共子结构
mcs_results = []
target_mol = molecules[0]
for i in range(1, len(molecules)):
    mcs_result = rdFMCS.FindMCS([target_mol, molecules[i]], matchValences=True)
    mcs_results.append(mcs_result)

# 从MCS结果中生成分子对象列表
mcs_molecules = [Chem.MolFromSmarts(mcs.smartsString) for mcs in mcs_results]

# 显示最大公共子结构图像
MolsToGridImage(mcs_molecules, molsPerRow=5, subImgSize=(500, 500), legends=[f'MCS {i + 1}' for i in range(len(mcs_molecules))]).save("/home/dell/wangzhen/TankBind-main/examples/HTVS/result/mcs_TYX.png")
