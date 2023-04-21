from rdkit import Chem
from rdkit.Chem import rdFMCS
from rdkit.Chem.Draw import MolsToGridImage

# 你的分子列表
smiles_list = [
    "FC(c1cc2cnc(nc2n(c1=O)C1CCCC1(C)O)NC1CCN(CC1)S(=O)(=O)C)F",
    "Cc1c(C)c2c(c(C)c1O)CC[C@@](C)(CCC[C@H](C)CCC[C@H](C)CCCC(C)C)O2",
    "Cn1c(NCc2ccnc(OC3CCCC3)c2)nc2c1c(=O)[nH]c(=O)n2C",
    "Cn1c(=O)c2c(ncn2CCC(=O)Nc2ccccn2)n(C)c1=O",
    "CN1C(=O)CC[C@H](NC(=O)Nc2cc(Cl)c(Br)cn2)[C@H]1c1ccnn1C",
    "CC1Oc2c(cccc2C(=O)N2CCN(C(=O)c3ccc(=O)[nH]c3)CC2)NC1=O",
    "Cn1cnc2c1c(=O)n(CC(=O)Nc1ccc(N3CCCCC3)nc1)c(=O)n2C",
    "COc1ccc(CNc2nc(N3CCC[C@H]3CO)ncc2C(=O)NCc2ncccn2)cc1Cl",
    "CC(C)(C)N1C(=O)CC(C(=O)Nc2ccc(F)c(C(N)=O)c2)C1c1ccc(Cl)c(F)c1",
    "Cn1cnc2c1c(=O)n(CC(=O)Nc1ccc(N3CCOCC3)nc1)c(=O)n2C",
    "CC(C)Oc1cc(NC(=O)[C@@H]2CCCC[C@H]2C(F)(F)F)ccc1C(N)=O",
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
MolsToGridImage(mcs_molecules, molsPerRow=5, subImgSize=(800, 800)).save("/home/dell/wangzhen/TankBind-main/examples/HTVS/result/mcs_filter_10.png")
