from rdkit import Chem
from rdkit.Chem import RDConfig
import os
import sys
sys.path.append(os.path.join(RDConfig.RDContribDir, 'SA_Score'))
# now you can import sascore!
import sascorer
import pandas as pd
import numpy as np
import rdkit
from rdkit import Chem
from rdkit.Chem import QED
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Draw, AllChem
from rdkit.ML.Descriptors import MoleculeDescriptors
def mol_to_feather(mol):
    mols=[]
    mols.append(mol)
    #计算SA
    SA=[]
    for mol in mols:
        m = sascorer.calculateScore(mol)
        SA.append(m)
    SA = pd.DataFrame(SA)
    SA.columns=['SA']

    #获取原子数
    atom_num=[]
    for mol in mols:
        m = mol.GetNumAtoms()
        atom_num.append(m)
    atom_num = pd.DataFrame(atom_num)
    atom_num.columns=['atom_num']

    #计算分子QED
    mol_qed=[]
    for mol in mols:
        m = rdkit.Chem.QED.qed(mol)
        mol_qed.append(m)
    mol_qed = pd.DataFrame(mol_qed)
    mol_qed.columns = ['qed']

    #计算描述符
    properties=[]
    for mol in mols:
        m = rdkit.Chem.QED.properties(mol)
        properties.append(m)
    properties = pd.DataFrame(properties)

    #des_list = ['MolWt', 'NumHAcceptors', 'NumHDonors', 'MolLogP', 'NumRotatableBonds']
    #其他描述符
    other_properties=[]
    des_list = ['fr_NH0', 'fr_NH1', 'fr_NH2', 'FractionCSP3', 'NumAliphaticRings']
    calculator = MoleculeDescriptors.MolecularDescriptorCalculator(des_list)
    for mol in mols:
        m = calculator.CalcDescriptors(mol)
        other_properties.append(m)
    other_properties = pd.DataFrame(other_properties)
    other_properties.columns=['fr_NH0', 'fr_NH1', 'fr_NH2', 'FractionCSP3', 'NumAliphaticRings']
    other_properties["fr_NH"] = other_properties["fr_NH0"] + other_properties["fr_NH1"] + other_properties["fr_NH2"]
    data=pd.concat([SA, atom_num, mol_qed, properties, other_properties["fr_NH"], other_properties['FractionCSP3'], other_properties['NumAliphaticRings']], axis=1)
    f_idx = ['datasets', 'smiles']
    f_x = [x for x in data.columns if x not in f_idx + ['SA', 'atom_num', 'qed', 'scores', 'shuffle']]
    data_pro = data[f_x]
    return data_pro
#显示所有列pd.set_option('display.max_columns',None)
"""显示data_pro所有列"""

