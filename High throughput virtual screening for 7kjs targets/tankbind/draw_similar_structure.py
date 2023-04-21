import pandas as pd
from rdkit import Chem
from rdkit.Chem import Draw

file_path = "/home/dell/wangzhen/TankBind-main/tankbind/output/merged/1.csv"
df = pd.read_csv(file_path)
filtered_df = df[(df['CDK2_pki'] > 6) & (df['CDK2_inhibitor'] == 'Yes')]

molecules = [Chem.MolFromSmiles(smiles) for smiles in filtered_df['SMILES']]

img = Draw.MolsToGridImage(molecules, molsPerRow=5, subImgSize=(500, 500))
img.save("/home/dell/wangzhen/TankBind-main/examples/HTVS/result/molecules_grid_image.png")
