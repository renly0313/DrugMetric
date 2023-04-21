import numpy as np
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import os
data = {
    'QED': {
        '显著差异的特征': {
            'p值<0.001': ['HIA', 'F(20%)', 'F(30%)', 'Caco-2', 'BBB', 'VDss', 'CYP1A2-inh', 'CYP1A2-sub', 'CYP2C19-inh', 'CYP2C19-sub', 'CYP2D6-inh', 'CYP2D6-sub', 'CYP3A4-sub', 'CL', 'Carcinogenicity', 'Respiratory', 'NR-ER-LBD', 'MW', 'Vol', 'nHA', 'nHD', 'TPSA', 'nRot', 'nHet', 'nRig', 'nStereo', 'Toxicophores', 'Synth', 'Natural Product-likeness', 'BMS'],
            'p值<0.01': ['Pgp-sub', 'H-HT', 'ROA', 'BCF', 'NR-ER', 'NR-PPAR-gamma', 'SureChEMBL', 'Skin_Sensitization'],
            'p值<0.05': ['LogD', 'MDCK', 'CYP3A4-inh', 'FDAMDD', 'IGC50', 'SR-MMP', 'SR-p53', 'Fsp3', 'MCE-18', 'Alarm_NMR']
        }
    },
    'ChemAIrank': {
        '显著差异的特征': {
            'p值<0.001': ['CYP2C19-sub', 'T1/2', 'ROA', 'FDAMDD', 'LC50DM', 'nRot', 'nRing', 'nStereo', 'Synth', 'Fsp3', 'MCE-18', 'Natural Product-likeness'],
            'p值<0.01': ['Caco-2', 'CYP3A4-sub', 'Carcinogenicity', 'Respiratory', 'Dense', 'nHD', 'TPSA', 'MaxRing', 'nHet', 'nRig', 'Flex'],
            'p值<0.05': ['Pgp-sub', 'MDCK', 'CYP3A4-inh', 'hERG', 'DILI', 'EI', 'BCF', 'NR-PPAR-gamma', 'nHA']
        }
    }
}

# 将数据转换为适合绘制热图的数据格式
rows = []
for method, categories in data.items():
    for p_value_category, features in categories['显著差异的特征'].items():
        for feature in features:
            rows.append([method, p_value_category, feature])

df = pd.DataFrame(rows, columns=['排序方法', 'p值分类', '特征'])
import os

def save_heatmap_merged(data, method, save_path, dpi=300):
    data_subset = data[data['排序方法'] == method]

    data_dict = {}
    for idx, row in data_subset.iterrows():
        data_dict[row['特征']] = 0
        if row['p值分类'] == 'p值<0.001':
            data_dict[row['特征']] = 3
        elif row['p值分类'] == 'p值<0.01':
            data_dict[row['特征']] = 2
        elif row['p值分类'] == 'p值<0.05':
            data_dict[row['特征']] = 1

    data_merged = pd.DataFrame(list(data_dict.items()), columns=['特征', '显著性差异'])

    cmap = mcolors.ListedColormap(['#FFFFFF', '#FFC300', '#FF5733', '#C70039'])

    heatmap_data = data_merged.pivot(index='特征', columns='显著性差异', values='显著性差异').fillna(0)

    fig, ax = plt.subplots(figsize=(1, 8))
    sns.heatmap(heatmap_data, cmap=cmap, cbar=False, yticklabels=True, xticklabels=False, linewidths=.5, ax=ax)

    # ax.set_title(f'{method} Significant Feature Heatmap')
    ax.set_ylabel('Feature')
    ax.set_xlabel('Significance Level')

    legend_elements = [mpatches.Patch(facecolor='#FFC300', label='p < 0.05'),
                       mpatches.Patch( facecolor='#FF5733', label='p < 0.01'),
                       mpatches.Patch(facecolor='#C70039', label='p < 0.001')]
    ax.legend(handles=legend_elements, bbox_to_anchor=(1.5, 1), loc='upper left', title='Significance Level')

    if not os.path.exists(save_path):
        os.makedirs(save_path)
    fig.savefig(os.path.join(save_path, f'{method}_heatmap.png'), dpi=dpi, bbox_inches='tight')
    plt.show()

save_path = '/home/dell/wangzhen/RealQED(2.17)/test/result/heatmap'

# 为QED排序绘制并保存热图
save_heatmap_merged(df, 'QED', save_path)

# 为ChemAIrank排序绘制并保存热图
save_heatmap_merged(df, 'ChemAIrank', save_path)


