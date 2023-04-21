#!/usr/bin/env python
# coding: utf-8
import numpy as np
import pickle
from matplotlib import pyplot as plt
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc
from itertools import cycle

path_drug = '../../train/data/latent_space_result/seprated_mean_1206_l64/seprated_clinical_drug_mean.pkl'  # path='/root/……/aus_openface.pkl'   pkl文件所在路径
f=open(path_drug, "rb")#文件所在路径
drug_mean=pickle.load(f, encoding='latin1')
path_chembl = '../../train/data/latent_space_result/seprated_mean_1206_l64/seprated_chembl_mean.pkl'  # path='/root/……/aus_openface.pkl'   pkl文件所在路径
f=open(path_chembl, "rb")#文件所在路径
chembl_mean=pickle.load(f, encoding='latin1')
path_zinc = '../../train/data/latent_space_result/seprated_mean_1206_l64/seprated_zinc_mean.pkl'  # path='/root/……/aus_openface.pkl'   pkl文件所在路径
f=open(path_zinc, "rb")#文件所在路径
zinc_mean=pickle.load(f, encoding='latin1')
path_gdb = '../../train/data/latent_space_result/seprated_mean_1206_l64/seprated_gdb_mean.pkl'  # path='/root/……/aus_openface.pkl'   pkl文件所在路径
f=open(path_gdb, "rb")#文件所在路径
gdb_mean=pickle.load(f, encoding='latin1')
X =np.vstack((drug_mean, chembl_mean, zinc_mean, gdb_mean))
# def calc_gmm(self, dim=10, calc_times=3, force=False):


y0=np.ones(3622)
y1=np.ones(3622)*3
y2=np.zeros(3622)
y3=np.ones(3622)*2


Y = np.hstack((y0,y1,y2,y3))
gmm_path = open(r'/home/dell/wangzhen/RealQED(2.17)/data/latent_space_result/seprated_mean_1206_l64/full.pkl','rb')
best_gmm  = pickle.load(gmm_path)

gmm = best_gmm
gmm_classes = gmm.predict(X)
gmm_classes
gmm_centers = gmm.means_
gmm_covariances = gmm.covariances_
#
# print(np.unique(gmm_classes[:3622], return_counts=True))
# print(np.unique(gmm_classes[3622:7244], return_counts=True))
# print(np.unique(gmm_classes[7244:10866], return_counts=True))
# print(np.unique(gmm_classes[10866:], return_counts=True))

points = gmm_centers
# 将标签二值化
y_predict = label_binarize(gmm_classes, classes=[0, 1, 2, 3])
y_true = label_binarize(Y, classes=[0, 1, 2, 3])
n_classes = y_predict.shape[1]
# 计算每一类的ROC
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_true[:, i], y_predict[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Compute micro-average ROC curve and ROC area（方法二）
fpr["micro"], tpr["micro"], _ = roc_curve(y_true.ravel(), y_predict.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

# Compute macro-average ROC curve and ROC area（方法一）
# First aggregate all false positive rates
all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

# Then interpolate all ROC curves at this points
mean_tpr = np.zeros_like(all_fpr)
for i in range(n_classes):
    mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

# Finally average it and compute AUC
mean_tpr /= n_classes
fpr["macro"] = all_fpr
tpr["macro"] = mean_tpr
roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

# Plot all ROC curves
lw=2
plt.figure()
plt.plot(fpr["micro"], tpr["micro"],
         label='average ROC curve (area = {0:0.2f})'
               ''.format(roc_auc["micro"]),
         color='deeppink', linestyle=':', linewidth=4)

# plt.plot(fpr["macro"], tpr["macro"],
#          label='macro-average ROC curve (area = {0:0.2f})'
#                ''.format(roc_auc["macro"]),
#          color='navy', linestyle=':', linewidth=4)
num_to_str = {
    0: "zinc",
    1: "clinical_drug",
    2: "gdb",
    3: "chembl",
}

colors = cycle(['aqua', 'darkorange', 'cornflowerblue','pink'])
for i, color in zip(range(n_classes), colors):
    datasets = num_to_str.get(i)
    plt.plot(fpr[i], tpr[i], color=color, lw=lw,
             label='ROC curve of {0} (area = {1:0.2f})'
             ''.format(datasets, roc_auc[i]))

plt.plot([0, 1], [0, 1], 'k--', lw=lw)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
# plt.title('Some extension of Receiver operating characteristic to multi-class')
plt.legend(loc="lower right")

import os
if not os.path.exists('/home/dell/wangzhen/RealQED(2.17)/test/result/AUROC'):
    os.makedirs('/home/dell/wangzhen/RealQED(2.17)/test/result/AUROC')
plt.savefig('/home/dell/wangzhen/RealQED(2.17)/test/result/AUROC/multi_class.png', dpi=300)

plt.show()
plt.show()
