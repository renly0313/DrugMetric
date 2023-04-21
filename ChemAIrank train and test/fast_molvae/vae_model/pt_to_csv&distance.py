import rdkit
import sys
sys.path.append('../../')
import time
import argparse
from argparse import Namespace
from logging import Logger
import torch
from fast_jtnn.datautils import get_data_processed
import pickle
import os
path_chembl = '../data/latent_space_result/pre_zinc250_vocab784/pre_zinc250_vocab784_total_mean_std.pkl'  # path='/root/……/aus_openface.pkl'   pkl文件所在路径
f=open(path_chembl, "rb")#文件所在路径
# chembl_distribution=pickle.load(f, encoding='latin1')[-1]
# print (inf)
# chembl_distribution_mu=chembl_distribution[0]
# chembl_distribution_std=chembl_distribution[1]
chembl_distribution=pickle.load(f, encoding='latin1')#读取pkl内容
chembl_distribution1=chembl_distribution[45]
chembl_distribution_mu1=chembl_distribution1[0]
chembl_distribution_std1=chembl_distribution1[1]
chembl_distribution2=chembl_distribution[1]
chembl_distribution_mu2=chembl_distribution2[0]
chembl_distribution_std2=chembl_distribution2[1]
chembl_distribution3=chembl_distribution[2]
chembl_distribution_mu3=chembl_distribution3[0]
chembl_distribution_std3=chembl_distribution3[1]
chembl_distribution4=chembl_distribution[3]
chembl_distribution_mu4=chembl_distribution4[0]
chembl_distribution_std4=chembl_distribution4[1]
chembl_distribution5=chembl_distribution[4]
chembl_distribution_mu5=chembl_distribution5[0]
chembl_distribution_std5=chembl_distribution5[1]
chembl_distribution6=chembl_distribution[5]
chembl_distribution_mu6=chembl_distribution6[0]
chembl_distribution_std6=chembl_distribution6[1]
chembl_distribution7=chembl_distribution[6]
chembl_distribution_mu7=chembl_distribution7[0]
chembl_distribution_std7=chembl_distribution7[1]
chembl_distribution8=chembl_distribution[7]
chembl_distribution_mu8=chembl_distribution8[0]
chembl_distribution_std8=chembl_distribution8[1]
chembl_distribution9=chembl_distribution[8]
chembl_distribution_mu9=chembl_distribution9[0]
chembl_distribution_std9=chembl_distribution9[1]
chembl_distribution10=chembl_distribution[9]
chembl_distribution_mu10=chembl_distribution10[0]
chembl_distribution_std10=chembl_distribution10[1]
# chembl_distribution11=chembl_distribution[10]
# chembl_distribution_mu11=chembl_distribution11[0]
# chembl_distribution_std11=chembl_distribution11[1]
chembl_distribution_mu = (chembl_distribution_mu1+chembl_distribution_mu2+chembl_distribution_mu3+chembl_distribution_mu4+chembl_distribution_mu5+chembl_distribution_mu6+chembl_distribution_mu7+chembl_distribution_mu8+chembl_distribution_mu9+chembl_distribution_mu10)/10
chembl_distribution_std = (chembl_distribution_std1+chembl_distribution_std2+chembl_distribution_std3+chembl_distribution_std4+chembl_distribution_std5+chembl_distribution_std6+chembl_distribution_std7+chembl_distribution_std8+chembl_distribution_std9+chembl_distribution_std10)/10
f.close()

path_clinical = '../data/latent_space_result/seprated_clinical_drug_mean_var/seprated_clinical_drug_mean_var.pkl'
f=open(path_clinical, "rb")
# clinical_drug_distribution=pickle.load(f, encoding='latin1')[-2]
# clinical_drug_distribution_mu=clinical_drug_distribution[0]
# clinical_drug_distribution_std=clinical_drug_distribution[1]
clinical_drug_distribution=pickle.load(f, encoding='latin1')#读取pkl内容
clinical_drug_distribution1=clinical_drug_distribution[0]
clinical_drug_distribution_mu1=clinical_drug_distribution1[0]
clinical_drug_distribution_std1=clinical_drug_distribution1[1]
clinical_drug_distribution2=clinical_drug_distribution[1]
clinical_drug_distribution_mu2=clinical_drug_distribution2[0]
clinical_drug_distribution_std2=clinical_drug_distribution2[1]
clinical_drug_distribution3=clinical_drug_distribution[2]
clinical_drug_distribution_mu3=clinical_drug_distribution3[0]
clinical_drug_distribution_std3=clinical_drug_distribution3[1]
clinical_drug_distribution4=clinical_drug_distribution[3]
clinical_drug_distribution_mu4=clinical_drug_distribution4[0]
clinical_drug_distribution_std4=clinical_drug_distribution4[1]
clinical_drug_distribution5=clinical_drug_distribution[4]
clinical_drug_distribution_mu5=clinical_drug_distribution5[0]
clinical_drug_distribution_std5=clinical_drug_distribution5[1]
clinical_drug_distribution6=clinical_drug_distribution[5]
clinical_drug_distribution_mu6=clinical_drug_distribution6[0]
clinical_drug_distribution_std6=clinical_drug_distribution6[1]
clinical_drug_distribution7=clinical_drug_distribution[6]
clinical_drug_distribution_mu7=clinical_drug_distribution7[0]
clinical_drug_distribution_std7=clinical_drug_distribution7[1]
clinical_drug_distribution8=clinical_drug_distribution[7]
clinical_drug_distribution_mu8=clinical_drug_distribution8[0]
clinical_drug_distribution_std8=clinical_drug_distribution8[1]
clinical_drug_distribution9=clinical_drug_distribution[8]
clinical_drug_distribution_mu9=clinical_drug_distribution9[0]
clinical_drug_distribution_std9=clinical_drug_distribution9[1]
clinical_drug_distribution10=clinical_drug_distribution[9]
clinical_drug_distribution_mu10=clinical_drug_distribution10[0]
clinical_drug_distribution_std10=clinical_drug_distribution10[1]
# clinical_drug_distribution11=clinical_drug_distribution[10]
# clinical_drug_distribution_mu11=clinical_drug_distribution11[0]
# clinical_drug_distribution_std11=clinical_drug_distribution11[1]
clinical_drug_distribution_mu = (clinical_drug_distribution_mu1+clinical_drug_distribution_mu2+clinical_drug_distribution_mu3+clinical_drug_distribution_mu4+clinical_drug_distribution_mu5+clinical_drug_distribution_mu6+clinical_drug_distribution_mu7+clinical_drug_distribution_mu8+clinical_drug_distribution_mu9+clinical_drug_distribution_mu10)/10
clinical_drug_distribution_std = (clinical_drug_distribution_std1+clinical_drug_distribution_std2+clinical_drug_distribution_std3+clinical_drug_distribution_std4+clinical_drug_distribution_std5+clinical_drug_distribution_std6+clinical_drug_distribution_std7+clinical_drug_distribution_std8+clinical_drug_distribution_std9+clinical_drug_distribution_std10)/10
f.close()

path_zinc = '../data/latent_space_result/seprated_zinc_rand1_mean_var/seprated_zinc_rand1_mean_var.pkl'
f=open(path_zinc, "rb")
# zinc_distribution=pickle.load(f, encoding='latin1')[-2]
# zinc_distribution_mu=zinc_distribution[0]
# zinc_distribution_std=zinc_distribution[1]
zinc_distribution=pickle.load(f, encoding='latin1')#读取pkl内容
# zinc_distribution=pickle.load(f, encoding='latin1')[-11:]#读取pkl内容
zinc_distribution1=zinc_distribution[0]
zinc_distribution_mu1=zinc_distribution1[0]
zinc_distribution_std1=zinc_distribution1[1]
zinc_distribution2=zinc_distribution[1]
zinc_distribution_mu2=zinc_distribution2[0]
zinc_distribution_std2=zinc_distribution2[1]
zinc_distribution3=zinc_distribution[2]
zinc_distribution_mu3=zinc_distribution3[0]
zinc_distribution_std3=zinc_distribution3[1]
zinc_distribution4=zinc_distribution[3]
zinc_distribution_mu4=zinc_distribution4[0]
zinc_distribution_std4=zinc_distribution4[1]
zinc_distribution5=zinc_distribution[4]
zinc_distribution_mu5=zinc_distribution5[0]
zinc_distribution_std5=zinc_distribution5[1]
zinc_distribution6=zinc_distribution[5]
zinc_distribution_mu6=zinc_distribution6[0]
zinc_distribution_std6=zinc_distribution6[1]
zinc_distribution7=zinc_distribution[6]
zinc_distribution_mu7=zinc_distribution7[0]
zinc_distribution_std7=zinc_distribution7[1]
zinc_distribution8=zinc_distribution[7]
zinc_distribution_mu8=zinc_distribution8[0]
zinc_distribution_std8=zinc_distribution8[1]
zinc_distribution9=zinc_distribution[8]
zinc_distribution_mu9=zinc_distribution9[0]
zinc_distribution_std9=zinc_distribution9[1]
zinc_distribution10=zinc_distribution[9]
zinc_distribution_mu10=zinc_distribution10[0]
zinc_distribution_std10=zinc_distribution10[1]
# zinc_distribution11=zinc_distribution[10]
# zinc_distribution_mu11=zinc_distribution11[0]
# zinc_distribution_std11=zinc_distribution11[1]
zinc_distribution_mu = (zinc_distribution_mu1+zinc_distribution_mu2+zinc_distribution_mu3+zinc_distribution_mu4+zinc_distribution_mu5+zinc_distribution_mu6+zinc_distribution_mu7+zinc_distribution_mu8+zinc_distribution_mu9+zinc_distribution_mu10)/10
zinc_distribution_std = (zinc_distribution_std1+zinc_distribution_std2+zinc_distribution_std3+zinc_distribution_std4+zinc_distribution_std5+zinc_distribution_std6+zinc_distribution_std7+zinc_distribution_std8+zinc_distribution_std9+zinc_distribution_std10)/10
f.close()

path_gdb = '../data/latent_space_result/seprated_gdb_rand1_mean_var/seprated_gdb_rand1_mean_var.pkl'
f=open(path_gdb, "rb")
# gdb_distribution=pickle.load(f, encoding='latin1')[-2]
# gdb_distribution_mu=gdb_distribution[0]
# gdb_distribution_std=gdb_distribution[1]
gdb_distribution=pickle.load(f, encoding='latin1')#读取pkl内容
gdb_distribution1=gdb_distribution[0]
gdb_distribution_mu1=gdb_distribution1[0]
gdb_distribution_std1=gdb_distribution1[1]
gdb_distribution2=gdb_distribution[1]
gdb_distribution_mu2=gdb_distribution2[0]
gdb_distribution_std2=gdb_distribution2[1]
gdb_distribution3=gdb_distribution[2]
gdb_distribution_mu3=gdb_distribution3[0]
gdb_distribution_std3=gdb_distribution3[1]
gdb_distribution4=gdb_distribution[3]
gdb_distribution_mu4=gdb_distribution4[0]
gdb_distribution_std4=gdb_distribution4[1]
gdb_distribution5=gdb_distribution[4]
gdb_distribution_mu5=gdb_distribution5[0]
gdb_distribution_std5=gdb_distribution5[1]
gdb_distribution6=gdb_distribution[5]
gdb_distribution_mu6=gdb_distribution6[0]
gdb_distribution_std6=gdb_distribution6[1]
gdb_distribution7=gdb_distribution[6]
gdb_distribution_mu7=gdb_distribution7[0]
gdb_distribution_std7=gdb_distribution7[1]
gdb_distribution8=gdb_distribution[7]
gdb_distribution_mu8=gdb_distribution8[0]
gdb_distribution_std8=gdb_distribution8[1]
gdb_distribution9=gdb_distribution[8]
gdb_distribution_mu9=gdb_distribution9[0]
gdb_distribution_std9=gdb_distribution9[1]
gdb_distribution10=gdb_distribution[9]
gdb_distribution_mu10=gdb_distribution10[0]
gdb_distribution_std10=gdb_distribution10[1]
# gdb_distribution11=gdb_distribution[10]
# gdb_distribution_mu11=gdb_distribution11[0]
# gdb_distribution_std11=gdb_distribution11[1]
gdb_distribution_mu = (gdb_distribution_mu1+gdb_distribution_mu2+gdb_distribution_mu3+gdb_distribution_mu4+gdb_distribution_mu5+gdb_distribution_mu6+gdb_distribution_mu7+gdb_distribution_mu8+gdb_distribution_mu9+gdb_distribution_mu10)/10
gdb_distribution_std = (gdb_distribution_std1+gdb_distribution_std2+gdb_distribution_std3+gdb_distribution_std4+gdb_distribution_std5+gdb_distribution_std6+gdb_distribution_std7+gdb_distribution_std8+gdb_distribution_std9+gdb_distribution_std10)/10
# gdb_distribution_mu = (gdb_distribution_mu1+gdb_distribution_mu2+gdb_distribution_mu3+gdb_distribution_mu4+gdb_distribution_mu5)/5
# gdb_distribution_std = (gdb_distribution_std1+gdb_distribution_std2+gdb_distribution_std3+gdb_distribution_std4+gdb_distribution_std5)/5
f.close()



def Wasserstein(mu1, mu2, std1, std2):
    p1 = torch.sum(torch.pow((mu1 - mu2),2),0)
    p2 = torch.sum(torch.pow(torch.pow(std1,1/2) - torch.pow(std2, 1/2),2) , 0)
    return p1+p2
#clinical_drug
clinical_drug_chembl_wd=Wasserstein(clinical_drug_distribution_mu, chembl_distribution_mu, clinical_drug_distribution_std, chembl_distribution_std)
print('clinical_drug_chembl之间的距离是', clinical_drug_chembl_wd)

clinical_drug_zinc_wd=Wasserstein(clinical_drug_distribution_mu, zinc_distribution_mu, clinical_drug_distribution_std, zinc_distribution_std)
print('clinical_drug_zinc之间的距离是', clinical_drug_zinc_wd)

clinical_drug_gdb_wd=Wasserstein(clinical_drug_distribution_mu, gdb_distribution_mu, clinical_drug_distribution_std, gdb_distribution_std)
print('clinical_drug_gdb之间的距离是', clinical_drug_gdb_wd)

#chembl
chembl_clinical_drug_wd=Wasserstein(chembl_distribution_mu, clinical_drug_distribution_mu, chembl_distribution_std, clinical_drug_distribution_std)
print('chembl_clinical_drug之间的距离是', chembl_clinical_drug_wd)

chembl_zinc_wd=Wasserstein(chembl_distribution_mu, zinc_distribution_mu, chembl_distribution_std, zinc_distribution_std)
print('chembl_zinc之间的距离是', chembl_zinc_wd)

chembl_gdb_wd=Wasserstein(chembl_distribution_mu, gdb_distribution_mu, chembl_distribution_std, gdb_distribution_std)
print('chembl_gdb之间的距离是', chembl_gdb_wd)

#zinc
zinc_clinical_drug_wd=Wasserstein(zinc_distribution_mu, clinical_drug_distribution_mu, zinc_distribution_std, clinical_drug_distribution_std)
print('zinc_clinical_drug之间的距离是', zinc_clinical_drug_wd)

zinc_chembl_wd=Wasserstein(zinc_distribution_mu, chembl_distribution_mu, zinc_distribution_std, chembl_distribution_std)
print('zinc_chembl之间的距离是', zinc_chembl_wd)

zinc_gdb_wd=Wasserstein(zinc_distribution_mu, gdb_distribution_mu, zinc_distribution_std, gdb_distribution_std)
print('zinc_gdb之间的距离是', zinc_gdb_wd)

#gdb
gdb_clinical_drug_wd=Wasserstein(gdb_distribution_mu, clinical_drug_distribution_mu, gdb_distribution_std, clinical_drug_distribution_std)
print('gdb_clinical_drug之间的距离是', gdb_clinical_drug_wd)

gdb_chembl_wd=Wasserstein(gdb_distribution_mu, chembl_distribution_mu, gdb_distribution_std, chembl_distribution_std)
print('gdb_chembl之间的距离是', gdb_chembl_wd)

gdb_zinc_wd=Wasserstein(gdb_distribution_mu, zinc_distribution_mu, gdb_distribution_std, zinc_distribution_std)
print('gdb_zinc之间的距离是', gdb_zinc_wd)