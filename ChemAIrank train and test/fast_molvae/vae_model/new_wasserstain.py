import numpy as np

import pickle
from wgpot import Wasserstein_GP
import torch
from scipy.linalg import sqrtm
import logging
logger = logging.getLogger(__name__)

logger.info("calculating gmm centers")
path_drug = '../data/latent_space_result/val_mean_1206_l64/seprated_chembl_val_mean.pkl'  # path='/root/……/aus_openface.pkl'   pkl文件所在路径
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
gmm_path = open('../data/latent_space_result/gmm/1206_l64/full.pkl','rb')
best_gmm  = pickle.load(gmm_path)
gmm = best_gmm
gmm_classes = best_gmm.predict(X)
gmm_centers = best_gmm.means_
gmm_covariances = gmm.covariances_
points = gmm_centers
gmm_centers = torch.tensor(gmm_centers)
# gmm_covariances = torch.tensor(gmm_covariances)

p0 = np.sqrt(np.linalg.norm((gmm_covariances[0]), 2))
p1 = np.sqrt(np.linalg.norm((gmm_covariances[1]), 2))
p2 = np.sqrt(np.linalg.norm((gmm_covariances[2]), 2))
p3 = np.sqrt(np.linalg.norm((gmm_covariances[3]), 2))

# def Wasserstein(mu1, mu2, cov1, cov2):#中文网站
#     p1 = torch.sum(torch.pow((mu1 - mu2),2),0)
#     p2 = torch.sum(torch.pow(torch.pow(abs(cov1),1/2) - torch.pow(abs(cov2), 1/2),2))
#     return p1+p2

# def Wasserstein(mu1, mu2, cov1, cov2):#fro范数
#     p1 = torch.sum(torch.pow((mu1 - mu2),2),0)
#     p2 = np.linalg.norm((cov1 - cov2), 'fro')
#     return p1+p2

# def Wasserstein(mu1, mu2, cov1, cov2):#spherical
#     p1 = torch.sum(torch.pow((mu1 - mu2),2),0)
#     p2 = torch.sum(torch.pow((cov1 - cov2),2),0)
#     return p1+p2



# gp_0 = (gmm_centers[0].numpy(), gmm_covariances[0].numpy())
# gp_1 = (gmm_centers[1].numpy(), gmm_covariances[1].numpy())
# # mu_0/mu_1 (ndarray (n, 1)) is the mean of one Gaussian Process
# # K_0/K_1 (ndarray (n, n)) is the covariance matrix of one
# # Gaussain Process
#
# wd_gp = Wasserstein_GP(gp_0, gp_1)
def Wasserstein(mu1, mu2, cov1, cov2):#点乘
    p1 = torch.sum(torch.pow((mu1 - mu2),2),0)
    p2 = np.trace(cov1) + np.trace(cov2) - 2*np.trace(sqrtm((np.dot(np.dot(sqrtm(cov2), cov1), sqrtm(cov2)))))
    d = np.sqrt(p1+p2)
    return d

num_to_str = {
    0: "zinc",
    1: "clinical_drug",
    2: "gdb",
    3: "chembl",
}
# for i, color in zip(range(gmm_centers), gmm_covariances):
#     datasets = num_to_str.get(i)
# #clinical_drug
clinical_drug_chembl_wd=Wasserstein(gmm_centers[1], gmm_centers[3], gmm_covariances[1], gmm_covariances[3])
print('clinical_drug_chembl之间的距离是', clinical_drug_chembl_wd)

clinical_drug_zinc_wd=Wasserstein(gmm_centers[1], gmm_centers[0], gmm_covariances[1], gmm_covariances[0])
print('clinical_drug_zinc之间的距离是', clinical_drug_zinc_wd)

clinical_drug_gdb_wd=Wasserstein(gmm_centers[1], gmm_centers[2], gmm_covariances[1], gmm_covariances[2])
print('clinical_drug_gdb之间的距离是', clinical_drug_gdb_wd)

#chembl
chembl_clinical_drug_wd=Wasserstein(gmm_centers[0], gmm_centers[1], gmm_covariances[0], gmm_covariances[1])
print('chembl_clinical_drug之间的距离是', chembl_clinical_drug_wd)

chembl_zinc_wd=Wasserstein(gmm_centers[0], gmm_centers[3], gmm_covariances[0], gmm_covariances[3])
print('chembl_zinc之间的距离是', chembl_zinc_wd)

chembl_gdb_wd=Wasserstein(gmm_centers[0], gmm_centers[2], gmm_covariances[0], gmm_covariances[2])
print('chembl_gdb之间的距离是', chembl_gdb_wd)

#zinc
zinc_clinical_drug_wd=Wasserstein(gmm_centers[3], gmm_centers[1], gmm_covariances[3], gmm_covariances[1])
print('zinc_clinical_drug之间的距离是', zinc_clinical_drug_wd)

zinc_chembl_wd=Wasserstein(gmm_centers[3], gmm_centers[0], gmm_covariances[3], gmm_covariances[0])
print('zinc_chembl之间的距离是', zinc_chembl_wd)

zinc_gdb_wd=Wasserstein(gmm_centers[3], gmm_centers[2], gmm_covariances[3], gmm_covariances[2])
print('zinc_gdb之间的距离是', zinc_gdb_wd)

#gdb
gdb_clinical_drug_wd=Wasserstein(gmm_centers[2], gmm_centers[1], gmm_covariances[2], gmm_covariances[1])
print('gdb_clinical_drug之间的距离是', gdb_clinical_drug_wd)

gdb_chembl_wd=Wasserstein(gmm_centers[2], gmm_centers[3], gmm_covariances[2], gmm_covariances[3])
print('gdb_chembl之间的距离是', gdb_chembl_wd)

gdb_zinc_wd=Wasserstein(gmm_centers[2], gmm_centers[0], gmm_covariances[2], gmm_covariances[0])
print('gdb_zinc之间的距离是', gdb_zinc_wd)

