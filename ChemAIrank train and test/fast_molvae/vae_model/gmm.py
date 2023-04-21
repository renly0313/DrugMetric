#!/usr/bin/env python
# coding: utf-8
import numpy as np
import pickle
from tqdm.auto import tqdm
import logging
from sklearn.mixture import GaussianMixture, BayesianGaussianMixture
logger = logging.getLogger(__name__)


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
dim=4
calc_times=100
logger.info("calculating gmm centers")
X =np.vstack((drug_mean, chembl_mean, zinc_mean, gdb_mean))
gmm_path = "../data/latent_space_result/BGMM/1206_l64/spherical.pkl"
# if gmm_path.exists():
#     logger.info(f"loading {gmm_path}")
#     with gmm_path.open("rb") as f:
#         best_gmm = pickle.load(f)
#         best_aic = best_gmm.aic(X)
# else:
best_aic = np.inf
pbar = tqdm(range(calc_times))
for i in pbar:
    gmm = GaussianMixture(n_components = dim, covariance_type="spherical", random_state=10, reg_covar=1,init_params='random').fit(X)
    if gmm.aic(X) < best_aic:
        best_aic = gmm.aic(X)
        best_gmm = gmm
    pbar.set_description(
        "[" + "⠸⠴⠦⠇⠋⠙"[i % 6] + "]" + f"{best_aic:.2f}")

with open(gmm_path, "wb") as f:
    pickle.dump(best_gmm, f)

logger.info(f"best aic : {best_aic}")
gmm = best_gmm
aic = best_aic
gmm_classes = best_gmm.predict(X)
gmm_centers = best_gmm.means_
gmm_covariances = gmm.covariances_
points = gmm_centers
