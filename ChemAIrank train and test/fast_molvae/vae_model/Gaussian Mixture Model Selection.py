import numpy as np
import itertools
import pickle
from scipy import linalg
import matplotlib.pyplot as plt
import matplotlib as mpl

from sklearn import mixture
import logging
logger = logging.getLogger(__name__)

logger.info("calculating gmm centers")
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

lowest_bic = np.infty
bic = []
n_components_range = range(1, 7)
cv_types = ["spherical", "tied", "diag", "full"]
for cv_type in cv_types:
    for n_components in n_components_range:
        # Fit a Gaussian mixture with EM
        gmm = mixture.GaussianMixture(
            n_components=n_components, covariance_type=cv_type
        )
        gmm.fit(X)
        bic.append(gmm.bic(X))
        if bic[-1] < lowest_bic:
            lowest_bic = bic[-1]
            best_gmm = gmm

bic = np.array(bic)
color_iter = itertools.cycle(["navy", "turquoise", "cornflowerblue", "darkorange"])
clf = best_gmm
bars = []

# Plot the BIC scores
plt.figure(figsize=(8, 6))
spl = plt.subplot(2, 1, 1)
for i, (cv_type, color) in enumerate(zip(cv_types, color_iter)):
    xpos = np.array(n_components_range) + 0.2 * (i - 2)
    bars.append(
        plt.bar(
            xpos,
            bic[i * len(n_components_range) : (i + 1) * len(n_components_range)],
            width=0.2,
            color=color,
        )
    )
plt.xticks(n_components_range)
plt.ylim([bic.min() * 1.01 - 0.01 * bic.max(), bic.max()])
plt.title("BIC score per model")
xpos = (
    np.mod(bic.argmin(), len(n_components_range))
    + 0.65
    + 0.2 * np.floor(bic.argmin() / len(n_components_range))
)
plt.text(xpos, bic.min() * 0.97 + 0.03 * bic.max(), "*", fontsize=14)
spl.set_xlabel("Number of components")
spl.legend([b[0] for b in bars], cv_types)

# Plot the winner
splot = plt.subplot(2, 1, 2)
Y_ = clf.predict(X)
for i, (mean, cov, color) in enumerate(zip(clf.means_, clf.covariances_, color_iter)):
    v, w = linalg.eigh(cov)
    if not np.any(Y_ == i):
        continue
    plt.scatter(X[Y_ == i, 0], X[Y_ == i, 1], 0.8, color=color)

    # Plot an ellipse to show the Gaussian component
    angle = np.arctan2(w[0][1], w[0][0])
    angle = 180.0 * angle / np.pi  # convert to degrees
    v = 2.0 * np.sqrt(2.0) * np.sqrt(v)
    ell = mpl.patches.Ellipse(mean, v[0], v[1], 180.0 + angle, color=color)
    ell.set_clip_box(splot.bbox)
    ell.set_alpha(0.5)
    splot.add_artist(ell)

plt.xticks(())
plt.yticks(())
plt.title(
    f"Selected GMM: {best_gmm.covariance_type} model, "
    f"{best_gmm.n_components} components"
)
plt.subplots_adjust(hspace=0.35, bottom=0.02)
plt.show()
