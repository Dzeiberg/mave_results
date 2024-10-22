import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from mave_calibration.skew_normal import density_utils
from tqdm import tqdm

def get_sample_density(X, results,sample_names):
    densities = []
    for result in tqdm(results,leave=False):
        iter_densities = [density_utils.joint_densities(X, result.get("component_params"),
                                                        result.get("weights")[i]).sum(0) \
                          for i in range(len(sample_names))]
        densities.append(iter_densities)
    D = np.stack(densities,axis=1)
    return D

def fit_fig(X,S,sample_names,ax, results=None,priors=[]):
    N_Samples = S.shape[1]
    std=X.std()
    rng = np.linspace(X.min() - 2 * std,X.max() + 2 * std,3000)
    palette = sns.color_palette("pastel", N_Samples)
    palette_3 = sns.color_palette("dark", N_Samples)
    palette_2 = sns.color_palette("bright", N_Samples)
    D = None
    if results is not None:
        D = get_sample_density(rng, results, sample_names)
    bins = np.linspace(X.min(),X.max(),25)
    for i in range(N_Samples):
        name = sample_names[i]
        label = f"{name} (n={S[:,i].sum():,d})"
        if len(priors) and (name == "gnomAD"):
            label += f"\n(median prior={np.quantile(priors,.5):.2f})"
        sns.histplot(X[S[:,i]],ax=ax[i],stat='density',color=palette[i],bins=bins,label=label)
        if D is not None:
            ax[i].plot(rng, D[i].mean(0),color=palette_3[i],)
            q = np.nanquantile(D[i], [0.025, .975], axis=0)
            ax[i].fill_between(rng, q[0], q[1], alpha=.5, color=palette_2[i])
        ax[i].legend(loc='upper left')
