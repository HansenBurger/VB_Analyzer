import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from modules.analyzers import AnalyzerBase, TimeMetrics, Correlations, Coherence

npy_path = r'src_data\simul_emp.npy'
npy_data = np.load(npy_path)
exp_0 = npy_data[0]

variance = TimeMetrics.global_variance(exp_0)
stability, synchrony = TimeMetrics.proxy_metastable_sync(exp_0)
kuramoto = TimeMetrics.kuramoto_index(exp_0)

print("Subject 1:\n")
print("\t Global Variance:\t{0}".format(round(variance, 3)))
print("\t Proxy Metastability:\t{0}".format(round(stability, 3)))
print("\t Proxy synchrony:\t{0}".format(round(synchrony, 3)))
print("\t Kuramoto index:\t{0}".format(round(kuramoto, 3)))

exp_0_partial = exp_0[0:25, :]

a, b = Coherence.cross_coherence(exp_0_partial[0],
                                 exp_0_partial[1],
                                 nperseg=64)

corr_base = AnalyzerBase.auto_matrix_genrator
corr_func_0 = Correlations.cross_correlation_coefficient
corr_func_1 = Correlations.temporal_covariance
corr_fucn_2 = Correlations.pearson_correlation
fig, (ax_0, ax_1, ax_2) = plt.subplots(1, 3, figsize=(16, 5))

coefficient = np.max(corr_base(exp_0_partial, corr_func_0), axis=2)
sns.heatmap(coefficient, ax=ax_0, cmap="coolwarm")
ax_0.set_title("Cross-correlation(Max)")

covariance = corr_base(exp_0_partial, corr_func_1)
sns.heatmap(covariance, ax=ax_1, cmap="rocket_r")
ax_1.set_title("Temporal covariance")

pearson = corr_base(exp_0_partial, corr_fucn_2)
sns.heatmap(pearson, ax=ax_2, cmap="coolwarm")
ax_2.set_title("Pearson correlation")

fig.tight_layout()
plt.show()
plt.close()
