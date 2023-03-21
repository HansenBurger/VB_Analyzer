import numpy as np
from modules.analyzers import TimeMetrics, Correlations
from matplotlib import pyplot as plt

npy_path = r'src_data\simul_emp.npy'
npy_data = np.load(npy_path)
exp_0 = npy_data[0]

# g_var = TimeMetrics.global_variance(exp_0)
# print("Subject 1's Global Variance:{0}".format(round(g_var, 3)))

# stable, sync = TimeMetrics.proxy_metastable_sync(exp_0)
a = exp_0[0:20, :]
# g_var = TimeMetrics.global_variance(a)
# stable, sync = TimeMetrics.proxy_metastable_sync(a)
# print("Subject 1's Global Variance:{0}".format(round(g_var, 3)))
# TimeMetrics.kuramoto_index(a)
# TimeMetrics.kuramoto_index(a, way='stft', nperseg=64, noverlap=32)
Correlations.cross_correlation_coefficient(a[0], a[1])
a = 1