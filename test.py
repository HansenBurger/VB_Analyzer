import numpy as np
from scipy.signal import correlate

in1 = np.array([0, 2, 0, 0, 2])
in2 = np.array([2, 0, 2])

n_in1, n_in2 = in1.shape[0], in2.shape[0]

a = n_in2 // 2

normalized_index = np.arange(1, n_in1 + n_in2, 1)
normalized_cond_ = lambda x: x if x < min(
    n_in1, n_in2) else n_in1 + n_in2 - x if x > max(n_in1, n_in2) else min(
        n_in1, n_in2)
normalized_array = np.vectorize(normalized_cond_)(normalized_index)

zm = lambda x: x - np.mean(x)
var_ = lambda x: np.sqrt(np.sum((zm(x))**2))
in1_, in2_ = zm(in1), zm(in2)

factor_1 = var_(in1_) * var_(in2_)
factor_2 = np.std(in1) * np.std(in2) * min(len(in1_), len(in2_))
factor_full = np.array(
    [min(i,
         min(n_in1, n_in2) - 1) + 1 for i in range(1, n_in1 + n_in2)])
c = correlate(in1_, in2_, mode='full')
c_1 = correlate(in1_, in2_, mode='same')

lags = np.arange(-3 + 1, 5)

a = 1