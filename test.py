import numpy as np
from scipy.signal import correlate

# 创建一个简单的二维数组
data = np.array([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
])

def cross_correlation_matrix(arr):
    n_rows = arr.shape[0]
    corr_matrix = np.zeros((n_rows, n_rows))
    
    for i in range(n_rows):
        for j in range(n_rows):
            row_i = arr[i] - np.mean(arr[i])
            row_j = arr[j] - np.mean(arr[j])
            normalization_factor = np.sqrt(np.sum(row_i**2) * np.sum(row_j**2))
            corr_matrix[i, j] = np.correlate(row_i, row_j) / normalization_factor
    
    return corr_matrix

result = cross_correlation_matrix(data)
print(result)