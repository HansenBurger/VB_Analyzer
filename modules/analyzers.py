import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from scipy.signal import stft, hilbert, get_window, correlate, coherence, signaltools


class AnalyzerBase():
    def __init__(self) -> None:
        pass

    @classmethod
    def assertions(cls, arr: np.ndarray, axis: int):
        pass

    @classmethod
    def zero_mean(cls, arr: np.ndarray, axis: int = 0):
        assert arr.shape[axis] > 1
        assert type(arr) == np.ndarray

        arr_mean = arr.mean(axis=axis, keepdims=True)
        arr_zero_mean = arr - arr_mean
        return arr_zero_mean

    @classmethod
    def hilbert_trans(cls, arr: np.ndarray, axis: int = 0):
        arr_zero_mean = cls.zero_mean(arr, axis)
        arr_norm = arr_zero_mean / np.std(arr, axis=axis, keepdims=True)
        analytic_signal = hilbert(arr_norm)
        phases = np.angle(analytic_signal)
        return phases

    @classmethod
    def stft_trans(cls,
                   arr: np.ndarray,
                   axis: int = 0,
                   fs: float = 1000,
                   nperseg: int = 128,
                   noverlap: int = 64):
        _, _, Zxx = stft(arr,
                         axis=axis,
                         fs=fs,
                         nperseg=nperseg,
                         noverlap=noverlap)
        phases = np.angle(Zxx)
        avg_phases = np.mean(phases, axis=1)  # 平均所有窗口的相位信息
        return avg_phases


class TimeMetrics(AnalyzerBase):
    @staticmethod
    def global_variance(arr: np.ndarray, axes: tuple = (0, 1)) -> float:
        '''
        Calculte the global variance of nodes output time series
        
        Args:
            arr: time series of nodes output \n
            axes: axis information of the input data, default: (s, t)
        
        Returns:
            float: global variance of nodes
        '''
        assert len(arr.shape) == 2, "Wrong input shape"
        arr_zero_mean = AnalyzerBase.zero_mean(arr, axes[1])
        global_var = np.var(arr_zero_mean)
        return global_var

    @staticmethod
    def proxy_metastable_sync(arr: np.ndarray, axes: tuple = (0, 1),
                              **kwargs) -> tuple:
        '''
        Calculate the spatial coherence of a group of nodes

        Args:
            arr: time series of nodes output \n
            axes: axis information of the input data, default: (s, t)

        Returns:
            tuple: metastablility, synchrony
        '''
        assert len(arr.shape) == 2, "Wrong input shape"
        arr_zero_mean = AnalyzerBase.zero_mean(arr, axes[0])
        proxy_space_sync = np.mean(np.abs(arr_zero_mean), axis=axes[0])
        metastablility = np.std(proxy_space_sync)
        synchrony = 1 / np.mean(proxy_space_sync)

        return metastablility, synchrony

    @staticmethod
    def kuramoto_index(arr: np.ndarray,
                       axes: tuple = (0, 1),
                       way: str = 'hilbert',
                       **kwargs) -> float:
        '''
        Calculate the Kuramoto index for nodes output time series

        Args:
            arr: time series of nodes output \n
            axes: axis information of the input data, default: (s, t) \n
            way: method of extracting phase from time series, default: "hilbert" \n
            kawrgs: other parameters for phase extraction

        Returns:
            float: Kuramoto index of nodes
        '''
        assert len(arr.shape) == 2, "Wrong input shape"
        trans_method = {
            'hilbert': AnalyzerBase.hilbert_trans,
            'stft': AnalyzerBase.stft_trans
        }
        arr_phases = trans_method(arr, axes[1], **kwargs)
        r_s = np.abs(np.mean(np.exp(1j * np.array(arr_phases)), axis=axes[0]))
        r_final = np.mean(r_s)
        return r_final


class Correlations(AnalyzerBase):
    @classmethod
    def cross_correlation_coefficient(cls, arr_0: np.ndarray,
                                      arr_1: np.ndarray):
        print(1)

    @classmethod
    def atuo_correlation_matrix(arr: np.ndarray,
                         mode: str = 'same',
                         axes: tuple = (0, 1)) -> np.ndarray:
        arr_zero_mean = AnalyzerBase.zero_mean(arr, axes[1])
        auto_correlate = correlate(arr_zero_mean, arr_zero_mean, mode='same')
        auto_coefficient = auto_correlate / np.sum(arr_zero_mean**2)
        a = 1
        pass

    @staticmethod
    def temporal_covariance(
        arr: np.ndarray, axes: tuple = (0, 1)) -> np.ndarray:
        pass

    @staticmethod
    def pearson_corr_coefficients(
        arr: np.ndarray, axes: tuple = (0, 1)) -> np.ndarray:
        pass


class Coherence(AnalyzerBase):
    @staticmethod
    def cross_coherence(arr):
        pass