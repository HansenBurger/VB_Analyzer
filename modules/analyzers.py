import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from scipy.signal import detrend, stft, hilbert, get_window, correlate, coherence, signaltools


class AnalyzerBase():
    def __init__(self) -> None:
        pass

    @staticmethod
    def assertions(arr: np.ndarray, axis: int):
        #TODO:
        pass

    @staticmethod
    def zero_mean(arr: np.ndarray, axis: int = 0):
        arr_mean = arr.mean(axis=axis, keepdims=True)
        arr_zero_mean = arr - arr_mean
        return arr_zero_mean

    @classmethod
    def hilbert_trans(cls, arr: np.ndarray, axis: int = 0, **kwargs):
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

    @staticmethod
    def auto_matrix_genrator(arr: np.ndarray, func: any,
                             axes: tuple = (0, 1)) -> np.ndarray:
        n_rows, n_cols = arr.shape[axes[0]], arr.shape[axes[0]]
        auto_grid = np.empty((n_rows, n_cols), dtype=object)

        for i in range(n_rows):
            for j in range(n_rows):
                auto_grid[i, j] = func(arr[i], arr[j])

        cell_shape = auto_grid[0, 0].shape

        if len(cell_shape) == 1 and cell_shape[0] == 1:
            corr_matrix = auto_grid
        else:
            corr_matrix = np.zeros((n_rows, n_cols, *cell_shape))
            for i in range(n_rows):
                for j in range(n_rows):
                    corr_matrix[i, j] = auto_grid[i, j]

        return corr_matrix


class TimeMetrics():
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
    def proxy_metastable_sync(arr: np.ndarray, axes: tuple = (0, 1)) -> tuple:
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
        arr_phases = trans_method[way](arr, axes[1], **kwargs)
        r_s = np.abs(np.mean(np.exp(1j * np.array(arr_phases)), axis=axes[0]))
        r_final = np.mean(r_s)
        return r_final


class Correlations():
    @staticmethod
    def cross_correlation_coefficient(arr_0: np.ndarray,
                                      arr_1: np.ndarray,
                                      mode: str = 'same') -> np.ndarray:
        '''
        Calculate two arrays' Cross-correlation coefficient

        Args:
            arr_0: input array 0 \n
            arr_1: input array 1 \n
            mode: mode setting used in `scipy.signal.correlate`, default: "same"

        Return:
            np.ndarray: cross-correlation coefficient
        '''
        std_0 = np.std(arr_0)
        std_1 = np.std(arr_1)
        arr_0_zero_mean = AnalyzerBase.zero_mean(arr_0)
        arr_1_zero_mean = AnalyzerBase.zero_mean(arr_1)
        correlation = correlate(arr_0_zero_mean, arr_1_zero_mean, mode=mode)

        def normalize_factor(mode: str = mode):
            n_0, n_1 = arr_0.shape[0], arr_1.shape[0]
            N = {
                'same': max(n_0, n_1),
                'valid': max(n_0, n_1) - min(n_0, n_1) + 1,
                'full': n_0 + n_1 - 1
            }
            normalized_factor = min(n_0, n_1) * np.ones(N[mode])
            # TODO: norm_func for full mode
            # normalized_factor = np.vectorize(norm_func)(norm_arr)
            return normalized_factor

        coefficient = correlation / (normalize_factor() * std_0 * std_1)
        return coefficient

    @staticmethod
    def temporal_covariance(arr_0: np.ndarray, arr_1: np.ndarray) -> float:
        '''
        Calculate two arrays' Temporal covariance

        Args:
            arr_0: input array 0 \n
            arr_1: input array 1
        
        Return:
            float: temporal covariance
        '''
        covariance = np.cov(arr_0, arr_1)[1, 0]
        return covariance

    @staticmethod
    def pearson_correlation(arr_0: np.ndarray, arr_1: np.ndarray) -> float:
        '''
        Calculate two arrays' pearson correlation coefficient

        Args:
            arr_0: input array 0 \n
            arr_1: input array 1
        
        Return:
            float: pearson correlation coefficient
        '''
        correlation, p = pearsonr(arr_0, arr_1)
        return correlation


class Coherence():
    @staticmethod
    def scipy_coherence(arr_0: np.ndarray, arr_1: np.ndarray) -> tuple:
        f, Cxy = coherence(arr_0, arr_1, fs=1000, nperseg=64, noverlap=32)
        return f, Cxy

    @staticmethod
    def cross_coherence(arr_0: np.ndarray,
                        arr_1: np.ndarray,
                        fs: float = 1,
                        window_name: str = 'hanning',
                        nperseg: int = 128,
                        noverlap: int = None) -> tuple:
        '''
        Calculate two arrays' coherence

        Args:
            arr_0: Input array 0
            arr_1: Input array 1
            fs: Sampling frequency
            window_name: name of window function, default: "hanning"
            nperseg: length of each segment
            noverlap: length of overlap

        Return:
            tuple: frequency array, coherence array
        '''

        noverlap = noverlap if noverlap else int(nperseg / 2)

        psd_0 = np.zeros(nperseg // 2 + 1)
        psd_1 = np.zeros(nperseg // 2 + 1)
        csd = np.zeros(nperseg // 2 + 1, dtype=complex)

        window = getattr(np, window_name)(nperseg)
        n_segments = (len(arr_0) - noverlap) // (nperseg - noverlap)

        for i in range(n_segments):
            p_start = i * (nperseg - noverlap)
            p_end = p_start + nperseg
            segment_0 = detrend(AnalyzerBase.zero_mean(arr_0[p_start:p_end]))
            segment_1 = detrend(AnalyzerBase.zero_mean(arr_1[p_start:p_end]))
            segment_0 = segment_0 * window
            segment_1 = segment_1 * window

            freq_0 = np.fft.rfft(segment_0)
            freq_1 = np.fft.rfft(segment_1)
            psd_0 += np.real(freq_0 * np.conj(freq_0))
            psd_1 += np.real(freq_1 * np.conj(freq_1))
            csd += freq_0 * np.conj(freq_1)

        psd_0 = psd_0 / n_segments
        psd_1 = psd_1 / n_segments
        csd = csd / n_segments

        coherence = np.abs(csd)**2 / (psd_0 * psd_1)
        f = np.fft.rfftfreq(nperseg, 1 / fs)

        return f, coherence