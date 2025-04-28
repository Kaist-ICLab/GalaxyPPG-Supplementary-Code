from typing import List

import numpy as np
from scipy import signal
import os
import pandas as pd
import sys
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)
from config import RESULTS_DIR, WINDOW_DIR


class SVDDenoising:
    def __init__(self):
        """
           Initialize SVD denoising algorithm with default parameters.

           Attributes
           ----------
           fs_galaxy : int
               Galaxy Watch sampling frequency (25 Hz)
           target_length_galaxy : int
               Target length for Galaxy Watch signal processing (200 samples)
           fs_e4 : int
               E4 device sampling frequency (64 Hz)
           target_length_e4 : int
               Target length for E4 signal processing (512 samples)
           min_singular_values : int
               Minimum number of singular values to retain
           energy_threshold : float
               Threshold for singular value energy retention (0.0 to 1.0)
           motion_threshold : float
               Threshold for motion artifact detection (0.0 to 1.0)
           lowcut : float
               Lower cutoff frequency for bandpass filter (Hz)
           highcut : float
               Higher cutoff frequency for bandpass filter (Hz)
           nfft : int
               Number of points for FFT computation
           """
        self.fs_galaxy = 25  # Hz
        self.target_length_galaxy = 200

        self.fs_e4 = 64  # Hz
        self.target_length_e4 = 512

        self.min_singular_values = 5
        self.energy_threshold = 0.7
        self.motion_threshold = 0.5

        self.lowcut = 0.5  # Hz
        self.highcut = 4  # Hz

        self.nfft = 1024
    def process_dataframe(self, df: pd.DataFrame,
                          ppg_col: str = 'ppg',
                          acc_cols: List[str] = ['acc_x', 'acc_y', 'acc_z'],
                          device_type: str = 'galaxy') -> pd.DataFrame:
        """
        Process PPG and accelerometer data from a DataFrame.

        Parameters
        ----------
        df : pandas.DataFrame
            Input DataFrame containing PPG and accelerometer data
        ppg_col : str, default='ppg'
            Name of the column containing PPG data
        acc_cols : list of str, default=['acc_x', 'acc_y', 'acc_z']
            Names of columns containing accelerometer data
        device_type : str, default='galaxy'
            Type of device ('galaxy' or 'e4')

        Returns
        -------
        pandas.DataFrame
            DataFrame with added column for denoised signal

       """
        # Validate input columns
        if ppg_col not in df.columns:
            raise ValueError(f"PPG column '{ppg_col}' not found in DataFrame")
        for col in acc_cols:
            if col not in df.columns:
                raise ValueError(f"Accelerometer column '{col}' not found in DataFrame")

        results = []
        for _, row in df.iterrows():
            try:
                ppg = np.array(row[ppg_col])
                acc_x = np.array(row[acc_cols[0]])
                acc_y = np.array(row[acc_cols[1]])
                acc_z = np.array(row[acc_cols[2]])

                fs = self.fs_galaxy if device_type.lower() == 'galaxy' else self.fs_e4
                denoised = self.denoise(ppg, acc_x, acc_y, acc_z, fs)

                results.append({
                    'denoised_signal': ';'.join(map(str, denoised))
                })
            except Exception as e:
                print(f"Error processing row: {str(e)}")
                results.append({
                    'denoised_signal': None
                })

        result_df = pd.concat([df, pd.DataFrame(results)], axis=1)
        return result_df
    def denoise(self, ppg, acc_x, acc_y, acc_z, fs):
        """
               Denoise PPG signal using SVD-based method.

               Parameters
               ----------
               ppg : numpy.ndarray
                   Raw PPG signal
               acc_x : numpy.ndarray
                   X-axis acceleration data
               acc_y : numpy.ndarray
                   Y-axis acceleration data
               acc_z : numpy.ndarray
                   Z-axis acceleration data
               fs : float
                   Sampling frequency in Hz

               Returns
               -------
               numpy.ndarray
                   Denoised PPG signal
        """
        try:
            ppg_processed, acc_x_processed, acc_y_processed, acc_z_processed = \
                self.preprocess_signals(ppg, acc_x, acc_y, acc_z, fs)

            if ppg_processed is None:
                return self.mean_normalize(ppg)

            H_ppg = self.build_hankel_matrix(ppg_processed)
            if H_ppg is None:
                return self.mean_normalize(ppg)

            U, S, Vt = np.linalg.svd(H_ppg, full_matrices=False)

            ppg_fft = np.fft.rfft(ppg_processed, n=self.nfft)
            acc_ffts = [
                np.fft.rfft(acc, n=self.nfft) for acc in
                [acc_x_processed, acc_y_processed, acc_z_processed]
            ]

            motion_weight = self.compute_motion_weights(ppg_fft, acc_ffts)

            H_reconstructed = self.select_singular_values(S, U, Vt, motion_weight)

            L, K = H_reconstructed.shape
            N = L + K - 1
            reconstructed = np.zeros(N)
            count = np.zeros(N)

            for i in range(L):
                for j in range(K):
                    n = i + j
                    reconstructed[n] += H_reconstructed[i, j]
                    count[n] += 1

            reconstructed /= count

            denoised = self.mean_normalize(reconstructed)

            return self.mean_normalize(denoised)

        except Exception as e:
            print({str(e)})
            return self.mean_normalize(ppg)
    def mean_normalize(self, x):
        if len(x) == 0:
            return x
        mean = np.mean(x)
        std_dev = np.std(x) + 1e-10
        return (x - mean) / std_dev
    def preprocess_signals(self, ppg, acc_x, acc_y, acc_z, fs):
        try:
            target_length = self.target_length_galaxy if fs == self.fs_galaxy else self.target_length_e4
            if len(ppg) != target_length:
                ppg = signal.resample(ppg, target_length)
            if len(acc_x) != target_length:
                acc_x = signal.resample(acc_x, target_length)
            if len(acc_y) != target_length:
                acc_y = signal.resample(acc_y, target_length)
            if len(acc_z) != target_length:
                acc_z = signal.resample(acc_z, target_length)
            nyq = 0.5 * fs
            low = self.lowcut / nyq
            high = self.highcut / nyq
            b, a = signal.butter(4, [low, high], btype='band')

            ppg_filtered = signal.filtfilt(b, a, ppg)
            acc_x_filtered = signal.filtfilt(b, a, acc_x)
            acc_y_filtered = signal.filtfilt(b, a, acc_y)
            acc_z_filtered = signal.filtfilt(b, a, acc_z)

            ppg_processed = self.mean_normalize(ppg_filtered)
            acc_x_processed = self.mean_normalize(acc_x_filtered)
            acc_y_processed = self.mean_normalize(acc_y_filtered)
            acc_z_processed = self.mean_normalize(acc_z_filtered)

            return ppg_processed, acc_x_processed, acc_y_processed, acc_z_processed

        except Exception as e:
            print(f" {str(e)}")
            return None, None, None, None
    def compute_motion_weights(self, ppg_fft, acc_ffts):
        try:
            ppg_magnitude = np.abs(ppg_fft)
            weights = []

            for acc_fft in acc_ffts:
                acc_magnitude = np.abs(acc_fft)
                min_len = min(len(ppg_magnitude), len(acc_magnitude))
                corr = np.corrcoef(ppg_magnitude[:min_len], acc_magnitude[:min_len])[0, 1]
                if np.isnan(corr):
                    corr = 0
                weight = 1 - min(abs(corr), self.motion_threshold)
                weights.append(weight)

            return np.mean(weights)

        except Exception as e:
            print(f"{str(e)}")
            return 0.5
    def select_singular_values(self, S, U, Vt, motion_weight):
        try:
            energy_ratio = np.cumsum(S ** 2) / np.sum(S ** 2)

            adjusted_threshold = self.energy_threshold * (1 + 0.1 * motion_weight)

            k = max(self.min_singular_values,
                    np.searchsorted(energy_ratio, adjusted_threshold) + 1)

            k = min(k, len(S))

            reconstructed = np.zeros_like(U @ np.diag(S) @ Vt)
            for i in range(k):
                adjusted_s = S[i] * (1 + 0.2 * motion_weight)
                reconstructed += adjusted_s * np.outer(U[:, i], Vt[i, :])

            return reconstructed

        except Exception as e:
            print(f"{str(e)}")
            return U @ np.diag(S) @ Vt
    def build_hankel_matrix(self, signal):
        try:
            N = len(signal)
            L = N // 2
            K = N - L + 1

            H = np.zeros((L, K))
            for i in range(L):
                H[i, :] = signal[i:i + K]

            return H

        except Exception as e:
            print( {str(e)})
            return None

def process_dataset():

    denoiser = SVDDenoising()

    files = [f for f in os.listdir(WINDOW_DIR) if f.endswith('GD.csv')]

    for filename in files:
        print( {filename})

        try:
            df = pd.read_csv(os.path.join(WINDOW_DIR, filename))

            results = {
                'denoisedGalaxy': [],
                'denoisedE4': []
            }

            for i, row in df.iterrows():
                try:
                    if pd.notna(row['galaxyPPG']) and pd.notna(row['galaxyACC']):
                        ppg = - np.array([float(x) for x in row['galaxyPPG'].split(';') if x.strip()])
                        acc = np.array([float(x) for x in row['galaxyACC'].split(';') if x.strip()]).reshape(-1, 3)

                        denoised = denoiser.denoise(
                            ppg, acc[:, 0], acc[:, 1], acc[:, 2], denoiser.fs_galaxy)
                        results['denoisedGalaxy'].append(';'.join(map(str, denoised)))
                    else:
                        results['denoisedGalaxy'].append(None)

                    if pd.notna(row['e4BVP']) and pd.notna(row['e4ACC']):
                        bvp = np.array([float(x) for x in row['e4BVP'].split(';') if x.strip()])
                        acc = np.array([float(x) for x in row['e4ACC'].split(';') if x.strip()]).reshape(-1, 3)

                        acc_resampled = np.array([
                            signal.resample(acc[:, i], len(bvp))
                            for i in range(3)
                        ]).T

                        denoised = denoiser.denoise(
                            bvp,
                            acc_resampled[:, 0],
                            acc_resampled[:, 1],
                            acc_resampled[:, 2],
                            denoiser.fs_e4
                        )
                        results['denoisedE4'].append(';'.join(map(str, denoised)))
                    else:
                        results['denoisedE4'].append(None)

                except Exception as e:
                    print( {str(e)})
                    results['denoisedGalaxy'].append(None)
                    results['denoisedE4'].append(None)

            for col, values in results.items():
                df[col] = values

            output_dir = os.path.join(RESULTS_DIR, 'SVD')
            os.makedirs(output_dir, exist_ok=True)
            output_file = os.path.join(output_dir, f"{os.path.splitext(filename)[0]}_denoised.csv")
            df.to_csv(output_file, index=False)

        except Exception as e:
            print( {str(e)})
            continue

if __name__ == "__main__":
    process_dataset()