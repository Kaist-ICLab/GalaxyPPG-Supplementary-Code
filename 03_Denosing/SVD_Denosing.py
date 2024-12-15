import numpy as np
from scipy import signal
import os
import pandas as pd
from config import RESULTS_DIR, WINDOW_DIR

class SVDDenoising:
    """
    Applies Singular Value Decomposition (SVD) based denoising to PPG and accelerometer signals.

    This class implements a denoising method using SVD on Hankel matrices constructed from the signals.
    It incorporates motion artifact mitigation by weighting singular values based on the correlation
    between PPG and accelerometer frequency spectra.
    """

    def __init__(self):
        """Initializes SVDDenoising parameters."""
        self.fs_galaxy = 25  # Hz, Sampling frequency for Galaxy Watch data
        self.target_length_galaxy = 200  # Target length for Galaxy Watch signals

        self.fs_e4 = 64  # Hz, Sampling frequency for Empatica E4 data
        self.target_length_e4 = 512  # Target length for Empatica E4 signals

        self.min_singular_values = 5  # Minimum number of singular values to retain
        self.energy_threshold = 0.7  # Energy threshold for singular value selection
        self.motion_threshold = 0.5  # Threshold for motion artifact detection

        self.lowcut = 0.5  # Hz, Lower cutoff frequency for bandpass filter
        self.highcut = 4  # Hz, Upper cutoff frequency for bandpass filter

        self.nfft = 1024  # Number of points for FFT

    def mean_normalize(self, x):
        """Mean normalizes a signal."""
        if not x.size:  # More robust empty array check
            return x
        mean = np.mean(x)
        std_dev = np.std(x) + 1e-10  # Avoid division by zero
        return (x - mean) / std_dev

    def preprocess_signals(self, ppg, acc_x, acc_y, acc_z, fs):
        """Preprocesses signals: resampling, bandpass filtering, and normalization."""
        try:
            target_length = self.target_length_galaxy if fs == self.fs_galaxy else self.target_length_e4
            ppg = signal.resample(ppg, target_length)
            acc_x = signal.resample(acc_x, target_length)
            acc_y = signal.resample(acc_y, target_length)
            acc_z = signal.resample(acc_z, target_length)

            nyq = 0.5 * fs
            low = self.lowcut / nyq
            high = self.highcut / nyq
            b, a = signal.butter(4, [low, high], btype='band')  # 4th order Butterworth

            ppg_filtered = signal.filtfilt(b, a, ppg)  # Zero-phase filtering
            acc_x_filtered = signal.filtfilt(b, a, acc_x)
            acc_y_filtered = signal.filtfilt(b, a, acc_y)
            acc_z_filtered = signal.filtfilt(b, a, acc_z)

            return (self.mean_normalize(ppg_filtered),
                    self.mean_normalize(acc_x_filtered),
                    self.mean_normalize(acc_y_filtered),
                    self.mean_normalize(acc_z_filtered))

        except Exception as e:
            print(f"Preprocessing error: {e}")
            return None, None, None, None

    def compute_motion_weights(self, ppg_fft, acc_ffts):
        """Computes motion weights based on correlation between PPG and accelerometer FFTs."""
        try:
            ppg_magnitude = np.abs(ppg_fft)
            weights = [1 - min(abs(np.corrcoef(ppg_magnitude[:min(len(ppg_magnitude), len(np.abs(acc_fft)))], np.abs(acc_fft)[:min(len(ppg_magnitude), len(np.abs(acc_fft)))] )[0, 1]), self.motion_threshold) if not np.isnan(np.corrcoef(ppg_magnitude[:min(len(ppg_magnitude), len(np.abs(acc_fft)))], np.abs(acc_fft)[:min(len(ppg_magnitude), len(np.abs(acc_fft)))] )[0, 1]) else 0 for acc_fft in acc_ffts]
            return np.mean(weights)
        except Exception as e:
            print(f"Motion weight computation error: {e}")
            return 0.5

    def select_singular_values(self, S, U, Vt, motion_weight):
        """Selects singular values based on energy and motion weight."""
        try:
            energy_ratio = np.cumsum(S ** 2) / np.sum(S ** 2)
            adjusted_threshold = self.energy_threshold * (1 + 0.1 * motion_weight)
            k = max(self.min_singular_values, np.searchsorted(energy_ratio, adjusted_threshold) + 1)
            k = min(k, len(S))
            reconstructed = np.zeros_like(U @ np.diag(S) @ Vt)
            for i in range(k):
                adjusted_s = S[i] * (1 + 0.2 * motion_weight)
                reconstructed += adjusted_s * np.outer(U[:, i], Vt[i, :])
            return reconstructed

        except Exception as e:
            print(f"Singular value selection error: {e}")
            return U @ np.diag(S) @ Vt

    def build_hankel_matrix(self, signal):
        """Builds a Hankel matrix from a signal."""
        try:
            N = len(signal)
            L = N // 2
            K = N - L + 1
            H = np.array([signal[i:i + K] for i in range(L)]) #More efficient hankel construction
            return H
        except Exception as e:
            print(f"Hankel matrix construction error: {e}")
            return None

    def denoise(self, ppg, acc_x, acc_y, acc_z, fs):
        """Denoises a PPG signal using SVD."""
        try:
            ppg_processed, acc_x_processed, acc_y_processed, acc_z_processed = self.preprocess_signals(ppg, acc_x, acc_y, acc_z, fs)
            if ppg_processed is None:
                return self.mean_normalize(ppg)

            H_ppg = self.build_hankel_matrix(ppg_processed)
            if H_ppg is None:
                return self.mean_normalize(ppg)

            U, S, Vt = np.linalg.svd(H_ppg, full_matrices=False)

            ppg_fft = np.fft.rfft(ppg_processed, n=self.nfft)
            acc_ffts = [np.fft.rfft(acc, n=self.nfft) for acc in [acc_x_processed, acc_y_processed, acc_z_processed]]

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
            return self.mean_normalize(reconstructed)

        except Exception as e:
            print(f"Denoising error: {e}")
            return self.mean_normalize(ppg)
def process_dataset():
    """Processes a dataset of PPG and accelerometer data, applying denoising."""
    denoiser = SVDDenoising()
    files = [f for f in os.listdir(WINDOW_DIR) if f.endswith('GD.csv')]

    for filename in files:
        print(f"Processing file: {filename}")
        try:
            df = pd.read_csv(os.path.join(WINDOW_DIR, filename))
            total_windows = len(df)
            print(f"Total windows: {total_windows}")

            results = {'denoisedGalaxy': [], 'denoisedE4': []}

            for i, row in df.iterrows():
                if i % 100 == 0:
                    print(f"Processing window: {i}/{total_windows}")

                try:
                    for sensor, prefix, fs, target_length in [
                        ('galaxy', 'galaxy', denoiser.fs_galaxy, denoiser.target_length_galaxy),
                        ('e4', 'e4', denoiser.fs_e4, denoiser.target_length_e4)
                    ]:
                        ppg_col = f"{prefix}PPG" if sensor == 'galaxy' else f"{prefix}BVP"
                        acc_col = f"{prefix}ACC"

                        if pd.notna(row[ppg_col]) and pd.notna(row[acc_col]):
                            ppg = np.array([float(x) for x in row[ppg_col].split(';') if x.strip()])
                            acc = np.array([float(x) for x in row[acc_col].split(';') if x.strip()]).reshape(-1, 3)

                            if sensor == 'e4':
                                acc = np.array([signal.resample(acc[:, i], len(ppg)) for i in range(3)]).T

                            denoised = denoiser.denoise(ppg, acc[:, 0], acc[:, 1], acc[:, 2], fs)
                            results[f'denoised{sensor.capitalize()}'].append(';'.join(map(str, denoised)))
                        else:
                            results[f'denoised{sensor.capitalize()}'].append(None)

                except Exception as e:
                    print(f"Window processing error: {e}")
                    results['denoisedGalaxy'].append(None)
                    results['denoisedE4'].append(None)

            for col, values in results.items():
                df[col] = values

            output_dir = os.path.join(RESULTS_DIR, 'SVD')
            os.makedirs(output_dir, exist_ok=True)
            output_file = os.path.join(output_dir, f"{os.path.splitext(filename)[0]}_denoised.csv")
            df.to_csv(output_file, index=False)
            print(f"File processed successfully. Output saved to: {output_file}")

        except FileNotFoundError:
            print(f"Error: File not found: {filename}")
        except pd.errors.ParserError:
            print(f"Error: Could not parse CSV file: {filename}. Check the file format.")
        except Exception as e:
            print(f"File processing error: {e}")
            continue
if __name__ == "__main__":
    process_dataset()