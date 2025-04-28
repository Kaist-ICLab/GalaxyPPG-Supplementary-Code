from typing import List
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, find_peaks, resample
import os
import sys
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)
from config import RESULTS_DIR, WINDOW_DIR

class KalmanDenoising:
    def __init__(self, device_type='galaxy'):
        """
               Initialize Kalman filter for PPG signal denoising.

               Parameters
               ----------
               device_type : str, default='galaxy'
                   Type of device ('galaxy' or 'e4')

               Attributes
               ----------
               ppg_fs : int
                   PPG sampling frequency (25 Hz for Galaxy, 64 Hz for E4)
               acc_fs : int
                   Accelerometer sampling frequency
               window_points : int
                   Number of points in processing window
               svd_threshold : float
                   Threshold for SVD-based noise detection
               signal_Q : float
                   Process noise covariance
               signal_R : float
                   Measurement noise covariance
               x_state : numpy.ndarray or None
                   Kalman filter state vector
               P_state : numpy.ndarray or None
                   State covariance matrix
               hr_min : int
                   Minimum expected heart rate in BPM
               hr_max : int
                   Maximum expected heart rate in BPM
        """
        if device_type == 'galaxy':
            self.ppg_fs = 25
            self.acc_fs = 25
            self.window_points = 200
        else:
            self.ppg_fs = 64
            self.acc_fs = 64 # Has benn upsampling
            self.window_points = 512

        self.device_type = device_type
        self.window_size = 8

        self.svd_threshold = 0.4

        self.signal_Q = 0.1
        self.signal_R = 1.0
        self.x_state = None
        self.P_state = None
        self.hr_min = 40
        self.hr_max = 200

    def process_dataframe(self, df: pd.DataFrame,
                          ppg_col: str = 'ppg',
                          acc_cols: List[str] = ['acc_x', 'acc_y', 'acc_z']) -> pd.DataFrame:
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

        Returns
        -------
        pandas.DataFrame
            DataFrame with added columns for denoised signal and heart rate

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
                acc_data = np.column_stack([np.array(row[col]) for col in acc_cols])

                result = self.process_signal(ppg, acc_data)

                results.append({
                    'denoised_signal': ';'.join(map(str, result['denoised_signal'])),
                    'heart_rate': result['heart_rate']
                })
            except Exception as e:
                print(f"Error processing row: {str(e)}")
                results.append({
                    'denoised_signal': None,
                    'heart_rate': None
                })

        result_df = pd.concat([df, pd.DataFrame(results)], axis=1)
        return result_df
    def process_signal(self, ppg, acc_data):
        """
               Process PPG signal using Kalman filtering.

               Parameters
               ----------
               ppg : numpy.ndarray
                   Raw PPG signal
               acc_data : numpy.ndarray
                   Accelerometer data with shape (n_samples, 3)

               Returns
               -------
               dict
                   Dictionary containing:
                   - 'denoised_signal': Processed PPG signal
                   - 'heart_rate': Estimated heart rate in BPM
        """
        ppg = self.normalize_signal(ppg)
        acc_data = np.apply_along_axis(self.normalize_signal, 0, acc_data)

        acc_resampled = self.resample_accelerometer(acc_data)

        filtered_signal = self.bandpass_filter(ppg)

        acc_magnitude = np.sqrt(np.sum(acc_resampled ** 2, axis=1))

        svd_signal = self.svd_denoising(
            filtered_signal,
            acc_resampled[:, 0],
            acc_resampled[:, 1],
            acc_resampled[:, 2]
        )


        final_signal = self.kalman_denoising(
            svd_signal,
            acc_resampled[:, 0],
            acc_resampled[:, 1],
            acc_resampled[:, 2]
        )


        hr = self.estimate_heart_rate(final_signal)

        return {
            'denoised_signal': final_signal,
            'heart_rate': hr,
            }

    def resample_accelerometer(self, acc_data):
        if self.device_type == 'e4':
            target_length = int(len(acc_data) * self.ppg_fs / self.acc_fs)
            resampled_acc = np.zeros((target_length, 3))
            for i in range(3):
                resampled_acc[:, i] = resample(acc_data[:, i], target_length)
            return resampled_acc
        return acc_data

    def bandpass_filter(self, signal):
        nyquist = self.ppg_fs * 0.5
        low = 0.5 / nyquist
        high = 4.0 / nyquist
        b, a = butter(2, [low, high], btype='band')
        return filtfilt(b, a, signal)

    def construct_hankel_matrix(self, signal):
        N = len(signal)
        L = N // 2
        K = N - L + 1
        H = np.zeros((L, K))

        for i in range(K):
            H[:, i] = signal[i:i + L]

        return H, L, K

    def auto_correlation_threshold(self, eigenvalues):
        eig = eigenvalues[:10]
        diff = -np.diff(eig)
        mean_diff = np.mean(diff)
        index = np.where(diff > mean_diff)[0]

        if len(index) > 0:
            ind = index[-1]
            p = diff[ind]
            if p > 2 * (eigenvalues[0] - eigenvalues[1]):
                num_aut = ind + 1
            else:
                num_aut = 10
        else:
            num_aut = 10

        return max(4, num_aut)

    def svd_denoising(self, ppg, acc_x, acc_y, acc_z):
        H_ppg, L, K = self.construct_hankel_matrix(ppg)
        H_x, _, _ = self.construct_hankel_matrix(acc_x)
        H_y, _, _ = self.construct_hankel_matrix(acc_y)
        H_z, _, _ = self.construct_hankel_matrix(acc_z)

        U_ppg, S_ppg, V_ppg = np.linalg.svd(H_ppg, full_matrices=False)
        U_x, S_x, _ = np.linalg.svd(H_x, full_matrices=False)
        U_y, S_y, _ = np.linalg.svd(H_y, full_matrices=False)
        U_z, S_z, _ = np.linalg.svd(H_z, full_matrices=False)

        ppg_aut = self.auto_correlation_threshold(S_ppg)
        acc_x_aut = self.auto_correlation_threshold(S_x)
        acc_y_aut = self.auto_correlation_threshold(S_y)
        acc_z_aut = self.auto_correlation_threshold(S_z)

        corr_x = (U_ppg[:, :ppg_aut].T @ U_x[:, :acc_x_aut]) ** 2
        corr_y = (U_ppg[:, :ppg_aut].T @ U_y[:, :acc_y_aut]) ** 2
        corr_z = (U_ppg[:, :ppg_aut].T @ U_z[:, :acc_z_aut]) ** 2

        max_corr = max(
            np.max(np.sum(corr_x, axis=1)),
            np.max(np.sum(corr_y, axis=1)),
            np.max(np.sum(corr_z, axis=1))
        )

        S_filtered = np.copy(S_ppg)
        if max_corr > self.svd_threshold:
            for i in range(len(S_filtered)):
                S_filtered[i] *= (1 - max_corr)

        H_filtered = U_ppg @ np.diag(S_filtered) @ V_ppg

        denoised_ppg = np.zeros(len(ppg))
        count = np.zeros(len(ppg))

        for i in range(K):
            denoised_ppg[i:i + L] += H_filtered[:, i]
            count[i:i + L] += 1

        denoised_ppg /= np.maximum(count, 1)

        return denoised_ppg

    def initialize_kalman(self, signal):
        if self.x_state is None:
            self.x_state = np.zeros(2)
            self.x_state[0] = signal[0]
            self.P_state = np.eye(2) * 1000

    def kalman_step(self, measurement, acc_magnitude):
        acc_threshold = np.mean(acc_magnitude) + 2 * np.std(acc_magnitude)
        self.signal_R = 10.0 if acc_magnitude > acc_threshold else 1.0

        dt = 1.0 / self.ppg_fs
        F = np.array([[1, dt],
                      [0, 1]])
        H = np.array([[1, 0]])

        x_pred = F @ self.x_state
        P_pred = F @ self.P_state @ F.T + np.eye(2) * self.signal_Q

        innovation = measurement - H @ x_pred
        S = H @ P_pred @ H.T + self.signal_R
        K = P_pred @ H.T / S

        self.x_state = x_pred + K.reshape(-1) * innovation
        self.P_state = (np.eye(2) - K.reshape(-1, 1) @ H) @ P_pred

        return self.x_state[0]

    def kalman_denoising(self, svd_signal, acc_x, acc_y, acc_z):
        self.initialize_kalman(svd_signal)

        acc_magnitude = np.sqrt(acc_x ** 2 + acc_y ** 2 + acc_z ** 2)

        kalman_signal = np.zeros_like(svd_signal)
        for i in range(len(svd_signal)):
            kalman_signal[i] = self.kalman_step(svd_signal[i], acc_magnitude[i])

        return kalman_signal

    def normalize_signal(self, signal):
        return (signal - np.mean(signal)) / (np.std(signal) + 1e-10)
    def estimate_heart_rate(self, signal):
        window = np.hanning(len(signal))
        windowed_signal = signal * window

        nfft = 2 ** 14
        X = np.abs(np.fft.fft(windowed_signal, nfft)) / np.sum(window)
        freq = np.fft.fftfreq(nfft, 1 / self.ppg_fs)

        mask = (freq >= self.hr_min / 60) & (freq <= self.hr_max / 60)
        X_masked = X[mask]
        freq_masked = freq[mask]

        peaks, properties = find_peaks(X_masked,
                                       distance=5,
                                       prominence=0.1 * np.max(X_masked))

        if len(peaks) > 0:
            peak_hrs = freq_masked[peaks] * 60
            peak_magnitudes = X_masked[peaks]

            weights = peak_magnitudes / np.max(peak_magnitudes)
            hr_scores = weights * (1 - np.abs(peak_hrs - 120) / 120)
            best_peak_idx = np.argmax(hr_scores)

            return peak_hrs[best_peak_idx]
        return 75


def process_dataset(participant_range=None, participant_list=None):
    """
    Process dataset with specified participant range or list.

    Args:
        participant_range (tuple): Range of participant numbers (start, end)
        participant_list (list): List of specific participant IDs
    """
    lowcut = 0.5
    highcut = 4.0

    if participant_range:
        start, end = participant_range
        participant_ids = [f'P{str(i).zfill(2)}' for i in range(start, end + 1)]
    elif participant_list:
        participant_ids = participant_list
    else:
        participant_ids = [f'P{str(i).zfill(2)}' for i in range(2, 25)]

    for participant_id in participant_ids:
        filename = f'{participant_id}_processed_GD.csv'
        file_path = os.path.join(WINDOW_DIR, filename)

        if not os.path.exists(file_path):
            print(f"File not found: {filename}")
            continue

        print(f"\nProcessing {participant_id}")

        try:
            df = pd.read_csv(file_path)

            results = {
                'denoisedGalaxy': [None] * len(df),
                'denoisedE4': [None] * len(df),
                'estimated_BPM_Galaxy': [None] * len(df),
                'estimated_BPM_E4': [None] * len(df),
                'BPM_error_Galaxy': [None] * len(df),
                'BPM_error_E4': [None] * len(df)
            }

            total_windows = len(df)
            for i, row in df.iterrows():
                try:
                    if i % 50 == 0:
                        progress = (i / total_windows) * 100
                        print(f"Progress: {progress:.1f}% ({i}/{total_windows})")

                    galaxy_ppg = - np.array([float(x) for x in row['galaxyPPG'].split(';') if x.strip()])
                    galaxy_acc = np.array([float(x) for x in row['galaxyACC'].split(';') if x.strip()]).reshape(-1, 3)

                    e4_bvp = np.array([float(x) for x in row['e4BVP'].split(';') if x.strip()])
                    e4_acc = np.array([float(x) for x in row['e4ACC'].split(';') if x.strip()]).reshape(-1, 3)

                    true_hr = row['gdHR']

                    # Process Galaxy Watch data
                    galaxy_denoiser = KalmanDenoising(device_type='galaxy')
                    galaxy_result = galaxy_denoiser.process_signal(
                        galaxy_ppg,
                        galaxy_acc
                    )
                    galaxy_denoised = galaxy_result['denoised_signal']
                    galaxy_bpm = galaxy_result['heart_rate']

                    # Process E4 data
                    e4_denoiser = KalmanDenoising(device_type='e4')
                    e4_result = e4_denoiser.process_signal(
                        e4_bvp,
                        e4_acc
                    )
                    e4_denoised = e4_result['denoised_signal']
                    e4_bpm = e4_result['heart_rate']

                    # Store results
                    results['denoisedGalaxy'][i] = ';'.join(map(str, galaxy_denoised))
                    results['denoisedE4'][i] = ';'.join(map(str, e4_denoised))
                    results['estimated_BPM_Galaxy'][i] = galaxy_bpm
                    results['estimated_BPM_E4'][i] = e4_bpm
                    results['BPM_error_Galaxy'][i] = abs(galaxy_bpm - true_hr)
                    results['BPM_error_E4'][i] = abs(e4_bpm - true_hr)

                except Exception as e:
                    print(f"Error processing window {i}: {str(e)}")
                    continue

            # Add results to dataframe
            for col, values in results.items():
                df[col] = values

            # Save processed data
            output_dir = os.path.join(RESULTS_DIR, 'Kalman')
            os.makedirs(output_dir, exist_ok=True)
            output_file = os.path.join(output_dir, f'{participant_id}_processed_GD_denoised.csv')
            df.to_csv(output_file, index=False)

        except Exception as e:
            print(f"Error processing participant {participant_id}: {str(e)}")
            continue


if __name__ == "__main__":
    # process_dataset(participant_range=(22, 24))
    # Alternative usage:
    # participant_list = ["P03"]
    # process_dataset(participant_list=participant_list)
    process_dataset()

