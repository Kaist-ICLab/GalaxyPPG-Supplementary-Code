import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, find_peaks, resample
import os
from config import RESULTS_DIR, WINDOW_DIR

class KalmanDenoising:
    def __init__(self, device_type='galaxy'):

        if device_type == 'galaxy':
            self.ppg_fs = 25
            self.acc_fs = 25
            self.window_points = 200
        else:
            self.ppg_fs = 64
            self.acc_fs = 64  # Has benn upsampling
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

        peaks, properties = find_peaks(X_masked,distance=5,
                                            prominence=0.1 * np.max(X_masked))

        if len(peaks) > 0:
            peak_hrs = freq_masked[peaks] * 60
            peak_magnitudes = X_masked[peaks]

            weights = peak_magnitudes / np.max(peak_magnitudes)
            hr_scores = weights * (1 - np.abs(peak_hrs - 120) / 120)
            best_peak_idx = np.argmax(hr_scores)

            return peak_hrs[best_peak_idx]
        return 75

    def process_signal(self, ppg, acc_data):

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


def process_dataset():
    window_data_dir = WINDOW_DIR
    result_dir = os.path.join(RESULTS_DIR, 'Kalman')
    os.makedirs(result_dir, exist_ok=True)

    denoiser_galaxy = KalmanDenoising(device_type='galaxy')
    denoiser_e4 = KalmanDenoising(device_type='e4')

    all_results = {
        'galaxy': {'hr_errors': []},
        'e4': {'hr_errors': []}
    }

    for filename in os.listdir(window_data_dir):
        if filename.endswith('_processed_GD.csv'):
            print(f"\nProcessing {filename}")
            df = pd.read_csv(os.path.join(window_data_dir, filename))

            results = {
                'galaxy': {
                    'denoised_signal': [],
                    'hr': [],
                    'error': []
                },
                'e4': {
                    'denoised_signal': [],
                    'hr': [],
                    'error': []
                }
            }

            for i, row in df.iterrows():

                try:
                    galaxy_ppg = np.array([float(x) for x in row['galaxyPPG'].split(';')])
                    galaxy_acc = np.array([float(x) for x in row['galaxyACC'].split(';')]).reshape(-1, 3)

                    galaxy_result = denoiser_galaxy.process_signal(galaxy_ppg, galaxy_acc)

                    results['galaxy']['denoised_signal'].append(
                        ';'.join(map(str, galaxy_result['denoised_signal']))
                    )
                    results['galaxy']['hr'].append(galaxy_result['heart_rate'])
                    results['galaxy']['error'].append(
                        abs(galaxy_result['heart_rate'] - row['gdHR'])
                    )

                    all_results['galaxy']['hr_errors'].append(
                        abs(galaxy_result['heart_rate'] - row['gdHR'])
                    )


                except Exception as e:
                    print(f"Error processing Galaxy data in window {i}: {str(e)}")
                    results['galaxy']['denoised_signal'].append('')
                    results['galaxy']['hr'].append(None)
                    results['galaxy']['error'].append(None)


                try:
                    e4_ppg = np.array([float(x) for x in row['e4BVP'].split(';')])
                    e4_acc = np.array([float(x) for x in row['e4ACC'].split(';')]).reshape(-1, 3)

                    e4_result = denoiser_e4.process_signal(e4_ppg, e4_acc)

                    results['e4']['denoised_signal'].append(
                        ';'.join(map(str, e4_result['denoised_signal']))
                    )
                    results['e4']['hr'].append(e4_result['heart_rate'])
                    results['e4']['error'].append(
                        abs(e4_result['heart_rate'] - row['gdHR'])
                    )

                    all_results['e4']['hr_errors'].append(
                        abs(e4_result['heart_rate'] - row['gdHR'])
                    )

                except Exception as e:
                    print(f"Error processing E4 data in window {i}: {str(e)}")
                    results['e4']['denoised_signal'].append('')
                    results['e4']['hr'].append(None)
                    results['e4']['error'].append(None)

            df['denoisedGalaxy'] = results['galaxy']['denoised_signal']
            df['estimatedHR_Galaxy'] = results['galaxy']['hr']
            df['HR_error_Galaxy'] = results['galaxy']['error']

            df['denoisedE4'] = results['e4']['denoised_signal']
            df['estimatedHR_E4'] = results['e4']['hr']
            df['HR_error_E4'] = results['e4']['error']

            output_file = os.path.join(
                result_dir,
                f'{os.path.splitext(filename)[0]}_denoised.csv'
            )
            df.to_csv(output_file, index=False)

            for device in ['galaxy', 'e4']:
                valid_errors = [e for e in results[device]['error'] if e is not None]

                if valid_errors:
                    print(f"\n{device.upper()} Statistics:")
                    print(f"Mean HR Error: {np.mean(valid_errors):.2f} BPM")
                    print(f"Std HR Error: {np.std(valid_errors):.2f} BPM")
                    print(f"Windows Processed: {len(valid_errors)}")

    stats = {}
    for device in ['galaxy', 'e4']:
        hr_errors = all_results[device]['hr_errors']

        if hr_errors:
            stats[device] = {
                'mean_hr_error': np.mean(hr_errors),
                'std_hr_error': np.std(hr_errors),
                'rmse': np.sqrt(np.mean(np.array(hr_errors) ** 2)),
                'windows_processed': len(hr_errors)
            }

    stats_df = pd.DataFrame.from_dict(stats, orient='index')
    stats_file = os.path.join(result_dir, 'overall_statistics.csv')
    stats_df.to_csv(stats_file)
   
if __name__ == "__main__":
    process_dataset()
