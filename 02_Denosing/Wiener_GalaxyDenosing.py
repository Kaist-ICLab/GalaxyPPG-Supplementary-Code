import numpy as np
from scipy import signal
from scipy.fft import fft, ifft
import os
import pandas as pd
from config import RESULTS_DIR, WINDOW_DIR


class WienerDenoising:
    def __init__(self):
        self.fs_galaxy = 25  # Hz
        self.target_length_galaxy = 200

        self.fs_e4_bvp = 64  # Hz
        self.target_length_e4 = 512

        self.fft_res = 1024
        self.wf_length = 15
        self.cutoff_freq_hz_hp = 0.5
        self.cutoff_freq_hz_lp = 4

        self.galaxy_prev_data = {
            'prev_ppg_fft': None,
            'w1_fft_history': [],
            'w2_fft_history': [],
            'prev_bpm_est': [],
            'range_idx': None
        }

        self.e4_prev_data = {
            'prev_ppg_fft': None,
            'w1_fft_history': [],
            'w2_fft_history': [],
            'prev_bpm_est': [],
            'range_idx': None
        }

    def _reset_prev_data(self):
        self.galaxy_prev_data = {
            'prev_ppg_fft': None,
            'w1_fft_history': [],
            'w2_fft_history': [],
            'prev_bpm_est': [],
            'range_idx': None
        }

        self.e4_prev_data = {
            'prev_ppg_fft': None,
            'w1_fft_history': [],
            'w2_fft_history': [],
            'prev_bpm_est': [],
            'range_idx': None
        }

    def historical_average(self, data, window_length):
        if len(data) < window_length:
            return np.mean(data, axis=0)
        return np.mean(data[-window_length:], axis=0)

    def phase_vocoder(self, prev_phase, cur_phase, freq_range):
        vocoder = np.zeros(20)
        for n in range(20):
            vocoder[n] = ((cur_phase - prev_phase) + (2 * np.pi * n)) / (2 * np.pi * 2)
        difference = vocoder - freq_range
        delta_idx = np.argmin(np.abs(difference))
        return vocoder[delta_idx]

    def process_galaxy(self, ppg, acc_x, acc_y, acc_z):
        try:
            ppg_filtered = self.bandpass_filter(ppg, self.cutoff_freq_hz_hp,
                                                self.cutoff_freq_hz_lp, self.fs_galaxy)
            acc_filtered = np.array([
                self.bandpass_filter(acc, self.cutoff_freq_hz_hp,
                                     self.cutoff_freq_hz_lp, self.fs_galaxy)
                for acc in [acc_x, acc_y, acc_z]
            ]).T

            ppg_norm = (ppg_filtered - ppg_filtered.min()) / (ppg_filtered.max() - ppg_filtered.min())
            acc_norm = (acc_filtered - acc_filtered.min(axis=0)) / (
                    acc_filtered.max(axis=0) - acc_filtered.min(axis=0))

            denoised, bpm_est, self.galaxy_prev_data = self.wiener_filter(
                ppg_norm,
                acc_norm[:, 0],
                acc_norm[:, 1],
                acc_norm[:, 2],
                self.galaxy_prev_data,
                self.fs_galaxy
            )

            return denoised, bpm_est

        except Exception as e:
            print({str(e)})
            return ppg, 0

    def process_e4(self, ppg, acc_x, acc_y, acc_z):
        try:
            ppg_filtered = self.bandpass_filter(ppg, self.cutoff_freq_hz_hp,
                                                self.cutoff_freq_hz_lp, self.fs_e4_bvp)
            acc_filtered = np.array([
                self.bandpass_filter(acc, self.cutoff_freq_hz_hp,
                                     self.cutoff_freq_hz_lp, self.fs_e4_bvp)
                for acc in [acc_x, acc_y, acc_z]
            ]).T

            ppg_norm = (ppg_filtered - ppg_filtered.min()) / (ppg_filtered.max() - ppg_filtered.min())
            acc_norm = (acc_filtered - acc_filtered.min(axis=0)) / (
                    acc_filtered.max(axis=0) - acc_filtered.min(axis=0))

            denoised, bpm_est, self.e4_prev_data = self.wiener_filter(
                ppg_norm,
                acc_norm[:, 0],
                acc_norm[:, 1],
                acc_norm[:, 2],
                self.e4_prev_data,
                self.fs_e4_bvp
            )

            return denoised, bpm_est

        except Exception as e:
            print({str(e)})
            return ppg, 0

    def bandpass_filter(self, data, lowcut, highcut, fs, order=4):
        nyquist = 0.5 * fs
        low = lowcut / nyquist
        high = highcut / nyquist
        b, a = signal.butter(order, [low, high], btype='band')
        y = signal.filtfilt(b, a, data)
        return y

    def wiener_filter(self, ppg, acc_x, acc_y, acc_z, prev_data, fs):

        b, a = signal.butter(4, [self.cutoff_freq_hz_hp, self.cutoff_freq_hz_lp],
                             btype='bandpass', fs=fs)

        min_len = 27
        if len(ppg) >= min_len and len(acc_x) >= min_len and len(acc_y) >= min_len and len(acc_z) >= min_len:
            ppg_filtered = signal.filtfilt(b, a, ppg)
            acc_x_filtered = signal.filtfilt(b, a, acc_x)
            acc_y_filtered = signal.filtfilt(b, a, acc_y)
            acc_z_filtered = signal.filtfilt(b, a, acc_z)
        else:
            ppg_filtered = ppg
            acc_x_filtered = acc_x
            acc_y_filtered = acc_y
            acc_z_filtered = acc_z

        ppg_fft = fft(ppg_filtered, self.fft_res)
        acc_x_fft = fft(acc_x_filtered, self.fft_res)
        acc_y_fft = fft(acc_y_filtered, self.fft_res)
        acc_z_fft = fft(acc_z_filtered, self.fft_res)

        freq_range = np.linspace(0, fs, self.fft_res)
        low_r = np.argmin(np.abs(freq_range - self.cutoff_freq_hz_hp))
        high_r = np.argmin(np.abs(freq_range - self.cutoff_freq_hz_lp))

        freq_range = freq_range[low_r:high_r]
        ppg_fft = ppg_fft[low_r:high_r]
        acc_x_fft = acc_x_fft[low_r:high_r]
        acc_y_fft = acc_y_fft[low_r:high_r]
        acc_z_fft = acc_z_fft[low_r:high_r]

        freq_range_ppg = freq_range.copy()
        if prev_data['prev_ppg_fft'] is not None:
            for ii in range(len(freq_range_ppg)):
                cur_phase = np.angle(ppg_fft[ii])
                prev_phase = np.angle(prev_data['prev_ppg_fft'][ii])
                freq_range_ppg[ii] = self.phase_vocoder(prev_phase, cur_phase, freq_range[ii])

        freq_range_ppg = np.convolve(freq_range_ppg, np.ones(3) / 3, mode='same')

        w1_fft = np.abs(ppg_fft) / np.max(np.abs(ppg_fft))
        w1_ppg_ave_fft_all = self.historical_average(prev_data['w1_fft_history'] + [w1_fft], self.wf_length)
        w1_ppg_ave_fft_all_norm = w1_ppg_ave_fft_all / np.max(w1_ppg_ave_fft_all)
        w1_acc_x_fft_norm = np.abs(acc_x_fft) / np.max(np.abs(acc_x_fft))
        w1_acc_y_fft_norm = np.abs(acc_y_fft) / np.max(np.abs(acc_y_fft))
        w1_acc_z_fft_norm = np.abs(acc_z_fft) / np.max(np.abs(acc_z_fft))
        wf1 = (1 - 1 / 3 * (w1_acc_x_fft_norm + w1_acc_y_fft_norm + w1_acc_z_fft_norm) / w1_ppg_ave_fft_all_norm)
        wf1[wf1 < 0] = -1
        w1_ppg_ave_fft_clean = np.abs(ppg_fft) * wf1
        w2_fft = np.abs(ppg_fft) / np.max(np.abs(ppg_fft))
        w2_ppg_ave_fft_all = self.historical_average(prev_data['w2_fft_history'] + [w2_fft], self.wf_length)
        w2_ppg_ave_fft_all_norm = w2_ppg_ave_fft_all / np.max(w2_ppg_ave_fft_all)
        w2_acc_x_fft_norm = np.abs(acc_x_fft) / np.max(np.abs(acc_x_fft))
        w2_acc_y_fft_norm = np.abs(acc_y_fft) / np.max(np.abs(acc_y_fft))
        w2_acc_z_fft_norm = np.abs(acc_z_fft) / np.max(np.abs(acc_z_fft))
        wf2 = w2_ppg_ave_fft_all_norm / (
                    ((w2_acc_x_fft_norm + w2_acc_y_fft_norm + w2_acc_z_fft_norm) / 3) + w2_ppg_ave_fft_all_norm)
        w2_ppg_ave_fft_clean = np.abs(ppg_fft) * wf2

        w1_ppg_ave_fft_clean = w1_ppg_ave_fft_clean / np.std(w1_ppg_ave_fft_clean)
        w2_ppg_ave_fft_clean = w2_ppg_ave_fft_clean / np.std(w2_ppg_ave_fft_clean)

        ppg_ave_fft_fin = w1_ppg_ave_fft_clean + w2_ppg_ave_fft_clean

        hist_int = 25
        if len(prev_data['prev_bpm_est']) > 15:
            hist_int = max(np.abs(np.diff(prev_data['prev_bpm_est']))) + 5

        if prev_data['range_idx'] is None:
            idx = np.argmax(ppg_ave_fft_fin)
            bpm_est = freq_range_ppg[idx] * 60
            range_idx = np.arange(
                max(0, idx - int(hist_int / ((freq_range[1] - freq_range[0]) * 60))),
                min(len(freq_range_ppg), idx + int(hist_int / ((freq_range[1] - freq_range[0]) * 60)))
            )
        else:
            idx = prev_data['range_idx'][np.argmax(ppg_ave_fft_fin[prev_data['range_idx']])]
            bpm_est = freq_range_ppg[idx] * 60
            range_idx = np.arange(
                max(0, idx - int(hist_int / ((freq_range[1] - freq_range[0]) * 60))),
                min(len(freq_range_ppg), idx + int(hist_int / ((freq_range[1] - freq_range[0]) * 60)))
            )

        if len(prev_data['prev_bpm_est']) > 5 and abs(bpm_est - prev_data['prev_bpm_est'][-1]) > 5:
            recent_bpm = np.array(prev_data['prev_bpm_est'][-5:])
            ddd = np.polyfit(range(len(recent_bpm)), recent_bpm, 1)
            predicted_bpm = np.polyval(ddd, len(recent_bpm))
            bpm_est = 0.8 * bpm_est + 0.2 * predicted_bpm

        if len(prev_data['prev_bpm_est']) > 6:
            mul = 0.1
            correction = np.sum(np.sign(np.array(prev_data['prev_bpm_est'][-6:]) -
                                        np.array(prev_data['prev_bpm_est'][-7:-1])) * mul)
            bpm_est += correction

        full_fft = np.zeros(self.fft_res, dtype=complex)
        full_fft[low_r:high_r] = ppg_ave_fft_fin * np.exp(1j * np.angle(ppg_fft))
        denoised_ppg = np.real(ifft(full_fft))[:len(ppg)]
        prev_data['prev_ppg_fft'] = ppg_fft
        prev_data['w1_fft_history'].append(w1_fft)
        prev_data['w2_fft_history'].append(w2_fft)
        prev_data['prev_bpm_est'].append(bpm_est)
        prev_data['range_idx'] = range_idx

        max_history = 30
        prev_data['w1_fft_history'] = prev_data['w1_fft_history'][-max_history:]
        prev_data['w2_fft_history'] = prev_data['w2_fft_history'][-max_history:]
        prev_data['prev_bpm_est'] = prev_data['prev_bpm_est'][-max_history:]

        return denoised_ppg, bpm_est, prev_data


def process_CR_dataset(dataset_name=None):

    if dataset_name:
        file_list = [f'{dataset_name}.csv']
    else:
        file_list = [f for f in os.listdir(WINDOW_DIR) if f.endswith('GD.csv')]

    all_galaxy_errors = []
    all_e4_errors = []

    for filename in file_list:
        if not os.path.exists(os.path.join(WINDOW_DIR, filename)):
            print({filename})
            continue

        current_dataset = os.path.splitext(filename)[0]
        print({current_dataset})

        try:
            df = pd.read_csv(os.path.join(WINDOW_DIR, filename))
            wiener = WienerDenoising()

            results = {
                'denoisedGalaxy': [None] * len(df),
                'denoisedE4': [None] * len(df),
                'estimated_BPM_Galaxy': [None] * len(df),
                'estimated_BPM_E4': [None] * len(df),
                'BPM_error_Galaxy': [None] * len(df),
                'BPM_error_E4': [None] * len(df)
            }

            for i, row in df.iterrows():
                try:
                    galaxy_ppg = np.array([float(x) for x in row['galaxyPPG'].split(';') if x.strip()])
                    galaxy_acc = np.array([float(x) for x in row['galaxyACC'].split(';') if x.strip()]).reshape(-1, 3)

                    e4_bvp = np.array([float(x) for x in row['e4BVP'].split(';') if x.strip()])
                    e4_acc = np.array([float(x) for x in row['e4ACC'].split(';') if x.strip()]).reshape(-1, 3)

                    e4_acc_resampled = np.array([
                        signal.resample(e4_acc[:, i], len(e4_bvp))
                        for i in range(3)
                    ]).T

                    true_hr = row['gdHR']

                    galaxy_denoised, galaxy_bpm = wiener.process_galaxy(
                        galaxy_ppg,
                        galaxy_acc[:, 0],
                        galaxy_acc[:, 1],
                        galaxy_acc[:, 2]
                    )

                    e4_denoised, e4_bpm = wiener.process_e4(
                        e4_bvp,
                        e4_acc_resampled[:, 0],
                        e4_acc_resampled[:, 1],
                        e4_acc_resampled[:, 2]
                    )

                    results['denoisedGalaxy'][i] = ';'.join(map(str, galaxy_denoised.tolist()))
                    results['denoisedE4'][i] = ';'.join(map(str, e4_denoised.tolist()))
                    results['estimated_BPM_Galaxy'][i] = galaxy_bpm
                    results['estimated_BPM_E4'][i] = e4_bpm
                    results['BPM_error_Galaxy'][i] = abs(galaxy_bpm - true_hr)
                    results['BPM_error_E4'][i] = abs(e4_bpm - true_hr)

                except Exception as e:
                    print({str(e)})
                    continue

            for col, values in results.items():
                df[col] = values

            output_dir = os.path.join(RESULTS_DIR, 'Wiener')
            os.makedirs(output_dir, exist_ok=True)
            output_file = os.path.join(output_dir, f'{current_dataset}_denoised.csv')
            df.to_csv(output_file, index=False)

            valid_galaxy_errors = [e for e in results['BPM_error_Galaxy'] if e is not None]
            valid_e4_errors = [e for e in results['BPM_error_E4'] if e is not None]

            if valid_galaxy_errors:
                all_galaxy_errors.extend(valid_galaxy_errors)

            if valid_e4_errors:
                all_e4_errors.extend(valid_e4_errors)


        except Exception as e:
            print({str(e)})
            continue


if __name__ == "__main__":
    process_CR_dataset()