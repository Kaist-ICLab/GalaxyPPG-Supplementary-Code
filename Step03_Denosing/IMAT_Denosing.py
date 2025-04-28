import os
from typing import List
import pandas as pd
from scipy.signal import find_peaks, butter, filtfilt
import sys
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)
from config import RESULTS_DIR, WINDOW_DIR
import numpy as np
from scipy import signal

class IMATDenoising:
    def __init__(self):
        """
                Initialize the IMAT denoising algorithm with default parameters.

                Attributes
                ----------
                fs_galaxy : int
                    Galaxy Watch sampling frequency (25 Hz)
                target_length_galaxy : int
                    Target length for Galaxy Watch signal processing (200 samples)
                fs_e4_bvp : int
                    E4 device BVP sampling frequency (64 Hz)
                target_length_e4 : int
                    Target length for E4 signal processing (512 samples)
                fs_e4_acc : int
                    E4 device accelerometer sampling frequency (32 Hz)
                window_count : int
                    Counter for processed windows
                max_windows : int
                    Maximum number of windows to process before resizing buffers
                sample : numpy.ndarray
                    Buffer for sample storage
                estimate : numpy.ndarray
                    Buffer for estimate storage
                previous_ppg : numpy.ndarray or None
                    Previous PPG signal for continuity
                previous_acc_x/y/z : numpy.ndarray or None
                    Previous accelerometer signals for continuity
                """
        # Galaxy
        self.fs_galaxy = 25
        self.target_length_galaxy = 200
        # E4
        self.fs_e4_bvp = 64
        self.target_length_e4 = 512
        self.fs_e4_acc = 32
        # Pre-Setting
        self.fs = None
        # Processing parameters
        self.window_count = 0
        self.max_windows = 2000
        self.sample = np.zeros(2000)
        self.estimate = np.zeros(2000)
        # Signal history
        self.previous_ppg = None
        self.previous_acc_x = None
        self.previous_acc_y = None
        self.previous_acc_z = None

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
            DataFrame with added columns for denoised signal and heart rate
        """
        if ppg_col not in df.columns:
            raise ValueError(f"PPG column '{ppg_col}' not found in DataFrame")
        for col in acc_cols:
            if col not in df.columns:
                raise ValueError(f"Accelerometer column '{col}' not found in DataFrame")

        results = []
        for _, row in df.iterrows():
            try:
                ppg = np.array(row[ppg_col])
                acc = [np.array(row[col]) for col in acc_cols]

                if device_type.lower() == 'galaxy':
                    ppg = -ppg
                    denoised, hr = self.process_galaxy(ppg, *acc)
                else:
                    denoised, hr = self.process_e4(ppg, *acc)

                results.append({
                    'denoised_signal': ';'.join(map(str, denoised)),
                    'heart_rate': hr
                })
            except Exception as e:
                print(f"Error processing row: {str(e)}")
                results.append({
                    'denoised_signal': None,
                    'heart_rate': None
                })

        result_df = pd.concat([df, pd.DataFrame(results)], axis=1)
        return result_df

    def process_galaxy(self, ppg, acc_x, acc_y, acc_z):
        self.fs = self.fs_galaxy
        return self._process_generic(ppg, acc_x, acc_y, acc_z, self.target_length_galaxy, self.fs_galaxy)

    def process_e4(self, ppg, acc_x, acc_y, acc_z):
        self.fs = self.fs_e4_bvp
        return self._process_generic(ppg, acc_x, acc_y, acc_z, self.target_length_e4, self.fs_e4_bvp)

    def _process_generic(self, ppg, acc_x, acc_y, acc_z, target_length, fs):
        def standardize_length(signal):
            current_length = len(signal)
            if current_length < target_length:
                padding = target_length - current_length
                return np.pad(signal, (0, padding), mode='edge')
            elif current_length > target_length:
                return signal[:target_length]
            return signal

        self.window_count += 1
        if self.window_count >= self.max_windows:
            self.max_windows *= 2
            self.sample = np.pad(self.sample, (0, self.max_windows), mode='constant')
            self.estimate = np.pad(self.estimate, (0, self.max_windows), mode='constant')

        try:
            ppg = standardize_length(np.array(ppg, dtype=np.float64))
            acc_x = standardize_length(np.array(acc_x, dtype=np.float64))
            acc_y = standardize_length(np.array(acc_y, dtype=np.float64))
            acc_z = standardize_length(np.array(acc_z, dtype=np.float64))

            if self.window_count == 1:
                self.Nprv, bpm0 = self.initialize_hr(ppg,fs)
                self.Nprv = bpm0 *4
                x2 = ppg
                z2 = x2
                x3 = x2
                self.previous_ppg = ppg.copy()
                self.previous_acc_x = acc_x.copy()
                self.previous_acc_y = acc_y.copy()
                self.previous_acc_z = acc_z.copy()
            else:
                x2 = self.ma_cancellation(ppg, acc_x, acc_y, acc_z, 30, 10, self.Nprv)

                extended_ppg = np.concatenate([self.previous_ppg, ppg])
                extended_acc_x = np.concatenate([self.previous_acc_x, acc_x])
                extended_acc_y = np.concatenate([self.previous_acc_y, acc_y])
                extended_acc_z = np.concatenate([self.previous_acc_z, acc_z])

                z2 = self.ma_cancellation(extended_ppg, extended_acc_x, extended_acc_y, extended_acc_z, 30, 10,
                                          self.Nprv)


                min_length = min(len(x2), len(z2))
                x2 = x2[:min_length]
                z2_current = z2[:min_length]
                x3 = (x2 + z2_current) / 2
                self.previous_ppg = ppg.copy()
                self.previous_acc_x = acc_x.copy()
                self.previous_acc_y = acc_y.copy()
                self.previous_acc_z = acc_z.copy()

            bpm = self.spectral_analysis( acc_x, acc_y, acc_z, self.Nprv, self.sample,
                                         self.estimate, self.window_count, x2, z2)

            self.Nprv = bpm * 4
            return x3, bpm

        except Exception as e:
            print(f"Error processing window {self.window_count}: {str(e)}")
            return ppg, self.Nprv / 4

    def ma_cancellation(self, ppg, acc_x, acc_y, acc_z, k, deg, Nprv):
        tool = len(ppg)
        y1 = self.refMA_generation_using_svd(acc_x, k, tool)
        y2 = self.refMA_generation_using_svd(acc_y, k, tool)
        y3 = self.refMA_generation_using_svd(acc_z, k, tool)

        s1 = np.ascontiguousarray(y1.reshape(k, tool).T)
        s2 = np.ascontiguousarray(y2.reshape(k, tool).T)
        s3 = np.ascontiguousarray(y3.reshape(k, tool).T)

        x2 = ppg - np.mean(ppg)
        en = np.sqrt(2 * np.mean(x2 ** 2)) * 2
        x2 = np.clip(x2, -en, en)

        x2_filtered = self.adaptive_ma_cancellation(s1, s2, s3, x2, k, deg, Nprv)

        return x2_filtered
    def refMA_generation_using_svd(self,sig, L, tool):

        N = tool
        if L > N // 2:
            L = N - L
        K = N - L + 1

        X = np.array([sig[i:i + L] for i in range(K)]).T

        S = np.dot(X, X.T)
        eigenvals, U = np.linalg.eig(S)
        idx = np.argsort(-eigenvals)
        U = np.real(U[:, idx])
        V = np.dot(X.T, U)

        p = []
        Lp, Kp = min(L, K), max(L, K)
        for j in range(min(L, K)):
            rca = np.outer(U[:, j], V[:, j])
            y = np.zeros(N, dtype=np.float64)

            for k in range(Lp - 1):
                y[k] = sum(
                    (1.0 / (k + 1)) * rca[m, k - m] for m in range(k + 1) if m < rca.shape[0] and k - m < rca.shape[1])

            for k in range(Lp - 1, Kp):
                y[k] = sum((1.0 / Lp) * rca[m, k - m] for m in range(Lp) if m < rca.shape[0] and k - m < rca.shape[1])

            for k in range(Kp, N):
                y[k] = sum((1.0 / (N - k)) * rca[m, k - m] for m in range(k - Kp + 2, min(N - Kp + 1, rca.shape[0])) if
                           m < rca.shape[0] and k - m < rca.shape[1])

            p.append(y)

        return np.concatenate(p)
    def adaptive_ma_cancellation(self, s1, s2, s3, x2, k, deg, Nprv):
        """
        Adaptive Motion Artifact Cancellation
        Args:
            s1, s2, s3: Reference signals from SVD decomposition (shape: [signal_length, k])
            x2: PPG signal to be filtered (shape: [signal_length,])
            k: Number of reference components
            deg: Filter length
            Nprv: Previous heart rate estimate * 4
        Returns:
            x2: Filtered PPG signal
        """
        ms = 0.005  # Step size parameter

        # Calculate main frequency components based on device sampling rate
        if self.fs == self.fs_galaxy:  # Galaxy (25Hz)
            fft_length = 6000
        else:  # E4 (64Hz)
            fft_length = 15000

        def get_freq_peaks(signal_col):
            """Get the dominant frequency for each column"""
            fft_result = np.abs(np.fft.fft(signal_col, fft_length))
            # Only look at first 800 points to match original code's frequency range
            return np.argmax(fft_result[:800])

        # Calculate dominant frequencies for each component
        frqs1 = np.array([get_freq_peaks(s1[:, i]) for i in range(s1.shape[1])])
        frqs2 = np.array([get_freq_peaks(s2[:, i]) for i in range(s2.shape[1])])
        frqs3 = np.array([get_freq_peaks(s3[:, i]) for i in range(s3.shape[1])])

        def lms_filter(x_ref, d, filter_length, step_size):
            """
            LMS Filter Implementation
            Args:
                x_ref: Reference signal
                d: Desired signal
                filter_length: Length of the filter
                step_size: Step size parameter
            """
            N = len(x_ref)
            w = np.zeros(filter_length)  # Filter coefficients
            y = np.zeros(N)  # Output signal

            # Ensure signals are float64 for precision
            x_ref = x_ref.astype(np.float64)
            d = d.astype(np.float64)

            for n in range(filter_length, N):
                # Get current input vector (reversed to match MATLAB order)
                x = x_ref[n - filter_length:n][::-1]
                # Calculate current output
                y[n] = np.dot(w, x)
                # Calculate error
                e = d[n] - y[n]
                # Update filter coefficients
                w = w + 2 * step_size * e * x

            return y

        # Limit k value as in original code
        k = min(20, k)
        r = 1

        # Process s1 components
        for i in range(k):
            if frqs1[i] < 250 or frqs1[i] > 800:
                filtered = lms_filter(s1[:, i], x2, deg, ms)
                x2 = x2 - filtered
            else:
                if Nprv - 50 <= frqs1[i] <= Nprv + 50:
                    filtered = lms_filter(s1[:, i] * r, x2, deg, ms)
                else:
                    filtered = lms_filter(s1[:, i], x2, deg, ms)
                x2 = x2 - filtered

        # Process s2 components
        for i in range(k):
            if frqs2[i] < 250 or frqs2[i] > 800:
                filtered = lms_filter(s2[:, i], x2, deg, ms)
                x2 = x2 - filtered
            else:
                if Nprv - 50 <= frqs2[i] <= Nprv + 50:
                    filtered = lms_filter(s2[:, i] * r, x2, deg, ms)
                else:
                    filtered = lms_filter(s2[:, i], x2, deg, ms)
                x2 = x2 - filtered

        # Process s3 components
        for i in range(k):
            if frqs3[i] < 250 or frqs3[i] > 800:
                filtered = lms_filter(s3[:, i], x2, deg, ms)
                x2 = x2 - filtered
            else:
                if Nprv - 50 <= frqs3[i] <= Nprv + 50:
                    filtered = lms_filter(s3[:, i] * r, x2, deg, ms)
                else:
                    filtered = lms_filter(s3[:, i], x2, deg, ms)
                x2 = x2 - filtered

        return x2

    def initialize_hr(self, ppg,fs):
        f = np.abs(np.fft.fft(ppg))
        freqs = np.fft.fftfreq(len(ppg), 1 / fs)

        valid_range = (freqs >= 1.2) & (freqs <= 1.5)

        f_in_range = f[valid_range]

        max_idx_in_range = np.argmax(f_in_range)

        Nprv = np.where(valid_range)[0][0] + max_idx_in_range

        return int(Nprv), freqs[Nprv] * 60

    def spectral_analysis(self, acc_x, acc_y, acc_z, Nprv, sample, estimate, window_count, x2, z2):
        """
        Analyze spectrum of PPG signal and estimate heart rate
        Args:
            acc_x/y/z: Acceleration signals for each axis
            Nprv: Previous heart rate estimate * 4
            sample: Sample flags array
            estimate: Estimation array
            window_count: Current window number
            x2: Current window PPG signal after MA cancellation
            z2: Extended window PPG signal after MA cancellation
        Returns:
            BPM: Estimated heart rate
        """
        # Initialize spectrum matrix (2 rows instead of 4 since we only process one PPG channel)
        spec = np.zeros((2, 800))  # Reduced from 4000 to 800 for efficiency

        # Calculate noise spectrum from acceleration signals
        acc_signals = np.stack([acc_x, acc_y, acc_z])
        acc_means = np.mean(acc_signals, axis=1, keepdims=True)
        acc_centered = acc_signals - acc_means

        # Adjust FFT length based on sampling rate
        if self.fs == self.fs_galaxy:  # Galaxy (25Hz)
            fft_length = 6000  # Reduced from 30000 for Galaxy's lower sampling rate
        else:  # E4 (64Hz)
            fft_length = 15000  # Adjusted for E4's higher sampling rate

        # Calculate noise spectrum
        noise = np.zeros((3, 800))
        for i in range(3):
            fft_result = np.abs(np.fft.fft(acc_centered[i], fft_length)) ** 2
            noise[i, :800] = fft_result[:800]  # Only keep first 800 points

        # Find noise peaks
        nsp = []
        for i in range(3):
            peaks, _ = find_peaks(noise[i], distance=20)
            if len(peaks) >= 2:
                peak_heights = noise[i][peaks]
                top_two_idx = np.argsort(peak_heights)[-2:]
                nsp.extend(peaks[top_two_idx])

        # Process signal using IMAT or FFT based on window count
        if window_count > 4:
            # For current window
            r = 1
            r = r + (r == 1) - (r == 2)  # Toggle between 1 and 2
            f = self.imat(x2, r * 2)
            spec[0, :800] = f[:800]

            # For extended window
            f = self.imat(z2, r * 2)
            spec[1, :800] = f[:800]
        else:
            # Use regular FFT for first few windows
            fft_result1 = np.abs(np.fft.fft(x2, fft_length)) ** 2
            spec[0, :800] = fft_result1[:800]

            fft_result2 = np.abs(np.fft.fft(z2, fft_length)) ** 2
            spec[1, :800] = fft_result2[:800]

        # Estimate heart rate using peak selection
        BPM = self.peakselection(Nprv, nsp, sample, estimate, window_count, spec)

        return BPM
    def peakselection(self, Nprv, nsp, sample, estimate, window_count, spec):
        """
        Select peaks to estimate heart rate
        Args:
            Nprv: Previous heart rate estimate * 4
            nsp: List of strong peak positions from noise signals
            sample: Sample flags array
            estimate: Estimation array
            window_count: Current window number
            spec: Spectrum matrix (2, 800)
                  First row: spectrum of current window PPG signal
                  Second row: spectrum of extended window PPG signal
        Returns:
            BPM: Estimated heart rate
        """
        # First check for strong peaks using watchdog
        bul, N = self.watchdog(spec, Nprv, nsp)

        if bul == 1:
            # If strong peak found, use watchdog result directly
            estimate[window_count] = N
            sample[window_count] = 1
        else:
            # If no strong peak found, proceed with peak analysis
            e = np.ceil(Nprv)
            q = self.findbigpeaks(spec, 0.6, e, nsp, 60, 60)

            # Process harmonics
            q = np.array(q)  # Ensure numpy array
            if len(q) >= 6:  # Check array length
                # Second harmonic processing
                q[2:4] = q[2:4] / 2
                # Third harmonic processing
                q[4:6] = q[4:6] / 3

                # Calculate initial estimate
                # Note: Only process first 4 values as we only have one PPG channel
                jam = 0
                k = 0
                for s in range(4):  # Only use first 4 values
                    if q[s] != 0:
                        jam += q[s]
                        k += 1

                # Protect against division by zero
                estimate[window_count] = jam / (k + (k == 0))
                if jam == 0:
                    estimate[window_count] = Nprv

        # Re-evaluate if new estimate differs too much from previous
        if abs(estimate[window_count] - Nprv) > 50:
            e = np.ceil(Nprv)
            q = self.findbigpeaks(spec, 0.6, e, nsp, 60, 60)
            q = np.array(q)
            if len(q) >= 6:
                q[2:4] = q[2:4] / 2  # Second harmonic
                q[4:6] = q[4:6] / 3  # Third harmonic

                # Recalculate estimate
                jam = 0
                k = 0
                v = []  # Store non-zero q values
                for s in range(4):  # Only use first 4 values
                    if q[s] != 0:
                        jam += q[s]
                        v.append(q[s])
                        k += 1

                estimate[window_count] = jam / (k + (k == 0))
                if jam == 0:
                    estimate[window_count] = Nprv  # If no dominant peak found

        # Check for zero value
        if estimate[window_count] == 0:
            estimate[window_count] = Nprv

        BPM = estimate[window_count] / 4

        return BPM
    def imat(self, y, L):
        """
        Precise implementation of IMAT algorithm
        Args:
            y: Input signal
            L: Window length parameter
        Returns:
            f: Power spectrum
        """
        iternum = 5  # Number of iterations

        i = np.arange(1, L * 10 + (L == 2) * 60 + 1)

        # Initialize mask and signal arrays
        msk = np.zeros(80, dtype=np.complex128)
        zsig = np.zeros(80, dtype=np.complex128)

        # Set mask and signal values - adjust indices to match MATLAB
        indices = (L * (i - 1) + 1).astype(int) - 1  # -1 to convert to 0-based indexing
        valid_indices = indices[indices < 80]
        msk[valid_indices] = 1
        zsig[valid_indices] = y[:len(valid_indices)]

        # Calculate initial FFT
        tsig = np.abs(np.fft.fft(zsig))
        beta = np.max(tsig) / 10
        alpha = 0.1
        landa = 0.2

        # Initialize signal
        x = np.zeros(80, dtype=np.complex128)
        y_signal = np.zeros(80, dtype=np.complex128)

        # Initial linear interpolation
        s = 0j
        for i in range(80):
            if msk[i] == 0:
                y_signal[i] = s
            else:
                y_signal[i] = zsig[i]
                s = zsig[i]

        # Iterative reconstruction
        for iter_idx in range(iternum):
            # Initialize interpolation signal
            xI = np.zeros(80, dtype=np.complex128)
            s = 0j

            # Linear interpolation
            for ii in range(80):
                if msk[ii] == 0:
                    xI[ii] = s
                else:
                    xI[ii] = x[ii]
                    s = x[ii]

            # Update signal
            xnew = x + landa * (y_signal - xI)
            tx = np.fft.fft(xnew)
            th = beta * np.exp(-alpha * iter_idx)

            # Threshold processing
            ttmsk = np.zeros(80, dtype=np.complex128)
            ttmsk[np.abs(tx) > th] = 1

            # Inverse FFT update
            x = np.fft.ifft(ttmsk * tx)

        # Calculate final power spectrum
        f = np.abs(np.fft.fft(x, 24000)) ** 2

        return f
    def findbigpeaks(self, spec, trsh, Nprv, nsp, s1, s2):
        """
        Find significant peaks in spectrum
        Args:
            spec: Spectrum matrix (2, 800)
                  First row: current window PPG1
                  Second row: extended window PPG1
            trsh: Threshold ratio (0.6)
            Nprv: Previous heart rate estimate * 4
            nsp: List of strong noise peak positions
            s1, s2: Search range parameters (both 60)
        Returns:
            pik: Array of found peaks [fundamental peaks, second harmonic peaks, third harmonic peaks]
        """
        # Initialize storage matrices - only need 2 rows for PPG1
        locs = np.zeros((2, 3))  # Fundamental peak positions
        hgt = np.zeros((2, 3))  # Fundamental peak heights
        locsh = np.zeros((2, 3))  # Second harmonic peak positions
        hgth = np.zeros((2, 3))  # Second harmonic peak heights
        locs3d = np.zeros((2, 3))  # Third harmonic peak positions
        hgt3d = np.zeros((2, 3))  # Third harmonic peak heights

        # Process each window's spectrum
        for j in range(2):  # 0: current window, 1: extended window
            # Get maximum value of current spectrum
            m = np.max(spec[j])

            # 1. Fundamental frequency range search
            b1 = int(max(Nprv - s1, 200))
            win = slice(b1, int(Nprv + s1))
            peaks, _ = find_peaks(spec[j, win], distance=5)
            if len(peaks) > 0:
                # Get peak heights and sort
                peak_heights = spec[j, win][peaks]
                sort_idx = np.argsort(-peak_heights)
                peak_heights = peak_heights[sort_idx]
                peaks = peaks[sort_idx]
                # Save top three peaks
                n_peaks = min(len(peaks), 3)
                locs[j, :n_peaks] = peaks[:n_peaks] + b1
                hgt[j, :n_peaks] = peak_heights[:n_peaks]

            # 2. Second harmonic range search
            b2 = int(max(2 * Nprv - s2, 400))
            whar = slice(b2, int(2 * Nprv + s2))
            peaks, _ = find_peaks(spec[j, whar], distance=5)
            if len(peaks) > 0:
                peak_heights = spec[j, whar][peaks]
                sort_idx = np.argsort(-peak_heights)
                peak_heights = peak_heights[sort_idx]
                peaks = peaks[sort_idx]
                n_peaks = min(len(peaks), 3)
                locsh[j, :n_peaks] = peaks[:n_peaks] + b2
                hgth[j, :n_peaks] = peak_heights[:n_peaks]

            # 3. Third harmonic range search
            w3d_start = int(3 * Nprv - 100)
            w3d_end = int(3 * Nprv + 100)
            w3d = slice(w3d_start, w3d_end)
            peaks, _ = find_peaks(spec[j, w3d], distance=5)
            if len(peaks) > 0:
                peak_heights = spec[j, w3d][peaks]
                sort_idx = np.argsort(-peak_heights)
                peak_heights = peak_heights[sort_idx]
                peaks = peaks[sort_idx]
                n_peaks = min(len(peaks), 3)
                locs3d[j, :n_peaks] = peaks[:n_peaks] + w3d_start
                hgt3d[j, :n_peaks] = peak_heights[:n_peaks]

        # Select significant peaks
        dmn = np.zeros(2)  # Fundamental peaks
        dmnh = np.zeros(2)  # Second harmonic peaks
        dmn3d = np.zeros(2)  # Third harmonic peaks

        # Set minimum threshold
        min_threshold = 1000

        # Use whichever is smaller: fixed minimum or percentage of max
        m_threshold = min(min_threshold, m * 0.1)

        # Process each window for peak selection
        for j in range(2):
            # Fundamental peak selection
            if hgt[j, 0] * trsh > hgt[j, 1] and hgt[j, 0] > m_threshold:
                dmn[j] = locs[j, 0]

            # Second harmonic peak selection
            if hgth[j, 0] * trsh > hgth[j, 1]:
                dmnh[j] = locsh[j, 0]

            # Third harmonic peak selection
            if hgt3d[j, 0] * trsh > hgt3d[j, 1]:
                dmn3d[j] = locs3d[j, 0]

        # Combine all selected peaks
        pik = np.concatenate([dmn.ravel(), dmnh.ravel(), dmn3d.ravel()])

        return pik
    def watchdog(self, spec, Nprv, nsp):
        """
        Monitor strong peaks in PPG signal
        Args:
            spec: Spectrum matrix (2, 800)
                spec[0]: 8-second window PPG spectrum
                spec[1]: 16-second(extended) window PPG spectrum
            Nprv: Previous heart rate estimate * 4
            nsp: List of strong peak positions from noise signals
        Returns:
            bul: Boolean flag indicating if valid strong peak found
            N: Estimated heart rate value
        """
        # Initialize parameters
        bul = 0
        N = 0
        dstns = 9  # Peak distance threshold
        trsh = 0.3  # Initial threshold
        Nprv = np.ceil(Nprv)
        nd = 5  # Noise distance threshold

        # Process 8-second window PPG signal
        y1 = spec[0, 50:800]  # Start from 50 to avoid low frequency noise
        peaks1, _ = find_peaks(y1)

        if len(peaks1) > 0:
            h1 = y1[peaks1]  # Peak heights
            # Sort in descending order
            sort_idx = np.argsort(-h1)
            h1 = h1[sort_idx]
            p1 = peaks1[sort_idx] + 50  # Add back the offset

            # Check strong peak conditions
            if len(h1) >= 3 and len(p1) >= 2:
                # Check if first peak is significantly larger than third peak
                if h1[0] * 0.33 > h1[2]:
                    # Check harmonic relationships
                    if abs(p1[0] - p1[1] / 2) < dstns:
                        if min(abs(np.array(p1[0]) - np.array(nsp))) >= nd and \
                                min(abs(np.array(p1[1]) - np.array(nsp))) >= nd:
                            N = (p1[0] + p1[1]) / 3
                            bul = 1

                    if abs(p1[1] - p1[0] / 2) < dstns:
                        if min(abs(np.array(p1[0]) - np.array(nsp))) >= nd and \
                                min(abs(np.array(p1[1]) - np.array(nsp))) >= nd:
                            N = (p1[0] + p1[1]) / 3
                            bul = 1

            # Dynamic threshold adjustment
            if len(p1) > 0:
                if abs(p1[0] - Nprv) < 50:
                    trsh = 0.4
                    if len(p1) > 1 and min(abs(p1[0] - np.array(p1[1:]) / 2)) < 5:
                        trsh = 0.5

            # Single peak detection
            if len(h1) >= 2 and h1[0] * trsh > h1[1]:
                if len(p1) > 0 and min(abs(p1[0] - np.array(nsp))) >= nd:
                    if p1[0] > 800:
                        N = p1[0] / 2
                    else:
                        N = p1[0]
                    bul = 1

        # If no significant peak found in 8-second window, check 16-second window
        if bul == 0:
            N = 0
            dstns = 9
            trsh = 0.4

            # Process 16-second window spectrum
            y2 = spec[1, 50:800]
            peaks2, _ = find_peaks(y2)

            if len(peaks2) > 0:
                h2 = y2[peaks2]
                sort_idx = np.argsort(-h2)
                h2 = h2[sort_idx]
                p2 = peaks2[sort_idx] + 50

                # Check strong peak conditions
                if len(h2) >= 3 and len(p2) >= 2:
                    if h2[0] * 0.33 > h2[2]:
                        if abs(p2[0] - p2[1] / 2) < dstns:
                            if min(abs(np.array(p2[0]) - np.array(nsp))) >= nd and \
                                    min(abs(np.array(p2[1]) - np.array(nsp))) >= nd:
                                N = (p2[0] + p2[1]) / 3
                                bul = 1

                        if abs(p2[1] - p2[0] / 2) < dstns:
                            if min(abs(np.array(p2[0]) - np.array(nsp))) >= nd and \
                                    min(abs(np.array(p2[1]) - np.array(nsp))) >= nd:
                                N = (p2[0] + p2[1]) / 3
                                bul = 1

                # Dynamic threshold adjustment(16-second window)
                if len(p2) > 0:
                    if abs(p2[0] - Nprv) < 50:
                        trsh = 0.4
                        if len(p2) > 1 and min(abs(p2[0] - np.array(p2[1:]) / 2)) < 5:
                            trsh = 0.5

                # Single peak detection(16-second window)
                if len(h2) >= 2 and h2[0] * trsh > h2[1]:
                    if len(p2) > 0 and min(abs(p2[0] - np.array(nsp))) >= nd:
                        if p2[0] > 800:
                            N = p2[0] / 2
                        else:
                            N = p2[0]
                        bul = 1

        # Final validation
        if N > 0:
            if min(abs(N - np.array(nsp))) < nd:
                bul = 0

        if abs(N - Nprv) > 100:
            N = Nprv + np.sign(N - Nprv) * 50
            bul = 0

        return bul, N
def bandpass_filter(data, lowcut, highcut, fs, order=4):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    y = filtfilt(b, a, data)
    return y
def process_dataset(participant_range=None, participant_list=None):

    all_errors = []
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
            print(f"{filename}")
            continue

        print(f"\n{participant_id}")

        try:
            df = pd.read_csv(file_path)
            galaxy_denoiser = IMATDenoising()
            e4_denoiser = IMATDenoising()

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
                        print(f" {progress:.1f}% ({i}/{total_windows})")

                    galaxy_ppg = np.array(row['galaxyPPG'].split(';'), dtype=float)
                    galaxy_acc = np.array(row['galaxyACC'].split(';'), dtype=float).reshape(-1, 3)

                    e4_bvp = np.array(row['e4BVP'].split(';'), dtype=float)
                    e4_acc = np.array(row['e4ACC'].split(';'), dtype=float).reshape(-1, 3)

                    true_hr = row['gdHR']

                    galaxy_ppg_filtered = bandpass_filter(galaxy_ppg, lowcut, highcut, galaxy_denoiser.fs_galaxy)
                    galaxy_acc_filtered = np.array([
                        bandpass_filter(galaxy_acc[:, i], lowcut, highcut, galaxy_denoiser.fs_galaxy)
                        for i in range(3)
                    ]).T

                    galaxy_ppg_norm = (galaxy_ppg_filtered - galaxy_ppg_filtered.min()) / (
                            galaxy_ppg_filtered.max() - galaxy_ppg_filtered.min())
                    galaxy_acc_norm = (galaxy_acc_filtered - galaxy_acc_filtered.min(axis=0)) / (
                            galaxy_acc_filtered.max(axis=0) - galaxy_acc_filtered.min(axis=0))

                    galaxy_denoised, galaxy_bpm = galaxy_denoiser.process_galaxy(
                        galaxy_ppg_norm,
                        galaxy_acc_norm[:, 0],
                        galaxy_acc_norm[:, 1],
                        galaxy_acc_norm[:, 2]
                    )

                    e4_bvp_filtered = bandpass_filter(e4_bvp, lowcut, highcut, e4_denoiser.fs_e4_bvp)
                    e4_acc_resampled = np.array([
                        signal.resample(e4_acc[:, i], len(e4_bvp))
                        for i in range(3)
                    ]).T

                    e4_acc_filtered = np.array([
                        bandpass_filter(e4_acc_resampled[:, i], lowcut, highcut, e4_denoiser.fs_e4_bvp)
                        for i in range(3)
                    ]).T

                    e4_bvp_norm = (e4_bvp_filtered - e4_bvp_filtered.min()) / (
                            e4_bvp_filtered.max() - e4_bvp_filtered.min())
                    e4_acc_norm = (e4_acc_filtered - e4_acc_filtered.min(axis=0)) / (
                            e4_acc_filtered.max(axis=0) - e4_acc_filtered.min(axis=0))

                    e4_denoised, e4_bpm = e4_denoiser.process_e4(
                        e4_bvp_norm,
                        e4_acc_norm[:, 0],
                        e4_acc_norm[:, 1],
                        e4_acc_norm[:, 2]
                    )

                    results['denoisedGalaxy'][i] = ';'.join(map(str, galaxy_denoised.tolist()))
                    results['denoisedE4'][i] = ';'.join(map(str, e4_denoised.tolist()))
                    results['estimated_BPM_Galaxy'][i] = galaxy_bpm
                    results['estimated_BPM_E4'][i] = e4_bpm
                    results['BPM_error_Galaxy'][i] = abs(galaxy_bpm - true_hr)
                    results['BPM_error_E4'][i] = abs(e4_bpm - true_hr)

                except Exception as e:
                    print(f" {str(e)}")
                    continue

            for col, values in results.items():
                df[col] = values

            output_dir = os.path.join(RESULTS_DIR, 'IMAT')
            os.makedirs(output_dir, exist_ok=True)
            output_file = os.path.join(output_dir, f'{participant_id}_processed_GD_denoised.csv')
            df.to_csv(output_file, index=False)

            for device in ['Galaxy', 'E4']:
                valid_errors = [e for e in results[f'BPM_error_{device}'] if e is not None]

        except Exception as e:
            print(f"{str(e)}")
            continue

if __name__ == "__main__":
    process_dataset(participant_range=(16, 21))
    # participant_list = ["P03"]
    # process_dataset(participant_list=participant_list)
    # process_dataset()