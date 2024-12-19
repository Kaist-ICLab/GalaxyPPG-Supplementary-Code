import pandas as pd
import numpy as np
import heartpy as hp
import os
import traceback
from heartpy.filtering import filter_signal
from heartpy.preprocessing import scale_data
from typing import Dict, Optional, Tuple
from collections import defaultdict

class PPGAnalyzer:
    def __init__(self):
        """Initialize PPG Analyzer with processing parameters"""
        # Sampling rates
        self.galaxy_fs = 25  # Galaxy sampling rate
        self.e4_fs = 64  # E4 sampling rate
        self.ecg_fs = 130  # ECG sampling rate

        # HR limits
        self.HR_MIN = 40
        self.HR_MAX = 200

        # Target sessions for analysis
        self.target_sessions = [
            'baseline',
            'tsst-speech',
            'screen-reading',
            'ssst-sing',
            'keyboard-typing',
            'mobile-typing',
            'standing',
            'walking',
            'jogging',
            'running'
        ]

        # Processing parameters
        self.processing_params = {
            'bandpass': {
                'low_cutoff': 0.5,
                'high_cutoff': 4.0,
                'order': 3
            },
            'enhancement': {
                'iterations': 1
            }
        }
    def process_signal(self, signal: np.ndarray, sample_rate: float) -> Tuple[
        Optional[Dict], Optional[Dict], Optional[np.ndarray]]:
        """Process PPG signal and detect peaks remains unchanged"""
        try:
            filtered_signal = filter_signal(
                signal,
                cutoff=[self.processing_params['bandpass']['low_cutoff'],
                        self.processing_params['bandpass']['high_cutoff']],
                sample_rate=sample_rate,
                filtertype='bandpass',
                order=self.processing_params['bandpass']['order']
            )
            scaled_signal = scale_data(filtered_signal)

            # scaled_signal = signal
            working_data, measures = hp.process(
                scaled_signal,
                sample_rate=sample_rate,
                windowsize=2,
                bpmmin=self.HR_MIN,
                bpmmax=self.HR_MAX,
                clean_rr=True,
                clean_rr_method='quotient-filter',
                interp_clipping=True,
                interp_threshold=1020,
            )

            if 'peaklist' in working_data and len(working_data['peaklist']) > 0:
                processed_peaks = self.post_process_peaks(
                    scaled_signal,
                    working_data['peaklist'],
                    sample_rate
                )

                if len(processed_peaks) >= 2:
                    working_data['peaklist'] = processed_peaks
                    working_data['ybeat'] = scaled_signal[processed_peaks]

                    intervals = np.diff(processed_peaks) / sample_rate * 1000

                    measures = {
                        'bpm': 60000 / np.mean(intervals),
                        'ibi': np.mean(intervals),
                        'sdnn': np.std(intervals, ddof=1),
                        'rmssd': np.sqrt(np.mean(np.diff(intervals) ** 2))
                    }

            return working_data, measures, scaled_signal
        except Exception as e:
            pass
            return None, None, None
        pass

    def analyze_data(self, df: pd.DataFrame) -> Dict:
        """
        Analyze PPG data from DataFrame

        Args:
            df: DataFrame containing PPG and ground truth data

        Returns:
            Dict: Analysis results for all sessions
        """
        all_results = {}

        for session in self.target_sessions:
            if session in df['session'].unique():
                session_results = self.analyze_session_data(df, session)
                if session_results:
                    all_results[session] = session_results

        return all_results

    def post_process_peaks(self, signal: np.ndarray, peaks: np.ndarray, sample_rate: float) -> np.ndarray:
        """
        Enhanced peak detection post-processing with improved missing peak detection algorithm
        Args:
            signal: Input PPG signal array
            peaks: Initially detected peak indices
            sample_rate: Sampling frequency in Hz
        Returns:
            np.ndarray: Refined peak indices
        """
        # Return original peaks if less than 2 peaks detected
        if len(peaks) < 2:
            return np.array(peaks, dtype=np.int64)

        peaks = np.array(peaks, dtype=np.int64)

        # Initialize physiological parameters based on cardiac timing
        MIN_INTERVAL = int(0.2 * sample_rate)  # Minimum allowed interval between peaks (200ms)
        MAX_INTERVAL = int(1.5 * sample_rate)  # Maximum allowed interval between peaks (1.5s)
        SEARCH_WINDOW = int(0.3 * sample_rate)  # Window size for local peak analysis

        def get_peak_characteristics(idx: int) -> Tuple[float, float, bool]:
            """
            Calculate peak prominence, relative height and validity
            Args:
                idx: Index position in signal to analyze
            Returns:
                Tuple of (prominence, relative_height, is_valid_peak)
            """
            if idx <= 0 or idx >= len(signal) - 1:
                return 0, 0, False

            # Define analysis window around peak
            left_idx = max(0, idx - SEARCH_WINDOW)
            right_idx = min(len(signal), idx + SEARCH_WINDOW)
            window = signal[left_idx:right_idx]

            # Calculate local statistics
            local_mean = np.mean(window)
            local_std = np.std(window)

            # Calculate peak prominence
            left_min = np.min(signal[left_idx:idx])
            right_min = np.min(signal[idx:right_idx])
            prominence = signal[idx] - max(left_min, right_min)

            # Calculate peak height relative to local signal statistics
            relative_height = (signal[idx] - local_mean) / local_std

            # Check if point is a valid peak
            is_peak = (signal[idx] > signal[idx - 1] and
                       signal[idx] > signal[idx + 1] and
                       relative_height > 0.8)

            return prominence, relative_height, is_peak

        def find_peak_in_range(start_idx: int, end_idx: int) -> Optional[int]:
            """
            Find most prominent peak within specified range
            Args:
                start_idx: Start index of search range
                end_idx: End index of search range
            Returns:
                Index of most prominent peak or None if no valid peak found
            """
            if end_idx - start_idx < MIN_INTERVAL:
                return None

            # Calculate local signal statistics
            range_signal = signal[start_idx:end_idx]
            range_mean = np.mean(range_signal)
            range_std = np.std(range_signal)
            min_height = range_mean + 0.5 * range_std

            best_peak = None
            best_prominence = 0

            # Search for most prominent peak in range
            for i in range(start_idx + 1, end_idx - 1):
                if (signal[i] > min_height and
                        signal[i] > signal[i - 1] and
                        signal[i] > signal[i + 1]):

                    prominence, _, is_peak = get_peak_characteristics(i)
                    if is_peak and prominence > best_prominence:
                        best_peak = i
                        best_prominence = prominence

            return best_peak

        # First pass: Filter peaks that are too close together
        filtered_peaks = []
        i = 0
        while i < len(peaks):
            current_peak = peaks[i]
            current_prominence, _, current_is_peak = get_peak_characteristics(current_peak)

            if not current_is_peak:
                i += 1
                continue

            # Compare with nearby peaks to select most prominent one
            j = i + 1
            while j < len(peaks) and peaks[j] - peaks[i] < MIN_INTERVAL:
                next_prominence, _, next_is_peak = get_peak_characteristics(peaks[j])
                if next_is_peak and next_prominence > current_prominence:
                    current_peak = peaks[j]
                    current_prominence = next_prominence
                j += 1

            filtered_peaks.append(current_peak)
            i = j if j > i + 1 else i + 1

        filtered_peaks = np.array(filtered_peaks)

        # Second pass: Detect missing peaks
        final_peaks = [filtered_peaks[0]]

        for i in range(1, len(filtered_peaks)):
            current_interval = filtered_peaks[i] - final_peaks[-1]

            # Check for missing peaks in large intervals
            if current_interval > MAX_INTERVAL:
                expected_peaks = int(current_interval / (0.6 * sample_rate))

                # Search for missing peaks at expected locations
                for j in range(1, expected_peaks):
                    expected_pos = final_peaks[-1] + int(j * current_interval / (expected_peaks + 1))

                    search_start = expected_pos - int(0.2 * sample_rate)
                    search_end = expected_pos + int(0.2 * sample_rate)

                    found_peak = find_peak_in_range(
                        max(final_peaks[-1] + MIN_INTERVAL, search_start),
                        min(filtered_peaks[i] - MIN_INTERVAL, search_end)
                    )

                    if found_peak is not None:
                        final_peaks.append(found_peak)

            final_peaks.append(filtered_peaks[i])

        # Final validation pass
        validated_peaks = []
        for peak in final_peaks:
            _, _, is_valid = get_peak_characteristics(peak)
            if is_valid:
                if not validated_peaks or peak - validated_peaks[-1] >= MIN_INTERVAL:
                    validated_peaks.append(peak)

        return np.array(validated_peaks, dtype=np.int64)

    def match_peaks_fcfs(self, ppg_peaks: np.ndarray, ecg_peaks: np.ndarray,
                         ppg_fs: float, tolerance_window: float = 0.5) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """
        Match PPG peaks to ECG R-peaks using First-Come-First-Serve principle

        Args:
            ppg_peaks: Array of PPG peak indices
            ecg_peaks: Array of ECG peak indices
            ppg_fs: PPG sampling frequency
            tolerance_window: Time window for peak matching (seconds)

        Returns:
            Tuple containing matched PPG peaks, matched ECG peaks, and matching statistics
        """
        try:
            # Ensure input arrays are numpy arrays
            ppg_peaks = np.asarray(ppg_peaks, dtype=np.int64)
            ecg_peaks = np.asarray(ecg_peaks, dtype=np.int64)

            # Convert peaks to time domain (seconds)
            ppg_times = ppg_peaks.astype(float) / ppg_fs
            ecg_times = ecg_peaks.astype(float) / self.ecg_fs

            matched_ppg = []
            matched_ecg = []
            unmatched_ppg = []

            # Process each PPG peak
            for i, ppg_time in enumerate(ppg_times):
                if len(ecg_times) == 0:
                    unmatched_ppg.append(ppg_peaks[i])
                    continue

                # Find closest ECG peak
                time_diffs = np.abs(ecg_times - ppg_time)
                min_diff_idx = np.argmin(time_diffs)

                # Match if within tolerance window
                if time_diffs[min_diff_idx] <= tolerance_window:
                    matched_ppg.append(ppg_peaks[i])
                    matched_ecg.append(ecg_peaks[min_diff_idx])
                    # Remove matched ECG peak from candidates
                    ecg_times = np.delete(ecg_times, min_diff_idx)
                    ecg_peaks = np.delete(ecg_peaks, min_diff_idx)
                else:
                    unmatched_ppg.append(ppg_peaks[i])

            # Convert to numpy arrays
            matched_ppg = np.array(matched_ppg, dtype=np.int64)
            matched_ecg = np.array(matched_ecg, dtype=np.int64)

            # Calculate statistics
            stats = {
                'total_ppg_peaks': len(ppg_peaks),
                'total_ecg_peaks': len(matched_ecg) + len(ecg_peaks),
                'matched_peaks': len(matched_ppg),
                'matching_rate': len(matched_ecg) / len(ecg_peaks) * 100 if len(ecg_peaks) > 0 else 0
            }

            return matched_ppg, matched_ecg, stats

        except Exception as e:
            print(f"Error in peak matching: {str(e)}")
            # Return empty arrays and zero stats on error
            return np.array([], dtype=np.int64), np.array([], dtype=np.int64), {
                'total_ppg_peaks': 0,
                'total_ecg_peaks': 0,
                'matched_peaks': 0,
                'matching_rate': 0
            }

    def summarize_results(self, results: Dict) -> Dict:
        """Summarize results across all sessions and participants"""
        summary = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

        for participant_id, participant_data in results.items():
            for session, session_data in participant_data.items():
                for device in ['PolarH10', 'Galaxy', 'E4']:
                    if device in session_data:
                        device_data = session_data[device]

                        # Process original metrics
                        for metric in ['HR', 'IBI', 'SDNN', 'RMSSD']:
                            if metric in device_data:
                                summary[session][device][metric].extend(device_data[metric])

                        # Process MAE metrics (only for Galaxy and E4)
                        if device in ['Galaxy', 'E4']:
                            for metric in ['HR_mae', 'IBI_mae', 'SDNN_mae', 'RMSSD_mae']:
                                if metric in device_data:
                                    summary[session][device][metric].extend(device_data[metric])

                        # Process PMR
                        if 'PMR' in device_data:
                            summary[session][device]['PMR'].extend(device_data['PMR'])

        # Calculate statistics
        for session in summary:
            for device in summary[session]:
                for metric in summary[session][device]:
                    values = summary[session][device][metric]
                    if values:
                        summary[session][device][metric] = {
                            'mean': np.mean(values),
                            'std': np.std(values) if len(values) > 1 else 0,
                            'count': len(values)
                        }

        return summary

    def analyze_session_data(self, df: pd.DataFrame, session: str) -> Dict:
        """Analyze data for a specific session"""
        session_results = {
            'Galaxy': defaultdict(list),
            'E4': defaultdict(list),
            'PolarH10': defaultdict(list)
        }

        session_data = df[df['session'] == session]

        for _, window in session_data.iterrows():
            #ECG HRV
            if not np.isnan(window['gdHR']):
                session_results['PolarH10']['HR'].append(window['gdHR'])
            if not np.isnan(window['gdIBI']):
                session_results['PolarH10']['IBI'].append(window['gdIBI'])
            if not np.isnan(window['gdSDNN']):
                session_results['PolarH10']['SDNN'].append(window['gdSDNN'])
            if not np.isnan(window['gdRMSSD']):
                session_results['PolarH10']['RMSSD'].append(window['gdRMSSD'])

            # Process Galaxy PPG
            try:
                galaxy_signal = np.array([float(x) for x in window['denoisedGalaxy'].split(';') if x.strip()])
                # galaxy_signal = np.array([float(x) for x in window['galaxyPPG'].split(';') if x.strip()])
                galaxy_results = self.process_signal(galaxy_signal, self.galaxy_fs)

                if galaxy_results[0] is not None:
                    working_data, metrics, _ = galaxy_results
                    ecg_peaks = np.array([float(x) for x in window['gdPeaks'].split(';') if x.strip()])

                    if len(working_data['peaklist']) >= 2 and len(ecg_peaks) >= 2:
                        if not np.isnan(metrics['bpm']):
                            session_results['Galaxy']['HR'].append(metrics['bpm'])
                        if not np.isnan(metrics['ibi']):
                            session_results['Galaxy']['IBI'].append(metrics['ibi'])
                        if not np.isnan(metrics['sdnn']):
                            session_results['Galaxy']['SDNN'].append(metrics['sdnn'])
                        if not np.isnan(metrics['rmssd']):
                            session_results['Galaxy']['RMSSD'].append(metrics['rmssd'])

                        # Store MAE
                        if not np.isnan(metrics['bpm']) and not np.isnan(window['gdHR']):
                            session_results['Galaxy']['HR_mae'].append(abs(metrics['bpm'] - window['gdHR']))
                        if not np.isnan(metrics['ibi']) and not np.isnan(window['gdIBI']):
                            session_results['Galaxy']['IBI_mae'].append(abs(metrics['ibi'] - window['gdIBI']))
                        if not np.isnan(metrics['sdnn']) and not np.isnan(window['gdSDNN']):
                            session_results['Galaxy']['SDNN_mae'].append(abs(metrics['sdnn'] - window['gdSDNN']))
                        if not np.isnan(metrics['rmssd']) and not np.isnan(window['gdRMSSD']):
                            session_results['Galaxy']['RMSSD_mae'].append(abs(metrics['rmssd'] - window['gdRMSSD']))

                        # Peak matching rate
                        matched_ppg, matched_ecg, _ = self.match_peaks_fcfs(
                            working_data['peaklist'],
                            ecg_peaks,
                            self.galaxy_fs
                        )
                        matching_rate = len(matched_ecg) / len(ecg_peaks) * 100 if len(ecg_peaks) > 0 else 0
                        session_results['Galaxy']['PMR'].append(matching_rate)

            except Exception as e:
                print(f"Error processing Galaxy PPG: {str(e)}")

            # Process E4 BVP (similar structure)
            try:
                e4_signal = np.array([float(x) for x in window['denoisedE4'].split(';') if x.strip()])
                # e4_signal = np.array([float(x) for x in window['e4BVP'].split(';') if x.strip()])
                e4_results = self.process_signal(e4_signal, self.e4_fs)

                if e4_results[0] is not None:
                    working_data, metrics, _ = e4_results
                    ecg_peaks = np.array([float(x) for x in window['gdPeaks'].split(';') if x.strip()])

                    if len(working_data['peaklist']) >= 2 and len(ecg_peaks) >= 2:
                        # Store original metrics
                        if not np.isnan(metrics['bpm']):
                            session_results['E4']['HR'].append(metrics['bpm'])
                        if not np.isnan(metrics['ibi']):
                            session_results['E4']['IBI'].append(metrics['ibi'])
                        if not np.isnan(metrics['sdnn']):
                            session_results['E4']['SDNN'].append(metrics['sdnn'])
                        if not np.isnan(metrics['rmssd']):
                            session_results['E4']['RMSSD'].append(metrics['rmssd'])

                        # Store MAE
                        if not np.isnan(metrics['bpm']) and not np.isnan(window['gdHR']):
                            session_results['E4']['HR_mae'].append(abs(metrics['bpm'] - window['gdHR']))
                        if not np.isnan(metrics['ibi']) and not np.isnan(window['gdIBI']):
                            session_results['E4']['IBI_mae'].append(abs(metrics['ibi'] - window['gdIBI']))
                        if not np.isnan(metrics['sdnn']) and not np.isnan(window['gdSDNN']):
                            session_results['E4']['SDNN_mae'].append(abs(metrics['sdnn'] - window['gdSDNN']))
                        if not np.isnan(metrics['rmssd']) and not np.isnan(window['gdRMSSD']):
                            session_results['E4']['RMSSD_mae'].append(abs(metrics['rmssd'] - window['gdRMSSD']))

                        # Peak matching rate
                        matched_ppg, matched_ecg, _ = self.match_peaks_fcfs(
                            working_data['peaklist'],
                            ecg_peaks,
                            self.e4_fs
                        )
                        matching_rate = len(matched_ecg) / len(ecg_peaks) * 100 if len(ecg_peaks) > 0 else 0
                        session_results['E4']['PMR'].append(matching_rate)

            except Exception as e:
                print(f"Error processing E4 BVP: {str(e)}")

        return session_results

    def save_results(self, summary: Dict, output_dir: str):
        """Save analysis results to CSV files"""
        os.makedirs(output_dir, exist_ok=True)

        # Create DataFrame for complete results
        rows = []
        for session in self.target_sessions:
            if session in summary:
                row = {'Activity': session}

                # Add metrics for each device
                for device in ['PolarH10', 'Galaxy', 'E4']:
                    if device in summary[session]:
                        # Add original metrics
                        for metric in ['HR', 'IBI', 'SDNN', 'RMSSD']:
                            if metric in summary[session][device]:
                                mean = summary[session][device][metric]['mean']
                                std = summary[session][device][metric]['std']
                                row[f'{device}_{metric}'] = f"{mean:.2f}±{std:.2f}"

                        # Add PMR for Galaxy and E4
                        if device in ['Galaxy', 'E4'] and 'PMR' in summary[session][device]:
                            mean = summary[session][device]['PMR']['mean']
                            row[f'{device}_PMR'] = f"{mean:.2f}"

                rows.append(row)

        # Save complete results
        complete_df = pd.DataFrame(rows)
        complete_df.to_csv(os.path.join(output_dir, 'complete_analysis_results.csv'), index=False)

        # Save MAE results separately
        mae_rows = []
        for session in self.target_sessions:
            if session in summary:
                row = {'Activity': session}
                for device in ['Galaxy', 'E4']:
                    if device in summary[session]:
                        for metric in ['HR_mae', 'IBI_mae', 'SDNN_mae', 'RMSSD_mae']:
                            if metric in summary[session][device]:
                                mean = summary[session][device][metric]['mean']
                                std = summary[session][device][metric]['std']
                                row[f'{device}_{metric}'] = f"{mean:.2f}±{std:.2f}"
                mae_rows.append(row)

        mae_df = pd.DataFrame(mae_rows)
        mae_df.to_csv(os.path.join(output_dir, 'mae_analysis_results.csv'), index=False)
# Data loading function
def load_participant_data(window_dir: str, results_dir: str, participant_id: str) -> pd.DataFrame:
    """
    Load and merge denoised and ground truth data for a participant

    Args:
        window_dir: Directory containing ground truth data
        results_dir: Directory containing denoised data
        participant_id: Participant ID to process

    Returns:
        pd.DataFrame: Merged dataset with both denoised and ground truth data
    """
    try:
        # Read denoised signal data and specify the Denoised algorithm
        denoised_file = os.path.join(results_dir, 'Wiener', f"{participant_id}_processed_GD_denoised.csv")
        ground_truth_file = os.path.join(window_dir, f"{participant_id}_processed_GD.csv")

        # Read the files
        denoised_df = pd.read_csv(denoised_file)
        ground_truth_df = pd.read_csv(ground_truth_file)
        denoised_df['denoisedGalaxy'] = denoised_df['denoisedGalaxy'].astype(str)
        denoised_df['denoisedE4'] = denoised_df['denoisedE4'].astype(str)
        # Drop the ground truth columns from denoised data if they exist
        columns_to_drop = ['gdHR', 'gdIBI', 'gdSDNN', 'gdRMSSD', 'gdPeaks']
        denoised_df = denoised_df.drop(columns=[col for col in columns_to_drop if col in denoised_df.columns])

        # Merge data based on windowNumber and session
        df = pd.merge(
            denoised_df,
            ground_truth_df[['windowNumber', 'session', 'gdHR', 'gdIBI', 'gdSDNN', 'gdRMSSD', 'gdPeaks']],
            on=['windowNumber', 'session'],
            how='left'
        )

        return df
    except Exception as e:
        print(f"Error loading data for participant {participant_id}: {str(e)}")
        traceback.print_exc()
        return pd.DataFrame()
def print_analysis(results):
    print("\n=== Per-Session Statistics ===")
    for session in results:
        print(f"\n{session}")
        for device in ['Galaxy', 'E4']:
            if device in results[session]:
                metrics = results[session][device]
                print(f"\n{device}:")
                for metric, stats in metrics.items():
                    print(f"{metric}: {stats['mean']:.2f} ± {stats['std']:.2f} (n={stats['count']})")
def main():
    from config import WINDOW_DIR, RESULTS_DIR
    # Initialize analyzer
    analyzer = PPGAnalyzer()
    output_dir = os.path.join(RESULTS_DIR, 'session_analysis')
    os.makedirs(output_dir, exist_ok=True)

    all_results = {}

    # Get list of all processed_GD files
    files = [f for f in os.listdir(WINDOW_DIR) if f.endswith('_processed_GD.csv')]

    for file in files:
        participant_id = file.split('_')[0]
        print(f"\nProcessing participant: {participant_id}")

        try:
            # Load data
            df = load_participant_data(WINDOW_DIR, RESULTS_DIR, participant_id)
            if not df.empty:
                # Process data
                results = analyzer.analyze_data(df)
                if results:
                    all_results[participant_id] = results
        except Exception as e:
            print(f"Error processing {participant_id}: {str(e)}")
            traceback.print_exc()

    # Summarize and save results
    summary = analyzer.summarize_results(all_results)
    analyzer.save_results(summary, output_dir)
    print(f"\nAnalysis complete. Results saved to: {output_dir}")
    print("\nSummary Statistics:")
    print_analysis(summary)

if __name__ == "__main__":
    main()




