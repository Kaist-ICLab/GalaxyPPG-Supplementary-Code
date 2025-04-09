import os
import pandas as pd
import numpy as np
import heartpy as hp
from typing import Dict, List, Tuple, Optional
import traceback
import sys
from collections import defaultdict
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import WINDOW_DIR, RESULTS_DIR, RAW_DIR
from PPG_Calculation_basing_HeartPy import


class HandPlacementMAEAnalyzer:
    """
    Analyzer for assessing the impact of hand position (left vs right)
    on heart rate metric accuracy compared to ECG ground truth.
    """

    def __init__(self):
        """Initialize the analyzer with the required parameters."""
        # Load metadata file containing hand placement information
        meta_path = os.path.join(RAW_DIR, 'META.csv')
        self.meta_df = pd.read_csv(meta_path)

        # Define target sessions to analyze
        self.target_sessions = [
            'baseline', 'tsst-speech', 'screen-reading', 'ssst-sing',
            'keyboard-typing', 'mobile-typing', 'standing', 'walking',
            'jogging', 'running'
        ]

        # Set output directory
        self.output_dir = os.path.join(RESULTS_DIR, 'hand_placement_analysis')
        os.makedirs(self.output_dir, exist_ok=True)

        # Set sampling rates
        self.galaxy_fs = 25  # Galaxy sampling rate
        self.e4_fs = 64  # E4 sampling rate

        # Separate participants by hand placement
        self.left_hand_users = self.meta_df[self.meta_df['GalaxyWatch'] == 'L']['UID'].tolist()
        self.right_hand_users = self.meta_df[self.meta_df['GalaxyWatch'] == 'R']['UID'].tolist()

        print(f"Left hand users: {len(self.left_hand_users)} participants")
        print(f"Right hand users: {len(self.right_hand_users)} participants")
    def load_participant_data(self, participant_id: str) -> pd.DataFrame:
        """
        Load processed data with ground truth ECG metrics for a participant

        Args:
            participant_id: ID of the participant to load data for

        Returns:
            pd.DataFrame: Processed data with ground truth metrics
        """
        try:
            # Load data with ground truth ECG metrics
            data_path = os.path.join(WINDOW_DIR, f"{participant_id}_processed_GD.csv")
            if not os.path.exists(data_path):
                print(f"Warning: No processed ground truth data found for {participant_id}")
                return pd.DataFrame()

            df = pd.read_csv(data_path)

            # Add hand placement info
            hand_placement = self.meta_df[self.meta_df['UID'] == participant_id]['GalaxyWatch'].values[0]
            df['HandPlacement'] = hand_placement

            return df
        except Exception as e:
            print(f"Error loading data for {participant_id}: {str(e)}")
            return pd.DataFrame()
    def analyze_participant_data(self, df: pd.DataFrame, participant_id: str) -> Dict[str, Dict]:
        """
        Analyze data for a single participant

        Args:
            df: DataFrame containing participant data
            participant_id: Participant ID

        Returns:
            Dictionary with analysis results
        """
        results = {
            'participant': participant_id,
            'hand_placement': df['HandPlacement'].iloc[0] if not df.empty else 'Unknown',
            'session_results': {},
            'device_results': {
                'Galaxy': {'HR': [], 'IBI': [], 'SDNN': [], 'RMSSD': [],
                           'HR_mae': [], 'IBI_mae': [], 'SDNN_mae': [], 'RMSSD_mae': []},
                'E4': {'HR': [], 'IBI': [], 'SDNN': [], 'RMSSD': [],
                       'HR_mae': [], 'IBI_mae': [], 'SDNN_mae': [], 'RMSSD_mae': []}
            }
        }

        # Process each session separately
        for session in self.target_sessions:
            session_data = df[df['session'] == session]

            if len(session_data) == 0:
                continue

            # Initialize session results
            results['session_results'][session] = {
                'Galaxy': {'HR': [], 'IBI': [], 'SDNN': [], 'RMSSD': [],
                           'HR_mae': [], 'IBI_mae': [], 'SDNN_mae': [], 'RMSSD_mae': []},
                'E4': {'HR': [], 'IBI': [], 'SDNN': [], 'RMSSD': [],
                       'HR_mae': [], 'IBI_mae': [], 'SDNN_mae': [], 'RMSSD_mae': []}
            }

            # Process each window in the session
            for _, window in session_data.iterrows():
                # Process Galaxy PPG
                try:
                    galaxy_signal = np.array([float(x) for x in window['galaxyPPG'].split(';') if x.strip()])
                    working_data, measures = self.process_ppg_signal(galaxy_signal, self.galaxy_fs)

                    if working_data is not None and measures is not None and len(working_data['peaklist']) >= 2:
                        # Calculate metrics
                        hr = measures['bpm']
                        ibi = measures['ibi']
                        sdnn = measures['sdnn']
                        rmssd = measures['rmssd']

                        # Calculate MAE
                        hr_mae = abs(hr - window['gdHR']) if not np.isnan(window['gdHR']) else np.nan
                        ibi_mae = abs(ibi - window['gdIBI']) if not np.isnan(window['gdIBI']) else np.nan
                        sdnn_mae = abs(sdnn - window['gdSDNN']) if not np.isnan(window['gdSDNN']) else np.nan
                        rmssd_mae = abs(rmssd - window['gdRMSSD']) if not np.isnan(window['gdRMSSD']) else np.nan

                        # Store metrics in session results
                        if not np.isnan(hr):
                            results['session_results'][session]['Galaxy']['HR'].append(hr)
                        if not np.isnan(ibi):
                            results['session_results'][session]['Galaxy']['IBI'].append(ibi)
                        if not np.isnan(sdnn):
                            results['session_results'][session]['Galaxy']['SDNN'].append(sdnn)
                        if not np.isnan(rmssd):
                            results['session_results'][session]['Galaxy']['RMSSD'].append(rmssd)

                        # Store MAE in session results
                        if not np.isnan(hr_mae):
                            results['session_results'][session]['Galaxy']['HR_mae'].append(hr_mae)
                        if not np.isnan(ibi_mae):
                            results['session_results'][session]['Galaxy']['IBI_mae'].append(ibi_mae)
                        if not np.isnan(sdnn_mae):
                            results['session_results'][session]['Galaxy']['SDNN_mae'].append(sdnn_mae)
                        if not np.isnan(rmssd_mae):
                            results['session_results'][session]['Galaxy']['RMSSD_mae'].append(rmssd_mae)

                        # Store metrics in device results
                        if not np.isnan(hr_mae):
                            results['device_results']['Galaxy']['HR_mae'].append(hr_mae)
                        if not np.isnan(ibi_mae):
                            results['device_results']['Galaxy']['IBI_mae'].append(ibi_mae)
                        if not np.isnan(sdnn_mae):
                            results['device_results']['Galaxy']['SDNN_mae'].append(sdnn_mae)
                        if not np.isnan(rmssd_mae):
                            results['device_results']['Galaxy']['RMSSD_mae'].append(rmssd_mae)
                except Exception as e:
                    pass

                # Process E4 BVP
                try:
                    e4_signal = np.array([float(x) for x in window['e4BVP'].split(';') if x.strip()])
                    working_data, measures = self.process_ppg_signal(e4_signal, self.e4_fs)

                    if working_data is not None and measures is not None and len(working_data['peaklist']) >= 2:
                        # Calculate metrics
                        hr = measures['bpm']
                        ibi = measures['ibi']
                        sdnn = measures['sdnn']
                        rmssd = measures['rmssd']

                        # Calculate MAE
                        hr_mae = abs(hr - window['gdHR']) if not np.isnan(window['gdHR']) else np.nan
                        ibi_mae = abs(ibi - window['gdIBI']) if not np.isnan(window['gdIBI']) else np.nan
                        sdnn_mae = abs(sdnn - window['gdSDNN']) if not np.isnan(window['gdSDNN']) else np.nan
                        rmssd_mae = abs(rmssd - window['gdRMSSD']) if not np.isnan(window['gdRMSSD']) else np.nan

                        # Store metrics in session results
                        if not np.isnan(hr):
                            results['session_results'][session]['E4']['HR'].append(hr)
                        if not np.isnan(ibi):
                            results['session_results'][session]['E4']['IBI'].append(ibi)
                        if not np.isnan(sdnn):
                            results['session_results'][session]['E4']['SDNN'].append(sdnn)
                        if not np.isnan(rmssd):
                            results['session_results'][session]['E4']['RMSSD'].append(rmssd)

                        # Store MAE in session results
                        if not np.isnan(hr_mae):
                            results['session_results'][session]['E4']['HR_mae'].append(hr_mae)
                        if not np.isnan(ibi_mae):
                            results['session_results'][session]['E4']['IBI_mae'].append(ibi_mae)
                        if not np.isnan(sdnn_mae):
                            results['session_results'][session]['E4']['SDNN_mae'].append(sdnn_mae)
                        if not np.isnan(rmssd_mae):
                            results['session_results'][session]['E4']['RMSSD_mae'].append(rmssd_mae)

                        # Store metrics in device results
                        if not np.isnan(hr_mae):
                            results['device_results']['E4']['HR_mae'].append(hr_mae)
                        if not np.isnan(ibi_mae):
                            results['device_results']['E4']['IBI_mae'].append(ibi_mae)
                        if not np.isnan(sdnn_mae):
                            results['device_results']['E4']['SDNN_mae'].append(sdnn_mae)
                        if not np.isnan(rmssd_mae):
                            results['device_results']['E4']['RMSSD_mae'].append(rmssd_mae)
                except Exception as e:
                    pass

        return results
    def calculate_aggregated_stats(self, values: List[float]) -> Dict[str, float]:
        """
        Calculate statistics for a list of values

        Args:
            values: List of values to calculate statistics for

        Returns:
            Dictionary of statistics
        """
        if not values:
            return {'mean': np.nan, 'std': np.nan, 'median': np.nan, 'min': np.nan, 'max': np.nan, 'count': 0}

        return {
            'mean': np.mean(values),
            'std': np.std(values) if len(values) > 1 else 0,
            'median': np.median(values),
            'min': np.min(values),
            'max': np.max(values),
            'count': len(values)
        }
    def run_analysis(self):
        """Run the hand placement analysis and generate reports"""
        # Collect results by hand placement
        left_hand_results = []
        right_hand_results = []

        # Process all participants
        for participant_id in self.meta_df['UID'].tolist():
            try:
                print(f"Processing participant: {participant_id}")
                df = self.load_participant_data(participant_id)

                if df.empty:
                    continue

                # Analyze participant data
                results = self.analyze_participant_data(df, participant_id)

                # Add to appropriate group
                if results['hand_placement'] == 'L':
                    left_hand_results.append(results)
                elif results['hand_placement'] == 'R':
                    right_hand_results.append(results)
            except Exception as e:
                print(f"Error processing participant {participant_id}: {str(e)}")
                traceback.print_exc()

        # Generate overall results
        self.generate_overall_report(left_hand_results, right_hand_results)

        # Generate session-specific results
        self.generate_session_report(left_hand_results, right_hand_results)
    def process_ppg_signal(self, signal: np.ndarray, sample_rate: float) -> Tuple[Optional[Dict], Optional[Dict]]:
        """
        Process PPG signal to extract heart rate metrics using HeartPy

        Args:
            signal: PPG signal array
            sample_rate: Sampling frequency in Hz

        Returns:
            Tuple of working_data and measures dictionaries
        """
        try:
            # Apply bandpass filter (same as in PPG_Calculation_basing_HeartPy.py)
            filtered = hp.filter_signal(
                signal,
                cutoff=[0.5, 4.0],
                sample_rate=sample_rate,
                filtertype='bandpass',
                order=3
            )

            # Scale the signal
            scaled = hp.scale_data(filtered)

            # Process with HeartPy
            working_data, measures = hp.process(
                scaled,
                sample_rate=sample_rate,
                bpmmin=40,
                bpmmax=200,
                clean_rr=True,
                clean_rr_method='quotient-filter',
                interp_clipping=True,
                interp_threshold=1020,
            )

            # Post-process peaks using same logic as in PPG_Calculation_basing_HeartPy.py
            if 'peaklist' in working_data and len(working_data['peaklist']) > 0:
                processed_peaks = self.post_process_peaks(
                    scaled,
                    working_data['peaklist'],
                    sample_rate
                )

                if len(processed_peaks) >= 2:
                    # Update peak list and recalculate measures
                    working_data['peaklist'] = processed_peaks
                    working_data['ybeat'] = scaled[processed_peaks]

                    intervals = np.diff(processed_peaks) / sample_rate * 1000  # in ms

                    # Recalculate metrics
                    measures = {
                        'bpm': 60000 / np.mean(intervals),
                        'ibi': np.mean(intervals),
                        'sdnn': np.std(intervals, ddof=1),
                        'rmssd': np.sqrt(np.mean(np.diff(intervals) ** 2))
                    }

            return working_data, measures

        except Exception as e:
            return None, None
    def post_process_peaks(self, signal: np.ndarray, peaks: np.ndarray, sample_rate: float) -> np.ndarray:
        """
        Enhanced peak detection post-processing
        (directly from PPG_Calculation_basing_HeartPy.py)

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
    def generate_overall_report(self, left_hand_results: List[Dict], right_hand_results: List[Dict]):
        """
        Generate overall report comparing left and right hand placement

        Args:
            left_hand_results: Results for participants with left hand placement
            right_hand_results: Results for participants with right hand placement
        """
        # Combine all MAE values for each device and metric
        devices = ['Galaxy', 'E4']
        metrics = ['HR_mae', 'IBI_mae', 'SDNN_mae', 'RMSSD_mae']

        # Organize data for reporting
        report_data = []

        for device in devices:
            for metric in metrics:
                # Collect all values for left hand
                left_values = []
                for result in left_hand_results:
                    left_values.extend(result['device_results'][device][metric])

                # Collect all values for right hand
                right_values = []
                for result in right_hand_results:
                    right_values.extend(result['device_results'][device][metric])

                # Calculate statistics
                left_stats = self.calculate_aggregated_stats(left_values)
                right_stats = self.calculate_aggregated_stats(right_values)

                # Calculate difference and significance
                mean_diff = left_stats['mean'] - right_stats['mean'] if not np.isnan(
                    left_stats['mean']) and not np.isnan(right_stats['mean']) else np.nan
                percent_diff = (mean_diff / right_stats['mean'] * 100) if not np.isnan(mean_diff) and right_stats[
                    'mean'] != 0 else np.nan
                better_hand = 'Right' if not np.isnan(mean_diff) and mean_diff > 0 else 'Left' if not np.isnan(
                    mean_diff) and mean_diff < 0 else 'Equal'

                # Add to report data
                report_data.append({
                    'Device': device,
                    'Metric': metric,
                    'Left_Mean': left_stats['mean'],
                    'Left_StdDev': left_stats['std'],
                    'Left_Count': left_stats['count'],
                    'Right_Mean': right_stats['mean'],
                    'Right_StdDev': right_stats['std'],
                    'Right_Count': right_stats['count'],
                    'Mean_Difference': mean_diff,
                    'Percent_Difference': percent_diff,
                    'Better_Hand': better_hand
                })

        # Create DataFrame and save to CSV
        report_df = pd.DataFrame(report_data)
        report_df.to_csv(os.path.join(self.output_dir, 'hand_placement_overall_report.csv'), index=False)

        # Print summary
        print("\nHand Placement Overall Report:")
        print(report_df[['Device', 'Metric', 'Left_Mean', 'Right_Mean', 'Better_Hand']])
    def generate_session_report(self, left_hand_results: List[Dict], right_hand_results: List[Dict]):
        """
        Generate session-specific report comparing left and right hand placement

        Args:
            left_hand_results: Results for participants with left hand placement
            right_hand_results: Results for participants with right hand placement
        """
        # Organize data by session
        devices = ['Galaxy', 'E4']
        metrics = ['HR_mae', 'IBI_mae', 'SDNN_mae', 'RMSSD_mae']

        # Initialize data structure
        session_data = defaultdict(list)

        # Process each session
        for session in self.target_sessions:
            for device in devices:
                for metric in metrics:
                    # Collect values for left hand
                    left_values = []
                    for result in left_hand_results:
                        if session in result['session_results']:
                            left_values.extend(result['session_results'][session][device][metric])

                    # Collect values for right hand
                    right_values = []
                    for result in right_hand_results:
                        if session in result['session_results']:
                            right_values.extend(result['session_results'][session][device][metric])

                    # Calculate statistics
                    left_stats = self.calculate_aggregated_stats(left_values)
                    right_stats = self.calculate_aggregated_stats(right_values)

                    # Calculate difference and significance
                    mean_diff = left_stats['mean'] - right_stats['mean'] if not np.isnan(
                        left_stats['mean']) and not np.isnan(right_stats['mean']) else np.nan
                    percent_diff = (mean_diff / right_stats['mean'] * 100) if not np.isnan(mean_diff) and right_stats[
                        'mean'] != 0 else np.nan
                    better_hand = 'Right' if not np.isnan(mean_diff) and mean_diff > 0 else 'Left' if not np.isnan(
                        mean_diff) and mean_diff < 0 else 'Equal'

                    # Add to session data
                    session_data[session].append({
                        'Session': session,
                        'Device': device,
                        'Metric': metric,
                        'Left_Mean': left_stats['mean'],
                        'Left_StdDev': left_stats['std'],
                        'Left_Count': left_stats['count'],
                        'Right_Mean': right_stats['mean'],
                        'Right_StdDev': right_stats['std'],
                        'Right_Count': right_stats['count'],
                        'Mean_Difference': mean_diff,
                        'Percent_Difference': percent_diff,
                        'Better_Hand': better_hand
                    })

        # Combine all session data
        all_session_data = []
        for session, data in session_data.items():
            all_session_data.extend(data)

        # Create DataFrame and save to CSV
        session_df = pd.DataFrame(all_session_data)
        session_df.to_csv(os.path.join(self.output_dir, 'hand_placement_session_report.csv'), index=False)

        # Create pivot table for easier comparison
        pivot_data = []
        for session in self.target_sessions:
            if session in session_data:
                for row in session_data[session]:
                    if row['Device'] == 'Galaxy' and row['Metric'] == 'HR_mae':  # Just use HR_mae for summary
                        pivot_data.append({
                            'Session': session,
                            'Left_Hand_Galaxy_HR_MAE': row['Left_Mean'],
                            'Right_Hand_Galaxy_HR_MAE': row['Right_Mean'],
                            'Better_Hand': row['Better_Hand'],
                            'Percent_Difference': row['Percent_Difference']
                        })
                    if row['Device'] == 'E4' and row['Metric'] == 'HR_mae':
                        # Update existing row
                        for i, existing in enumerate(pivot_data):
                            if existing['Session'] == session:
                                pivot_data[i]['Left_Hand_E4_HR_MAE'] = row['Left_Mean']
                                pivot_data[i]['Right_Hand_E4_HR_MAE'] = row['Right_Mean']
                                pivot_data[i]['E4_Better_Hand'] = row['Better_Hand']
                                pivot_data[i]['E4_Percent_Difference'] = row['Percent_Difference']
                                break

        # Create pivot DataFrame and save
        pivot_df = pd.DataFrame(pivot_data)
        pivot_df.to_csv(os.path.join(self.output_dir, 'hand_placement_session_summary.csv'), index=False)

        # Print summary
        print("\nHand Placement Session Summary:")
        print(pivot_df[['Session', 'Left_Hand_Galaxy_HR_MAE', 'Right_Hand_Galaxy_HR_MAE', 'Better_Hand']])

def main():
    analyzer = HandPlacementMAEAnalyzer()
    analyzer.run_analysis()

if __name__ == "__main__":
    main()