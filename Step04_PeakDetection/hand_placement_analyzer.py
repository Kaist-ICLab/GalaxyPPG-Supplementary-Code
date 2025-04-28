import os
import pandas as pd
import numpy as np
import heartpy as hp
from typing import Dict, List, Tuple, Optional
import traceback
import sys
from scipy.stats import ttest_rel, wilcoxon
from collections import defaultdict

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import WINDOW_DIR, RESULTS_DIR, RAW_DIR


class HandPlacementMAEAnalyzer:
    """
    Analyzer for assessing the impact of hand position (left vs right)
    on heart rate metric accuracy compared to ECG ground truth.
    This version compares left vs right placement within each participant,
    regardless of which device is worn on which hand.
    """
    def __init__(self):
        """Initialize the analyzer with the required parameters."""
        # Load metadata file containing hand placement information
        meta_path = os.path.join(RAW_DIR, 'Meta.csv')
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

        print(f"Left hand Galaxy users: {len(self.left_hand_users)} participants")
        print(f"Right hand Galaxy users: {len(self.right_hand_users)} participants")

        # Initialize metrics to analyze
        self.metrics = ['HR', 'IBI', 'SDNN', 'RMSSD']


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
            df['GalaxyHandPlacement'] = hand_placement

            return df
        except Exception as e:
            print(f"Error loading data for {participant_id}: {str(e)}")
            return pd.DataFrame()


    def analyze_participant_data(self, df: pd.DataFrame, participant_id: str) -> Dict[str, Dict]:
        """
        Analyze data for a single participant, comparing left vs right hand MAE

        Args:
            df: DataFrame containing participant data
            participant_id: Participant ID

        Returns:
            Dictionary with analysis results for left and right hand
        """
        # Initialize result structure
        results = {
            'participant': participant_id,
            'galaxy_hand': df['GalaxyHandPlacement'].iloc[0] if not df.empty else 'Unknown',
            'session_results': {},
            'left_hand_maes': {metric: [] for metric in self.metrics},
            'right_hand_maes': {metric: [] for metric in self.metrics}
        }

        # Process each session separately
        for session in self.target_sessions:
            session_data = df[df['session'] == session]

            if len(session_data) == 0:
                continue

            # Initialize session results
            results['session_results'][session] = {
                'left_hand_maes': {metric: [] for metric in self.metrics},
                'right_hand_maes': {metric: [] for metric in self.metrics}
            }

            # Process each window in the session
            for _, window in session_data.iterrows():
                galaxy_hand = window['GalaxyHandPlacement']

                # Process Galaxy PPG
                try:
                    galaxy_signal = np.array([float(x) for x in window['galaxyPPG'].split(';') if x.strip()])
                    galaxy_working_data, galaxy_measures = self.process_ppg_signal(galaxy_signal, self.galaxy_fs)

                    # Process E4 BVP
                    e4_signal = np.array([float(x) for x in window['e4BVP'].split(';') if x.strip()])
                    e4_working_data, e4_measures = self.process_ppg_signal(e4_signal, self.e4_fs)

                    # Only proceed if both devices have valid measures
                    if (galaxy_working_data is not None and galaxy_measures is not None and
                            e4_working_data is not None and e4_measures is not None and
                            len(galaxy_working_data['peaklist']) >= 2 and
                            len(e4_working_data['peaklist']) >= 2):

                        # Calculate MAEs for each metric for both devices
                        galaxy_maes = {}
                        e4_maes = {}

                        for metric in self.metrics:
                            # Map metric name to HeartPy keys
                            hp_key = metric.lower() if metric != 'HR' else 'bpm'
                            gd_key = f"gd{metric}"

                            if hp_key in galaxy_measures and hp_key in e4_measures and gd_key in window:
                                # Calculate MAE
                                galaxy_mae = abs(galaxy_measures[hp_key] - window[gd_key]) if not np.isnan(
                                    window[gd_key]) else np.nan
                                e4_mae = abs(e4_measures[hp_key] - window[gd_key]) if not np.isnan(
                                    window[gd_key]) else np.nan

                                # Store MAEs
                                galaxy_maes[metric] = galaxy_mae
                                e4_maes[metric] = e4_mae

                        # Assign MAEs to left or right hand based on GalaxyWatch placement
                        left_maes = galaxy_maes if galaxy_hand == 'L' else e4_maes
                        right_maes = e4_maes if galaxy_hand == 'L' else galaxy_maes

                        # Store MAEs in results
                        for metric in self.metrics:
                            if metric in left_maes and not np.isnan(left_maes[metric]):
                                results['session_results'][session]['left_hand_maes'][metric].append(left_maes[metric])
                                results['left_hand_maes'][metric].append(left_maes[metric])

                            if metric in right_maes and not np.isnan(right_maes[metric]):
                                results['session_results'][session]['right_hand_maes'][metric].append(right_maes[metric])
                                results['right_hand_maes'][metric].append(right_maes[metric])

                except Exception as e:
                    # Silent exception - we'll just skip this window
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
        # Store participant results
        all_participant_results = []

        # Process all participants
        for participant_id in self.meta_df['UID'].tolist():
            try:
                print(f"Processing participant: {participant_id}")
                df = self.load_participant_data(participant_id)

                if df.empty:
                    continue

                # Analyze participant data
                results = self.analyze_participant_data(df, participant_id)
                all_participant_results.append(results)

            except Exception as e:
                print(f"Error processing participant {participant_id}: {str(e)}")
                traceback.print_exc()

        # Generate reports
        self.generate_overall_report(all_participant_results)
        self.generate_session_report(all_participant_results)
        self.generate_handedness_summary(all_participant_results)


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


    def generate_overall_report(self, all_participant_results: List[Dict]):
        """
        Generate an overall report comparing left vs. right hand placement
        across all participants and metrics.
        """
        # Initialize results for each metric
        metric_results = {metric: {'left': [], 'right': []} for metric in self.metrics}

        # Collect all MAE values across participants
        for result in all_participant_results:
            for metric in self.metrics:
                metric_results[metric]['left'].extend(result['left_hand_maes'][metric])
                metric_results[metric]['right'].extend(result['right_hand_maes'][metric])

        # Build report rows
        report_rows = []
        for metric in self.metrics:
            left_vals = metric_results[metric]['left']
            right_vals = metric_results[metric]['right']

            # Calculate statistics
            left_stats = self.calculate_aggregated_stats(left_vals)
            right_stats = self.calculate_aggregated_stats(right_vals)

            # Compare left vs right
            diff = left_stats['mean'] - right_stats['mean'] if not np.isnan(left_stats['mean']) and not np.isnan(
                right_stats['mean']) else np.nan
            better_hand = 'Right' if diff > 0 else 'Left' if diff < 0 else 'Equal' if diff == 0 else 'Unknown'

            # Add to report
            report_rows.append({
                'Metric': metric,
                'Left_Mean_MAE': left_stats['mean'],
                'Left_StdDev': left_stats['std'],
                'Left_Count': left_stats['count'],
                'Right_Mean_MAE': right_stats['mean'],
                'Right_StdDev': right_stats['std'],
                'Right_Count': right_stats['count'],
                'Difference': diff,
                'Better_Hand': better_hand
            })

        # Create and save report
        report_df = pd.DataFrame(report_rows)
        report_df.to_csv(os.path.join(self.output_dir, 'overall_hand_placement_comparison.csv'), index=False)
        print(f"Overall report saved to {os.path.join(self.output_dir, 'overall_hand_placement_comparison.csv')}")


    def generate_session_report(self, all_participant_results: List[Dict]):
        """
        Generate a session-specific report comparing left vs. right hand placement
        across all metrics.
        """
        # Initialize structure to hold session-specific MAEs
        session_maes = {session: {metric: {'left': [], 'right': []} for metric in self.metrics}
                        for session in self.target_sessions}

        # Collect MAEs by session
        for result in all_participant_results:
            for session, session_data in result['session_results'].items():
                for metric in self.metrics:
                    session_maes[session][metric]['left'].extend(session_data['left_hand_maes'][metric])
                    session_maes[session][metric]['right'].extend(session_data['right_hand_maes'][metric])

        # Build report rows
        report_rows = []
        for session in self.target_sessions:
            for metric in self.metrics:
                left_vals = session_maes[session][metric]['left']
                right_vals = session_maes[session][metric]['right']

                # Calculate statistics
                left_stats = self.calculate_aggregated_stats(left_vals)
                right_stats = self.calculate_aggregated_stats(right_vals)

                # Compare left vs right
                diff = left_stats['mean'] - right_stats['mean'] if not np.isnan(left_stats['mean']) and not np.isnan(
                    right_stats['mean']) else np.nan
                better_hand = 'Right' if diff > 0 else 'Left' if diff < 0 else 'Equal' if diff == 0 else 'Unknown'

                # Calculate statistical significance if possible
                p_value = np.nan
                if left_stats['count'] > 1 and right_stats['count'] > 1:
                    try:
                        _, p_value = wilcoxon(left_vals, right_vals, alternative='two-sided')
                    except:
                        p_value = np.nan

                # Add to report
                report_rows.append({
                    'Session': session,
                    'Metric': metric,
                    'Left_Mean_MAE': left_stats['mean'],
                    'Left_StdDev': left_stats['std'],
                    'Left_Count': left_stats['count'],
                    'Right_Mean_MAE': right_stats['mean'],
                    'Right_StdDev': right_stats['std'],
                    'Right_Count': right_stats['count'],
                    'Difference': diff,
                    'Better_Hand': better_hand,
                    'p_value': p_value,
                    'Significant': p_value < 0.05 if not np.isnan(p_value) else False
                })

        # Create and save report
        report_df = pd.DataFrame(report_rows)
        report_df.to_csv(os.path.join(self.output_dir, 'session_hand_placement_comparison.csv'), index=False)
        print(f"Session report saved to {os.path.join(self.output_dir, 'session_hand_placement_comparison.csv')}")


    def generate_handedness_summary(self, all_participant_results: List[Dict]):
        """
        Generate participant-level summary comparing left and right hand placement
        for HR MAE with paired statistical testing.

        This analysis is more statistically sound because it uses paired comparisons
        within each participant, controlling for individual differences.
        """
        # Collect participant-level means for HR MAE
        participant_means = []

        for result in all_participant_results:
            participant_id = result['participant']

            # Calculate mean HR MAE for left and right hand
            left_hr_maes = result['left_hand_maes']['HR']
            right_hr_maes = result['right_hand_maes']['HR']

            if len(left_hr_maes) > 0 and len(right_hr_maes) > 0:
                left_mean = np.mean(left_hr_maes)
                right_mean = np.mean(right_hr_maes)

                participant_means.append({
                    'Participant': participant_id,
                    'GalaxyHandPlacement': result['galaxy_hand'],
                    'Left_Mean_HR_MAE': left_mean,
                    'Right_Mean_HR_MAE': right_mean,
                    'Left_Count': len(left_hr_maes),
                    'Right_Count': len(right_hr_maes),
                    'Difference': left_mean - right_mean,
                    'Better_Hand': 'Right' if left_mean > right_mean else 'Left' if right_mean > left_mean else 'Equal'
                })

        # Create and save participant means
        means_df = pd.DataFrame(participant_means)
        means_df.to_csv(os.path.join(self.output_dir, 'participant_hand_placement_means.csv'), index=False)

        # Perform paired t-test on participant means
        if len(participant_means) > 1:
            left_means = [p['Left_Mean_HR_MAE'] for p in participant_means]
            right_means = [p['Right_Mean_HR_MAE'] for p in participant_means]

            t_stat, p_value = ttest_rel(left_means, right_means)

            print(f"\nPaired t-test results for Left vs Right hand placement (HR MAE):")
            print(f"Number of participants with valid data: {len(participant_means)}")
            print(f"Left hand mean HR MAE: {np.mean(left_means):.2f} ± {np.std(left_means):.2f}")
            print(f"Right hand mean HR MAE: {np.mean(right_means):.2f} ± {np.std(right_means):.2f}")
            print(f"t-statistic: {t_stat:.4f}")
            print(f"p-value: {p_value:.4f}")
            print(f"Conclusion: {'Significant difference (p<0.05)' if p_value < 0.05 else 'No significant difference'}")

            # Save summary statistics
            summary = {
                'Analysis': 'Paired t-test for Left vs Right hand placement (HR MAE)',
                'Number_of_Participants': len(participant_means),
                'Left_Hand_Mean_HR_MAE': np.mean(left_means),
                'Left_Hand_StdDev': np.std(left_means),
                'Right_Hand_Mean_HR_MAE': np.mean(right_means),
                'Right_Hand_StdDev': np.std(right_means),
                't_statistic': t_stat,
                'p_value': p_value,
                'Significant': p_value < 0.05
            }

            pd.DataFrame([summary]).to_csv(os.path.join(self.output_dir, 'hand_placement_paired_ttest.csv'), index=False)
            print(f"Paired t-test results saved to {os.path.join(self.output_dir, 'hand_placement_paired_ttest.csv')}")


def main():
    analyzer = HandPlacementMAEAnalyzer()
    analyzer.run_analysis()
if __name__ == "__main__":
    main()