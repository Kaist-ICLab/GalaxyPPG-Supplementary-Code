import os
import pandas as pd
import numpy as np
import heartpy as hp
from heartpy.filtering import remove_baseline_wander, filter_signal
from heartpy.preprocessing import enhance_ecg_peaks
from typing import Tuple, Optional

class ECGAnalyzer:
    def __init__(self):
        """
        Initialize the ECG analyzer with processing parameters
        """
        self.ecg_fs = 130  # ECG sampling frequency in Hz
        self.HR_MIN = 40  # Minimum heart rate in bpm
        self.HR_MAX = 200  # Maximum heart rate in bpm

    def process_ecg_signal(self, ecg_signal: np.ndarray, low_cutoff: float,
                           high_cutoff: float) -> Tuple[dict, dict, float]:
        """
        Process ECG signal and calculate heart rate metrics

        Args:
            ecg_signal: Raw ECG signal array
            low_cutoff: Lower frequency cutoff for bandpass filter
            high_cutoff: Upper frequency cutoff for bandpass filter

        Returns:
            Tuple containing working data, measures, and current heart rate
        """
        # Remove baseline wander
        ecg_filtered = remove_baseline_wander(ecg_signal, self.ecg_fs)

        # Apply bandpass filter
        filtered_signal = filter_signal(
            ecg_filtered,
            cutoff=[low_cutoff, high_cutoff],
            sample_rate=self.ecg_fs,
            filtertype='bandpass',
            order=3
        )

        # Enhance peaks
        filtered_signal = enhance_ecg_peaks(
            filtered_signal,
            self.ecg_fs,
            iterations=1
        )

        # Normalize signal
        filtered_signal = (filtered_signal - np.mean(filtered_signal)) / np.std(filtered_signal)

        # Process with HeartPy
        working_data, measures = hp.process(
            filtered_signal,
            sample_rate=self.ecg_fs,
            bpmmin=self.HR_MIN,
            bpmmax=self.HR_MAX,
            clean_rr=True
        )

        # Post-process peaks if they exist
        if 'peaklist' in working_data:
            processed_peaks = self.post_process_peaks(
                filtered_signal,
                working_data['peaklist'],
                self.ecg_fs
            )

            if len(processed_peaks) >= 2:
                working_data['peaklist'] = processed_peaks
                working_data['ybeat'] = filtered_signal[processed_peaks]

                # Calculate RR intervals in ms
                rr_intervals = np.diff(processed_peaks) / self.ecg_fs * 1000

                # Calculate metrics
                current_hr = 60000 / np.mean(rr_intervals)

                # Update measures
                measures = {
                    'bpm': current_hr,
                    'ibi': np.mean(rr_intervals),
                    'sdnn': np.std(rr_intervals, ddof=1) if len(rr_intervals) > 1 else np.nan,
                    'rmssd': np.sqrt(np.mean(np.diff(rr_intervals) ** 2)) if len(rr_intervals) > 2 else np.nan,
                }

                return working_data, measures, current_hr

        # Return defaults if processing failed
        return working_data, {"bpm": 0, "ibi": np.nan, "sdnn": np.nan, "rmssd": np.nan}, 0

    def process_dataframe(self, df: pd.DataFrame, filter_params: pd.DataFrame) -> pd.DataFrame:
        """
        Process ECG data from DataFrame

        Args:
            df: DataFrame containing ECG data
            filter_params: DataFrame containing filter parameters

        Returns:
            DataFrame with added ECG analysis results
        """
        # Add result columns
        result_columns = ['gdHR', 'gdIBI', 'gdSDNN', 'gdRMSSD', 'gdPeaks']
        for col in result_columns:
            df[col] = None

        total_windows = 0
        failed_windows = 0

        for idx, row in df.iterrows():
            if pd.notna(row['polarECG']) and pd.notna(row['polarHR']):
                total_windows += 1
                try:
                    # Get ECG signal
                    ecg_signal = np.array([float(x) for x in row['polarECG'].split(';')])

                    # Get filter parameters
                    params = filter_params[
                        (filter_params['windowNumber'] == row['windowNumber']) &
                        (filter_params['session'] == row['session'])
                        ].iloc[0]

                    # Process signal
                    working_data, measures, current_hr = self.process_ecg_signal(
                        ecg_signal,
                        params['lowCutoff'],
                        params['highCutoff']
                    )

                    if 'peaklist' in working_data and len(working_data['peaklist']) >= 2:
                        # Update metrics
                        df.at[idx, 'gdHR'] = current_hr
                        df.at[idx, 'gdIBI'] = measures['ibi']
                        df.at[idx, 'gdSDNN'] = measures['sdnn']
                        df.at[idx, 'gdRMSSD'] = measures['rmssd']
                        df.at[idx, 'gdPeaks'] = ';'.join(map(str, working_data['peaklist']))
                    else:
                        failed_windows += 1

                except Exception as e:
                    print(f"Error processing window {row['windowNumber']}: {str(e)}")
                    failed_windows += 1

        # Calculate success rate
        success_rate = ((total_windows - failed_windows) / total_windows * 100) if total_windows > 0 else 0
        processing_stats = {
            'success_rate': success_rate,
            'total_windows': total_windows,
            'failed_windows': failed_windows
        }

        return df, processing_stats


    def post_process_peaks(self, signal: np.ndarray, peaks: np.ndarray, sample_rate: float) -> np.ndarray:
        """
        Enhanced ECG R-peak detection with exact same logic as WINDOW_ECG
        """
        if len(peaks) < 2:
            return peaks

        def find_peak_in_window(signal_window: np.ndarray, start_idx: int) -> Optional[int]:
            """Find the most likely R-peak in a given window"""
            local_mean = np.mean(signal_window)
            local_std = np.std(signal_window)
            local_threshold = local_mean + local_std

            potential_peaks = []
            for i in range(1, len(signal_window) - 1):
                if (signal_window[i] > signal_window[i - 1] and
                        signal_window[i] > signal_window[i + 1] and
                        signal_window[i] > local_threshold):
                    potential_peaks.append((start_idx + i, signal_window[i]))

            if potential_peaks:
                return max(potential_peaks, key=lambda x: x[1])[0]
            return None

        def get_adaptive_threshold(signal_segment: np.ndarray) -> float:
            """Calculate adaptive threshold based on signal segment"""
            segment_mean = np.mean(signal_segment)
            segment_std = np.std(signal_segment)
            return segment_mean + 1.5 * segment_std

        # First pass: Remove too close peaks
        min_distance = int(0.2 * sample_rate)
        processed_peaks = []

        for peak in peaks:
            if not processed_peaks or (peak - processed_peaks[-1]) >= min_distance:
                processed_peaks.append(peak)

        if len(processed_peaks) < 2:
            return np.array(processed_peaks)

        # Calculate initial RR interval statistics
        rr_intervals = np.diff(processed_peaks)
        mean_rr = np.mean(rr_intervals)

        # Second pass: Check for missing peaks using RR prediction
        final_peaks = [processed_peaks[0]]
        search_window_size = int(0.2 * sample_rate)

        for i in range(1, len(processed_peaks)):
            current_peak = processed_peaks[i]
            last_peak = final_peaks[-1]
            current_rr = current_peak - last_peak

            if current_rr > 1.5 * mean_rr:
                n_missing = int(current_rr / mean_rr) - 1

                for j in range(1, n_missing + 1):
                    expected_position = last_peak + int(j * mean_rr)
                    window_start = max(0, expected_position - search_window_size)
                    window_end = min(len(signal), expected_position + search_window_size)
                    search_window = signal[window_start:window_end]
                    potential_peak = find_peak_in_window(search_window, window_start)

                    if potential_peak is not None:
                        if ((potential_peak - final_peaks[-1]) >= min_distance and
                                (current_peak - potential_peak) >= min_distance):
                            local_window = signal[max(0, potential_peak - 20):
                                                  min(len(signal), potential_peak + 21)]
                            if signal[potential_peak] > get_adaptive_threshold(local_window):
                                final_peaks.append(potential_peak)

            if (current_peak - final_peaks[-1]) >= min_distance:
                final_peaks.append(current_peak)
                if len(final_peaks) > 2:
                    recent_rr = np.diff(final_peaks[-3:])
                    mean_rr = np.mean(recent_rr)

        # Final validation pass
        validated_peaks = []
        for peak in final_peaks:
            window_start = max(0, peak - search_window_size)
            window_end = min(len(signal), peak + search_window_size)
            local_window = signal[window_start:window_end]

            if signal[peak] > get_adaptive_threshold(local_window):
                if not validated_peaks or (peak - validated_peaks[-1]) >= min_distance:
                    validated_peaks.append(peak)

        return np.array(validated_peaks, dtype=np.int64)
def load_participant_data(window_data_dir: str, participant_id: str) -> pd.DataFrame:
    """
    Load participant data from files

    Args:
        window_data_dir: Directory containing window data files
        participant_id: ID of participant to load

    Returns:
        DataFrame containing participant data
    """
    try:
        input_file = os.path.join(window_data_dir, f"{participant_id}_processed.csv")
        if not os.path.exists(input_file):
            print(f"File not found: {input_file}")
            return pd.DataFrame()
        df = pd.read_csv(input_file)
        df = df.rename(columns={'trueHR': 'polarHR'})
        return df

    except Exception as e:
        print(f"Error loading data for participant {participant_id}: {str(e)}")
        return pd.DataFrame()
def load_filter_parameters(param_dir: str, participant_id: str) -> pd.DataFrame:
    """
    Load filter parameters from file

    Args:
        param_dir: Directory containing parameter files
        participant_id: ID of participant to load parameters for

    Returns:
        DataFrame containing filter parameters
    """
    param_file = os.path.join(param_dir, f"{participant_id}_filter_params.csv")
    if not os.path.exists(param_file):
        raise FileNotFoundError(f"Parameter file not found for participant {participant_id}")
    return pd.read_csv(param_file)
def main():
    from config import BASE_DIR

    window_data_dir = os.path.join(BASE_DIR, 'WindowData')
    param_dir = os.path.join('filter_parameters')

    # Get list of participants
    participants = [f.split('_')[0] for f in os.listdir(window_data_dir)
                    if f.endswith('_processed.csv')]

    analyzer = ECGAnalyzer()

    for participant_id in participants:
        print(f"\nProcessing participant {participant_id}...")

        try:
            # Load data
            df = load_participant_data(window_data_dir, participant_id)
            if df.empty:
                continue

            # Load filter parameters
            filter_params = load_filter_parameters(param_dir, participant_id)

            # Process data
            processed_df, stats = analyzer.process_dataframe(df, filter_params)

            # Save results
            output_file = os.path.join(window_data_dir, f"{participant_id}_processed_GD.csv")
            processed_df.to_csv(output_file, index=False)

        except Exception as e:
            print(f"Error processing participant {participant_id}: {str(e)}")

if __name__ == '__main__':
    main()

