import os
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from abc import ABC, abstractmethod
from config import BASE_DIR, RAW_DIR

class OverallSamplingRateAnalyzer:
    """Main analyzer class that coordinates device-specific analyzers"""

    def __init__(self):
        self.analyzers = {
            'galaxy': GalaxyWatchAnalyzer(),
            'e4': E4Analyzer(),
            'polar': PolarH10Analyzer()
        }
        self.output_path = os.path.join(BASE_DIR, 'SamplingRateAnalysis')
        os.makedirs(self.output_path, exist_ok=True)
        self.all_results = []  # Store all results for global statistics
        self.all_sampling_periods = {}

    def calculate_global_sampling_statistics(self) -> Dict:
        """
        Calculate global sampling statistics by dividing total samples by total time.
        The function now includes additional checks to ensure that session and device data
        are valid before accessing their values.
        """
        global_stats = {}

        # Track total samples and total time for each device-signal pair
        device_signal_totals = {}

        # Iterate through all participants and sessions
        for result in self.all_results:
            # Get the sessions dictionary safely (skip if missing or None)
            sessions = result.get('sessions', {})
            if sessions is None:
                continue
            for session_name, session_data in sessions.items():
                # Ensure that session_data is a dictionary
                if not isinstance(session_data, dict):
                    continue
                for device_name, device_data in session_data.items():
                    # Skip if device_data is None or not a dictionary
                    if not device_data or not isinstance(device_data, dict):
                        continue

                    # Use .get to safely access 'duration' (set default to 0 if missing)
                    session_duration = device_data.get('duration', 0)
                    # If duration is zero (or invalid), skip processing for this device in the session
                    if not session_duration:
                        continue

                    # Safely get the signals dictionary; skip if missing or None
                    signals = device_data.get('signals', {})
                    if signals is None:
                        continue

                    for signal_type, signal_stats in signals.items():
                        # Skip if signal_stats is None or missing expected keys
                        if not signal_stats or 'actual_samples' not in signal_stats:
                            continue

                        key = f"{device_name}_{signal_type}"

                        # Initialize totals for this device-signal pair if not already present
                        if key not in device_signal_totals:
                            device_signal_totals[key] = {
                                'total_samples': 0,
                                'total_time': 0,
                                'expected_rate': signal_stats.get('expected_rate', 0)
                            }

                        # Add the session's sample counts and duration to the running totals
                        device_signal_totals[key]['total_samples'] += signal_stats['actual_samples']
                        device_signal_totals[key]['total_time'] += session_duration

        # Calculate overall sampling rates and coverage ratios for each device-signal combination
        for key, totals in device_signal_totals.items():
            if totals['total_time'] > 0:
                actual_rate = totals['total_samples'] / totals['total_time']
                device, signal = key.split('_', 1)
                global_stats[key] = {
                    'mean_rate': actual_rate,
                    'total_samples': totals['total_samples'],
                    'total_time_seconds': totals['total_time'],
                    'expected_rate': totals['expected_rate'],
                    'coverage_ratio': actual_rate / totals['expected_rate'] if totals['expected_rate'] > 0 else 0
                }
            else:
                global_stats[key] = {
                    'error': 'No valid time data'
                }

        return global_stats

    def process_participant(self, participant: str) -> Dict:

        print(f"\nProcessing participant: {participant}")

        results = {
            'participant': participant,
            'missing_files': [],
            'column_errors': [],
            'sessions': {}
        }

        # Load data for each device
        data = {}
        for device_name, analyzer in self.analyzers.items():
            device_path = os.path.join(RAW_DIR, participant,
                                       'GalaxyWatch' if device_name == 'galaxy' else
                                       'E4' if device_name == 'e4' else 'PolarH10')

            for signal_type, config in analyzer.get_signal_configs().items():
                file_path = os.path.join(device_path, config['file'])
                key = f"{device_name}_{signal_type}"

                if os.path.exists(file_path):
                    try:
                        df = pd.read_csv(file_path)
                        if analyzer.validate_columns(df, config['columns'], config['file']):
                            if device_name == 'polar' and 'phoneTimestamp' in df.columns:
                                df['timestamp'] = analyzer.adjust_timestamp(df['phoneTimestamp'])
                            else:
                                df['timestamp'] = analyzer.adjust_timestamp(df['timestamp'])

                            # Sort by timestamp to ensure correct period calculation
                            df = df.sort_values('timestamp')

                            # Calculate sampling periods for this file and add to global collection
                            device_signal_key = f"{device_name}_{signal_type}"
                            if len(df) > 1:
                                periods = np.diff(df['timestamp'].values) / 1000.0  # Convert to seconds
                                # Initialize dictionary entry if it doesn't exist
                                if device_signal_key not in self.all_sampling_periods:
                                    self.all_sampling_periods[device_signal_key] = []
                                # Append periods from this file
                                self.all_sampling_periods[device_signal_key].extend(periods)

                            data[key] = df
                        else:
                            results['column_errors'].append(f"{key} (Invalid columns)")
                    except Exception as e:
                        results['missing_files'].append(f"{key} (Read Error: {str(e)})")
                else:
                    results['missing_files'].append(key)

        # Process sessions
        events_df = pd.read_csv(os.path.join(RAW_DIR, participant, 'event.csv'))

        for session in self.analyzers['galaxy'].valid_sessions:  # Use any analyzer's valid_sessions
            session_events = events_df[events_df['session'] == session]

            if len(session_events) >= 2:
                enter_event = session_events[session_events['status'].str.upper() == 'ENTER'].iloc[0]
                exit_event = session_events[session_events['status'].str.upper() == 'EXIT'].iloc[0]

                results['sessions'][session] = {}
                for device_name, analyzer in self.analyzers.items():
                    results['sessions'][session][device_name] = analyzer.analyze_session_sampling(
                        data,
                        session,
                        enter_event['timestamp'],
                        exit_event['timestamp']
                    )

        return results


    def generate_reports(self, all_results: List[Dict]):
        """Generate comprehensive reports for all devices"""
        # Create DataFrames for reports
        availability_data = []
        sampling_data = []

        for result in all_results:
            # Process data availability issues
            for issue in result['missing_files'] + result['column_errors']:
                availability_data.append({
                    'Participant': result['participant'],
                    'Issue_Type': 'Missing File' if issue in result['missing_files'] else 'Column Error',
                    'Detail': issue
                })

            # Process sampling rate data
            for session, session_data in result['sessions'].items():
                for device, device_data in session_data.items():
                    # Skip if device_data is None
                    if device_data is None:
                        continue

                    # Skip if signals key is missing
                    if 'signals' not in device_data:
                        continue

                    for signal_type, signal_stats in device_data['signals'].items():
                        # Skip if signal_stats is None or has error
                        if signal_stats is None or 'error' in signal_stats:
                            continue

                        record = {
                            'Participant': result['participant'],
                            'Session': session,
                            'Device': device,
                            'Signal': signal_type,
                            'Duration_Seconds': device_data['duration'],
                            'Expected_Rate': signal_stats['expected_rate'],
                            'Actual_Rate': signal_stats['actual_rate'],
                            'Expected_Samples': signal_stats['expected_samples'],
                            'Actual_Samples': signal_stats['actual_samples'],
                            'Coverage_Ratio': signal_stats['coverage_ratio']
                        }

                        if 'axis_samples' in signal_stats:
                            for axis, count in signal_stats['axis_samples'].items():
                                record[f'{axis}_samples'] = count
                        elif 'valid_samples' in signal_stats:
                            record['valid_samples'] = signal_stats['valid_samples']

                        sampling_data.append(record)

        # Save reports
        if availability_data:
            pd.DataFrame(availability_data).to_csv(
                os.path.join(self.output_path, 'data_availability_report.csv'),
                index=False
            )

        if sampling_data:
            df_sampling = pd.DataFrame(sampling_data)

            # Save detailed report
            df_sampling.to_csv(
                os.path.join(self.output_path, 'detailed_sampling_analysis.csv'),
                index=False
            )

            # Generate summary by device and signal type
            summary = df_sampling.groupby(['Device', 'Signal']).agg({
                'Actual_Rate': ['mean', 'std', 'min', 'max'],
                'Expected_Rate': 'first',
                'Coverage_Ratio': ['mean', 'min', 'max'],
                'Actual_Samples': 'sum'
            }).round(2)

            summary.to_csv(os.path.join(self.output_path, 'sampling_rate_summary.csv'))

            # Generate session-wise summary
            session_summary = df_sampling.groupby(['Session', 'Device', 'Signal']).agg({
                'Actual_Rate': ['mean', 'std'],
                'Coverage_Ratio': 'mean',
                'Actual_Samples': 'sum'
            }).round(2)

            session_summary.to_csv(os.path.join(self.output_path, 'session_sampling_summary.csv'))

        # Generate and save global sampling statistics
        global_stats = self.calculate_global_sampling_statistics()
        global_stats_data = []

        for key, stats in global_stats.items():
            if 'error' not in stats:
                device, signal = key.split('_', 1)
                record = {
                    'Device': device,
                    'Signal': signal,
                    'Expected_Rate': stats['expected_rate'],
                    'Actual_Rate': stats['mean_rate'],
                    'Total_Samples': stats['total_samples'],
                    'Total_Time_Seconds': stats['total_time_seconds'],
                    'Coverage_Ratio': stats['coverage_ratio']
                }
                global_stats_data.append(record)

        if global_stats_data:
            df_global_stats = pd.DataFrame(global_stats_data)
            df_global_stats.to_csv(
                os.path.join(self.output_path, 'global_sampling_statistics.csv'),
                index=False
            )
            print(
                f"Generated global sampling statistics from {sum([stats['total_samples'] for key, stats in global_stats.items() if 'error' not in stats])} samples")
            # Add sampling period analysis
            period_stats = {}
            for device_signal_key, periods in self.all_sampling_periods.items():
                if periods:
                    # Filter out unreasonable periods (e.g., negative or extremely large)
                    valid_periods = np.array(periods)
                    valid_periods = valid_periods[
                        (valid_periods > 0) & (valid_periods < 10)]  # Max 10 seconds between samples

                    if len(valid_periods) > 0:
                        period_stats[device_signal_key] = {
                            'mean_period': np.mean(valid_periods),
                            'std_period': np.std(valid_periods),
                            'median_period': np.median(valid_periods),
                            'min_period': np.min(valid_periods),
                            'max_period': np.max(valid_periods),
                            'samples_analyzed': len(valid_periods)
                        }

            # Save period statistics
            if period_stats:
                period_stats_data = []
                for key, stats in period_stats.items():
                    device, signal = key.split('_', 1)
                    record = {
                        'Device': device,
                        'Signal': signal,
                        'Mean_Period_Seconds': stats['mean_period'],
                        'STD_Period_Seconds': stats['std_period'],
                        'Median_Period_Seconds': stats['median_period'],
                        'Min_Period_Seconds': stats['min_period'],
                        'Max_Period_Seconds': stats['max_period'],
                        'Samples_Analyzed': stats['samples_analyzed']
                    }
                    period_stats_data.append(record)

                df_period_stats = pd.DataFrame(period_stats_data)
                df_period_stats.to_csv(
                    os.path.join(self.output_path, 'sampling_period_statistics.csv'),
                    index=False
                )
                print(
                    f"Generated sampling period statistics from {sum([stats['samples_analyzed'] for stats in period_stats.values()])} intervals")

    def process_specific_participant(self, participant_id: str):
        """Process data for a specific participant"""
        if os.path.isdir(os.path.join(RAW_DIR, participant_id)):
            result = self.process_participant(participant_id)
            if result:
                self.generate_reports([result])
                print(f"\nAnalysis completed for {participant_id}. Results saved in: {self.output_path}")
        else:
            print(f"Participant {participant_id} not found in dataset")

    def process_all_participants(self):
        """Process all participants and generate reports"""
        participants = [d for d in os.listdir(RAW_DIR)
                        if os.path.isdir(os.path.join(RAW_DIR, d))
                        and d.startswith('P') and d != 'P01']

        self.all_results = []  # Store all results for global statistics
        total_participants = len(participants)

        print(f"Starting analysis for {total_participants} participants...")

        for i, participant in enumerate(sorted(participants), 1):
            print(f"\nProcessing participant {participant} ({i}/{total_participants})")
            try:
                result = self.process_participant(participant)
                if result:
                    self.all_results.append(result)
            except Exception as e:
                print(f"Error processing participant {participant}: {str(e)}")

        # Generate comprehensive reports
        if self.all_results:
            self.generate_reports(self.all_results)
            print(f"\nAnalysis completed. Results saved in: {self.output_path}")
            print(f"Successfully processed {len(self.all_results)} out of {total_participants} participants")
        else:
            print("No valid results to generate reports")


class BaseSamplingAnalyzer(ABC):
    """Base class for sampling rate analysis"""
    """
        Abstract base class for sampling rate analysis.

        This class defines the interface for analyzing sampling rates of different
        wearable devices. Implementations should provide device-specific logic.
    """

    def __init__(self, device_name: str):
        self.device_name = device_name
        self.output_path = os.path.join(BASE_DIR, 'SamplingRateAnalysis')
        os.makedirs(self.output_path, exist_ok=True)

        # Valid sessions list
        self.valid_sessions = [
            'adaptation', 'baseline', 'tsst-prep', 'tsst-speech',
            'meditation-1', 'screen-reading', 'ssst-prep', 'ssst-sing',
            'meditation-2', 'keyboard-typing', 'rest-1', 'mobile-typing',
            'rest-2', 'standing', 'rest-3', 'walking', 'rest-4',
            'jogging', 'rest-5', 'running'
        ]

    @abstractmethod
    def get_signal_configs(self) -> Dict:
        """Return signal configurations for the device"""
        pass

    @abstractmethod
    def adjust_timestamp(self, timestamp: pd.Series) -> pd.Series:
        """Adjust timestamps based on device type"""
        pass

    def calculate_sampling_rate(self,
                                timestamps: np.ndarray,
                                duration: float,
                                expected_rate: float) -> Dict[str, float]:
        """Calculate actual sampling rate based on number of samples and duration"""
        if duration <= 0 or len(timestamps) < 2:
            return {
                'actual_rate': 0,
                'expected_samples': 0,
                'actual_samples': len(timestamps),
                'coverage_ratio': 0
            }

        actual_rate = len(timestamps) / duration
        expected_samples = duration * expected_rate

        return {
            'actual_rate': actual_rate,
            'expected_samples': expected_samples,
            'actual_samples': len(timestamps),
            'coverage_ratio': len(timestamps) / expected_samples if expected_samples > 0 else 0
        }

    def validate_columns(self, df: pd.DataFrame, expected_columns: List[str], file_name: str) -> bool:
        """Validate DataFrame columns"""
        missing_columns = [col for col in expected_columns if col not in df.columns]
        if missing_columns:
            print(f"Warning: Missing columns in {file_name}: {missing_columns}")
            return False
        return True

    def analyze_session_sampling(self,
                                 data: Dict[str, pd.DataFrame],
                                 session: str,
                                 start_time: int,
                                 end_time: int) -> Dict:
        """Analyze sampling rates for all signals within a session"""
        duration = (end_time - start_time) / 1000  # Convert to seconds
        session_stats = {
            'duration': duration,
            'signals': {}
        }

        for signal_type, config in self.get_signal_configs().items():
            key = f"{self.device_name}_{signal_type}"
            if key in data:
                df = data[key]
                mask = (df['timestamp'] >= start_time) & (df['timestamp'] <= end_time)
                session_data = df[mask]

                if len(session_data) > 0:
                    stats = self.calculate_sampling_rate(
                        session_data['timestamp'].values,
                        duration,
                        config['rate']
                    )

                    value_cols = config['value_col']
                    if isinstance(value_cols, list):
                        stats['axis_samples'] = {
                            col: len(session_data[session_data[col].notna()])
                            for col in value_cols
                        }
                    else:
                        stats['valid_samples'] = len(session_data[session_data[value_cols].notna()])

                    session_stats['signals'][signal_type] = {
                        'expected_rate': config['rate'],
                        **stats
                    }
                else:
                    session_stats['signals'][signal_type] = {
                        'error': 'No data in session window'
                    }

        return session_stats  # Return the stats regardless of whether we processed any signals


def calculate_sampling_periods(self,
                               timestamps: np.ndarray) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    Calculate sampling periods (time between consecutive samples)
    Returns the periods array and statistics
    """
    if len(timestamps) < 2:
        return np.array([]), {
            'mean': 0,
            'std': 0,
            'min': 0,
            'max': 0,
            'median': 0
        }

    # Calculate periods in seconds
    periods = np.diff(timestamps) / 1000.0

    # Filter out unreasonable periods (e.g., negative or extremely large)
    valid_periods = periods[periods > 0]

    if len(valid_periods) == 0:
        return np.array([]), {
            'mean': 0,
            'std': 0,
            'min': 0,
            'max': 0,
            'median': 0
        }

    stats = {
        'mean': np.mean(valid_periods),
        'std': np.std(valid_periods),
        'min': np.min(valid_periods),
        'max': np.max(valid_periods),
        'median': np.median(valid_periods)
    }

    return valid_periods, stats


class GalaxyWatchAnalyzer(BaseSamplingAnalyzer):
    """Galaxy Watch specific analyzer"""

    def __init__(self):
        super().__init__('galaxy')

    def get_signal_configs(self) -> Dict:
        return {
            'ppg': {
                'rate': 25,
                'file': 'PPG.csv',
                'columns': ['dataReceived', 'timestamp', 'ppg', 'status'],
                'value_col': 'ppg'
            },
            'acc': {
                'rate': 25,
                'file': 'ACC.csv',
                'columns': ['dataReceived', 'timestamp', 'x', 'y', 'z'],
                'value_col': ['x', 'y', 'z']
            },
            'hr': {
                'rate': 1,
                'file': 'HR.csv',
                'columns': ['dataReceived', 'timestamp', 'hr', 'hrStatus', 'ibi', 'ibiStatus'],
                'value_col': 'hr'
            },
            'skintemp': {
                'rate': 1 / 60,
                'file': 'SkinTemp.csv',
                'columns': ['dataReceived', 'timestamp', 'ambientTemp', 'objectTemp', 'status'],
                'value_col': ['ambientTemp', 'objectTemp']
            }
        }

    def adjust_timestamp(self, timestamp: pd.Series) -> pd.Series:
        return timestamp  # No adjustment needed for Galaxy Watch


class E4Analyzer(BaseSamplingAnalyzer):
    """E4 specific analyzer"""

    def __init__(self):
        super().__init__('e4')

    def get_signal_configs(self) -> Dict:
        return {
            'bvp': {
                'rate': 64,
                'file': 'BVP.csv',
                'columns': ['value', 'timestamp'],
                'value_col': 'value'
            },
            'acc': {
                'rate': 32,
                'file': 'ACC.csv',
                'columns': ['x', 'y', 'z', 'timestamp'],
                'value_col': ['x', 'y', 'z']
            },
            'temp': {
                'rate': 4,
                'file': 'TEMP.csv',
                'columns': ['value', 'timestamp'],
                'value_col': 'value'
            },
            'hr': {
                'rate': 1,
                'file': 'HR.csv',
                'columns': ['value', 'timestamp'],
                'value_col': 'value'
            }
        }

    def adjust_timestamp(self, timestamp: pd.Series) -> pd.Series:
        return timestamp / 1000  # Convert microseconds to milliseconds


class PolarH10Analyzer(BaseSamplingAnalyzer):
    """PolarH10 specific analyzer"""

    def __init__(self):
        super().__init__('polar')

    def get_signal_configs(self) -> Dict:
        return {
            'ecg': {
                'rate': 130,
                'file': 'ECG.csv',
                'columns': ['phoneTimestamp', 'sensorTimestamp', 'ecg'],
                'value_col': 'ecg'
            },
            'acc': {
                'rate': 200,
                'file': 'ACC.csv',
                'columns': ['phoneTimestamp', 'sensorTimestamp', 'X', 'Y', 'Z'],
                'value_col': ['X', 'Y', 'Z']
            },
            'hr': {
                'rate': 1,
                'file': 'HR.csv',
                'columns': ['phoneTimestamp', 'hr', 'hrv'],
                'value_col': 'hr'
            }
        }

    def adjust_timestamp(self, timestamp: pd.Series) -> pd.Series:
        return timestamp - 32400000  # Subtract 9 hours


def main():
    try:
        analyzer = OverallSamplingRateAnalyzer()
        analyzer.process_all_participants()
        # Example Galaxy
        # analyzer = GalaxyWatchAnalyzer()
        # Specific Participant
        # analyzer.process_specific_participant()
    except Exception as e:
        print(f"Error in main execution: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()

