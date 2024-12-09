import os
import pandas as pd
import numpy as np
from typing import Dict, List
from config import BASE_DIR, RAW_DIR


class SamplingRateAnalyzer:
    def __init__(self):
        # Define expected sampling rates and corresponding columns for each signal
        self.signal_configs = {
            'galaxy': {
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
            },
            'e4': {
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
            },
            'polar': {
                'ecg': {
                    'rate': 130,
                    'file': 'ECG.csv',
                    'columns': ['phoneTimestamp', 'sensorTimestamp', 'timestamp', 'ecg'],
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
        }

        # Valid sessions list
        self.valid_sessions = [
            'adaptation', 'baseline', 'tsst-prep', 'tsst-speech',
            'meditation-1', 'screen-reading', 'ssst-prep', 'ssst-sing',
            'meditation-2', 'keyboard-typing', 'rest-1', 'mobile-typing',
            'rest-2', 'standing', 'rest-3', 'walking', 'rest-4',
            'jogging', 'rest-5', 'running'
        ]

        # Output directory
        self.output_path = os.path.join(BASE_DIR, 'SamplingRateAnalysis')
        os.makedirs(self.output_path, exist_ok=True)

    def _adjust_timestamp(self, device: str, timestamp: pd.Series) -> pd.Series:
        """
        Adjust timestamps based on device type
        """
        if device == 'polar':
            return timestamp - 32400000  # Subtract 9 hours
        elif device == 'e4':
            return timestamp / 1000  # Convert microseconds to milliseconds
        return timestamp  # Galaxy Watch needs no adjustment

    def calculate_sampling_rate(self,
                                timestamps: np.ndarray,
                                duration: float,
                                expected_rate: float) -> Dict[str, float]:
        """
        Calculate actual sampling rate based on number of samples and duration
        """
        if duration <= 0 or len(timestamps) < 2:
            return {
                'actual_rate': 0,
                'expected_samples': 0,
                'actual_samples': len(timestamps),
                'coverage_ratio': 0
            }

        # Calculate actual sampling rate
        actual_rate = len(timestamps) / duration
        expected_samples = duration * expected_rate

        return {
            'actual_rate': actual_rate,
            'expected_samples': expected_samples,
            'actual_samples': len(timestamps),
            'coverage_ratio': len(timestamps) / expected_samples if expected_samples > 0 else 0
        }

    def validate_columns(self, df: pd.DataFrame, expected_columns: List[str], file_name: str) -> bool:
        """
        Validate that DataFrame contains all expected columns
        """
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
        """
        Analyze sampling rates for all signals within a session
        """
        duration = (end_time - start_time) / 1000  # Convert to seconds
        session_stats = {
            'duration': duration,
            'signals': {}
        }

        for device, device_configs in self.signal_configs.items():
            session_stats['signals'][device] = {}

            for signal_type, config in device_configs.items():
                key = f"{device}_{signal_type}"
                if key in data:
                    df = data[key]
                    mask = (df['timestamp'] >= start_time) & (df['timestamp'] <= end_time)
                    session_data = df[mask]

                    if len(session_data) > 0:
                        # Calculate sampling rate statistics
                        stats = self.calculate_sampling_rate(
                            session_data['timestamp'].values,
                            duration,
                            config['rate']
                        )

                        # Add signal-specific information
                        value_cols = config['value_col']
                        if isinstance(value_cols, list):  # For accelerometer data
                            stats['axis_samples'] = {
                                col: len(session_data[session_data[col].notna()])
                                for col in value_cols
                            }
                        else:  # For single-channel signals
                            stats['valid_samples'] = len(session_data[session_data[value_cols].notna()])

                        session_stats['signals'][device][signal_type] = {
                            'expected_rate': config['rate'],
                            **stats
                        }
                    else:
                        session_stats['signals'][device][signal_type] = {
                            'error': 'No data in session window'
                        }

        return session_stats

    def process_participant(self, participant: str) -> Dict:
        """Process single participant data"""
        try:
            print(f"\nProcessing participant: {participant}")

            # Initialize data storage
            data = {}
            missing_files = []
            column_errors = []

            # Load all files based on configuration
            for device, device_configs in self.signal_configs.items():
                device_path = os.path.join(RAW_DIR, participant,
                                           'GalaxyWatch' if device == 'galaxy' else
                                           'E4' if device == 'e4' else 'PolarH10')

                for signal_type, config in device_configs.items():
                    file_path = os.path.join(device_path, config['file'])
                    key = f"{device}_{signal_type}"

                    if os.path.exists(file_path):
                        try:
                            df = pd.read_csv(file_path)
                            if self.validate_columns(df, config['columns'], config['file']):
                                # Standardize timestamp column and adjust values
                                if device == 'polar' and 'phoneTimestamp' in df.columns:
                                    df['timestamp'] = self._adjust_timestamp(device, df['phoneTimestamp'])
                                else:
                                    df['timestamp'] = self._adjust_timestamp(device, df['timestamp'])
                                data[key] = df
                            else:
                                column_errors.append(f"{key} (Invalid columns)")
                        except Exception as e:
                            missing_files.append(f"{key} (Read Error: {str(e)})")
                    else:
                        missing_files.append(key)

            # Read events file
            events_df = pd.read_csv(os.path.join(RAW_DIR, participant, 'event.csv'))

            # Process each session
            results = {
                'participant': participant,
                'missing_files': missing_files,
                'column_errors': column_errors,
                'sessions': {}
            }

            for session in self.valid_sessions:
                session_events = events_df[events_df['session'] == session]

                if len(session_events) >= 2:
                    enter_event = session_events[session_events['status'].str.upper() == 'ENTER'].iloc[0]
                    exit_event = session_events[session_events['status'].str.upper() == 'EXIT'].iloc[0]

                    results['sessions'][session] = self.analyze_session_sampling(
                        data,
                        session,
                        enter_event['timestamp'],
                        exit_event['timestamp']
                    )

            return results

        except Exception as e:
            print(f"Error processing participant {participant}: {str(e)}")
            return None

    def generate_reports(self, all_results: List[Dict]):
        """Generate comprehensive reports"""
        # 1. Data Availability Report
        availability_data = []
        for result in all_results:
            if result['missing_files'] or result['column_errors']:
                for issue in result['missing_files']:
                    availability_data.append({
                        'Participant': result['participant'],
                        'Issue_Type': 'Missing File',
                        'Detail': issue
                    })
                for issue in result['column_errors']:
                    availability_data.append({
                        'Participant': result['participant'],
                        'Issue_Type': 'Column Error',
                        'Detail': issue
                    })

        if availability_data:
            pd.DataFrame(availability_data).to_csv(
                os.path.join(self.output_path, 'data_availability_report.csv'),
                index=False
            )

        # 2. Detailed Sampling Rate Analysis
        sampling_data = []
        for result in all_results:
            for session, session_data in result['sessions'].items():
                for device, device_data in session_data['signals'].items():
                    for signal_type, signal_stats in device_data.items():
                        if 'error' not in signal_stats:
                            record = {
                                'Participant': result['participant'],
                                'Session': session,
                                'Device': device,
                                'Signal': signal_type,
                                'Duration_Seconds': session_data['duration'],
                                'Expected_Rate': signal_stats['expected_rate'],
                                'Actual_Rate': signal_stats['actual_rate'],
                                'Expected_Samples': signal_stats['expected_samples'],
                                'Actual_Samples': signal_stats['actual_samples'],
                                'Coverage_Ratio': signal_stats['coverage_ratio']
                            }

                            # Add axis-specific information for accelerometer data
                            if 'axis_samples' in signal_stats:
                                for axis, count in signal_stats['axis_samples'].items():
                                    record[f'{axis}_samples'] = count
                            elif 'valid_samples' in signal_stats:
                                record['valid_samples'] = signal_stats['valid_samples']

                            sampling_data.append(record)

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

            summary.to_csv(
                os.path.join(self.output_path, 'sampling_rate_summary.csv')
            )

            # Generate session-wise summary
            session_summary = df_sampling.groupby(['Session', 'Device', 'Signal']).agg({
                'Actual_Rate': ['mean', 'std'],
                'Coverage_Ratio': 'mean',
                'Actual_Samples': 'sum'
            }).round(2)

            session_summary.to_csv(
                os.path.join(self.output_path, 'session_sampling_summary.csv')
            )

            # Print summary to console
            print("\nSampling Rate Analysis Summary:")
            print("=" * 80)
            for (device, signal), stats in summary.iterrows():
                print(f"\nDevice: {device}, Signal: {signal}")
                print(f"Expected Rate: {stats['Expected_Rate']['first']:.2f} Hz")
                print(f"Actual Rate: {stats['Actual_Rate']['mean']:.2f} ± {stats['Actual_Rate']['std']:.2f} Hz")
                print(f"Coverage Ratio: {stats['Coverage_Ratio']['mean']:.2f}% "
                      f"(Range: {stats['Coverage_Ratio']['min']:.2f}% - {stats['Coverage_Ratio']['max']:.2f}%)")
                print(f"Total Samples: {stats['Actual_Samples']['sum']}")

    def process_all_participants(self):
        """Process all participants and generate reports"""
        participants = [d for d in os.listdir(RAW_DIR)
                        if os.path.isdir(os.path.join(RAW_DIR, d))
                        and d.startswith('P')]

        all_results = []
        for participant in sorted(participants):
            result = self.process_participant(participant)
            if result:
                all_results.append(result)

        # Generate reports
        self.generate_reports(all_results)


def main():
    analyzer = SamplingRateAnalyzer()
    analyzer.process_all_participants()
    print(f"\nAnalysis completed. Results saved in: {analyzer.output_path}")
if __name__ == '__main__':
    main()