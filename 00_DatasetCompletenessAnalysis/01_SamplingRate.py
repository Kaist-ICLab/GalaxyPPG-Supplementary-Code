import os
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from abc import ABC, abstractmethod
from config import BASE_DIR, RAW_DIR

class SamplingRateAnalyzer:
    """Main analyzer class that coordinates device-specific analyzers"""

    def __init__(self):
        self.analyzers = {
            'galaxy': GalaxyWatchAnalyzer(),
            'e4': E4Analyzer(),
            'polar': PolarH10Analyzer()
        }
        self.output_path = os.path.join(BASE_DIR, 'SamplingRateAnalysis')
        os.makedirs(self.output_path, exist_ok=True)

    def process_participant(self, participant: str) -> Dict:
        """Process single participant data"""
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
                    for signal_type, signal_stats in device_data['signals'].items():
                        if 'error' not in signal_stats:
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
                        and d.startswith('P')]

        all_results = []
        total_participants = len(participants)

        print(f"Starting analysis for {total_participants} participants...")

        for i, participant in enumerate(sorted(participants), 1):
            print(f"\nProcessing participant {participant} ({i}/{total_participants})")
            try:
                result = self.process_participant(participant)
                if result:
                    all_results.append(result)
            except Exception as e:
                print(f"Error processing participant {participant}: {str(e)}")

        # Generate comprehensive reports
        if all_results:
            self.generate_reports(all_results)
            print(f"\nAnalysis completed. Results saved in: {self.output_path}")
            print(f"Successfully processed {len(all_results)} out of {total_participants} participants")
        else:
            print("No valid results to generate reports")
class BaseSamplingAnalyzer(ABC):
    """Base class for sampling rate analysis"""
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

        return session_stats
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

    def adjust_timestamp(self, timestamp: pd.Series) -> pd.Series:
        return timestamp - 32400000  # Subtract 9 hours

def main():
    try:
        analyzer = SamplingRateAnalyzer()
        analyzer.process_all_participants()
    except Exception as e:
        print(f"Error in main execution: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()