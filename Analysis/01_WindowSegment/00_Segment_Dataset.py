import os
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from Analysis.config import RAW_DIR, BASE_DIR

class WindowProcessor:
    def __init__(self, dataset_path: str):

        self.dataset_path = dataset_path
        self.output_path = os.path.join(BASE_DIR, 'WindowData')
        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)
        self.window_size = 8000
        self.window_step = 2000

    def extract_window_data(self,
                            df: pd.DataFrame,
                            start_time: int,
                            end_time: int,
                            timestamp_col: str,
                            value_cols: List[str],
                            resample_rate: Optional[int] = None) -> Optional[np.ndarray]:

        try:
            mask = (df[timestamp_col] >= start_time) & (df[timestamp_col] <= end_time)
            window_data = df[mask]

            if window_data.empty:
                return None

            values = window_data[value_cols].values

            if resample_rate is not None:
                original_samples = len(values)
                target_samples = int((end_time - start_time) * resample_rate / 1000)

                if original_samples > 1:
                    resampled_values = []
                    for col in range(values.shape[1]):
                        original_times = np.linspace(start_time, end_time, original_samples)
                        target_times = np.linspace(start_time, end_time, target_samples)
                        resampled_col = np.interp(target_times, original_times, values[:, col])
                        resampled_values.append(resampled_col)
                    values = np.column_stack(resampled_values)

            return values

        except Exception as e:
            print(f"Error in extract_window_data: {str(e)}")
            return None

    def create_window(self,
                      participant: str,
                      session_name: str,
                      window_number: int,
                      window_start: int,
                      window_end: int,
                      data: Dict[str, pd.DataFrame]) -> Dict:
        window_data = {
            'participant': participant,
            'session': session_name,
            'windowNumber': window_number,
            'startTime': window_start,
            'endTime': window_end
        }

        signals_config = {
            'galaxyPPG': ('galaxy_ppg', ['ppg'], 25),
            'e4BVP': ('e4_bvp', ['value'], 64),
            'polarECG': ('polar_ecg', ['ecg'], None),
            'galaxyACC': ('galaxy_acc', ['x', 'y', 'z'], 25),
            'e4ACC': ('e4_acc', ['x', 'y', 'z'], 64),
            'polarACC': ('polar_acc', ['X', 'Y', 'Z'], None)
        }

        for signal_name, (data_key, cols, target_rate) in signals_config.items():
            try:
                signal_data = self.extract_window_data(
                    data[data_key],
                    window_start,
                    window_end,
                    'timestamp',
                    cols,
                    target_rate
                )

                if signal_data is not None:
                    if len(cols) == 1:
                        window_data[signal_name] = ';'.join(map(str, signal_data.flatten()))
                    else:
                        window_data[signal_name] = ';'.join(';'.join(map(str, row)) for row in signal_data)
                else:
                    window_data[signal_name] = None
            except Exception as e:
                print(f"Warning: Error processing {signal_name}: {str(e)}")
                window_data[signal_name] = None

        try:
            hr_data = self.extract_window_data(
                data['polar_hr'], window_start, window_end, 'timestamp', ['hr']
            )
            window_data['trueHR'] = float(np.mean(hr_data)) if hr_data is not None else None
        except Exception as e:
            window_data['trueHR'] = None

        window_data['isPadded'] = any(v is None for k, v in window_data.items()
                                      if k not in ['participant', 'session', 'windowNumber',
                                                   'startTime', 'endTime'])

        return window_data

    def get_valid_sessions(self, events: pd.DataFrame) -> List[Dict]:

        valid_sessions = []
        events_sorted = events.sort_values('timestamp')

        for i in range(len(events_sorted) - 1):
            current_event = events_sorted.iloc[i]
            next_event = events_sorted.iloc[i + 1]

            if current_event['status'].upper() == 'ENTER':
                session = {
                    'name': current_event['session'],
                    'start': current_event['timestamp'],
                    'end': None
                }

                exit_events = events_sorted[
                    (events_sorted['session'] == current_event['session']) &
                    (events_sorted['status'].str.upper() == 'EXIT') &
                    (events_sorted['timestamp'] >= current_event['timestamp'])
                    ]

                if not exit_events.empty:
                    session['end'] = exit_events.iloc[0]['timestamp']
                    session_duration = (session['end'] - session['start']) / 1000

                    if session_duration >= self.window_size / 1000:
                        valid_sessions.append(session)

            if current_event['status'].upper() == 'EXIT' and next_event['status'].upper() == 'ENTER':
                transition_period = {
                    'name': f"TP_{current_event['session']}_to_{next_event['session']}",
                    'start': current_event['timestamp'],
                    'end': next_event['timestamp']
                }

                transition_duration = (transition_period['end'] - transition_period['start']) / 1000
                if transition_duration >= self.window_size / 1000:
                    valid_sessions.append(transition_period)

        valid_sessions.sort(key=lambda x: x['start'])

        for session in valid_sessions:
            duration = (session['end'] - session['start']) / 1000
            print(f"Found session: {session['name']}, Duration: {duration:.2f} seconds")

        return valid_sessions

    def process_participant(self, participant: str):
        """
        Process a single participant's data.

        Args:
            participant (str): The participant ID.
        """
        try:
            print(f"Processing participant: {participant}")

            # Read participant data
            data, events = self.read_participant_data(participant)

            # Skip if critical data is missing
            if data is None or events is None:
                print(f"Skipping participant {participant} due to missing critical files.")
                return

            # Process valid sessions
            valid_sessions = self.get_valid_sessions(events)

            all_windows = []
            global_window_count = 1

            for session in valid_sessions:
                print(f"\nProcessing session: {session['name']}")
                print(f"Duration: {(session['end'] - session['start']) / 1000:.2f} seconds")

                window_starts = range(
                    int(session['start']),
                    int(session['end'] - self.window_size + 1),
                    int(self.window_step)
                )

                session_windows = 0
                for window_start in window_starts:
                    window_data = self.create_window(
                        participant,
                        session['name'],
                        global_window_count,
                        window_start,
                        window_start + self.window_size,
                        data
                    )
                    if window_data:
                        all_windows.append(window_data)
                        global_window_count += 1
                        session_windows += 1

                print(f"Created {session_windows} windows for session {session['name']}")

            if all_windows:
                output_file = os.path.join(self.output_path, f"{participant}_processed.csv")
                df = pd.DataFrame(all_windows)
                columns = [
                    'participant', 'session', 'windowNumber', 'startTime', 'endTime',
                    'trueHR', 'galaxyPPG', 'galaxyACC', 'e4BVP', 'e4ACC',
                    'polarECG', 'polarACC', 'isPadded'
                ]
                df = df.reindex(columns=columns)
                df.to_csv(output_file, index=False)
                print(f"\nSaved processed data to: {output_file}")
                print(f"Total windows created: {len(all_windows)}")

                session_counts = df['session'].value_counts()
                print("\nWindows per session:")
                for session, count in session_counts.items():
                    print(f"{session}: {count} windows")

        except Exception as e:
            print(f"Error processing participant {participant}: {str(e)}")

    def _adjust_timestamps(self, data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:

        adjusted_data = {}
        for key, df in data.items():
            df = df.copy()
            if 'polar' in key:
                if 'phoneTimestamp' in df.columns:
                    df['timestamp'] = df['phoneTimestamp'] - 32400000
            elif 'e4' in key:
                if 'timestamp' in df.columns:
                    df['timestamp'] = df['timestamp'] / 1000
            adjusted_data[key] = df
        return adjusted_data

    def read_participant_data(self, participant: str) -> Tuple[Dict[str, pd.DataFrame], pd.DataFrame]:
        """
        Reads participant data from the dataset folder. Checks if files exist before reading them.

        Args:
            participant (str): The participant ID.

        Returns:
            Tuple[Dict[str, pd.DataFrame], pd.DataFrame]: A dictionary of data and the event DataFrame.
        """
        participant_path = os.path.join(self.dataset_path, participant)
        data = {}

        def safe_read_csv(file_path: str) -> pd.DataFrame:
            """
            Safely reads a CSV file. If the file does not exist, returns an empty DataFrame.

            Args:
                file_path (str): The path to the CSV file.

            Returns:
                pd.DataFrame: The read DataFrame, or an empty DataFrame if the file does not exist.
            """
            if os.path.exists(file_path):
                return pd.read_csv(file_path)
            else:
                print(f"Warning: File not found: {file_path}")
                return pd.DataFrame()  # Return an empty DataFrame to prevent errors

        # Read each data file safely, skipping files that don't exist
        data['polar_acc'] = safe_read_csv(os.path.join(participant_path, 'PolarH10', 'ACC.csv'))
        data['polar_ecg'] = safe_read_csv(os.path.join(participant_path, 'PolarH10', 'ECG.csv'))
        data['polar_hr'] = safe_read_csv(os.path.join(participant_path, 'PolarH10', 'HR.csv'))
        data['galaxy_acc'] = safe_read_csv(os.path.join(participant_path, 'GalaxyWatch', 'ACC.csv'))
        data['galaxy_ppg'] = safe_read_csv(os.path.join(participant_path, 'GalaxyWatch', 'PPG.csv'))
        data['e4_acc'] = safe_read_csv(os.path.join(participant_path, 'E4', 'ACC.csv'))
        data['e4_bvp'] = safe_read_csv(os.path.join(participant_path, 'E4', 'BVP.csv'))
        events = safe_read_csv(os.path.join(participant_path, 'Event.csv'))

        # Raise a warning if the events DataFrame is empty
        if events.empty:
            raise ValueError(f"Warning: No event data found for participant {participant}")

        # Adjust timestamps if data is available
        data = self._adjust_timestamps(data)
        return data, events

    def process_all_participants(self):
        participants = [d for d in os.listdir(self.dataset_path)
                        if os.path.isdir(os.path.join(self.dataset_path, d))
                        and d.startswith('P')]

        for participant in participants:
            self.process_participant(participant)

    def process_specific_participant(self, participant: str):
        if os.path.isdir(os.path.join(self.dataset_path, participant)):
            self.process_participant(participant)
        else:
            print(f"Participant {participant} not found in dataset")

def main():
    dataset_path = RAW_DIR
    processor = WindowProcessor(dataset_path)
    # processor.process_specific_participant("P03")
    processor.process_all_participants()

if __name__ == '__main__':
    main()