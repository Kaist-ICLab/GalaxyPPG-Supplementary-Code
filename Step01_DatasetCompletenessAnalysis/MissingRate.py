import os
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from config import BASE_DIR, RAW_DIR

class MissingRateAnalyzer:
    """
        Analyzer for assessing signal quality and missing data in PPG recordings.

        This class provides functionality to analyze the quality and completeness of
        physiological signals from different wearable devices (Galaxy Watch, E4, Polar H10).
    """
    def __init__(self):
        # Initialize dictionaries with descriptions for Galaxy PPG and HR status codes
        self.galaxy_ppg_status_desc = {
            -999: "Higher priority sensor operating",
            0: "Normal value",
            500: "STATUS interface not supported"
        }

        self.galaxy_hr_status_desc = {
            -999: "Higher priority sensor operating",
            -99: "Flush called but no data",
            -10: "PPG signal too weak/excessive movement",
            -8: "Weak PPG signal/movement",
            -3: "Wearable detached",
            -2: "Wearable movement detected",
            0: "Initial state/higher priority sensor",
            1: "Successful measurement"
        }

        self.output_path = os.path.join(BASE_DIR, 'MissingRateAnalysis')
        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)
    def check_signal_quality(self, signal_data: np.ndarray, signal_type: str) -> Tuple[float, Dict]:
        """
                Check the quality of different types of signals (PPG, ECG, ACC, HR).

                Args:
                    signal_data (np.ndarray): The signal data to analyze.
                    signal_type (str): Type of signal ('PPG', 'ECG', 'ACC', 'HR').

                Returns:
                    Tuple: Missing rate percentage and a dictionary of quality metrics.
        """
        total_samples = len(signal_data)
        nan_count = np.sum(np.isnan(signal_data))
        zero_count = np.sum(signal_data == 0)

        if signal_type in ['PPG', 'ECG']:
            signal_range = np.ptp(signal_data[~np.isnan(signal_data)])
            weak_signal = signal_range < np.std(signal_data[~np.isnan(signal_data)]) * 0.1


            quality_metrics = {
                'nan_rate': (nan_count / total_samples) * 100,
                'zero_rate': (zero_count / total_samples) * 100,
                'weak_signal_detected': weak_signal,
            }

        elif signal_type == 'ACC':
            acc_magnitude = np.sqrt(np.sum(signal_data ** 2, axis=1))
            invalid_acc = np.sum(acc_magnitude > 10)

            quality_metrics = {
                'nan_rate': (nan_count / total_samples) * 100,
                'zero_rate': (zero_count / total_samples) * 100,
                'invalid_acc_samples': invalid_acc,
                'invalid_acc_rate': (invalid_acc / total_samples) * 100
            }

        elif signal_type == 'HR':
            invalid_hr = np.sum((signal_data < 40) | (signal_data > 200))

            quality_metrics = {
                'nan_rate': (nan_count / total_samples) * 100,
                'zero_rate': (zero_count / total_samples) * 100,
                'invalid_hr_samples': invalid_hr,
                'invalid_hr_rate': (invalid_hr / total_samples) * 100
            }

        missing_rate = (nan_count + zero_count) / total_samples * 100

        return missing_rate, quality_metrics
    def analyze_participant(self, participant: str) -> Dict:
        """
              Analyze the signal quality for a single participant.

              Args:
                  participant (str): The participant's folder name.

              Returns:
                  Dict: Analysis results for different devices and signals.
              """
        participant_path = os.path.join(RAW_DIR, participant)
        results = {'participant': participant, 'galaxy': {}}

        try:
            ppg_file = os.path.join(participant_path, 'GalaxyWatch', 'PPG.csv')
            if os.path.exists(ppg_file):
                galaxy_ppg = pd.read_csv(ppg_file)
                if len(galaxy_ppg) > 0:
                    ppg_missing = sum(galaxy_ppg['status'] != 0) / len(galaxy_ppg) * 100
                    results['galaxy']['ppg_missing_rate'] = ppg_missing
                    results['galaxy']['ppg_status_dist'] = galaxy_ppg['status'].value_counts().to_dict()
                else:
                    print(f"Warning: Empty PPG file for {participant}")
                    results['galaxy']['ppg_missing_rate'] = 100
                    results['galaxy']['ppg_status_dist'] = {}
            else:
                print(f"Warning: No PPG file for {participant}")
                results['galaxy']['ppg_missing_rate'] = None
                results['galaxy']['ppg_status_dist'] = None

            acc_file = os.path.join(participant_path, 'GalaxyWatch', 'ACC.csv')
            if os.path.exists(acc_file):
                galaxy_acc = pd.read_csv(acc_file)
                if len(galaxy_acc) > 0:
                    acc_data = np.column_stack((galaxy_acc['x'], galaxy_acc['y'], galaxy_acc['z']))
                    acc_missing, acc_metrics = self.check_signal_quality(acc_data, 'ACC')
                    results['galaxy']['acc_missing_rate'] = acc_missing
                    results['galaxy']['acc_metrics'] = acc_metrics
                else:
                    print(f"Warning: Empty ACC file for {participant}")
                    results['galaxy']['acc_missing_rate'] = 100
                    results['galaxy']['acc_metrics'] = None
            else:
                print(f"Warning: No ACC file for {participant}")
                results['galaxy']['acc_missing_rate'] = None
                results['galaxy']['acc_metrics'] = None

            hr_file = os.path.join(participant_path, 'GalaxyWatch', 'HR.csv')
            if os.path.exists(hr_file):
                galaxy_hr = pd.read_csv(hr_file)
                if len(galaxy_hr) > 0:
                    hr_missing = sum(galaxy_hr['hrStatus'] != 1) / len(galaxy_hr) * 100
                    results['galaxy']['hr_missing_rate'] = hr_missing
                    results['galaxy']['hr_status_dist'] = galaxy_hr['hrStatus'].value_counts().to_dict()
                else:
                    print(f"Warning: Empty HR file for {participant}")
                    results['galaxy']['hr_missing_rate'] = 100
                    results['galaxy']['hr_status_dist'] = {}
            else:
                print(f"Warning: No HR file for {participant}")
                results['galaxy']['hr_missing_rate'] = None
                results['galaxy']['hr_status_dist'] = None

        except Exception as e:
            print(f"Error processing Galaxy data for {participant}: {str(e)}")

        try:
            results['e4'] = {}
            bvp_file = os.path.join(participant_path, 'E4', 'BVP.csv')
            if os.path.exists(bvp_file):
                e4_bvp = pd.read_csv(bvp_file)
                if len(e4_bvp) > 0:
                    bvp_missing, bvp_metrics = self.check_signal_quality(e4_bvp['value'].values, 'PPG')
                    results['e4']['bvp_missing_rate'] = bvp_missing
                    results['e4']['bvp_metrics'] = bvp_metrics
                else:
                    print(f"Warning: Empty BVP file for {participant}")
                    results['e4']['bvp_missing_rate'] = 100
                    results['e4']['bvp_metrics'] = None
            else:
                print(f"Warning: No BVP file for {participant}")
                results['e4']['bvp_missing_rate'] = None
                results['e4']['bvp_metrics'] = None

            acc_file = os.path.join(participant_path, 'E4', 'ACC.csv')
            if os.path.exists(acc_file):
                e4_acc = pd.read_csv(acc_file)
                if len(e4_acc) > 0:
                    acc_data = np.column_stack((e4_acc['x'], e4_acc['y'], e4_acc['z']))
                    acc_missing, acc_metrics = self.check_signal_quality(acc_data, 'ACC')
                    results['e4']['acc_missing_rate'] = acc_missing
                    results['e4']['acc_metrics'] = acc_metrics
                else:
                    print(f"Warning: Empty ACC file for {participant}")
                    results['e4']['acc_missing_rate'] = 100
                    results['e4']['acc_metrics'] = None
            else:
                print(f"Warning: No ACC file for {participant}")
                results['e4']['acc_missing_rate'] = None
                results['e4']['acc_metrics'] = None

            hr_file = os.path.join(participant_path, 'E4', 'HR.csv')
            if os.path.exists(hr_file):
                e4_hr = pd.read_csv(hr_file)
                if len(e4_hr) > 0:
                    hr_values = e4_hr['value'].values
                    hr_missing, hr_metrics = self.check_signal_quality(hr_values, 'HR')
                    results['e4']['hr_missing_rate'] = hr_missing
                    results['e4']['hr_metrics'] = hr_metrics

                    valid_hr_mask = (hr_values >= 40) & (hr_values <= 200)
                    invalid_hr_rate = (1 - np.mean(valid_hr_mask)) * 100
                    results['e4']['hr_invalid_rate'] = invalid_hr_rate

                else:
                    print(f"Warning: Empty HR file for {participant}")
                    results['e4']['hr_missing_rate'] = 100
                    results['e4']['hr_metrics'] = None
                    results['e4']['hr_invalid_rate'] = 100
            else:
                print(f"Warning: No HR file for {participant}")
                results['e4']['hr_missing_rate'] = None
                results['e4']['hr_metrics'] = None
                results['e4']['hr_invalid_rate'] = None

        except Exception as e:
            print(f"Error processing E4 data for {participant}: {str(e)}")
            results['e4'] = None

        try:
            results['polar'] = {}
            ecg_file = os.path.join(participant_path, 'PolarH10', 'ECG.csv')
            if os.path.exists(ecg_file):
                polar_ecg = pd.read_csv(ecg_file)
                if len(polar_ecg) > 0:
                    ecg_missing, ecg_metrics = self.check_signal_quality(polar_ecg['ecg'].values, 'ECG')
                    results['polar']['ecg_missing_rate'] = ecg_missing
                    results['polar']['ecg_metrics'] = ecg_metrics
                else:
                    print(f"Warning: Empty ECG file for {participant}")
                    results['polar']['ecg_missing_rate'] = 100
                    results['polar']['ecg_metrics'] = None
            else:
                print(f"Warning: No ECG file for {participant}")
                results['polar']['ecg_missing_rate'] = None
                results['polar']['ecg_metrics'] = None

            acc_file = os.path.join(participant_path, 'PolarH10', 'ACC.csv')
            if os.path.exists(acc_file):
                polar_acc = pd.read_csv(acc_file)
                if len(polar_acc) > 0:
                    acc_data = np.column_stack((polar_acc['X'], polar_acc['Y'], polar_acc['Z']))
                    acc_missing, acc_metrics = self.check_signal_quality(acc_data, 'ACC')
                    results['polar']['acc_missing_rate'] = acc_missing
                    results['polar']['acc_metrics'] = acc_metrics
                else:
                    print(f"Warning: Empty ACC file for {participant}")
                    results['polar']['acc_missing_rate'] = 100
                    results['polar']['acc_metrics'] = None
            else:
                print(f"Warning: No ACC file for {participant}")
                results['polar']['acc_missing_rate'] = None
                results['polar']['acc_metrics'] = None

            hr_file = os.path.join(participant_path, 'PolarH10', 'HR.csv')
            if os.path.exists(hr_file):
                polar_hr = pd.read_csv(hr_file)
                if len(polar_hr) > 0:
                    hr_missing, hr_metrics = self.check_signal_quality(polar_hr['hr'].values, 'HR')
                    results['polar']['hr_missing_rate'] = hr_missing
                    results['polar']['hr_metrics'] = hr_metrics
                else:
                    print(f"Warning: Empty HR file for {participant}")
                    results['polar']['hr_missing_rate'] = 100
                    results['polar']['hr_metrics'] = None
            else:
                print(f"Warning: No HR file for {participant}")
                results['polar']['hr_missing_rate'] = None
                results['polar']['hr_metrics'] = None

        except Exception as e:
            print(f"Error processing PolarH10 data for {participant}: {str(e)}")
            results['polar'] = None

        return results
    def analyze_all_participants(self):
        """
                Analyze the signal quality for all participants in the data directory.
        """
        participants = [d for d in os.listdir(RAW_DIR)
                        if os.path.isdir(os.path.join(RAW_DIR, d))
                        and d.startswith('P')]

        all_results = []
        for participant in participants:
            try:
                print(f"Processing {participant}...")
                result = self.analyze_participant(participant)
                all_results.append(result)
            except Exception as e:
                print(f"Error processing {participant}: {str(e)}")

        self.save_results(all_results)
        return all_results
    def save_results(self, all_results: List[Dict]):
        """
               Save the summarized and detailed results to CSV files.

               Args:
                   all_results (List[Dict]): List of results for each participant.
       """

        device_stats = {
            'galaxy': {'ppg': [], 'acc': [], 'hr': []},
            'e4': {'bvp': [], 'acc': [], 'hr': []},
            'polar': {'ecg': [], 'acc': [], 'hr': []}
        }

        for result in all_results:
            if result.get('galaxy'):
                if 'ppg_missing_rate' in result['galaxy']:
                    device_stats['galaxy']['ppg'].append(result['galaxy']['ppg_missing_rate'])
                if 'acc_missing_rate' in result['galaxy']:
                    device_stats['galaxy']['acc'].append(result['galaxy']['acc_missing_rate'])
                if 'hr_missing_rate' in result['galaxy']:
                    device_stats['galaxy']['hr'].append(result['galaxy']['hr_missing_rate'])

            if result.get('e4'):
                if 'bvp_missing_rate' in result['e4']:
                    device_stats['e4']['bvp'].append(result['e4']['bvp_missing_rate'])
                if 'acc_missing_rate' in result['e4']:
                    device_stats['e4']['acc'].append(result['e4']['acc_missing_rate'])
                if 'hr_missing_rate' in result['e4']:
                    device_stats['e4']['hr'].append(result['e4']['hr_missing_rate'])

            if result.get('polar'):
                if 'ecg_missing_rate' in result['polar']:
                    device_stats['polar']['ecg'].append(result['polar']['ecg_missing_rate'])
                if 'acc_missing_rate' in result['polar']:
                    device_stats['polar']['acc'].append(result['polar']['acc_missing_rate'])
                if 'hr_missing_rate' in result['polar']:
                    device_stats['polar']['hr'].append(result['polar']['hr_missing_rate'])

        summary_stats = []
        for device, signals in device_stats.items():
            for signal_type, rates in signals.items():
                if rates:
                    valid_rates = [r for r in rates if r is not None]
                    if valid_rates:
                        summary_stats.append({
                            'Device': device,
                            'Signal': signal_type,
                            'Average Missing Rate (%)': np.mean(valid_rates),
                            'Std Missing Rate': np.std(valid_rates),
                            'Median Missing Rate (%)': np.median(valid_rates),
                            'Min Missing Rate (%)': np.min(valid_rates),
                            'Max Missing Rate (%)': np.max(valid_rates),
                            'Sample Count': len(valid_rates)
                        })

        summary_df = pd.DataFrame(summary_stats)
        summary_df.to_csv(os.path.join(self.output_path, 'signal_quality_summary.csv'), index=False)

        detailed_results = []
        for result in all_results:
            participant_detail = {'Participant': result['participant']}

            if result.get('galaxy'):
                participant_detail.update({
                    'Galaxy_PPG_Missing': result['galaxy'].get('ppg_missing_rate'),
                    'Galaxy_ACC_Missing': result['galaxy'].get('acc_missing_rate'),
                    'Galaxy_HR_Missing': result['galaxy'].get('hr_missing_rate')
                })
                if 'ppg_status_dist' in result['galaxy']:
                    for status, count in result['galaxy']['ppg_status_dist'].items():
                        participant_detail[f'Galaxy_PPG_Status_{status}'] = count
                if 'hr_status_dist' in result['galaxy']:
                    for status, count in result['galaxy']['hr_status_dist'].items():
                        participant_detail[f'Galaxy_HR_Status_{status}'] = count

            if result.get('e4'):
                participant_detail.update({
                    'E4_BVP_Missing': result['e4'].get('bvp_missing_rate'),
                    'E4_ACC_Missing': result['e4'].get('acc_missing_rate'),
                    'E4_HR_Missing': result['e4'].get('hr_missing_rate')
                })
                if 'hr_invalid_rate' in result['e4']:
                    participant_detail['E4_HR_Invalid_Rate'] = result['e4']['hr_invalid_rate']

            if result.get('polar'):
                participant_detail.update({
                    'Polar_ECG_Missing': result['polar'].get('ecg_missing_rate'),
                    'Polar_ACC_Missing': result['polar'].get('acc_missing_rate'),
                    'Polar_HR_Missing': result['polar'].get('hr_missing_rate')
                })

            detailed_results.append(participant_detail)

        detailed_df = pd.DataFrame(detailed_results)
        detailed_df.to_csv(os.path.join(self.output_path, 'signal_quality_detailed.csv'), index=False)
def main():
    analyzer = MissingRateAnalyzer()
    results = analyzer.analyze_all_participants()
    print("Analysis completed. Results saved in MissingRateAnalysis directory.")
if __name__ == "__main__":
    main()