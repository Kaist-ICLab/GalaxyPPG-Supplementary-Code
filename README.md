# GalaxyPPG Supplementary Code 
### Welcome to the GalaxyPPG Supplementary Code repository. 

This repository provides supplementary code for the GalaxyPPG dataset paper, 
demonstrating basic processing pipelines and potential use cases. 
The code implementations are for demonstration and reproducibility purposes only 
and do not represent optimal solutions for ECG and PPG signal processing. 
# Installation
```bash
git clone https://github.com/Kaist-ICLab/GalaxyPPG-Supplementary-Code.git
cd GalaxyPPG-Supplementary-Code
pip install -r requirements.txt
```
# Repository Structure 
- Step01 Dataset Completeness Analysis
- Step02 Pre-Processing
- Step03 Denoising
- Step04 Peak Detection
- Jupyter Notebook 
## 01 Dataset Completeness Analysis
Located in [Step01_DatasetCompletenessAnalysis](Step01_DatasetCompletenessAnalysis), this section contains scripts for assessing dataset completeness.
### [MissingRate.py](Step01_DatasetCompletenessAnalysis%2FMissingRate.py): 
- Purpose: Calculate and analyze missing data rates across all signals.
- Expected Output: A summary CSV file indicating the percentage of missing data per participant and device signal.
### Usage Guide:
```python
from MissingRate import MissingRateAnalyzer

# Initialize the analyzer
>>> analyzer = MissingRateAnalyzer()

# Analyze a single participant
>>> result = analyzer.analyze_participant("P01")
>>> print(result['galaxy']['ppg_missing_rate'])
21.43

# Get detailed metrics for the PPG signal
>>> ppg_metrics = result['galaxy']['ppg_metrics']
>>> print(ppg_metrics['nan_rate'])
15.2
>>> print(ppg_metrics['weak_signal_detected'])
False

# Analyze all participants and save results
>>> all_results = analyzer.analyze_all_participants()

# Access summary statistics for different devices
>>> for the device in ['galaxy', 'e4', 'polar']:
...     if the device in result:
...         for signal in result[device]:
...             if 'missing_rate' in signal:
...                 print(f"{device} {signal}: {result[device][signal]}")
galaxy ppg_missing_rate: 21.43
galaxy acc_missing_rate: 0.52
galaxy hr_missing_rate: 18.76
e4 bvp_missing_rate: 0.84
e4 acc_missing_rate: 0.31
polar ecg_missing_rate: 0.12
```
### [SamplingRate.py](Step01_DatasetCompletenessAnalysis%2FSamplingRate.py):
- Purpose: Validate and analyze sampling rates for each device’s signals.
- Expected Outputs:
  - detailed_sampling_analysis.csv: Detailed per-session sampling info.
  - global_sampling_statistics: Global Sampling Rate.
  - sampling_rate_summary.csv: Summary of observed vs. expected sampling rates.
  - session_sampling_summary.csv: Session-level statistics.
  - data_availability_report.csv: Highlights any data availability issues.
### Classes:
- BaseSamplingAnalyzer: Abstract base class defining core analysis methods
- Device-specific analyzers:
  - GalaxyWatchAnalyzer 
  - E4Analyzer
  - PolarH10Analyzer
- OverallSamplingRateAnalyzer: Main coordinator class managing device analyzers
### Basic Usage:
``` python
# Basic usage of OverallSamplingRateAnalyzer
from SamplingRate import OverallSamplingRateAnalyzer

# Initialize the analyzer
>>> analyzer = OverallSamplingRateAnalyzer()

# Process all participants
>>> analyzer.process_all_participants()
Starting analysis for 24 participants...
Processing participant P01 (1/24)
Processing participant P02 (2/24)
...
Analysis completed. Results saved in: /path/to/SamplingRateAnalysis

# Process specific participant
>>> analyzer.process_specific_participant("P03")
Processing participant: P03
Analysis completed for P03. Results saved in: /path/to/SamplingRateAnalysis
```
### Device-Specific Analyzers Usage Examples
```python
# Device-specific analyzers
from SamplingRate import GalaxyWatchAnalyzer, E4Analyzer, PolarH10Analyzer

# Galaxy Watch Analysis
>>> galaxy_analyzer = GalaxyWatchAnalyzer()
>>> configs = galaxy_analyzer.get_signal_configs()
>>> print(configs['ppg']['rate'])
25  # Expected sampling rate for PPG

# Example of processing Galaxy Watch data
>>> import pandas as pd
>>> data = {
...     'timestamp': [1000, 1040, 1080, 1120],
...     'ppg': [100, 102, 98, 103]
... }
>>> df = pd.DataFrame(data)
>>> duration = 0.12  # seconds
>>> stats = galaxy_analyzer.calculate_sampling_rate(
...     df['timestamp'].values,
...     duration,
...     expected_rate=25
... )
>>> print(stats)
{
    'actual_rate': 33.33,
    'expected_samples': 3,
    'actual_samples': 4,
    'coverage_ratio': 1.33
}

# E4 Analysis
>>> e4_analyzer = E4Analyzer()
>>> configs = e4_analyzer.get_signal_configs()
>>> print(configs['bvp']['rate'])
64  # Expected sampling rate for BVP

# Polar H10 Analysis
>>> polar_analyzer = PolarH10Analyzer()
>>> configs = polar_analyzer.get_signal_configs()
>>> print(configs['ecg']['rate'])
130  # Expected sampling rate for ECG

# Example of timestamp adjustment for each device
>>> timestamps = pd.Series([1000000, 2000000, 3000000])
>>> # E4 timestamps (microseconds to milliseconds)
>>> e4_adjusted = e4_analyzer.adjust_timestamp(timestamps)
>>> print(e4_adjusted.iloc[0])
1000.0
```
## 02 Pre Processing

Located in [Step02_PreProcessing](Step02_PreProcessing), this section handles signal segmentation.

### [Window_Segment.py](Step02_PreProcessing%2FWindow_Segment.py): 
Purpose: Segment signals into fixed-length windows for downstream analysis.
Expected Outputs: A CSV file containing segmented signals (PPG, ECG, ACC) for each participant and session.
- Configurable parameters:
  - window_size: Duration of each analysis window (default: 8 seconds)
  - window_step: Sliding step between consecutive windows (default: 2 seconds)
```python
from window_segment import WindowProcessor
import os
import pandas as pd

# Basic Usage Example
>>> # Initialize the processor with the dataset path
>>> dataset_path = "/path/to/dataset"
>>> processor = WindowProcessor(dataset_path)

# Example 1: Process a specific participant
>>> # Process data for participant P01
>>> processor.process_specific_participant("P01")
Processing participant: P01
Found session: baseline, Duration: 300.50 seconds
Found session: tsst-speech, Duration: 180.25 seconds
...
Created 146 windows for session baseline
Created 89 windows for session tsst-speech
...
Saved processed data to: /path/to/output/P01_processed.csv
Total windows created: 235

# Example 2: Process all participants
>>> # Process all participants in the dataset
>>> processor.process_all_participants()
Processing participant: P01
...
Processing participant: P24

# Example 3: Custom window configuration
>>> # Create a processor with custom window parameters
>>> processor = WindowProcessor(dataset_path)
>>> processor.window_size = 10000  # 10 seconds
>>> processor.window_step = 5000   # 5 seconds
>>> processor.process_specific_participant("P01")

# Example 4: Accessing processed data
>>> # Read processed windows
>>> output_file = os.path.join(processor.output_path, "P01_processed.csv")
>>> processed_data = pd.read_csv(output_file)
>>> 
>>> # View window information
>>> print(processed_data[['session', 'windowNumber', 'startTime', 'endTime']].head())
         session  windowNumber     startTime       endTime
0      baseline            1  1621234567890  1621234575890
1      baseline            2  1621234569890  1621234577890
2      baseline            3  1621234571890  1621234579890
3      baseline            4  1621234573890  1621234581890
4      baseline            5  1621234575890  1621234583890

# Example 5: Processing specific sessions
>>> # Access validation events
>>> data, events = processor.read_participant_data("P01")
>>> valid_sessions = processor.get_valid_sessions(events)
>>> 
>>> # Print session information
>>> for session in valid_sessions:
...     print(f"Session: {session['name']}")
...     print(f"Start: {session['start']}")
...     print(f"End: {session['end']}")
...     print(f"Duration: {(session['end'] - session['start'])/1000:.2f} seconds")
Session: baseline
Start: 1621234567890
End: 1621234867890
Duration: 300.00 seconds
...

# Example 6: Extract window data
>>> # Extract specific window data
>>> window_data = processor.create_window(
...     participant="P01",
...     session_name="baseline",
...     window_number=1,
...     window_start=1621234567890,
...     window_end=1621234575890,
...     data=data
... )
>>> 
>>> # Access window data
>>> print(f"Session: {window_data['session']}")
>>> print(f"Window Number: {window_data['windowNumber']}")
>>> print(f"Heart Rate: {window_data['polarHR']}")
Session: baseline
Window Number: 1
Heart Rate: 72.5
```
## 03 Denoising 
Located in [Step03_Denosing](Step03_Denosing), this section handles signal denoising.

For optimal performance, we recommend referring to the original papers for algorithm details and considering state-of-the-art approaches in the field.
### [IMAT_Denosing.py](Step03_Denosing%2FIMAT_Denosing.py)
This file implements the Iterative Method with Adaptive Thresholding (IMAT) approach for motion artifact reduction, based on spectral analysis and sparse signal reconstruction
#### Usage Example 1: DataFrame-based Processing
``` python
from imat_denoising import IMATDenoising
import numpy as np
import pandas as pd

# Initialize the denoiser
>>> denoiser = IMATDenoising()

# Process Galaxy Watch data (25Hz sampling rate)
>>> ppg_data = np.random.randn(200)  # 8 seconds at 25Hz 
>>> acc_data = np.random.randn(200, 3)  # 3-axis accelerometer data

# Process signal
>>> denoised_signal, heart_rate = denoiser.process_galaxy(
...     ppg_signal,    # PPG signal array
...     acc_x,         # X-axis acceleration array
...     acc_y,         # Y-axis acceleration array 
...     acc_z          # Z-axis acceleration array
... )

# Process full DataFrame
>>> df = pd.DataFrame({
...     'ppg': [';'.join(map(str, ppg_data))],
...     'acc_x': [';'.join(map(str, acc_data[:,0]))],
...     'acc_y': [';'.join(map(str, acc_data[:,1]))],
...     'acc_z': [';'.join(map(str, acc_data[:,2]))]
... })

>>> processed_df = denoiser.process_dataframe(
...     df,
...     ppg_col='ppg',
...     acc_cols=['acc_x', 'acc_y', 'acc_z']
... )

The `process_dataframe` method returns a DataFrame with additional columns:
- `denoised_signal`: Denoised PPG signal as semicolon-separated values
- `heart_rate`: Estimated heart rate in BPM
```
#### Usage Example 2: File-based Processing
``` python
from imat_denoising import process_dataset

# Process specific participants
>>> process_dataset(participant_range=(22, 24))

# Process a list of participants  
>>> participant_list = ["P03", "P04"]
>>> process_dataset(participant_list=participant_list)

# Process all participants
>>> process_dataset()

The file-based processing creates the following output files:
- `{participant_id}_processed_GD_denoised.csv`: Contains denoised signals and heart rate estimates
- Location: `RESULTS_DIR/IMAT/{filename}`
```
#### References
Mashhadi, M.B., Asadi, E., Eskandari, M., Kiani, S. and Marvasti, F., 2015. Heart rate tracking using wrist-type photoplethysmographic (PPG) signals during physical exercise with simultaneous accelerometry. IEEE Signal Processing Letters, 23(2), pp.227-231.
### [Wiener_Denosing.py](Step03_Denosing%2FWiener_Denosing.py)
This file provides the Wiener filtering implementation with adaptive noise estimation for PPG signals during physical activity
#### Usage Example 1: DataFrame-based Processing
``` python
from wiener_denoising import WienerDenoising  
import numpy as np
import pandas as pd

# Initialize denoiser
>>> denoiser = WienerDenoising()

# Process Galaxy Watch data (25Hz sampling rate)
>>> ppg_data = np.random.randn(200)  # 8 seconds at 25Hz
>>> acc_data = np.random.randn(200, 3)  # 3-axis accelerometer data 

# Process signal
>>> denoised_signal, heart_rate = denoiser.process_galaxy(
...     ppg_data,    # PPG signal array
...     acc_data[:,0],  # X-axis acceleration array
...     acc_data[:,1],  # Y-axis acceleration array  
...     acc_data[:,2]   # Z-axis acceleration array
... )

# Process full DataFrame
>>> df = pd.DataFrame({
...     'ppg': [';'.join(map(str, ppg_data))],
...     'acc_x': [';'.join(map(str, acc_data[:,0]))], 
...     'acc_y': [';'.join(map(str, acc_data[:,1]))],
...     'acc_z': [';'.join(map(str, acc_data[:,2]))]
... })

>>> processed_df = denoiser.process_dataframe(
...     df,
...     ppg_col='ppg',
...     acc_cols=['acc_x', 'acc_y', 'acc_z']
... )

The `process_dataframe` method returns a DataFrame with additional columns:
- `denoised_signal`: Denoised PPG signal as semicolon-separated values
- `heart_rate`: Estimated heart rate in BPM
```
#### Usage Example 2: File-based Processing
``` python
from wiener_denoising import process_dataset

# Process specific participants
>>> process_dataset(participant_range=(22, 24))

# Process a list of participants
>>> participant_list = ["P03", "P04"]
>>> process_dataset(participant_list=participant_list)

# Process all participants
>>> process_dataset()

The file-based processing creates the following output files:
- `{participant_id}_processed_GD_denoised.csv`: Contains denoised signals and heart rate estimates  
- Location: `RESULTS_DIR/Wiener/{filename}`
```
#### References
WFPV: Temko, A., 2017. Accurate heart rate monitoring during physical exercises using PPG. IEEE Transactions on Biomedical Engineering, 64(9), pp.2016-2024.
### [Kalman_Denoising.py](Step03_Denosing%2FKalman_Denoising.py)
This file contains the Kalman filter implementation for adaptive motion artifact reduction
#### Usage Example 1: DataFrame-based Processing
``` python
from kalman_denoising import KalmanDenoising
import numpy as np
import pandas as pd

# Initialize denoiser for Galaxy Watch
>>> denoiser = KalmanDenoising(device_type='galaxy')

# Create sample data
>>> ppg_data = np.random.randn(200)  # 8 seconds at 25Hz
>>> acc_data = np.random.randn(200, 3)  # 3-axis accelerometer data

# Process signal
>>> result = denoiser.process_signal(ppg_data, acc_data)
>>> denoised_signal = result['denoised_signal']
>>> heart_rate = result['heart_rate']

# Process full DataFrame
>>> df = pd.DataFrame({
...     'ppg': [';'.join(map(str, ppg_data))],
...     'acc_x': [';'.join(map(str, acc_data[:,0]))],
...     'acc_y': [';'.join(map(str, acc_data[:,1]))],
...     'acc_z': [';'.join(map(str, acc_data[:,2]))]
... })

>>> processed_df = denoiser.process_dataframe(
...     df,
...     ppg_col='ppg',
...     acc_cols=['acc_x', 'acc_y', 'acc_z']
... )

The `process_dataframe` method returns a DataFrame with additional columns:
- `denoised_signal`: Denoised PPG signal as semicolon-separated values
- `heart_rate`: Estimated heart rate in BPM

``` 
#### Usage Example 2: File-based Processing
``` python
from kalman_denoising import process_dataset

# Process specific participants
>>> process_dataset(participant_range=(22, 24))

# Process a list of participants
>>> participant_list = ["P03", "P04"]
>>> process_dataset(participant_list=participant_list)

# Process all participants
>>> process_dataset()

The file-based processing creates the following output files:
- `{participant_id}_processed_GD_denoised.csv`: Contains denoised signals and heart rate estimates
- Location: `RESULTS_DIR/Kalman/{filename}`

```
#### References
Seyedtabaii, S.S.A.L. and Seyedtabaii, L., 2008. Kalman filter based adaptive reduction of motion artifact from photoplethysmographic signal. World Acad. Sci. Eng. Technol, 37(2), pp.173-176.
### [SVD_Denosing.py](Step03_Denosing%2FSVD_Denosing.py)
This file implements Singular Value Decomposition-based signal decomposition for artifact removal
#### Usage Example 1: DataFrame-based Processing
``` python
from svd_denoising import SVDDenoising
import numpy as np
import pandas as pd

# Initialize denoiser
>>> denoiser = SVDDenoising()

# Process Galaxy Watch data (25Hz sampling rate)
>>> ppg_data = np.random.randn(200)  # 8 seconds at 25Hz
>>> acc_data = np.random.randn(200, 3)  # 3-axis accelerometer data

# Process signal
>>> denoised_signal, heart_rate = denoiser.denoise(
...     ppg_data,     # PPG signal array
...     acc_data[:,0],  # X-axis acceleration array
...     acc_data[:,1],  # Y-axis acceleration array
...     acc_data[:,2],  # Z-axis acceleration array
...     fs=25         # Sampling frequency
... )

# Process full DataFrame
>>> df = pd.DataFrame({
...     'ppg': [';'.join(map(str, ppg_data))],
...     'acc_x': [';'.join(map(str, acc_data[:,0]))],
...     'acc_y': [';'.join(map(str, acc_data[:,1]))],
...     'acc_z': [';'.join(map(str, acc_data[:,2]))]
... })

>>> processed_df = denoiser.process_dataframe(
...     df,
...     ppg_col='ppg',
...     acc_cols=['acc_x', 'acc_y', 'acc_z']
... )

The `process_dataframe` method returns a DataFrame with additional columns:
- `denoised_signal`: Denoised PPG signal as semicolon-separated values
- `heart_rate`: Estimated heart rate in BPM
```
#### Usage Example 2: File-based Processing
``` python
from svd_denoising import process_dataset

# Process specific participants
>>> process_dataset(participant_range=(22, 24))

# Process a list of participants
>>> participant_list = ["P03", "P04"]
>>> process_dataset(participant_list=participant_list)

# Process all participants
>>> process_dataset()

The file-based processing creates the following output files:
- `{participant_id}_processed_GD_denoised.csv`: Contains denoised signals and heart rate estimates
- Location: `RESULTS_DIR/SVD/{filename}`
```
#### References
Reddy, K.A. and Kumar, V.J., 2007, May. Motion artifact reduction in photoplethysmographic signals using singular value decomposition. In 2007 IEEE Instrumentation & Measurement Technology Conference IMTC 2007 (pp. 1-4). IEEE.



# 04 Peak Detection
Located in [Step04_PeakDetection](Step04_PeakDetection), this section uses the HeartPy toolkit (https://python-heart-rate-analysis-toolkit.readthedocs.io/en/latest/) to detect peaks and compute HR/HRV metrics.

### [ECG_Calculation_basing_HeartPy.py](Step04_PeakDetection%2FECG_Calculation_basing_HeartPy.py): 
This file calculates HRV metrics from ECG signals using HeartPy.
It also utilizes customized bandpass filtering parameters from [filter_parameters](Step04_PeakDetection%2Ffilter_parameters).

This module calculates key ECG and HRV metrics:

- Heart Rate (HR): Beats per minute 
- Inter-Beat Interval (IBI): Time between consecutive heartbeats in milliseconds 
- SDNN: Standard deviation of NN intervals, reflects overall heart rate variability 
- RMSSD: Root mean square of successive differences, indicates beat-to-beat variability 
- R-peak Locations: Sample points where R-peaks were detected

#### Usage Example 1: Using DataFrame 
``` python
from ECG_Calculation_basing_HeartPy import ECGAnalyzer

# Initialize analyzer
analyzer = ECGAnalyzer()

# Create sample DataFrame
>>> import pandas as pd
>>> df = pd.DataFrame({
...     'polarECG': ['0.1;0.2;0.3;0.4;0.5', '0.2;0.3;0.4;0.5;0.6'],
...     'polarHR': [75, 80],
...     'windowNumber': [1, 2],
...     'session': ['baseline', 'baseline']
... })

>>> filter_params = pd.DataFrame({
...     'windowNumber': [1, 2],
...     'session': ['baseline', 'baseline'],
...     'lowCutoff': [5, 5],
...     'highCutoff': [15, 15]
... })

# Process data and examine all metrics
>>> processed_df, stats = analyzer.process_dataframe(df, filter_params)

# Accessing different metrics
>>> print("Heart Rate (BPM):", processed_df['gdHR'].iloc[0])
Heart Rate (BPM): 75.2

>>> print("Inter-Beat Interval (ms):", processed_df['gdIBI'].iloc[0])
Inter-Beat Interval (ms): 798.5

>>> print("SDNN (ms):", processed_df['gdSDNN'].iloc[0])
SDNN (ms): 45.3

>>> print("RMSSD (ms):", processed_df['gdRMSSD'].iloc[0])
RMSSD (ms): 35.8

>>> print("R-peak Locations:", processed_df['gdPeaks'].iloc[0])
R-peak Locations: 32;152;272;392;512
```
#### Usage Example 2: Processing from Files 
``` python
from ECG_Calculation_basing_HeartPy import ECGAnalyzer, load_participant_data, load_filter_parameters

# Initialize analyzer
analyzer = ECGAnalyzer()

# Load and process data
>>> participant_id = "P01"
>>> df = load_participant_data("path/to/window_data", participant_id)
>>> filter_params = load_filter_parameters("path/to/filter_params", participant_id)
>>> processed_df, stats = analyzer.process_dataframe(df, filter_params)

# Example of analyzing metrics across sessions
>>> metrics_summary = processed_df.groupby('session').agg({
...     'gdHR': ['mean', 'std'],
...     'gdIBI': ['mean', 'std'],
...     'gdSDNN': 'mean',
...     'gdRMSSD': 'mean'
... }).round(2)

>>> print("\nMetrics Summary by Session:")
>>> print(metrics_summary)
                 gdHR          gdIBI          gdSDNN  gdRMSSD
                mean    std    mean    std    mean    mean
session                                                    
baseline        72.5   3.2    827.4  35.6    42.3    38.5
exercise        95.3   8.7    630.2  57.8    65.7    45.2
recovery        78.4   4.1    765.3  40.2    48.9    40.1

# Accessing specific windows
>>> print("\nSingle Window Analysis:")
>>> window = processed_df.iloc[0]
>>> print(f"Window {window['windowNumber']} in {window['session']}:")
>>> print(f"- Heart Rate: {window['gdHR']:.1f} BPM")
>>> print(f"- IBI: {window['gdIBI']:.1f} ms")
>>> print(f"- SDNN: {window['gdSDNN']:.1f} ms")
>>> print(f"- RMSSD: {window['gdRMSSD']:.1f} ms")
>>> print(f"- Peak Count: {len(str(window['gdPeaks']).split(';'))}")
```

### [PPG_Calculation_basing_HeartPy.py](Step04_PeakDetection%2FPPG_Calculation_basing_HeartPy.py): 
This file handles the processing of denoised signals using HeartPy and also measures the difference between PPG-derived and ECG-derived HRV metrics.

This module calculates the following metrics for both Galaxy Watch and E4:

- Heart Rate (HR): Beats per minute
- Inter-Beat Interval (IBI): Time between consecutive heartbeats in milliseconds 
- SDNN: Standard deviation of NN intervals for overall variability assessment 
- RMSSD: Root mean square of successive differences for short-term variability 
- Peak Matching Rate (PMR): Percentage of PPG peaks matching ECG R-peaks 
- Mean Absolute Error (MAE): Average absolute difference from ECG reference

#### Usage Example 1: Using DataFrame 
``` python
from PPG_Calculation_basing_HeartPy import PPGAnalyzer

# Initialize analyzer
analyzer = PPGAnalyzer()

# Create sample DataFrame
>>> import pandas as pd
>>> df = pd.DataFrame({
...     'denoisedGalaxy': ['0.1;0.2;0.3;0.4;0.5', '0.2;0.3;0.4;0.5;0.6'],
...     'denoisedE4': ['0.15;0.25;0.35;0.45;0.55', '0.25;0.35;0.45;0.55;0.65'],
...     'gdHR': [75, 80],
...     'gdPeaks': ['100;200;300', '150;250;350'],
...     'session': ['baseline', 'baseline']
... })

# Process data and examine metrics
>>> results = analyzer.analyze_data(df)

# Access device-specific metrics
>>> baseline_results = results['baseline']

# Galaxy Watch metrics
>>> print("\nGalaxy Watch Metrics:")
>>> print(f"Heart Rate (BPM): {baseline_results['Galaxy']['HR'][0]}")
Heart Rate (BPM): 74.8
>>> print(f"IBI (ms): {baseline_results['Galaxy']['IBI'][0]}")
IBI (ms): 802.1
>>> print(f"SDNN (ms): {baseline_results['Galaxy']['SDNN'][0]}")
SDNN (ms): 42.5
>>> print(f"RMSSD (ms): {baseline_results['Galaxy']['RMSSD'][0]}")
RMSSD (ms): 38.2
>>> print(f"Peak Matching Rate (%): {baseline_results['Galaxy']['PMR'][0]}")
Peak Matching Rate (%): 95.3

# E4 metrics
>>> print("\nE4 Metrics:")
>>> print(f"Heart Rate (BPM): {baseline_results['E4']['HR'][0]}")
Heart Rate (BPM): 75.2
>>> print(f"IBI (ms): {baseline_results['E4']['IBI'][0]}")
IBI (ms): 797.8
>>> print(f"SDNN (ms): {baseline_results['E4']['SDNN'][0]}")
SDNN (ms): 43.1
>>> print(f"RMSSD (ms): {baseline_results['E4']['RMSSD'][0]}")
RMSSD (ms): 37.9
>>> print(f"Peak Matching Rate (%): {baseline_results['E4']['PMR'][0]}")
Peak Matching Rate (%): 94.8
```
#### Usage Example 2: Processing from Files 
``` python
from PPG_Calculation_basing_HeartPy import PPGAnalyzer, load_participant_data

# Initialize analyzer
analyzer = PPGAnalyzer()

# Load and process data
>>> participant_id = "P01"
>>> df = load_participant_data("path/to/window_dir", "path/to/results_dir", participant_id)
>>> results = analyzer.analyze_data(df)

# Generate summary statistics 
>>> summary = analyzer.summarize_results(results)

# Example of accessing summarized metrics across sessions
>>> print("\nActivity-wise Analysis:")

BASELINE Session:

Galaxy Metrics:
Mean HR: 72.5 ± 3.2 BPM
Mean IBI: 827.4 ± 35.6 ms  
Mean SDNN: 42.3 ms
Mean RMSSD: 38.5 ms
Mean PMR: 95.2%

E4 Metrics:
Mean HR: 73.1 ± 3.4 BPM
Mean IBI: 821.3 ± 37.8 ms
Mean SDNN: 43.1 ms 
Mean RMSSD: 39.2 ms
Mean PMR: 94.8%

WALKING Session:

Galaxy Metrics:
Mean HR: 95.3 ± 8.7 BPM
Mean IBI: 630.2 ± 57.8 ms
Mean SDNN: 65.7 ms
Mean RMSSD: 45.2 ms 
Mean PMR: 92.5%

E4 Metrics:
Mean HR: 96.2 ± 8.9 BPM
Mean IBI: 623.7 ± 58.2 ms
Mean SDNN: 67.3 ms
Mean RMSSD: 46.8 ms
Mean PMR: 91.8%

RUNNING Session:

Galaxy Metrics: 
Mean HR: 154.2 ± 12.5 BPM
Mean IBI: 389.1 ± 31.5 ms
Mean SDNN: 82.3 ms
Mean RMSSD: 51.6 ms
Mean PMR: 85.7%

E4 Metrics:
Mean HR: 155.8 ± 13.1 BPM
Mean IBI: 385.1 ± 32.7 ms
Mean SDNN: 84.5 ms
Mean RMSSD: 53.2 ms
Mean PMR: 84.2%

# Save detailed results to CSV files
>>> analyzer.save_results(summary, "output/directory")
Analysis complete. Results saved to: output/directory/complete_analysis_results.csv
Analysis complete. Results saved to: output/directory/mae_analysis_results.csv
```

### [raw_ppg_heartpy.py](Step04_PeakDetection/raw_ppg_heartpy.py)

This file implements PPG signal processing and HRV metric calculation from raw PPG signals using HeartPy, without any denoising preprocessing. It processes both Galaxy Watch PPG and Empatica E4 BVP signals to calculate heart rate and HRV metrics, comparing them against ECG ground truth.

Key features:
- Processes both Galaxy Watch (25 Hz) and E4 (64 Hz) PPG signals
- Calculates HR, IBI, SDNN, RMSSD from PPG peaks
- Implements peak matching with ECG R-peaks to calculate Peak Matching Rate (PMR)
- Provides session-by-session analysis across different activities
- Compares metrics against ECG ground truth

#### Usage Example:

```python
from raw_ppg_heartpy import rawPPGHeartPy

# Initialize analyzer 
analyzer = rawPPGHeartPy()

# Create sample DataFrame
import pandas as pd
df = pd.DataFrame({
    'galaxyPPG': ['1.2;1.3;1.4;1.5;1.6', '1.3;1.4;1.5;1.6;1.7'],
    'e4BVP': ['0.1;0.2;0.3;0.4;0.5', '0.2;0.3;0.4;0.5;0.6'],
    'gdHR': [75, 80],
    'gdIBI': [800, 750],
    'gdSDNN': [45, 50],
    'gdRMSSD': [35, 40],
    'gdPeaks': ['32;152;272;392;512', '35;155;275;395;515'],
    'session': ['baseline', 'baseline']
})

# Process data and get results
results = analyzer.analyze_data(df)

# Access metrics for a specific session and device
baseline_results = results['baseline']

# Galaxy Watch metrics
print("Galaxy HR (BPM):", np.mean(baseline_results['Galaxy']['HR']))
print("Galaxy IBI (ms):", np.mean(baseline_results['Galaxy']['IBI']))
print("Galaxy SDNN (ms):", np.mean(baseline_results['Galaxy']['SDNN']))
print("Galaxy RMSSD (ms):", np.mean(baseline_results['Galaxy']['RMSSD']))
print("Galaxy PMR (%):", np.mean(baseline_results['Galaxy']['PMR']))

# E4 metrics  
print("E4 HR (BPM):", np.mean(baseline_results['E4']['HR']))
print("E4 IBI (ms):", np.mean(baseline_results['E4']['IBI']))
print("E4 SDNN (ms):", np.mean(baseline_results['E4']['SDNN']))
print("E4 RMSSD (ms):", np.mean(baseline_results['E4']['RMSSD']))  
print("E4 PMR (%):", np.mean(baseline_results['E4']['PMR']))
```

The results are saved to CSV files in the results directory:
- complete_analysis_results.csv: Contains all metrics across sessions and devices
- mae_analysis_results.csv: Contains Mean Absolute Error compared to ECG ground truth


### [hand_placement_analyzer.py](Step04_PeakDetection/hand_placement_analyzer.py)

This module analyzes the impact of device placement (left vs right hand) on PPG signal quality and heart rate metric accuracy. It processes both Galaxy Watch and E4 data to compare measurement accuracy based on hand placement, using ECG data from PolarH10 as ground truth.

Key features:
- Analyzes left vs right hand placement effects on PPG measurements
- Compares HR, IBI, SDNN, RMSSD accuracy across hand placements
- Performs statistical analysis (paired t-tests) to assess placement significance
- Generates comprehensive reports for overall, session-specific, and participant-level analyses
- Uses Meta.csv to determine device placement for each participant

#### Usage Example:

```python
from hand_placement_analyzer import HandPlacementMAEAnalyzer

# Initialize analyzer
analyzer = HandPlacementMAEAnalyzer()

# Run complete analysis
analyzer.run_analysis()

# Process specific participant
analyzer.process_participant_data(df, "P02")

# Access participant data
participant_data = analyzer.load_participant_data("P02")

# Example of accessing results manually
results = analyzer.analyze_participant_data(participant_data, "P02")
print("Left hand HR MAE:", np.mean(results['left_hand_maes']['HR']))
print("Right hand HR MAE:", np.mean(results['right_hand_maes']['HR']))
```

The analyzer generates three main report files in the results directory:

1. overall_hand_placement_comparison.csv:
- Aggregated metrics comparing left vs right placement
- Overall MAE statistics for each metric
- Better performing hand for each metric

2. session_hand_placement_comparison.csv:
- Session-specific comparisons of hand placement effects
- Statistical significance tests for each session
- MAE differences between left and right placement

3. participant_hand_placement_means.csv:
- Individual participant-level analysis
- Mean HR MAE for left and right placement
- Participant-specific better performing hand
- Paired t-test results for placement significance

These reports help understand how device placement affects measurement accuracy across different activities and participants.

Example output structure:
```json
{
    "participant": "P02",
    "galaxy_hand": "R",
    "session_results": {
        "baseline": {
            "left_hand_maes": {
                "HR": [2.5, 3.1, 2.8],
                "IBI": [25.3, 28.4, 26.1],
                "SDNN": [12.4, 13.2, 11.9],
                "RMSSD": [15.6, 16.2, 14.8]
            },
            "right_hand_maes": {
                "HR": [2.1, 2.8, 2.4],
                "IBI": [22.1, 24.5, 23.2],
                "SDNN": [11.2, 12.1, 10.8],
                "RMSSD": [14.2, 15.1, 13.7]
            }
        }
        // Additional sessions...
    }
}
```