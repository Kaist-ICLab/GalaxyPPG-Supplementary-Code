# GalaxyPPG Supplementary Code 
### Welcome to the GalaxyPPG Supplementary Code repository. 

This repository provides supplementary code for the GalaxyPPG dataset paper, 
demonstrating basic processing pipelines and potential use cases. 
The code implementations are for demonstration and reproducibility purposes only 
and do not represent optimal solutions for ECG and PPG signal processing. 
### We encourage researchers to develop and apply more advanced algorithms using this dataset.
The provided code assists in data completeness checks, pre-processing, denoising, and peak detection, helping researchers efficiently navigate and analyze the dataset.
# Installation
```bash
git clone https://github.com/Kaist-ICLab/GalaxyPPG-Supplementary-Code.git
cd GalaxyPPG-Supplementary-Code
pip install -r requirements.txt
```
# Repository Structure
- 01 Dataset Completeness Analysis
- 02 Pre-Processing
- 03 Denoising
- 04 Peak Detection
## 01 Dataset Completeness Analysis
Located in [01_DatasetCompletenessAnalysis](01_DatasetCompletenessAnalysis), this section contains scripts for assessing dataset completeness.
### [MissingRate.py](00_RawDataCheck%2F00_MissingRate.py): 
- Purpose: Calculate and analyze missing data rates across all signals.
- Expected Output: A summary CSV file indicating the percentage of missing data per participant and device signal.
### [SamplingRate.py](01_DatasetCompletenessAnalysis%2FSamplingRate.py):
- Purpose: Validate and analyze sampling rates for each device’s signals.
- Expected Outputs:
  - detailed_sampling_analysis.csv: Detailed per-session sampling info.
  - sampling_rate_summary.csv: Summary of observed vs. expected sampling rates.
  - session_sampling_summary.csv: Session-level statistics.
  - data_availability_report.csv: Highlights any data availability issues.
### Key Classes:
- BaseSamplingAnalyzer: Abstract base class defining core analysis methods
- Device-specific analyzers:
  - GalaxyWatchAnalyzer 
  - E4Analyzer
  - PolarH10Analyzer
- OverallSamplingRateAnalyzer: Main coordinator class managing device analyzers
### Basic Usage:
``` python
from SamplingRate import OverallSamplingRateAnalyzer

# Create an analyzer instance
analyzer = OverallSamplingRateAnalyzer()

# Process all participant's
analyzer.process_all_participants()

# Process specific participant
analyzer.process_specific_participant("P01")
```
### Device-Specific Analyzers Usage Examples
```python
# Galaxy Watch Analyzer
from SamplingRate import GalaxyWatchAnalyzer

galaxy_analyzer = GalaxyWatchAnalyzer()
# Expected sampling rates:
# PPG: 25Hz
# ACC: 25Hz
# HR: 1Hz
# Skin Temperature: 1/60Hz (once per minute)

# E4 Analyzer
from SamplingRate import E4Analyzer

e4_analyzer = E4Analyzer()
# Expected sampling rates:
# BVP: 64Hz
# ACC: 32Hz
# Temperature: 4Hz
# HR: 1Hz

# Polar H10 Analyzer
from SamplingRate import PolarH10Analyzer

polar_analyzer = PolarH10Analyzer()
# Expected sampling rates:
# ECG: 130Hz
# ACC: 200Hz
# HR: 1Hz
```
## 02 Pre Processing

Located in [02_PreProcessing](02_PreProcessing), this section handles signal segmentation.

### [Window_Segment.py](02_PreProcessing%2FWindow_Segment.py): 
Purpose: Segment signals into fixed-length windows for downstream analysis.

Expected Outputs: A CSV file containing segmented signals (PPG, ECG, ACC) for each participant and session.

- Configurable parameters:
  - window_size: Duration of each analysis window (default: 8 seconds)
  - window_step: Sliding step between consecutive windows (default: 2 seconds)
## 03 Denoising 
Located in [03_Denosing](03_Denosing), this section handles signal denoising.

For optimal performance, we recommend referring to the original papers for algorithm details and considering state-of-the-art approaches in the field.

### [IMAT_Denosing.py](03_Denosing%2FIMAT_Denosing.py)

This file implements the Iterative Method with Adaptive Thresholding (IMAT) approach for motion artifact reduction, based on spectral analysis and sparse signal reconstruction
### Usage Guide
```python
from denoising.imat import IMATDenoising
# Initialize the denoiser
denoiser = IMATDenoising()

# Process Galaxy Watch data (25Hz sampling rate)
denoised_signal, heart_rate = denoiser.process_galaxy(
    ppg_signal,    # PPG signal array
    acc_x,         # X-axis acceleration array 
    acc_y,         # Y-axis acceleration array
    acc_z          # Z-axis acceleration array
)

# Process E4 data (64Hz sampling rate)
denoised_signal, heart_rate = denoiser.process_e4(
    bvp_signal,    # BVP signal array
    acc_x,         # X-axis acceleration array
    acc_y,         # Y-axis acceleration array
    acc_z          # Z-axis acceleration array
)
```
### [Wiener_Denosing.py](03_Denosing%2FWiener_Denosing.py)
This file provides the Wiener filtering implementation with adaptive noise estimation for PPG signals during physical activity
### Usage Guide
```` python
from denoising.wiener import WienerDenoising

# Initialize the denoiser
denoiser = WienerDenoising()

# Process Galaxy Watch data
denoised_signal, heart_rate = denoiser.process_galaxy(
    ppg,           # Raw PPG signal
    acc_x, acc_y, acc_z  # Tri-axial acceleration data
)

# Process E4 data 
denoised_signal, heart_rate = denoiser.process_e4(
    ppg,           # Raw PPG signal
    acc_x, acc_y, acc_z  # Tri-axial acceleration data
)
````
### [Kalman_Denoising.py](03_Denosing%2FKalman_Denoising.py)
This file contains the Kalman filter implementation for adaptive motion artifact reduction
### Usage Guide
```` python
from denoising.kalman import KalmanDenoising

# Initialize with device specification
denoiser = KalmanDenoising(device_type='galaxy')  # or 'e4'

# Process signal
result = denoiser.process_signal(
    ppg,           # PPG signal
    acc_data       # Acceleration data (n x 3 array)
)

denoised_signal = result['denoised_signal']
heart_rate = result['heart_rate']
```` 
### [SVD_Denosing.py](03_Denosing%2FSVD_Denosing.py)
This file implements Singular Value Decomposition-based signal decomposition for artifact removal
### Usage Guide
```` python
from denoising.svd import SVDDenoising

# Initialize the denoiser
denoiser = SVDDenoising()

# Process data
denoised = denoiser.denoise(
    ppg,           # PPG signal
    acc_x, acc_y, acc_z,  # Tri-axial acceleration data
    fs             # Sampling rate (25 for Galaxy Watch, 64 for E4)
)
```` 
## REF
This implementation draws from the following foundational works:

- Mashhadi et al. (2015) - IMAT algorithm

- Temko et al. (2017) - Wiener filtering approach

- Seyedtabaii & Seyedtabaii (2008) - Kalman filtering method

- Reddy et al. (2008) - SVD-based artifact reduction
# 04 Peak Detection
Located in [04_PeakDetection](04_PeakDetection).
This section uses the HeartPy toolkit (https://python-heart-rate-analysis-toolkit.readthedocs.io/en/latest/) to detect peaks and compute HR/HRV metrics.

### [ECG_Calculation_basing_HeartPy.py](04_PeakDetection%2FECG_Calculation_basing_HeartPy.py): 
This file calculates HRV metrics from ECG signals using HeartPy.
And it utilizes customized bandpass filtering parameters from [filter_parameters](01_WindowSegment%2Ffilter_parameters).
### [PPG_Calculation_basing_HeartPy.py](04_PeakDetection%2FPPG_Calculation_basing_HeartPy.py): 
This file handles the processing of denoised signals using HeartPy and also measures the difference between PPG-derived and ECG-derived HRV metrics.

### Usage Guide
#### ECG
```` python
from ECG_Calculation_basing_HeartPy import ECGAnalyzer
# Initialize processor
analyzer = ECGAnalyzer(window_data_dir='path/to/window/data')
````
#### PPG
```` python
from PPG_Calculation_basing_HeartPy import PPGAnalyzer
# Initialize analyzer
analyzer = PPGAnalyzer(window_dir='path/to/window/data')
````