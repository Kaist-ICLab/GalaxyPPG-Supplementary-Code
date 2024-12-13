# GalaxyPPG Supplementary Code
This repository contains tools and scripts for processing and analyzing the GalaxyPPG dataset, focusing on PPG signal processing and heart rate variability analysis.
## 01 Dataset Completeness Analysis
Located in [01_DatasetCompletenessAnalysis](01_DatasetCompletenessAnalysis), this section contains scripts for assessing dataset completeness.

### [MissingRate.py](00_RawDataCheck%2F00_MissingRate.py): 

This file calculates and analyzes missing data rates across all signals

### [SamplingRate.py](01_DatasetCompletenessAnalysis%2FSamplingRate.py):

This file validates and analyzes sampling rates across device signals

- BaseSamplingAnalyzer: Abstract base class defining core analysis methods

- Device-specific analyzers:
  - GalaxyWatchAnalyzer 
  - E4Analyzer
  - PolarH10Analyzer

- SamplingRateAnalyzer: Main coordinator class managing device analyzers


### Basic Usage:
``` python
from SamplingRate import OverallSamplingRateAnalyzer

# Create analyzer instance
analyzer = OverallSamplingRateAnalyzer()

# Process all participants
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

The analyzer generates reports in the SamplingRateAnalysis directory:
- detailed_sampling_analysis.csv: Detailed sampling data
- sampling_rate_summary.csv: Summary by device and signal type
- session_sampling_summary.csv: Session-wise analysis
- data_availability_report.csv: Data availability issues


## 02 Pre Processing

Located in [02_PreProcessing](02_PreProcessing), this section handles signal segmentation.

### [Window_Segment.py](02_PreProcessing%2FWindow_Segment.py): 
This file implements windowed segmentation of physiological signals.

- Configurable parameters:
  - window_size: Duration of each analysis window (default: 8 seconds)
  - window_step: Sliding step between consecutive windows (default: 2 seconds)

And it outputs a CSV file containing segmented signals for all modalities.

## 03 Denoising 

### [IMAT_GalaxyDenosing.py](02_Denosing%2FIMAT_GalaxyDenosing.py): 

This file implements the Iterative Method with Adaptive Thresholding (IMAT) approach for motion artifact reduction, based on spectral analysis and sparse signal reconstruction
### [Wiener_GalaxyDenosing.py](02_Denosing%2FWiener_GalaxyDenosing.py): 
This file provides the Wiener filtering implementation with adaptive noise estimation for PPG signals during physical activity
### [Kalman_GalaxyDenoising.py](02_Denosing%2FKalman_GalaxyDenoising.py): 
This file contains the Kalman filter implementation for adaptive motion artifact reduction
### [SVD_GalaxyDenosing.py](02_Denosing%2FSVD_GalaxyDenosing.py): 
This file implements Singular Value Decomposition-based signal decomposition for artifact removal
### Usage Guide( Taking IMAT as Example )
#### Initialize the denoiser
```python
from denoising.imat import IMATDenoising
denoiser = IMATDenoising()
```
#### Process the signal
```python
### For Galaxy Watch data (25Hz)
denoised_signal, heart_rate = denoiser.process_galaxy(
    ppg_signal,    # Raw PPG signal array
    acc_x,         # X-axis acceleration array
    acc_y,         # Y-axis acceleration array
    acc_z          # Z-axis acceleration array
)
### For E4 data (64Hz)
denoised_signal, heart_rate = denoiser.process_e4(
    bvp_signal,    # BVP signal array
    acc_x,         # X-axis acceleration array
    acc_y,         # Y-axis acceleration array
    acc_z          # Z-axis acceleration array
)
```

This implementation draws from the following foundational works:

- Mashhadi et al. (2015) - IMAT algorithm

- Temko et al. (2017) - Wiener filtering approach

- Seyedtabaii & Seyedtabaii (2008) - Kalman filtering method

- Reddy et al. (2008) - SVD-based artifact reduction
# 04 Peak Detection

Located in [04_PeakDetection](04_PeakDetection).
### [ECG_Calculation_basing_HeartPy.py](04_PeakDetection%2FECG_Calculation_basing_HeartPy.py): 
This file calculates HRV metrics from ECG signals using HeartPy.
And it utilizes customized bandpass filtering parameters from [filter_parameters](01_WindowSegment%2Ffilter_parameters).
### [PPG_Calculation_basing_HeartPy.py](04_PeakDetection%2FPPG_Calculation_basing_HeartPy.py): 
This file handles the processing of denoised signals using HeartPy and also measures the difference between PPG-derived and ECG-derived HRV metrics.
