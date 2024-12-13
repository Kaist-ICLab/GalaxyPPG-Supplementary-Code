# GalaxyPPG Supplementary Code
This repository contains tools and scripts for processing and analyzing the GalaxyPPG dataset, focusing on PPG signal processing and heart rate variability analysis.
# Dataset Completeness Analysis
Located in [00_DatasetCompletenessAnalysis](00_DatasetCompletenessAnalysis), this section contains scripts for assessing dataset completeness:

### [00_MissingRate.py](00_RawDataCheck%2F00_MissingRate.py): 

Calculates and analyzes missing data rates across all signals

### [01_SamplingRate.py](00_DatasetCompletenessAnalysis%2F01_SamplingRate.py):

Validates and analyzes sampling rates across device signals

BaseSamplingAnalyzer: Abstract base class defining core analysis methods

Device-specific analyzers:
- GalaxyWatchAnalyzer 
- E4Analyzer
- PolarH10Analyzer

SamplingRateAnalyzer: Main coordinator class managing device analyzers

# Pre Processing

Located in [01_PreProcessing](01_PreProcessing), this section handles signal segmentation:
## Segmentation

[00_Window_Segment.py](01_PreProcessing%2F00_Window_Segment.py): Implements windowed segmentation of physiological signals

Configurable parameters:

window_size: Duration of each analysis window (default: 8 seconds)

window_step: Sliding step between consecutive windows (default: 2 seconds)

Outputs a CSV file containing segmented signals for all modalities



# Denoising 

## IMAT

[IMAT_GalaxyDenosing.py](02_Denosing%2FIMAT_GalaxyDenosing.py): 

Implements the Iterative Method with Adaptive Thresholding (IMAT) approach for motion artifact reduction, based on spectral analysis and sparse signal reconstruction
## Wiener
[Wiener_GalaxyDenosing.py](02_Denosing%2FWiener_GalaxyDenosing.py): 

Provides the Wiener filtering implementation with adaptive noise estimation for PPG signals during physical activity
## Kalman
[Kalman_GalaxyDenoising.py](02_Denosing%2FKalman_GalaxyDenoising.py): 

Contains the Kalman filter implementation for adaptive motion artifact reduction
## SVD
[SVD_GalaxyDenosing.py](02_Denosing%2FSVD_GalaxyDenosing.py): 

Implements Singular Value Decomposition-based signal decomposition for artifact removal

## Usage Guide( Taking IMAT as Example )
### Initialize the denoiser
```python
from denoising.imat import IMATDenoising
denoiser = IMATDenoising()
```
### Process the signal
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

Mashhadi et al. (2015) - IMAT algorithm

Temko et al. (2017) - Wiener filtering approach

Seyedtabaii & Seyedtabaii (2008) - Kalman filtering method

Reddy et al. (2008) - SVD-based artifact reduction
# ECG & PPG Peak Detection -> HRV

Located in [03_PeakDetection](03_PeakDetection)
[00_ECG_Calculation_basing_HeartPy.py](03_PeakDetection%2F00_ECG_Calculation_basing_HeartPy.py): 

Calculates HRV metrics from ECG signals using HeartPy

Utilizes customized bandpass filtering parameters from [filter_parameters](01_WindowSegment%2Ffilter_parameters)

[01_PPG_Calculation_basing_HeartPy.py](03_PeakDetection%2F01_PPG_Calculation_basing_HeartPy.py): 

handles the processing of denoised signals using HeartPy
It also measures the difference between PPG-derived and ECG-derived HRV metrics
