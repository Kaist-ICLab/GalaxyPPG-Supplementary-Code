# GalaxyPPG Dataset Processing Tools
This repository contains tools and scripts for processing and analyzing the GalaxyPPG dataset, focusing on PPG signal processing and heart rate variability analysis.

# Raw Data Validation
Located in [00_RawDataCheck](00_RawDataCheck), this section contains scripts for assessing dataset completeness and quality:

[00_MissingRate.py](00_RawDataCheck%2F00_MissingRate.py): Calculates and analyzes missing data rates across all signals

[01_SamplingRate.py](00_RawDataCheck%2F01_SamplingRate.py): Validates sampling rates for different sensor modalities

# Pre Processing

Located in [01_WindowSegment](01_WindowSegment), this section handles signal segmentation and heart rate analysis:

[00_Segment_Dataset.py](01_WindowSegment%2F00_Segment_Dataset.py): Implements windowed segmentation of physiological signals

Configurable parameters:

window_size: Duration of each analysis window (default: 8 seconds)

window_step: Sliding step between consecutive windows (default: 2 seconds)

Outputs a CSV file containing segmented signals for all modalities

# ECG Calculation -> HRV

[01_ECG_Calculation_basing_HeartPy.py](01_WindowSegment%2F01_ECG_Calculation_basing_HeartPy.py): Calculates HRV metrics from ECG signals using HeartPy

Utilizes customized bandpass filtering parameters from [filter_parameters](01_WindowSegment%2Ffilter_parameters)

Note: ECG R-peak detection traditionally operates within a 5-15Hz bandpass range. However, when analyzing short 8-second windows, we observed that this narrow range could occasionally lead to signal attenuation that impedes reliable peak detection. Our implementation therefore employs an expanded bandpass range for consistent HRV analysis across our dataset's short-window segments.

# Denoising Alogorithm

[IMAT_GalaxyDenosing.py](02_Denosing%2FIMAT_GalaxyDenosing.py): Implements the Iterative Method with Adaptive Thresholding (IMAT) approach for motion artifact reduction, based on spectral analysis and sparse signal reconstruction

[Wiener_GalaxyDenosing.py](02_Denosing%2FWiener_GalaxyDenosing.py): Provides the Wiener filtering implementation with adaptive noise estimation for PPG signals during physical activity

[Kalman_GalaxyDenoising.py](02_Denosing%2FKalman_GalaxyDenoising.py): Contains the Kalman filter implementation for adaptive motion artifact reduction

[SVD_GalaxyDenosing.py](02_Denosing%2FSVD_GalaxyDenosing.py): Implements Singular Value Decomposition-based signal decomposition for artifact removal

This implementation draws from the following foundational works:

Mashhadi et al. (2015) - IMAT algorithm

Temko et al. (2017) - Wiener filtering approach

Seyedtabaii & Seyedtabaii (2008) - Kalman filtering method

Reddy et al. (2008) - SVD-based artifact reduction
# PPG Calculation -> HRV

Located in [03_PPG_HRV](03_PPG_HRV) [HRV_PPG_HeartPy.py](03_PPG_HRV%2FHRV_PPG_HeartPy.py) handles the processing of denoised signals using HeartPy

It also measures the difference between PPG-derived and ECG-derived HRV metrics
