import os
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(BASE_DIR, 'Dataset')
WINDOW_DIR = os.path.join(BASE_DIR, 'WindowData')
RESULTS_DIR = os.path.join(BASE_DIR, 'results', 'Denoised_Result')
DENOISING_ALGORITHMS = ['Wiener','IMAT','Kalman','SVD']



