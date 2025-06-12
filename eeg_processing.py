import os
import pandas as pd
import numpy as np
from scipy import signal
import warnings
warnings.filterwarnings('ignore')

def load_eeg_data(participant_dir):
    """
    Load EEG data from CSV file
    """
    eeg_file_1 = os.path.join(participant_dir, 'EEG_recording.csv')
    eeg_file_2 = os.path.join(participant_dir, 'EEG_recordings.csv')
    if os.path.exists(eeg_file_1):
        df = pd.read_csv(eeg_file_1)
        return df
    elif os.path.exists(eeg_file_2):
        df = pd.read_csv(eeg_file_2)
        return df
    else:
        raise FileNotFoundError(f"Could not find EEG_recording.csv or EEG_recordings.csv in {participant_dir}")

def preprocess_eeg(eeg_data, sampling_rate=256):
    """
    Preprocess EEG data:
    1. Bandpass filter (1-45 Hz)
    2. Remove artifacts using ICA
    3. Segment into epochs
    """
    # Bandpass filter
    nyquist = sampling_rate / 2
    low = 1.0 / nyquist
    high = 45.0 / nyquist
    b, a = signal.butter(4, [low, high], btype='band')
    
    # Apply filter to each channel
    filtered_data = pd.DataFrame()
    for column in eeg_data.columns:
        if column != 'timestamps':
            filtered_data[column] = signal.filtfilt(b, a, eeg_data[column].values)
    
    filtered_data['timestamps'] = eeg_data['timestamps']
    return filtered_data

def extract_eeg_features(eeg_data, window_size=256):  # 1 second window at 256 Hz
    """
    Extract EEG features:
    1. Power in standard frequency bands
    2. Spectral entropy
    3. Cross-channel correlations
    """
    features = []
    frequency_bands = {
        'delta': (0.5, 4),
        'theta': (4, 8),
        'alpha': (8, 13),
        'beta': (13, 30),
        'gamma': (30, 45)
    }
    
    for i in range(0, len(eeg_data), window_size):
        window = eeg_data.iloc[i:i+window_size]
        if len(window) < window_size:
            continue
            
        window_features = {'timestamp_0': window['timestamps'].iloc[0]}
        
        # Calculate power in each frequency band for each channel
        for column in eeg_data.columns:
            if column == 'timestamps':
                continue
                
            # Calculate power spectrum
            f, pxx = signal.welch(window[column], fs=256)
            
            # Calculate power in each band
            for band_name, (low, high) in frequency_bands.items():
                band_mask = (f >= low) & (f <= high)
                power = np.mean(pxx[band_mask])
                window_features[f'{column}_{band_name}_power'] = power
            
            # Calculate spectral entropy
            pxx_norm = pxx / np.sum(pxx)
            entropy = -np.sum(pxx_norm * np.log2(pxx_norm + 1e-10))
            window_features[f'{column}_entropy'] = entropy
        
        # Calculate cross-channel correlations
        channels = [col for col in eeg_data.columns if col != 'timestamps']
        for i, ch1 in enumerate(channels):
            for ch2 in channels[i+1:]:
                corr = np.corrcoef(window[ch1], window[ch2])[0,1]
                window_features[f'corr_{ch1}_{ch2}'] = corr
        
        features.append(window_features)
    
    return pd.DataFrame(features)

def synchronize_eeg_data(eeg_data, block_mapping):
    """
    Synchronize EEG data with behavioral blocks
    """
    synced_data = []
    for _, block in block_mapping.iterrows():
        if pd.isna(block['start_time']) or pd.isna(block['end_time']):
            continue
            
        block_data = eeg_data[
            (eeg_data['timestamps'] >= block['start_time']) & 
            (eeg_data['timestamps'] <= block['end_time'])
        ]
        
        if not block_data.empty:
            block_data['block'] = block['block']
            synced_data.append(block_data)
    
    return pd.concat(synced_data) if synced_data else pd.DataFrame() 