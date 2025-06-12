import pandas as pd
import numpy as np
import os
from scipy import signal
import warnings
warnings.filterwarnings('ignore')

def load_physiological_data(participant_dir, data_type, side='Left'):
    """
    Load physiological data from CSV files
    data_type: 'EDA', 'HR', 'IBI', 'TEMP', 'BVP', 'ACC'
    side: 'Left' or 'Right'
    """
    file_map = {
        'EDA': f'{side}_EDA.csv',
        'HR': f'{side}_HR.csv',
        'IBI': f'{side}_IBI.csv',
        'TEMP': f'{side}_TEMP.csv',
        'BVP': f'{side}_BVP.csv',
        'ACC': f'{side}_ACC.csv'
    }
    
    file_path = os.path.join(participant_dir, file_map[data_type])
    if not os.path.exists(file_path):
        print(f"[WARNING] Could not find {file_path}. Skipping {data_type} for {side}.")
        return None
    
    df = pd.read_csv(file_path)
    print(f"[DEBUG] Loaded {data_type} ({side}) from {file_path}: shape={df.shape}")
    print(df.head())
    # If 'timestamp' is missing, generate it from 'start_time_unix' and 'sampling_rate'
    if 'timestamp' not in df.columns:
        if 'start_time_unix' in df.columns and 'sampling_rate' in df.columns:
            start_time = df['start_time_unix'].iloc[0]
            sampling_rate = df['sampling_rate'].iloc[0]
            df['timestamp'] = start_time + np.arange(len(df)) / sampling_rate
            print(f"[DEBUG] Generated timestamp for {data_type} ({side}): shape={df.shape}")
            print(df[['timestamp']].head())
        else:
            print(f"[WARNING] Could not generate timestamp for {file_path}. Skipping.")
            return None
    return df

def preprocess_temp(temp_data, sampling_rate=4):
    """
    Preprocess temperature data:
    1. Remove artifacts using moving median filter
    2. Calculate rate of change
    3. Extract features
    """
    # Apply moving median filter to remove spikes
    window_size = sampling_rate * 5  # 5-second window
    filtered_temp = pd.Series(temp_data['TEMP']).rolling(
        window=window_size, center=True
    ).median().fillna(method='ffill').fillna(method='bfill').reset_index(drop=True)
    
    # Calculate rate of change
    temp_diff = np.diff(filtered_temp, prepend=filtered_temp.iloc[0])
    temp_diff = pd.Series(temp_diff).reset_index(drop=True)
    
    # Ensure all arrays are the same length
    n = len(temp_data)
    filtered_temp = filtered_temp.iloc[:n]
    temp_diff = temp_diff.iloc[:n]
    
    # Debug prints
    print(f"[DEBUG] preprocess_temp lengths: timestamp={len(temp_data['timestamp'])}, raw_temp={len(temp_data['TEMP'])}, filtered_temp={len(filtered_temp)}, temp_rate={len(temp_diff)}")
    
    return pd.DataFrame({
        'timestamp': temp_data['timestamp'].reset_index(drop=True),
        'raw_temp': temp_data['TEMP'].reset_index(drop=True),
        'filtered_temp': filtered_temp,
        'temp_rate': temp_diff
    })

def extract_temp_features(temp_data, window_size=30):
    """
    Extract temperature features:
    1. Mean temperature
    2. Temperature variance
    3. Rate of change statistics
    """
    features = []
    for i in range(0, len(temp_data), window_size):
        window = temp_data.iloc[i:i+window_size]
        if len(window) > 0:
            features.append({
                'timestamp': window['timestamp'].iloc[0],
                'mean_temp': window['filtered_temp'].mean(),
                'temp_std': window['filtered_temp'].std(),
                'max_temp_rate': window['temp_rate'].max(),
                'min_temp_rate': window['temp_rate'].min()
            })
    print(f"[DEBUG] extract_temp_features: num_windows={len(features)}")
    df = pd.DataFrame(features)
    print(f"[DEBUG] extract_temp_features DataFrame shape: {df.shape}")
    return df

def preprocess_bvp(bvp_data, sampling_rate=64):
    """
    Preprocess BVP data:
    1. Bandpass filter (0.5-4 Hz)
    """
    # Bandpass filter
    nyquist = sampling_rate / 2
    low = 0.5 / nyquist
    high = 4.0 / nyquist
    b, a = signal.butter(4, [low, high], btype='band')
    filtered_bvp = signal.filtfilt(b, a, bvp_data['BVP'].values)
    
    return pd.DataFrame({
        'timestamp': bvp_data['timestamp'],
        'raw_bvp': bvp_data['BVP'],
        'filtered_bvp': filtered_bvp
    })

def extract_bvp_features(bvp_data, window_size=30, sampling_rate=64):
    """
    Extract BVP features:
    1. Mean pulse rate
    2. Pulse rate variability
    3. Mean pulse amplitude
    4. Pulse amplitude variability
    """
    features = []
    for i in range(0, len(bvp_data), window_size):
        window = bvp_data.iloc[i:i+window_size]
        if len(window) > 1:
            # Find peaks in filtered BVP
            peaks, _ = signal.find_peaks(window['filtered_bvp'], distance=sampling_rate/2)
            pulse_amplitude = window['filtered_bvp'].iloc[peaks] if len(peaks) > 0 else []
            if len(peaks) > 1:
                pulse_intervals = np.diff(window.index[peaks]) / sampling_rate
                pulse_rate = 60 / pulse_intervals  # Convert to BPM
            else:
                pulse_rate = []
            features.append({
                'timestamp': window['timestamp'].iloc[0],
                'mean_pulse_rate': np.mean(pulse_rate) if len(pulse_rate) > 0 else 0,
                'pulse_rate_std': np.std(pulse_rate) if len(pulse_rate) > 0 else 0,
                'mean_pulse_amplitude': np.mean(pulse_amplitude) if len(pulse_amplitude) > 0 else 0,
                'pulse_amplitude_std': np.std(pulse_amplitude) if len(pulse_amplitude) > 0 else 0
            })
    return pd.DataFrame(features)

def preprocess_acc(acc_data, sampling_rate=32):
    """
    Preprocess accelerometer data:
    1. Calculate magnitude
    2. Apply low-pass filter
    3. Calculate activity counts
    """
    # Calculate magnitude
    acc_magnitude = np.sqrt(
        acc_data['ACC_X']**2 + 
        acc_data['ACC_Y']**2 + 
        acc_data['ACC_Z']**2
    )
    
    # Low-pass filter
    nyquist = sampling_rate / 2
    cutoff = 2.0 / nyquist  # 2 Hz cutoff
    b, a = signal.butter(4, cutoff, btype='low')
    filtered_acc = signal.filtfilt(b, a, acc_magnitude)
    
    # Calculate activity counts (sum of absolute differences)
    activity_counts = np.sum(np.abs(np.diff(filtered_acc)))
    
    return pd.DataFrame({
        'timestamp': acc_data['timestamp'],
        'acc_magnitude': acc_magnitude,
        'filtered_acc': filtered_acc,
        'activity_count': activity_counts
    })

def extract_acc_features(acc_data, window_size=30):
    """
    Extract accelerometer features:
    1. Mean magnitude
    2. Standard deviation
    3. Activity counts
    4. Peak frequency
    """
    features = []
    for i in range(0, len(acc_data), window_size):
        window = acc_data.iloc[i:i+window_size]
        if len(window) > 0:
            # Calculate frequency domain features
            f, pxx = signal.welch(window['filtered_acc'])
            peak_freq = f[np.argmax(pxx)]
            
            features.append({
                'timestamp': window['timestamp'].iloc[0],
                'mean_acc': window['acc_magnitude'].mean(),
                'acc_std': window['acc_magnitude'].std(),
                'activity_count': window['activity_count'].sum(),
                'peak_frequency': peak_freq
            })
    return pd.DataFrame(features)

def synchronize_physiological_data(physio_data, block_mapping):
    """
    Synchronize physiological data with behavioral blocks
    """
    synced_data = []
    for _, block in block_mapping.iterrows():
        if pd.isna(block['start_time']) or pd.isna(block['end_time']):
            continue
        block_data = physio_data[
            (physio_data['timestamp'] >= block['start_time']) & 
            (physio_data['timestamp'] <= block['end_time'])
        ]
        if not block_data.empty:
            block_data['block'] = block['block']
            synced_data.append(block_data)
    if synced_data:
        result = pd.concat(synced_data)
        print(f"[DEBUG] synchronize_physiological_data: result shape {result.shape}")
        return result
    else:
        print(f"[DEBUG] synchronize_physiological_data: no data to sync")
        return pd.DataFrame() 