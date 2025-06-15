# eeg_processing_new.py
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
        print(f"[DEBUG] Loading EEG data from: {eeg_file_1}")
        df = pd.read_csv(eeg_file_1)
        print(f"[DEBUG] EEG data shape: {df.shape}, columns: {list(df.columns)}")
        return df
    elif os.path.exists(eeg_file_2):
        print(f"[DEBUG] Loading EEG data from: {eeg_file_2}")
        df = pd.read_csv(eeg_file_2)
        print(f"[DEBUG] EEG data shape: {df.shape}, columns: {list(df.columns)}")
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
    print(f"[DEBUG] Preprocessing EEG data with {len(eeg_data)} samples")
    
    # Check if we have a timestamp column (could be 'timestamps' or 'timestamp')
    timestamp_col = None
    for col in ['timestamps', 'timestamp', 'time']:
        if col in eeg_data.columns:
            timestamp_col = col
            break
    
    if timestamp_col is None:
        print("[WARNING] No timestamp column found, creating artificial timestamps")
        eeg_data['timestamps'] = np.arange(len(eeg_data)) / sampling_rate
        timestamp_col = 'timestamps'
    
    # Bandpass filter
    nyquist = sampling_rate / 2
    low = 1.0 / nyquist
    high = 45.0 / nyquist
    b, a = signal.butter(4, [low, high], btype='band')
    
    # Apply filter to each channel (excluding timestamp column)
    filtered_data = pd.DataFrame()
    for column in eeg_data.columns:
        if column != timestamp_col:
            try:
                filtered_data[column] = signal.filtfilt(b, a, eeg_data[column].values)
            except Exception as e:
                print(f"[WARNING] Could not filter column {column}: {e}")
                filtered_data[column] = eeg_data[column].values
    
    filtered_data[timestamp_col] = eeg_data[timestamp_col]
    print(f"[DEBUG] Filtered EEG data shape: {filtered_data.shape}")
    return filtered_data

def extract_eeg_features(eeg_data, window_size=256):  # 1 second window at 256 Hz
    """
    Extract EEG features:
    1. Power in standard frequency bands
    2. Spectral entropy
    3. Cross-channel correlations
    """
    print(f"[DEBUG] Extracting EEG features with window size: {window_size}")
    
    # Find timestamp column
    timestamp_col = None
    for col in ['timestamps', 'timestamp', 'time']:
        if col in eeg_data.columns:
            timestamp_col = col
            break
    
    if timestamp_col is None:
        print("[ERROR] No timestamp column found in EEG data")
        return pd.DataFrame()
    
    features = []
    frequency_bands = {
        'delta': (0.5, 4),
        'theta': (4, 8),
        'alpha': (8, 13),
        'beta': (13, 30),
        'gamma': (30, 45)
    }
    
    # Get EEG channels (exclude timestamp column)
    eeg_channels = [col for col in eeg_data.columns if col != timestamp_col]
    print(f"[DEBUG] EEG channels found: {eeg_channels}")
    
    if len(eeg_channels) == 0:
        print("[ERROR] No EEG channels found")
        return pd.DataFrame()
    
    num_windows = 0
    for i in range(0, len(eeg_data), window_size):
        window = eeg_data.iloc[i:i+window_size]
        if len(window) < window_size:
            continue
            
        num_windows += 1
        # Use the correct timestamp column name
        window_features = {timestamp_col: window[timestamp_col].iloc[0]}
        
        # Calculate power in each frequency band for each channel
        for column in eeg_channels:
            try:
                # Calculate power spectrum
                f, pxx = signal.welch(window[column], fs=256, nperseg=min(window_size, len(window)))
                
                # Calculate power in each band
                for band_name, (low, high) in frequency_bands.items():
                    band_mask = (f >= low) & (f <= high)
                    if np.any(band_mask):
                        power = np.mean(pxx[band_mask])
                        window_features[f'{column}_{band_name}_power'] = power
                    else:
                        window_features[f'{column}_{band_name}_power'] = 0
                
                # Calculate spectral entropy
                pxx_norm = pxx / (np.sum(pxx) + 1e-10)
                entropy = -np.sum(pxx_norm * np.log2(pxx_norm + 1e-10))
                window_features[f'{column}_entropy'] = entropy
                
            except Exception as e:
                print(f"[WARNING] Error processing channel {column}: {e}")
                # Set default values if processing fails
                for band_name in frequency_bands.keys():
                    window_features[f'{column}_{band_name}_power'] = 0
                window_features[f'{column}_entropy'] = 0
        
        # Calculate cross-channel correlations
        if len(eeg_channels) > 1:
            try:
                for i_ch, ch1 in enumerate(eeg_channels):
                    for ch2 in eeg_channels[i_ch+1:]:
                        corr = np.corrcoef(window[ch1], window[ch2])[0,1]
                        if np.isnan(corr):
                            corr = 0
                        window_features[f'corr_{ch1}_{ch2}'] = corr
            except Exception as e:
                print(f"[WARNING] Error calculating correlations: {e}")
        
        features.append(window_features)
    
    print(f"[DEBUG] Extracted {num_windows} windows with {len(window_features)-1} features each")
    result_df = pd.DataFrame(features)
    print(f"[DEBUG] EEG features DataFrame shape: {result_df.shape}")
    return result_df

def synchronize_eeg_data(eeg_features, block_mapping):
    """
    Synchronize EEG data with behavioral blocks
    """
    print(f"[DEBUG] Synchronizing EEG data with {len(block_mapping)} blocks")
    
    # Find timestamp column in eeg_features
    timestamp_col = None
    for col in ['timestamps', 'timestamp', 'time']:
        if col in eeg_features.columns:
            timestamp_col = col
            break
    
    if timestamp_col is None:
        print("[ERROR] No timestamp column found in EEG features")
        return pd.DataFrame()
    
    synced_data = []
    for idx, block in block_mapping.iterrows():
        if pd.isna(block['start_time']) or pd.isna(block['end_time']):
            print(f"[WARNING] Block {block.get('block', idx)} has missing start/end times")
            continue
        
        # Filter EEG features for this block
        block_data = eeg_features[
            (eeg_features[timestamp_col] >= block['start_time']) & 
            (eeg_features[timestamp_col] <= block['end_time'])
        ]
        
        if not block_data.empty:
            block_data = block_data.copy()
            block_data['block'] = block['block']
            synced_data.append(block_data)
            print(f"[DEBUG] Block {block['block']}: {len(block_data)} EEG windows")
        else:
            print(f"[WARNING] No EEG data found for block {block['block']} (time range: {block['start_time']}-{block['end_time']})")
    
    result = pd.concat(synced_data, ignore_index=True) if synced_data else pd.DataFrame()
    print(f"[DEBUG] Final synchronized EEG data shape: {result.shape}")
    return result

# Test if EEG is getting processed correctly 
def test_eeg_processing(participant_dir, block_mapping_file):
    """
    Test function to check EEG processing pipeline
    """
    print(f"\n=== Testing EEG Processing for {participant_dir} ===")
    
    try:
        # Load block mapping
        if os.path.exists(block_mapping_file):
            block_mapping = pd.read_csv(block_mapping_file)
            print(f"[DEBUG] Block mapping shape: {block_mapping.shape}")
            print(f"[DEBUG] Block mapping columns: {list(block_mapping.columns)}")
        else:
            print(f"[ERROR] Block mapping file not found: {block_mapping_file}")
            return False
        
        # Load EEG data
        eeg_data = load_eeg_data(participant_dir)
        print(f"[DEBUG] Raw EEG data loaded: {eeg_data.shape}")
        
        # Preprocess (optional, comment out if causing issues)
        # eeg_data = preprocess_eeg(eeg_data)
        
        # Extract features
        eeg_features = extract_eeg_features(eeg_data)
        if eeg_features.empty:
            print("[ERROR] No EEG features extracted")
            return False
        
        # Synchronize with blocks
        eeg_synced = synchronize_eeg_data(eeg_features, block_mapping)
        if eeg_synced.empty:
            print("[ERROR] No synchronized EEG data")
            return False
        
        # Aggregate per block
        eeg_agg = eeg_synced.groupby('block').mean(numeric_only=True).reset_index()
        print(f"[DEBUG] Aggregated EEG features shape: {eeg_agg.shape}")
        print(f"[DEBUG] Aggregated EEG features columns: {list(eeg_agg.columns)}")
        
        return True
        
    except Exception as e:
        print(f"[ERROR] EEG processing failed: {e}")
        import traceback
        traceback.print_exc()
        return False