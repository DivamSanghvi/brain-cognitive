import pandas as pd
import numpy as np
import os
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Import processing modules
from behavioral_processing import load_behavioral_data, extract_behavioral_features
from physiological_processing import (
    load_physiological_data, preprocess_temp, extract_temp_features,
    preprocess_bvp, extract_bvp_features, preprocess_acc, extract_acc_features,
    synchronize_physiological_data
)
from eeg_processing import load_eeg_data, extract_eeg_features, synchronize_eeg_data

def aggregate_physiological_features(physio_data, block_mapping):
    """
    Aggregate physiological features per block
    """
    block_features = []
    
    for _, block in block_mapping.iterrows():
        if pd.isna(block['start_time']) or pd.isna(block['end_time']):
            continue
            
        # Get data for this block
        block_data = physio_data[
            (physio_data['timestamp_0'] >= block['start_time']) & 
            (physio_data['timestamp_0'] <= block['end_time'])
        ]
        
        if block_data.empty:
            continue
            
        # Calculate block-level features
        features = {
            'block': block['block'],
            'condition': block['sensory_condition'],
            'participant': block['participant']
        }
        
        # Add mean and std for each numeric feature
        for col in block_data.columns:
            if col not in ['timestamp_0', 'timestamp_6', 'timestamp_12', 'block', 'condition', 'participant']:
                if pd.api.types.is_numeric_dtype(block_data[col]):
                    features[f'{col}_mean'] = block_data[col].mean()
                    features[f'{col}_std'] = block_data[col].std()
        
        block_features.append(features)
    
    return pd.DataFrame(block_features)

def main():
    # Set up paths
    data_dir = "data"
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)
    
    # Get list of participants
    participants = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d)) and not d.startswith('Excluded_ID_')]
    print(f"\nFound {len(participants)} participants")
    
    all_features = []
    for participant in participants:
        print(f"\nProcessing participant {participant}")
        participant_dir = os.path.join(data_dir, participant)
        
        # Load and process behavioral data
        behavioral_data = load_behavioral_data(participant_dir)
        if behavioral_data is None:
            print(f"[WARNING] Skipping {participant} (no behavioral data)")
            continue
            
        # Load block time mapping
        block_mapping_file = os.path.join(participant_dir, 'block_time_mapping.csv')
        if not os.path.exists(block_mapping_file):
            print(f"[WARNING] Skipping {participant} (no block time mapping)")
            continue
            
        block_mapping = pd.read_csv(block_mapping_file)
        print(f"[DEBUG] Loaded block time mapping from {block_mapping_file}: shape={block_mapping.shape}")
        print(block_mapping.head())
        
        # Extract behavioral features
        behavioral_features = extract_behavioral_features(behavioral_data)
        behavioral_features['participant'] = participant
        print(f"Extracted behavioral features: {behavioral_features.shape}")
        
        # Process physiological data
        physio_features = []
        
        # Process temperature data
        temp_data = load_physiological_data(participant_dir, 'TEMP')
        if temp_data is not None:
            temp_processed = preprocess_temp(temp_data)
            temp_features = extract_temp_features(temp_processed)
            temp_features['participant'] = participant
            physio_features.append(temp_features)
            print(f"Processed temperature data: {temp_features.shape}")
        
        # Process BVP data
        bvp_data = load_physiological_data(participant_dir, 'BVP')
        if bvp_data is not None:
            bvp_processed = preprocess_bvp(bvp_data)
            bvp_features = extract_bvp_features(bvp_processed)
            bvp_features['participant'] = participant
            physio_features.append(bvp_features)
            print(f"Processed BVP data: {bvp_features.shape}")
        
        # Process accelerometer data
        acc_data = load_physiological_data(participant_dir, 'ACC')
        if acc_data is not None:
            acc_processed = preprocess_acc(acc_data)
            acc_features = extract_acc_features(acc_processed)
            acc_features['participant'] = participant
            physio_features.append(acc_features)
            print(f"Processed accelerometer data: {acc_features.shape}")
        
        # Combine physiological features
        if physio_features:
            physio_df = pd.concat(physio_features, axis=1)
            print(f"Combined physiological features: {physio_df.shape}")
            
            # Debug: Check for duplicate column names
            print("Column names in physio_df:", physio_df.columns.tolist())
            print("Duplicate columns:", physio_df.columns[physio_df.columns.duplicated()].tolist())
            
            # Rename duplicate columns
            physio_df.columns = [f"{col}_{i}" if physio_df.columns.tolist().count(col) > 1 else col for i, col in enumerate(physio_df.columns)]
            print("Renamed columns in physio_df:", physio_df.columns.tolist())
            
            # Aggregate physiological features per block
            block_mapping['participant'] = participant
            physio_aggregated = aggregate_physiological_features(physio_df, block_mapping)
            print(f"Aggregated physiological features: {physio_aggregated.shape}")
            
            # Merge with behavioral features
            participant_features = pd.merge(
                behavioral_features,
                physio_aggregated,
                on=['block', 'participant'],
                how='outer'
            )
        else:
            participant_features = behavioral_features
        
        # Load and process EEG data
        eeg_data = load_eeg_data(participant_dir)
        if eeg_data is not None:
            eeg_features = extract_eeg_features(eeg_data)
            eeg_features['participant'] = participant
            print(f"Processed EEG data: {eeg_features.shape}")
            
            # Synchronize EEG data with block mapping
            eeg_synced = synchronize_eeg_data(eeg_features, block_mapping)
            print(f"Synchronized EEG data: {eeg_synced.shape}")
            
            # Merge EEG features
            participant_features = pd.merge(
                participant_features,
                eeg_synced,
                on=['block', 'participant'],
                how='left'
            )
        
        all_features.append(participant_features)
        print(f"Final feature matrix for {participant}: {participant_features.shape}")
    
    # Combine all participants
    if all_features:
        feature_matrix = pd.concat(all_features, ignore_index=True)
        print(f"\nFinal feature matrix shape: {feature_matrix.shape}")
        
        # Save feature matrix
        feature_matrix.to_csv(os.path.join(results_dir, 'nback_features.csv'), index=False)
        print(f"Saved feature matrix to {os.path.join(results_dir, 'nback_features.csv')}")
    else:
        print("No features were extracted from any participant")

if __name__ == "__main__":
    main() 