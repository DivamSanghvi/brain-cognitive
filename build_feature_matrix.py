import os
import pandas as pd
import numpy as np
from behavioral_processing import load_behavioral_data, extract_behavioral_features
from physiological_processing import (
    load_physiological_data, preprocess_temp, extract_temp_features,
    preprocess_bvp, extract_bvp_features, preprocess_acc, extract_acc_features,
    synchronize_physiological_data
)
from eeg_processing import load_eeg_data, extract_eeg_features, synchronize_eeg_data

def aggregate_physiological_features(physio_data, block_mapping, participant):
    """
    Aggregate physiological features per block
    """
    block_features = []
    for _, block in block_mapping.iterrows():
        if pd.isna(block['start_time']) or pd.isna(block['end_time']):
            continue
        block_data = physio_data[
            (physio_data['timestamp'] >= block['start_time']) & 
            (physio_data['timestamp'] <= block['end_time'])
        ]
        if block_data.empty:
            continue
        features = {'block': block['block'], 'participant': participant}
        for col in block_data.columns:
            if col not in ['timestamp', 'block'] and pd.api.types.is_numeric_dtype(block_data[col]):
                features[f'{col}_mean'] = block_data[col].mean()
                features[f'{col}_std'] = block_data[col].std()
        block_features.append(features)
    return pd.DataFrame(block_features)

def main():
    data_dir = "data"
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)
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
        block_mapping_file = os.path.join(participant_dir, 'block_time_mapping.csv')
        if not os.path.exists(block_mapping_file):
            print(f"[WARNING] Skipping {participant} (no block time mapping)")
            continue
        block_mapping = pd.read_csv(block_mapping_file)
        # Extract behavioral features
        behavioral_features = extract_behavioral_features(behavioral_data)
        behavioral_features['participant'] = participant
        # Process physiological data (TEMP, BVP, ACC, both sides)
        physio_features = []
        for side in ['Left', 'Right']:
            temp_data = load_physiological_data(participant_dir, 'TEMP', side=side)
            if temp_data is not None:
                temp_processed = preprocess_temp(temp_data)
                temp_features = extract_temp_features(temp_processed)
                temp_features['participant'] = participant
                temp_features['side'] = side
                temp_agg = aggregate_physiological_features(temp_features, block_mapping, participant)
                # Add side info to column names except for 'block' and 'participant'
                temp_agg = temp_agg.rename(columns={col: f'{side}_TEMP_{col}' if col not in ['block', 'participant'] else col for col in temp_agg.columns})
                physio_features.append(temp_agg)
            bvp_data = load_physiological_data(participant_dir, 'BVP', side=side)
            if bvp_data is not None:
                bvp_processed = preprocess_bvp(bvp_data)
                bvp_features = extract_bvp_features(bvp_processed)
                bvp_features['participant'] = participant
                bvp_features['side'] = side
                bvp_agg = aggregate_physiological_features(bvp_features, block_mapping, participant)
                bvp_agg = bvp_agg.rename(columns={col: f'{side}_BVP_{col}' if col not in ['block', 'participant'] else col for col in bvp_agg.columns})
                physio_features.append(bvp_agg)
            acc_data = load_physiological_data(participant_dir, 'ACC', side=side)
            if acc_data is not None:
                acc_processed = preprocess_acc(acc_data)
                acc_features = extract_acc_features(acc_processed)
                acc_features['participant'] = participant
                acc_features['side'] = side
                acc_agg = aggregate_physiological_features(acc_features, block_mapping, participant)
                acc_agg = acc_agg.rename(columns={col: f'{side}_ACC_{col}' if col not in ['block', 'participant'] else col for col in acc_agg.columns})
                physio_features.append(acc_agg)
        # Merge all physiological features on ['block', 'participant']
        if physio_features:
            from functools import reduce
            physio_df = reduce(lambda left, right: pd.merge(left, right, on=['block', 'participant'], how='outer'), physio_features)
        else:
            physio_df = pd.DataFrame()
        # Merge behavioral and physiological features
        if not physio_df.empty:
            participant_features = pd.merge(behavioral_features, physio_df, left_on=['block', 'participant'], right_on=[physio_df.columns[0], physio_df.columns[1]], how='left')
        else:
            participant_features = behavioral_features
        # Load and process EEG data
        try:
            eeg_data = load_eeg_data(participant_dir)
            eeg_features = extract_eeg_features(eeg_data)
            eeg_features['participant'] = participant
            eeg_synced = synchronize_eeg_data(eeg_features, block_mapping)
            if not eeg_synced.empty:
                # Aggregate EEG features per block
                eeg_agg = eeg_synced.groupby('block').mean().reset_index()
                eeg_agg['participant'] = participant
                participant_features = pd.merge(participant_features, eeg_agg, on=['block', 'participant'], how='left')
        except Exception as e:
            print(f"[WARNING] Skipping EEG for {participant}: {e}")
        all_features.append(participant_features)
        print(f"Final feature matrix for {participant}: {participant_features.shape}")
    # Combine all participants
    if all_features:
        feature_matrix = pd.concat(all_features, ignore_index=True)
        print(f"\nFinal feature matrix shape: {feature_matrix.shape}")
        feature_matrix.to_csv(os.path.join(results_dir, 'nback_features.csv'), index=False)
        print(f"Saved feature matrix to {os.path.join(results_dir, 'nback_features.csv')}")
    else:
        print("No features were extracted from any participant")

if __name__ == "__main__":
    main() 