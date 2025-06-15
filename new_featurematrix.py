# main_processing_new.py
import os
import pandas as pd
import numpy as np
from behavioral_processing import load_behavioral_data, extract_behavioral_features
from physiological_processing import (
    load_physiological_data, preprocess_temp, extract_temp_features,
    preprocess_bvp, extract_bvp_features, preprocess_acc, extract_acc_features,
    synchronize_physiological_data
)
from new_eeg import load_eeg_data, extract_eeg_features, synchronize_eeg_data, test_eeg_processing

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
    eeg_success_count = 0
    eeg_fail_count = 0
    
    for participant in participants:
        print(f"\n{'='*50}")
        print(f"Processing participant {participant}")
        print(f"{'='*50}")
        
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
        print(f"[DEBUG] Block mapping loaded: {block_mapping.shape}")
        
        # Extract behavioral features
        behavioral_features = extract_behavioral_features(behavioral_data)
        behavioral_features['participant'] = participant
        print(f"[DEBUG] Behavioral features: {behavioral_features.shape}")
        
        # Process physiological data (TEMP, BVP, ACC, both sides)
        physio_features = []
        for side in ['Left', 'Right']:
            # TEMP
            temp_data = load_physiological_data(participant_dir, 'TEMP', side=side)
            if temp_data is not None:
                temp_processed = preprocess_temp(temp_data)
                temp_features = extract_temp_features(temp_processed)
                temp_features['participant'] = participant
                temp_features['side'] = side
                temp_agg = aggregate_physiological_features(temp_features, block_mapping, participant)
                temp_agg = temp_agg.rename(columns={col: f'{side}_TEMP_{col}' if col not in ['block', 'participant'] else col for col in temp_agg.columns})
                physio_features.append(temp_agg)
            
            # BVP
            bvp_data = load_physiological_data(participant_dir, 'BVP', side=side)
            if bvp_data is not None:
                bvp_processed = preprocess_bvp(bvp_data)
                bvp_features = extract_bvp_features(bvp_processed)
                bvp_features['participant'] = participant
                bvp_features['side'] = side
                bvp_agg = aggregate_physiological_features(bvp_features, block_mapping, participant)
                bvp_agg = bvp_agg.rename(columns={col: f'{side}_BVP_{col}' if col not in ['block', 'participant'] else col for col in bvp_agg.columns})
                physio_features.append(bvp_agg)
            
            # ACC
            acc_data = load_physiological_data(participant_dir, 'ACC', side=side)
            if acc_data is not None:
                acc_processed = preprocess_acc(acc_data)
                acc_features = extract_acc_features(acc_processed)
                acc_features['participant'] = participant
                acc_features['side'] = side
                acc_agg = aggregate_physiological_features(acc_features, block_mapping, participant)
                acc_agg = acc_agg.rename(columns={col: f'{side}_ACC_{col}' if col not in ['block', 'participant'] else col for col in acc_agg.columns})
                physio_features.append(acc_agg)
        
        # Merge all physiological features
        if physio_features:
            from functools import reduce
            physio_df = reduce(lambda left, right: pd.merge(left, right, on=['block', 'participant'], how='outer'), physio_features)
            print(f"[DEBUG] Physiological features: {physio_df.shape}")
        else:
            physio_df = pd.DataFrame()
            print(f"[DEBUG] No physiological features")
        
        # Merge behavioral and physiological features
        if not physio_df.empty:
            participant_features = pd.merge(behavioral_features, physio_df, on=['block', 'participant'], how='left')
        else:
            participant_features = behavioral_features
        
        print(f"[DEBUG] Features before EEG: {participant_features.shape}")
        
        # Load and process EEG data with detailed debugging
        eeg_added = False
        try:
            print(f"\n--- EEG Processing for {participant} ---")
            
            # Test EEG processing first
            eeg_test_success = test_eeg_processing(participant_dir, block_mapping_file)
            
            if eeg_test_success:
                # Proceed with actual EEG processing
                eeg_data = load_eeg_data(participant_dir)
                eeg_features = extract_eeg_features(eeg_data)
                
                if not eeg_features.empty:
                    eeg_features['participant'] = participant
                    eeg_synced = synchronize_eeg_data(eeg_features, block_mapping)
                    
                    if not eeg_synced.empty:
                        # Aggregate EEG features per block
                        eeg_agg = eeg_synced.groupby('block').mean(numeric_only=True).reset_index()
                        eeg_agg['participant'] = participant
                        
                        print(f"[DEBUG] EEG aggregated features: {eeg_agg.shape}")
                        print(f"[DEBUG] EEG feature columns: {[col for col in eeg_agg.columns if col not in ['block', 'participant']][:10]}...")
                        
                        # Merge with existing features
                        before_shape = participant_features.shape
                        participant_features = pd.merge(participant_features, eeg_agg, on=['block', 'participant'], how='left')
                        after_shape = participant_features.shape
                        
                        print(f"[DEBUG] Features after EEG merge: {before_shape} -> {after_shape}")
                        
                        if after_shape[1] > before_shape[1]:
                            eeg_added = True
                            eeg_success_count += 1
                            print(f"[SUCCESS] EEG features added successfully! Added {after_shape[1] - before_shape[1]} columns")
                        else:
                            print(f"[WARNING] EEG merge didn't add columns")
                    else:
                        print(f"[WARNING] No synchronized EEG data")
                else:
                    print(f"[WARNING] No EEG features extracted")
            
            if not eeg_added:
                eeg_fail_count += 1
                print(f"[WARNING] EEG processing failed for {participant}")
                
        except Exception as e:
            eeg_fail_count += 1
            print(f"[ERROR] EEG processing failed for {participant}: {e}")
            import traceback
            traceback.print_exc()
        
        all_features.append(participant_features)
        print(f"\nFinal feature matrix for {participant}: {participant_features.shape}")
        print(f"EEG added: {eeg_added}")
        
        # Show some sample column names
        non_id_cols = [col for col in participant_features.columns if col not in ['block', 'participant']]
        print(f"Sample feature columns: {non_id_cols[:10]}...")
    
    # Final summary
    print(f"\n{'='*60}")
    print(f"PROCESSING SUMMARY")
    print(f"{'='*60}")
    print(f"Total participants processed: {len(all_features)}")
    print(f"EEG processing successful: {eeg_success_count}")
    print(f"EEG processing failed: {eeg_fail_count}")
    
    # Combine all participants
    if all_features:
        feature_matrix = pd.concat(all_features, ignore_index=True)
        print(f"\nFinal feature matrix shape: {feature_matrix.shape}")
        
        # Check for EEG columns in final matrix
        eeg_columns = [col for col in feature_matrix.columns if any(band in col.lower() for band in ['delta', 'theta', 'alpha', 'beta', 'gamma', 'entropy', 'corr'])]
        print(f"EEG-related columns found: {len(eeg_columns)}")
        if eeg_columns:
            print(f"Sample EEG columns: {eeg_columns[:10]}...")
        
        # Save results
        feature_matrix.to_csv(os.path.join(results_dir, 'nback_features_new.csv'), index=False)
        print(f"Saved feature matrix to {os.path.join(results_dir, 'nback_features_new.csv')}")
        
        # Save column info for debugging
        column_info = pd.DataFrame({
            'column_name': feature_matrix.columns,
            'non_null_count': feature_matrix.count(),
            'data_type': feature_matrix.dtypes
        })
        column_info.to_csv(os.path.join(results_dir, 'feature_columns_info_new.csv'), index=False)
        print(f"Saved column info to {os.path.join(results_dir, 'feature_columns_info_new.csv')}")
        
    else:
        print("No features were extracted from any participant")

if __name__ == "__main__":
    main()