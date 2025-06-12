import pandas as pd
import numpy as np
import os
from pathlib import Path
from physiological_processing import (
    load_physiological_data,
    preprocess_temp,
    extract_temp_features,
    preprocess_bvp,
    extract_bvp_features,
    preprocess_acc,
    extract_acc_features,
    synchronize_physiological_data
)
from eeg_processing import (
    load_eeg_data,
    preprocess_eeg,
    extract_eeg_features,
    synchronize_eeg_data
)
from feature_combination import create_feature_matrix

def get_valid_participants(experiment_dir):
    """
    Get list of valid participant folders (excluding Excluded_ID_* folders and non-directories)
    """
    all_items = os.listdir(experiment_dir)
    valid_folders = [f for f in all_items 
                     if os.path.isdir(os.path.join(experiment_dir, f))
                     and not f.startswith('Excluded_ID_')]
    return sorted(valid_folders)

def extract_behavioral_data(df):
    """
    Extract key behavioral data from the n-back responses
    """
    # Initialize lists to store extracted data
    trials_data = []
    
    # Process each trial
    for trial in range(1, len(df) + 1):
        trial_data = {
            'trial_number': trial,
            'block': df.loc[trial-1, 'Block'],
            'n_back_level': None,  # Will be determined from the trial list
            'sensory_condition': None,  # Will be determined from the session
            'accuracy': df.loc[trial-1, 'Correct'],
            'reaction_time': None,  # Will be extracted from RT columns
            'participant_id': df.loc[trial-1, 'Subject'],
            'session': df.loc[trial-1, 'Session']
        }
        
        # Determine n-back level from trial list
        for i in range(101, 105):
            if df.loc[trial-1, f'TrialList{i}'] == 1:
                trial_data['n_back_level'] = i - 100
                break
        
        # Determine sensory condition
        if df.loc[trial-1, 'NewMusicSession'] == 1:
            trial_data['sensory_condition'] = 'new_music'
        elif df.loc[trial-1, 'NoMusic'] == 1:
            trial_data['sensory_condition'] = 'no_music'
        elif df.loc[trial-1, 'RelaxSession'] == 1:
            trial_data['sensory_condition'] = 'relax_music'
        elif df.loc[trial-1, 'StressSession'] == 1:
            trial_data['sensory_condition'] = 'stress_music'
        
        # Get reaction time
        for i in range(100, 105):
            rt = df.loc[trial-1, f'Stimulus{i}.RT']
            if pd.notna(rt):
                trial_data['reaction_time'] = rt
                break
        
        trials_data.append(trial_data)
    
    return pd.DataFrame(trials_data)

def load_behavioral_data(participant_dir):
    """
    Load and process n_back_responses.csv for a participant
    """
    responses_file = os.path.join(participant_dir, 'n_back_responses.csv')
    if not os.path.exists(responses_file):
        raise FileNotFoundError(f"Could not find {responses_file}")
    
    df = pd.read_csv(responses_file)
    return extract_behavioral_data(df)

def load_tags(participant_dir):
    tags_file = os.path.join(participant_dir, 'tags.csv')
    if not os.path.exists(tags_file):
        raise FileNotFoundError(f"Could not find {tags_file}")
    with open(tags_file, 'r') as f:
        timestamps = [float(line.strip()) for line in f if line.strip()]
    return timestamps

def synchronize_blocks_with_tags(behavioral_df, tags):
    # Get unique blocks in order
    blocks = behavioral_df['block'].unique()
    block_sync = []
    for i, block in enumerate(blocks):
        start_time = tags[i] if i < len(tags) else None
        end_time = tags[i+1] if (i+1) < len(tags) else None
        block_sync.append({
            'block': block,
            'start_time': start_time,
            'end_time': end_time
        })
    return pd.DataFrame(block_sync)

def process_participant_data(participant_dir):
    """
    Process all data for a single participant
    """
    try:
        # 1. Process behavioral data
        behavioral_data = load_behavioral_data(participant_dir)
        behavioral_file = os.path.join(participant_dir, 'processed_behavioral_data.csv')
        behavioral_data.to_csv(behavioral_file, index=False)
        print(f"Processed behavioral data: {behavioral_file}")

        # 2. Load and process tags
        tags = load_tags(participant_dir)
        block_mapping = synchronize_blocks_with_tags(behavioral_data, tags)
        block_mapping_file = os.path.join(participant_dir, 'block_time_mapping.csv')
        block_mapping.to_csv(block_mapping_file, index=False)
        print(f"Created block mapping: {block_mapping_file}")

        # 3. Process physiological data
        # Left side
        left_temp = load_physiological_data(participant_dir, 'TEMP', side='Left')
        if left_temp is not None and 'timestamp' in left_temp.columns:
            left_temp_processed = preprocess_temp(left_temp)
            left_temp_features = extract_temp_features(left_temp_processed)
            left_temp_sync = synchronize_physiological_data(left_temp_features, block_mapping)
            left_temp_file = os.path.join(participant_dir, 'processed_left_temp.csv')
            left_temp_sync.to_csv(left_temp_file, index=False)
            print(f"Processed left temperature data: {left_temp_file}")
        else:
            print(f"[WARNING] Skipping left TEMP for {participant_dir} (missing or invalid data)")

        left_bvp = load_physiological_data(participant_dir, 'BVP', side='Left')
        if left_bvp is not None and 'timestamp' in left_bvp.columns:
            left_bvp_processed = preprocess_bvp(left_bvp)
            left_bvp_features = extract_bvp_features(left_bvp_processed)
            left_bvp_sync = synchronize_physiological_data(left_bvp_features, block_mapping)
            left_bvp_file = os.path.join(participant_dir, 'processed_left_bvp.csv')
            left_bvp_sync.to_csv(left_bvp_file, index=False)
            print(f"Processed left BVP data: {left_bvp_file}")
        else:
            print(f"[WARNING] Skipping left BVP for {participant_dir} (missing or invalid data)")

        left_acc = load_physiological_data(participant_dir, 'ACC', side='Left')
        if left_acc is not None and 'timestamp' in left_acc.columns:
            left_acc_processed = preprocess_acc(left_acc)
            left_acc_features = extract_acc_features(left_acc_processed)
            left_acc_sync = synchronize_physiological_data(left_acc_features, block_mapping)
            left_acc_file = os.path.join(participant_dir, 'processed_left_acc.csv')
            left_acc_sync.to_csv(left_acc_file, index=False)
            print(f"Processed left accelerometer data: {left_acc_file}")
        else:
            print(f"[WARNING] Skipping left ACC for {participant_dir} (missing or invalid data)")

        # Right side
        right_temp = load_physiological_data(participant_dir, 'TEMP', side='Right')
        if right_temp is not None and 'timestamp' in right_temp.columns:
            right_temp_processed = preprocess_temp(right_temp)
            right_temp_features = extract_temp_features(right_temp_processed)
            right_temp_sync = synchronize_physiological_data(right_temp_features, block_mapping)
            right_temp_file = os.path.join(participant_dir, 'processed_right_temp.csv')
            right_temp_sync.to_csv(right_temp_file, index=False)
            print(f"Processed right temperature data: {right_temp_file}")
        else:
            print(f"[WARNING] Skipping right TEMP for {participant_dir} (missing or invalid data)")

        right_bvp = load_physiological_data(participant_dir, 'BVP', side='Right')
        if right_bvp is not None and 'timestamp' in right_bvp.columns:
            right_bvp_processed = preprocess_bvp(right_bvp)
            right_bvp_features = extract_bvp_features(right_bvp_processed)
            right_bvp_sync = synchronize_physiological_data(right_bvp_features, block_mapping)
            right_bvp_file = os.path.join(participant_dir, 'processed_right_bvp.csv')
            right_bvp_sync.to_csv(right_bvp_file, index=False)
            print(f"Processed right BVP data: {right_bvp_file}")
        else:
            print(f"[WARNING] Skipping right BVP for {participant_dir} (missing or invalid data)")

        right_acc = load_physiological_data(participant_dir, 'ACC', side='Right')
        if right_acc is not None and 'timestamp' in right_acc.columns:
            right_acc_processed = preprocess_acc(right_acc)
            right_acc_features = extract_acc_features(right_acc_processed)
            right_acc_sync = synchronize_physiological_data(right_acc_features, block_mapping)
            right_acc_file = os.path.join(participant_dir, 'processed_right_acc.csv')
            right_acc_sync.to_csv(right_acc_file, index=False)
            print(f"Processed right accelerometer data: {right_acc_file}")
        else:
            print(f"[WARNING] Skipping right ACC for {participant_dir} (missing or invalid data)")

        # 4. Process EEG data
        eeg_data = load_eeg_data(participant_dir)
        if eeg_data is not None and 'timestamp' in eeg_data.columns:
            eeg_processed = preprocess_eeg(eeg_data)
            eeg_features = extract_eeg_features(eeg_processed)
            eeg_sync = synchronize_eeg_data(eeg_features, block_mapping)
            eeg_file = os.path.join(participant_dir, 'processed_eeg.csv')
            eeg_sync.to_csv(eeg_file, index=False)
            print(f"Processed EEG data: {eeg_file}")
        else:
            print(f"[WARNING] Skipping EEG for {participant_dir} (missing or invalid data)")

        return True
    except Exception as e:
        print(f"Error processing participant data: {e}")
        return False

if __name__ == "__main__":
    # Set up paths
    experiment_1_dir = "data"
    
    # Get valid participants
    valid_participants = get_valid_participants(experiment_1_dir)
    print(f"Found {len(valid_participants)} valid participants in Experiment 1")
    
    # Process each participant
    processed = 0
    for participant in valid_participants:
        participant_dir = os.path.join(experiment_1_dir, participant)
        print(f"\nProcessing participant: {participant}")
        if process_participant_data(participant_dir):
            processed += 1
    
    print(f"\nProcessed data for {processed}/{len(valid_participants)} participants")
    
    # Create feature matrix
    print("\nCreating feature matrix...")
    feature_matrix = create_feature_matrix(experiment_1_dir)
    feature_matrix_file = os.path.join(experiment_1_dir, 'feature_matrix.csv')
    feature_matrix.to_csv(feature_matrix_file, index=False)
    print(f"Feature matrix saved to: {feature_matrix_file}") 