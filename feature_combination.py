import pandas as pd
import numpy as np
import os

def load_processed_data(participant_dir):
    """
    Load all processed data for a participant
    """
    data = {}
    
    # Load behavioral data
    behavioral_file = os.path.join(participant_dir, 'processed_behavioral_data.csv')
    if os.path.exists(behavioral_file):
        data['behavioral'] = pd.read_csv(behavioral_file)
    
    # Load physiological data
    for side in ['left', 'right']:
        # Temperature
        temp_file = os.path.join(participant_dir, f'processed_{side}_temp.csv')
        if os.path.exists(temp_file):
            data[f'{side}_temp'] = pd.read_csv(temp_file)
        
        # BVP
        bvp_file = os.path.join(participant_dir, f'processed_{side}_bvp.csv')
        if os.path.exists(bvp_file):
            data[f'{side}_bvp'] = pd.read_csv(bvp_file)
        
        # Accelerometer
        acc_file = os.path.join(participant_dir, f'processed_{side}_acc.csv')
        if os.path.exists(acc_file):
            data[f'{side}_acc'] = pd.read_csv(acc_file)
    
    # Load EEG data
    eeg_file = os.path.join(participant_dir, 'processed_eeg.csv')
    if os.path.exists(eeg_file):
        data['eeg'] = pd.read_csv(eeg_file)
    
    return data

def combine_features(processed_data):
    """
    Combine features from all modalities into a single DataFrame
    """
    # Start with behavioral data as the base
    combined = processed_data['behavioral'].copy()
    
    # Add physiological features
    for side in ['left', 'right']:
        # Temperature features
        if f'{side}_temp' in processed_data:
            temp_data = processed_data[f'{side}_temp']
            # Merge on block only
            combined = pd.merge(
                combined,
                temp_data.drop(columns=['timestamp'], errors='ignore'),
                on=['block'],
                how='left',
                suffixes=('', f'_{side}_temp')
            )
        
        # BVP features
        if f'{side}_bvp' in processed_data:
            bvp_data = processed_data[f'{side}_bvp']
            # Merge on block only
            combined = pd.merge(
                combined,
                bvp_data.drop(columns=['timestamp'], errors='ignore'),
                on=['block'],
                how='left',
                suffixes=('', f'_{side}_bvp')
            )
        
        # Accelerometer features
        if f'{side}_acc' in processed_data:
            acc_data = processed_data[f'{side}_acc']
            # Merge on block only
            combined = pd.merge(
                combined,
                acc_data.drop(columns=['timestamp'], errors='ignore'),
                on=['block'],
                how='left',
                suffixes=('', f'_{side}_acc')
            )
    
    # Add EEG features
    if 'eeg' in processed_data:
        eeg_data = processed_data['eeg']
        # Merge on block only
        combined = pd.merge(
            combined,
            eeg_data.drop(columns=['timestamp'], errors='ignore'),
            on=['block'],
            how='left',
            suffixes=('', '_eeg')
        )
    
    # Fill NaN values with appropriate defaults
    numeric_columns = combined.select_dtypes(include=[np.number]).columns
    combined[numeric_columns] = combined[numeric_columns].fillna(0)
    
    return combined

def create_feature_matrix(experiment_dir):
    """
    Create a feature matrix for all participants in an experiment
    """
    all_features = []
    
    # Get valid participants
    all_folders = os.listdir(experiment_dir)
    valid_participants = [f for f in all_folders if not f.startswith('Excluded_ID_')]
    
    for participant in valid_participants:
        participant_dir = os.path.join(experiment_dir, participant)
        try:
            # Load processed data
            processed_data = load_processed_data(participant_dir)
            
            # Combine features
            participant_features = combine_features(processed_data)
            
            # Add participant ID
            participant_features['participant_id'] = participant
            
            all_features.append(participant_features)
            
        except Exception as e:
            print(f"Error processing {participant}: {e}")
    
    # Combine all participants
    if all_features:
        return pd.concat(all_features, ignore_index=True)
    else:
        return pd.DataFrame()

if __name__ == "__main__":
    # Set up paths
    experiment_1_dir = "Experiment_1"
    
    # Create feature matrix
    feature_matrix = create_feature_matrix(experiment_1_dir)
    
    # Save feature matrix
    output_file = os.path.join(experiment_1_dir, 'feature_matrix.csv')
    feature_matrix.to_csv(output_file, index=False)
    print(f"Created and saved feature matrix: {output_file}") 