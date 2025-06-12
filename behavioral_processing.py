import pandas as pd
import numpy as np
import os

def load_behavioral_data(participant_dir):
    """
    Load behavioral data from CSV files
    """
    # Look for behavioral data files
    behavioral_file = 'processed_behavioral_data.csv'
    file_path = os.path.join(participant_dir, behavioral_file)
    
    if not os.path.exists(file_path):
        print(f"[WARNING] No behavioral data found in {participant_dir}")
        return None
    
    # Load the behavioral file
    df = pd.read_csv(file_path)
    print(f"[DEBUG] Loaded behavioral data from {file_path}: shape={df.shape}")
    print(df.head())
    
    # Load block time mapping
    block_time_file = 'block_time_mapping.csv'
    block_time_path = os.path.join(participant_dir, block_time_file)
    
    if os.path.exists(block_time_path):
        block_time_df = pd.read_csv(block_time_path)
        print(f"[DEBUG] Loaded block time mapping from {block_time_path}: shape={block_time_df.shape}")
        print(block_time_df.head())
        
        # Merge block time mapping with behavioral data
        df = pd.merge(df, block_time_df, on='block', how='left')
        print(f"[DEBUG] Merged behavioral data with block time mapping: shape={df.shape}")
        print(df.head())
    else:
        print(f"[WARNING] No block time mapping found in {participant_dir}")
    
    return df

def extract_behavioral_features(behavioral_data):
    """
    Extract behavioral features from the data
    """
    # Group by block and sensory_condition
    features = []
    
    for (block, sensory_condition), group in behavioral_data.groupby(['block', 'sensory_condition']):
        # Calculate accuracy
        accuracy = (group['accuracy'] == 1).mean()
        
        # Calculate reaction time statistics
        rt_mean = group['reaction_time'].mean()
        rt_std = group['reaction_time'].std()
        rt_median = group['reaction_time'].median()
        
        # Calculate number of trials
        n_trials = len(group)
        
        # Store features
        features.append({
            'block': block,
            'condition': sensory_condition,
            'accuracy': accuracy,
            'rt_mean': rt_mean,
            'rt_std': rt_std,
            'rt_median': rt_median,
            'n_trials': n_trials
        })
    
    return pd.DataFrame(features) 