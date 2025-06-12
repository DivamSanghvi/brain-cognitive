import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os

def load_and_clean_data(file_path):
    """Load and clean the feature matrix data."""
    df = pd.read_csv(file_path)
    
    # Clean missing values
    df['n_back_level'] = df['n_back_level'].ffill()
    df['sensory_condition'] = df['sensory_condition'].fillna('baseline')
    
    # Convert reaction time to seconds
    df['reaction_time'] = df['reaction_time'] / 1000
    
    # Remove trials with zero reaction time (missed responses)
    df = df[df['reaction_time'] > 0]
    
    return df

def calculate_descriptive_stats(df):
    """Calculate descriptive statistics for key metrics."""
    stats_dict = {}
    
    # Overall statistics
    stats_dict['overall'] = {
        'accuracy': df['accuracy'].describe(),
        'reaction_time': df['reaction_time'].describe()
    }
    
    # Statistics by n-back level
    stats_dict['by_nback'] = df.groupby('n_back_level').agg({
        'accuracy': ['mean', 'std', 'count'],
        'reaction_time': ['mean', 'std']
    })
    
    # Statistics by sensory condition
    stats_dict['by_condition'] = df.groupby('sensory_condition').agg({
        'accuracy': ['mean', 'std', 'count'],
        'reaction_time': ['mean', 'std']
    })
    
    return stats_dict

def calculate_correlations(df):
    """Calculate correlations between key metrics."""
    # Calculate correlation between accuracy and reaction time
    corr = df[['accuracy', 'reaction_time']].corr()
    
    # Calculate correlation by n-back level
    corr_by_nback = df.groupby('n_back_level')[['accuracy', 'reaction_time']].corr()
    
    return corr, corr_by_nback

def create_visualizations(df, output_dir='figures'):
    """Create initial visualizations of the data."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Set style
    plt.style.use('default')
    
    # 1. Accuracy by n-back level
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='n_back_level', y='accuracy', data=df)
    plt.title('Accuracy by N-back Level')
    plt.xlabel('N-back Level')
    plt.ylabel('Accuracy')
    plt.savefig(os.path.join(output_dir, 'accuracy_by_nback.png'))
    plt.close()
    
    # 2. Reaction time by n-back level
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='n_back_level', y='reaction_time', data=df)
    plt.title('Reaction Time by N-back Level')
    plt.xlabel('N-back Level')
    plt.ylabel('Reaction Time (s)')
    plt.savefig(os.path.join(output_dir, 'rt_by_nback.png'))
    plt.close()
    
    # 3. Accuracy by sensory condition
    plt.figure(figsize=(12, 6))
    sns.boxplot(x='sensory_condition', y='accuracy', data=df)
    plt.title('Accuracy by Sensory Condition')
    plt.xlabel('Sensory Condition')
    plt.ylabel('Accuracy')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'accuracy_by_condition.png'))
    plt.close()
    
    # 4. Reaction time by sensory condition
    plt.figure(figsize=(12, 6))
    sns.boxplot(x='sensory_condition', y='reaction_time', data=df)
    plt.title('Reaction Time by Sensory Condition')
    plt.xlabel('Sensory Condition')
    plt.ylabel('Reaction Time (s)')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'rt_by_condition.png'))
    plt.close()
    
    # 5. Scatter plot of accuracy vs reaction time
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x='reaction_time', y='accuracy', alpha=0.5)
    plt.title('Accuracy vs Reaction Time')
    plt.xlabel('Reaction Time (s)')
    plt.ylabel('Accuracy')
    plt.savefig(os.path.join(output_dir, 'accuracy_vs_rt.png'))
    plt.close()

def main():
    # Load and clean data
    df = load_and_clean_data('Experiment_1/feature_matrix.csv')
    
    # Calculate descriptive statistics
    stats_dict = calculate_descriptive_stats(df)
    
    # Calculate correlations
    corr, corr_by_nback = calculate_correlations(df)
    
    # Print descriptive statistics
    print("\nOverall Statistics:")
    print("\nAccuracy:")
    print(stats_dict['overall']['accuracy'])
    print("\nReaction Time:")
    print(stats_dict['overall']['reaction_time'])
    
    print("\nStatistics by N-back Level:")
    print(stats_dict['by_nback'])
    
    print("\nStatistics by Sensory Condition:")
    print(stats_dict['by_condition'])
    
    print("\nCorrelation between Accuracy and Reaction Time:")
    print(corr)
    
    print("\nCorrelation by N-back Level:")
    print(corr_by_nback)
    
    # Create visualizations
    create_visualizations(df)

if __name__ == "__main__":
    main() 