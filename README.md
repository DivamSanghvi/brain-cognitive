# Multimodal n-back Dataset Analysis

This project provides tools for analyzing multimodal physiological and behavioral data from n-back cognitive tasks. The analysis pipeline processes various data streams including EEG, temperature, blood volume pulse (BVP), and accelerometer data, synchronizes them with behavioral data, and creates a unified feature matrix for further analysis.

## Project Structure

```
.
├── analyze_nback.py           # Main analysis script
├── physiological_processing.py # Functions for processing physiological signals
├── eeg_processing.py         # Functions for processing EEG data
├── feature_combination.py    # Functions for combining features
├── requirements.txt          # Python dependencies
└── Experiment_1/            # Data directory
    ├── A1/                  # Participant directory
    │   ├── EEG_recording.csv
    │   ├── Left_TEMP.csv
    │   ├── Right_TEMP.csv
    │   └── ...
    └── ...
```

## Data Processing Pipeline

The analysis pipeline consists of the following steps:

1. **Behavioral Data Processing**
   - Load and process n-back task responses
   - Extract accuracy and reaction time metrics

2. **Physiological Signal Processing**
   - Process temperature data (TEMP)
   - Process blood volume pulse data (BVP)
   - Process accelerometer data (ACC)
   - Extract relevant features from each signal

3. **EEG Data Processing**
   - Apply bandpass filtering (1-45 Hz)
   - Extract power in standard frequency bands
   - Calculate spectral entropy and cross-channel correlations

4. **Data Synchronization**
   - Synchronize all data streams with behavioral blocks
   - Create block-level feature vectors

5. **Feature Combination**
   - Combine features from all modalities
   - Create a unified feature matrix for analysis

## Usage

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Run the analysis pipeline:
   ```bash
   python analyze_nback.py
   ```

This will:
- Process all participants' data in Experiment_1
- Create processed data files for each participant
- Generate a feature matrix combining all modalities

## Output Files

For each participant, the following files are generated:
- `processed_behavioral_data.csv`: Processed behavioral metrics
- `block_time_mapping.csv`: Block/timestamp synchronization
- `processed_left_temp.csv`: Processed temperature data (left wrist)
- `processed_right_temp.csv`: Processed temperature data (right wrist)
- `processed_left_bvp.csv`: Processed BVP data (left wrist)
- `processed_right_bvp.csv`: Processed BVP data (right wrist)
- `processed_left_acc.csv`: Processed accelerometer data (left wrist)
- `processed_right_acc.csv`: Processed accelerometer data (right wrist)
- `processed_eeg.csv`: Processed EEG data

The final output is:
- `feature_matrix.csv`: Combined feature matrix for all participants

## Features Extracted

### Behavioral Features
- Accuracy
- Reaction time
- N-back level
- Sensory condition

### Physiological Features
- Temperature: mean, variance, rate of change
- BVP: pulse rate, pulse amplitude, variability
- Accelerometer: movement magnitude, activity counts, frequency features

### EEG Features
- Power in standard frequency bands (delta, theta, alpha, beta, gamma)
- Spectral entropy
- Cross-channel correlations

## Notes

- The pipeline processes data from both wrists (left and right) for physiological signals
- EEG data is processed with a 1-second window (256 samples at 256 Hz)
- All features are synchronized with behavioral blocks using timestamps
- Excluded participants (folders starting with 'Excluded_ID_') are automatically skipped 