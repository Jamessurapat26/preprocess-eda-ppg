import pandas as pd
import numpy as np
import neurokit2 as nk
import glob
from pathlib import Path
from datetime import datetime
import json

def load_json(file_path: Path) -> dict:
    """Load JSON file"""
    with open(file_path, "r", encoding="utf-8") as file:
        data = json.load(file)
    return data['label']

# Constants
SAMPLING_RATE = 100
RAW_PATH = Path('Raw/ppg')
PROCESSED_PATH = Path('Processed/ppg')
Label = load_json("label.json")


def process_timestamp(df: pd.DataFrame) -> pd.DataFrame:
    """
    Process and correct timestamp values in the dataframe
    
    Args:
        df: Input dataframe with timestamp column
        
    Returns:
        DataFrame with corrected timestamps
    """
    # Convert Unix timestamp to datetime
    df['DateTime'] = pd.to_datetime(df['LocalTimestamp'] , unit='s').dt.tz_localize('UTC').dt.tz_convert('Asia/Bangkok')

    
    return df

def get_subject_id_from_filename(filename: str) -> str:
    """Extract subject ID from filename (e.g., 'S01_PPG.csv' -> 'S01')"""
    # Convert to uppercase to match label.json format
    return filename.split('_')[0].upper()

def get_label_for_subject(subject_id: str, labels: list) -> dict:
    """Get label information for a specific subject"""
    # Convert input subject_id to uppercase for comparison
    subject_id = subject_id.upper()
    for label in labels:
        if label['id'].upper() == subject_id:
            return label
    return None

def process_ppg_file(file_path: Path) -> None:
    """Process a single PPG file with label information"""
    try:
        # Get subject ID and corresponding label
        subject_id = get_subject_id_from_filename(file_path.name)
        subject_label = get_label_for_subject(subject_id, Label)
        
        if not subject_label:
            print(f"Warning: No label found for subject {subject_id}")
            return
            
        # Load and clean data
        data = pd.read_csv(file_path)
        data = data.dropna().reset_index(drop=True)
        
        # Process timestamp
        data = process_timestamp(data)
        
        # Process PPG signal
        signals, info = nk.ppg_process(data['PG'], sampling_rate=SAMPLING_RATE)
        signals = pd.DataFrame(signals)
        
        # Calculate HRV frequency metrics
        peaks = signals['PPG_Peaks']
        hrv_indices = nk.hrv_frequency(peaks, sampling_rate=SAMPLING_RATE, show=False)
        
        # Add HRV metrics to signals DataFrame
        for column in hrv_indices.columns:
            signals[f'HRV_{column}'] = hrv_indices[column].iloc[0]
        
        signals['DateTime'] = data['DateTime']
        
        # Add label information
        for key, value in subject_label.items():
            signals[key] = value
            
        if len(signals) != len(data):
            print(f"Warning: Length mismatch in {file_path.name}")
            return
        
        # Ensure output directory exists
        PROCESSED_PATH.mkdir(parents=True, exist_ok=True)
        
        # Save processed data
        output_path = PROCESSED_PATH / file_path.name
        signals.to_csv(output_path, index=False)
        print(f'Successfully processed {file_path.name} for subject {subject_id}')
        
    except Exception as e:
        print(f'Error processing {file_path.name}: {str(e)}')

def main():
    """Main function to process all PPG files"""
    # Get list of files to process
    files = list(RAW_PATH.glob('*.csv'))
    
    if not files:
        print(f'No CSV files found in {RAW_PATH}')
        return
        
    print(f'Found {len(files)} files to process')
    
    # Process each file
    for file_path in files:
        process_ppg_file(file_path)

if __name__ == '__main__':
    main()