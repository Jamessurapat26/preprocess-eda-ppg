import pandas as pd
import numpy as np
from pathlib import Path

PROCESSED_PATH = Path('Processed/eda')
COMBINED_PATH = Path('Combined/eda')

# Concatenate all PPG files into a single DataFrame
def concat_ppg_files(file_list: list) -> pd.DataFrame:
    """Concatenate all PPG files into a single DataFrame"""
    # Initialize an empty list to store DataFrames
    dataframes = []
    
    # Iterate over each file in the list
    for file in file_list:
        # Load the PPG data from the file
        ppg_data = pd.read_csv(file)
        
        # Append the DataFrame to the list
        dataframes.append(ppg_data)
    
    # Concatenate all DataFrames in the list
    combined_data = pd.concat(dataframes, ignore_index=True)
    
    return combined_data

def main():
    """Main function to concatenate all PPG files"""
    # Get list of files to concatenate
    files = list(PROCESSED_PATH.glob('*.csv'))
    
    if not files:
        print(f'No CSV files found in {PROCESSED_PATH}')
        return
        
    print(f'Found {len(files)} files to concatenate')
    
    # Concatenate all PPG files
    combined_data = concat_ppg_files(files)
    
    # Ensure output directory exists
    COMBINED_PATH.mkdir(parents=True, exist_ok=True)
    
    # Save the combined data
    output_path = COMBINED_PATH / 'combined_eda_data.csv'
    combined_data.to_csv(output_path, index=False)
    
    print(f'Successfully concatenated {len(files)} eda files')

if __name__ == '__main__':
    main()