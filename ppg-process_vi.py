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
    return data.get('label', [])


# Constants
SAMPLING_RATE = 100
WINDOW_SIZE = 300  # 5 นาที (300 วินาที)
RAW_PATH = Path('Raw/ppg')
PROCESSED_PATH = Path('Processed/ppg')
Label = load_json("label.json")


def process_timestamp(df: pd.DataFrame) -> pd.DataFrame:
    """Process timestamp values"""
    if 'LocalTimestamp' not in df.columns:
        print(f"⚠️ Warning: 'LocalTimestamp' column missing in data")
        return df

    df = df.copy()
    df['DateTime'] = pd.to_datetime(df['LocalTimestamp'], unit='s', errors='coerce')

    if df['DateTime'].isna().sum() > 0:
        print(f"⚠️ Warning: Some DateTime values could not be converted")

    df['DateTime'] = df['DateTime'].dt.tz_localize('UTC').dt.tz_convert('Asia/Bangkok')
    return df


def get_subject_id_from_filename(filename: str) -> str:
    """Extract subject ID from filename"""
    return filename.split('_')[0].upper()


def get_label_for_subject(subject_id: str, labels: list) -> dict:
    """Get label for a specific subject"""
    subject_id = subject_id.upper()
    for label in labels:
        if label.get('id', '').upper() == subject_id:
            return label
    return None


def process_ppg_file(file_path: Path) -> None:
    """Process a single PPG file with label information"""
    try:
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
        signals, info = nk.eda_process(data['PG'], sampling_rate=SAMPLING_RATE)
        signals = pd.DataFrame(signals)

        signals['DateTime'] = data['DateTime']


        # Get the PPG signal column (try to identify it or use the second column)
        # Try common PPG column names first
        ppg_columns = [col for col in data.columns if
                       any(ppg in col.lower() for ppg in ['ppg', 'pg', 'photoplethysmography', 'pulse'])]

        if ppg_columns:
            ppg_column = ppg_columns[0]
            print(f"Using column '{ppg_column}' as PPG signal")
        else:
            # If no clear PPG column, use the first numeric column that's not the timestamp
            numeric_cols = [col for col in data.columns if data[col].dtype.kind in 'if' and col != 'DateTime']
            if numeric_cols:
                ppg_column = numeric_cols[0]
                print(f"Using column '{ppg_column}' as PPG signal (best guess)")
            else:
                # Last resort: use the second column (idx 1) if it exists
                if len(data.columns) > 1:
                    ppg_column = data.columns[1]
                    print(f"Using column '{ppg_column}' as PPG signal (fallback to second column)")
                else:
                    raise ValueError(f"Cannot identify PPG signal column in {file_path.name}")

        # Ensure PPG data is numeric, replace any non-numeric values with NaN
        data[ppg_column] = pd.to_numeric(data[ppg_column], errors='coerce')
        if data[ppg_column].isna().any():
            data[ppg_column] = data[ppg_column].interpolate(method='linear')
            print(f"Filled {data[ppg_column].isna().sum()} NaN values in PPG signal")

        # Process PPG signal
        signals, info = nk.ppg_process(data[ppg_column], sampling_rate=SAMPLING_RATE)
        signals = pd.DataFrame(signals)
        signals['DateTime'] = data['DateTime'].values

        # Extract peaks for HRV analysis
        peaks = signals['PPG_Peaks'].values
        peak_indices = np.where(peaks == 1)[0]
        print(f"Found {len(peak_indices)} peaks in {file_path.name}")

        # Create a data structure to hold all processed data
        processed_data = signals.copy()

        # Merge PPG signal processing data with original data
        # Get the signals without DateTime to avoid duplicate columns
        signals_without_dt = signals.drop('DateTime', axis=1, errors='ignore')

        # Since signals has the same length as data, we can just add the columns
        for col in signals_without_dt.columns:
            processed_data[col] = signals_without_dt[col].values

        # Only proceed with HRV if we have enough peaks
        if len(peak_indices) > 10:  # Need enough peaks for meaningful HRV
            # Compute all HRV metrics on the entire signal
            hrv_time = nk.hrv_time(peak_indices, sampling_rate=SAMPLING_RATE, show=False)
            hrv_freq = nk.hrv_frequency(peak_indices, sampling_rate=SAMPLING_RATE, show=False)
            hrv_nonlinear = nk.hrv_nonlinear(peak_indices, sampling_rate=SAMPLING_RATE, show=False)

            # Combine all HRV metrics
            hrv_all = pd.concat([hrv_time, hrv_freq, hrv_nonlinear], axis=1)

            # Add HRV metrics to all rows
            # These are global metrics for the entire recording, so we replicate them for each row
            hrv_metrics = ['HRV_ULF', 'HRV_VLF', 'HRV_LF', 'HRV_HF', 'HRV_VHF', 'HRV_TP',
                           'HRV_LFHF', 'HRV_LFn', 'HRV_HFn', 'HRV_LnHF']

            for col in hrv_all.columns:
                if col in hrv_metrics:
                    processed_data[col] = hrv_all[col].values[0]

            # Ensure all required HRV metrics exist, set to NaN if they don't
            for metric in hrv_metrics:
                if metric not in processed_data.columns:
                    print(f"Adding missing metric {metric} with NaN values")
                    processed_data[metric] = np.nan

            # Add beat-to-beat HRV metrics if possible
            if len(peak_indices) > 2:  # Need at least 3 peaks for RR intervals
                # Calculate RR intervals
                rr_intervals = np.diff(peak_indices) / SAMPLING_RATE

                # Convert to timezone-naive datetime
                rr_df = pd.DataFrame({
                    'RR': rr_intervals,
                    'DateTime': pd.to_datetime(data['DateTime'].iloc[peak_indices[1:]].dt.tz_localize(None))
                })

                # Compute rolling HRV metrics for shorter windows
                window_size = 300  # seconds
                print(f"Computing rolling HRV metrics with {window_size}s window")

                # Create columns for rolling metrics
                processed_data['RR_Mean'] = np.nan
                processed_data['RMSSD'] = np.nan
                processed_data['SDNN'] = np.nan

                # Use a faster vectorized approach for larger datasets
                if len(data) > 10000:
                    sample_indices = np.arange(0, len(data), 100)
                    print(f"Large dataset detected. Sampling {len(sample_indices)} points for efficiency.")
                else:
                    sample_indices = range(len(data))

                for i in sample_indices:
                    # Convert to timezone-naive datetime for comparison
                    current_time = data['DateTime'].iloc[i].tz_localize(None)
                    window_start = current_time - pd.Timedelta(seconds=window_size / 2)
                    window_end = current_time + pd.Timedelta(seconds=window_size / 2)

                    # Get RR intervals in the current window
                    mask = (rr_df['DateTime'] >= window_start) & (rr_df['DateTime'] <= window_end)
                    window_rr = rr_df.loc[mask, 'RR'].values

                    # Only compute if we have enough RR intervals
                    if len(window_rr) > 5:  # Need at least a few beats for meaningful HRV
                        # Simple time domain metrics
                        processed_data.at[i, 'RR_Mean'] = np.mean(window_rr)
                        processed_data.at[i, 'RMSSD'] = np.sqrt(np.mean(np.diff(window_rr) ** 2)) if len(
                            window_rr) > 1 else np.nan
                        processed_data.at[i, 'SDNN'] = np.std(window_rr)

                # Fill NaN values with interpolation
                for col in ['RR_Mean', 'RMSSD', 'SDNN']:
                    processed_data[col] = processed_data[col].interpolate(method='linear')
        else:
            print(f"Not enough peaks in {file_path.name} for HRV analysis. Setting HRV metrics to NaN.")
            # Set all HRV metrics to NaN if we don't have enough peaks
            hrv_metrics = ['HRV_ULF', 'HRV_VLF', 'HRV_LF', 'HRV_HF', 'HRV_VHF', 'HRV_TP',
                           'HRV_LFHF', 'HRV_LFn', 'HRV_HFn', 'HRV_LnHF',
                           'RR_Mean', 'RMSSD', 'SDNN']
            for metric in hrv_metrics:
                processed_data[metric] = np.nan


        #resample data to 1 second
        # แยกคอลัมน์ที่เป็น object ออกมา
        object_cols = processed_data.select_dtypes(include=['object'])
        processed_data.set_index('DateTime', inplace=True)

        numeric_data = processed_data.select_dtypes(include=['number'])
        resampled_data = numeric_data.resample('1s').mean(numeric_only=True)

        final_data = pd.concat([resampled_data, object_cols], axis=1)
        processed_data = final_data

        # Add subject label information
        for key, value in subject_label.items():
            processed_data[key] = value

        # Save processed data
        PROCESSED_PATH.mkdir(parents=True, exist_ok=True)
        output_path = PROCESSED_PATH / file_path.name
        processed_data.to_csv(output_path, index=True)
        print(f'Successfully processed {file_path.name} for subject {subject_id}')


    except Exception as e:
        print(f'Error processing {file_path.name}: {str(e)}')
        print(traceback.format_exc())

def main():
    """Main function to process all PPG files"""
    files = list(RAW_PATH.glob('*.csv'))

    if not files:
        print(f'❌ No CSV files found in {RAW_PATH}')
        return

    print(f'🔍 Found {len(files)} files to process')

    for file_path in files:
        process_ppg_file(file_path)


if __name__ == '__main__':
    main()