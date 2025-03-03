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
    """
    df['DateTime'] = pd.to_datetime(df['LocalTimestamp'], unit='s').dt.tz_localize('UTC').dt.tz_convert('Asia/Bangkok')
    return df


def get_subject_id_from_filename(filename: str) -> str:
    """Extract subject ID from filename (e.g., 'S01_PPG.csv' -> 'S01')"""
    return filename.split('_')[0].upper()


def get_label_for_subject(subject_id: str, labels: list) -> dict:
    """Get label information for a specific subject"""
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

        # กำหนดขนาดหน้าต่างเวลาเป็น 5 นาที (300 วินาที)
        WINDOW_SIZE = 300

        # ดึงค่า PPG Peaks และ DateTime
        peaks = signals['PPG_Peaks']
        timestamps = pd.to_datetime(data['DateTime'])
        signals['DateTime'] = timestamps

        # สร้าง DataFrame สำหรับเก็บ HRV
        hrv_results = []

        # หาเวลาจุดเริ่มต้นและจุดสิ้นสุดของข้อมูล
        start_time = timestamps.iloc[0]
        end_time = timestamps.iloc[-1]

        # วนลูปแบ่งข้อมูลเป็นช่วงๆ ละ 5 นาที
        while start_time < end_time:
            window_end = start_time + pd.Timedelta(seconds=WINDOW_SIZE)
            mask = (timestamps >= start_time) & (timestamps < window_end)
            window_peaks = peaks[mask]

            if len(window_peaks) > 1:  # ต้องมีมากกว่า 1 peak เพื่อคำนวณ HRV
                hrv_indices = nk.hrv_frequency(window_peaks, sampling_rate=SAMPLING_RATE, show=False)
                hrv_indices['StartTime'] = start_time
                hrv_indices['EndTime'] = window_end
                hrv_results.append(hrv_indices)

            start_time = window_end  # เลื่อนช่วงเวลาไปที่ถัดไป

        # รวมผลลัพธ์ทั้งหมดเป็น DataFrame
        if hrv_results:
            hrv_df = pd.concat(hrv_results, ignore_index=True)

            # Merge ค่าที่ได้กลับไปยัง signals
            signals = signals.merge(hrv_df, left_on='DateTime', right_on='StartTime', how='left')

            # เพิ่ม label ข้อมูล
            for key, value in subject_label.items():
                signals[key] = value

            # ตรวจสอบความสอดคล้องของขนาดข้อมูล
            if len(signals) != len(data):
                print(f"Warning: Length mismatch in {file_path.name}")

            # บันทึกผลลัพธ์
            PROCESSED_PATH.mkdir(parents=True, exist_ok=True)
            output_path = PROCESSED_PATH / file_path.name
            signals.to_csv(output_path, index=False)
            print(f'Successfully processed {file_path.name} for subject {subject_id}')
        else:
            print(f'No valid HRV data for {file_path.name}')

    except Exception as e:
        print(f'Error processing {file_path.name}: {str(e)}')

def main():
    """Main function to process all PPG files"""
    files = list(RAW_PATH.glob('*.csv'))

    if not files:
        print(f'No CSV files found in {RAW_PATH}')
        return

    print(f'Found {len(files)} files to process')

    for file_path in files:
        process_ppg_file(file_path)


if __name__ == '__main__':
    main()
