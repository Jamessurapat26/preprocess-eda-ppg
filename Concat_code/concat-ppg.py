import pandas as pd
import numpy as np
from pathlib import Path

PROCESSED_PATH = Path('Processed/ppg')
COMBINED_PATH = Path('Combined/ppg')


# กำหนด dtype ให้เป็น string ก่อนแล้วแปลงทีหลัง
def concat_ppg_files(file_list: list) -> pd.DataFrame:
    """Concatenate all PPG files into a single DataFrame"""
    dataframes = []

    for file in file_list:
        try:
            ppg_data = pd.read_csv(file, dtype=str, low_memory=False)  # อ่านทุกคอลัมน์เป็น string
            dataframes.append(ppg_data)
        except Exception as e:
            print(f"Error reading {file}: {e}")

    if not dataframes:
        print("No valid dataframes to concatenate.")
        return pd.DataFrame()

    combined_data = pd.concat(dataframes, ignore_index=True)

    # แปลงคอลัมน์ 16 และ 17 ให้เป็น numeric (ถ้าเป็นไปได้)
    cols_to_convert = [16, 17]
    for col in cols_to_convert:
        col_name = combined_data.columns[col] if col < len(combined_data.columns) else None
        if col_name:
            combined_data[col_name] = pd.to_numeric(combined_data[col_name],
                                                    errors='coerce')  # แปลงค่าที่เป็นตัวเลข ถ้าไม่ได้ให้เป็น NaN

    return combined_data


def main():
    """Main function to concatenate all PPG files"""
    files = list(PROCESSED_PATH.glob('*.csv'))

    if not files:
        print(f'No CSV files found in {PROCESSED_PATH}')
        return

    print(f'Found {len(files)} files to concatenate')

    combined_data = concat_ppg_files(files)

    if combined_data.empty:
        print("No data to save.")
        return

    COMBINED_PATH.mkdir(parents=True, exist_ok=True)

    output_path = COMBINED_PATH / 'combined_ppg_data.csv'

    try:
        combined_data.to_csv(output_path, index=False)
        print(f'Successfully concatenated {len(files)} PPG files and saved to {output_path}')
    except Exception as e:
        print(f"Error saving file: {e}")


if __name__ == '__main__':
    main()
