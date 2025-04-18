import pandas as pd
import numpy as np
from pathlib import Path
import os

# Path ของไฟล์ที่ต้องการรวม
PROCESSED_PATH_EA = Path('Combined/eda')
PROCESSED_PATH_PG = Path('Combined/ppg')
COMBINED_PATH = Path('Combined/eda_ppg')

def load_all_csv_from_dir(directory_path: Path) -> pd.DataFrame:
    """Load all CSV files from a directory and combine them"""
    if not directory_path.exists():
        print(f"Error: Directory not found - {directory_path}")
        return pd.DataFrame()
    
    if not directory_path.is_dir():
        print(f"Error: {directory_path} is not a directory")
        return pd.DataFrame()
    
    # List all CSV files in the directory
    csv_files = list(directory_path.glob('*.csv'))
    
    if not csv_files:
        print(f"No CSV files found in {directory_path}")
        return pd.DataFrame()
    
    print(f"Found {len(csv_files)} CSV files in {directory_path}")
    
    # Load and combine all CSV files
    all_data = []
    for file_path in csv_files:
        try:
            # Explicitly check if file is accessible
            if not os.access(file_path, os.R_OK):
                print(f"Permission denied accessing {file_path}")
                continue
                
            df = pd.read_csv(file_path, low_memory=False)
            
            # Convert DateTime column to datetime if it exists - with better error handling
            if 'DateTime' in df.columns:
                try:
                    # Try multiple datetime parsing approaches
                    try:
                        # First try the default parsing
                        df['DateTime'] = pd.to_datetime(df['DateTime'])
                    except:
                        # If that fails, try with format='mixed' which is more flexible
                        df['DateTime'] = pd.to_datetime(df['DateTime'], format='mixed')
                except Exception as dt_error:
                    print(f"  - Warning: DateTime conversion error in {file_path.name}: {str(dt_error)}")
                    print(f"  - Attempting alternative datetime parsing...")
                    
                    # Try to handle timezone formats that might be causing issues
                    df['DateTime'] = df['DateTime'].apply(lambda x: 
                        pd.to_datetime(x.split('+')[0] if isinstance(x, str) and '+' in x else x)
                    )
            
            # สำหรับคอลัมน์ที่มีชื่อลงท้ายด้วย _eda หรือ _ppg ให้ตรวจสอบรูปแบบข้อมูล
            for col_suffix in ['_eda', '_ppg']:
                for col_prefix in ['gender', 'type', 'sleep', 'bmi', 'bmi_category', 'id']:
                    col_name = f"{col_prefix}{col_suffix}"
                    if col_name in df.columns:
                        # แน่ใจว่าข้อมูลในคอลัมน์เหล่านี้มีรูปแบบที่เหมาะสม
                        if col_prefix in ['bmi', 'sleep']:
                            # พยายามแปลงเป็นตัวเลขถ้าเป็นไปได้
                            try:
                                df[col_name] = pd.to_numeric(df[col_name], errors='coerce')
                            except:
                                pass
            
            print(f"  - Loaded {file_path.name}: {df.shape[0]} rows, {df.shape[1]} columns")
            all_data.append(df)
        except Exception as e:
            print(f"  - Error loading {file_path.name}: {str(e)}")
    
    if not all_data:
        print(f"No data could be loaded from {directory_path}")
        return pd.DataFrame()
    
    # Combine all dataframes
    combined_df = pd.concat(all_data, ignore_index=True)
    print(f"Combined data from {directory_path}: {combined_df.shape[0]} rows, {combined_df.shape[1]} columns")
    
    return combined_df

def merge_eda_ppg(eda_df: pd.DataFrame, ppg_df: pd.DataFrame) -> pd.DataFrame:
    """Merge EDA and PPG data on DateTime and combine duplicate columns"""
    if eda_df.empty or ppg_df.empty:
        print("Warning: One or both dataframes are empty. Skipping merge.")
        return pd.DataFrame()

    # Ensure DateTime column exists in both dataframes
    for df, name in [(eda_df, 'EDA'), (ppg_df, 'PPG')]:
        if 'DateTime' not in df.columns:
            print(f"Error: DateTime column missing in {name} data")
            return pd.DataFrame()
    
    # ข้อมูลคอลัมน์ที่ต้องการรวม
    duplicate_columns = [
        ('gender_eda', 'gender_ppg', 'gender'),
        ('type_eda', 'type_ppg', 'type'),
        ('sleep_eda', 'sleep_ppg', 'sleep'),
        ('bmi_eda', 'bmi_ppg', 'bmi'),
        ('bmi_category_eda', 'bmi_category_ppg', 'bmi_category'),
        ('id_eda', 'id_ppg', 'id')
    ]
    
    # เปลี่ยนชื่อคอลัมน์ก่อน merge ถ้ามีคอลัมน์นั้น
    for eda_col, ppg_col, new_col in duplicate_columns:
        if eda_col in eda_df.columns:
            eda_df = eda_df.rename(columns={eda_col: f"{new_col}_eda"})
        if ppg_col in ppg_df.columns:
            ppg_df = ppg_df.rename(columns={ppg_col: f"{new_col}_ppg"})
    
    # Standardize DateTime formats
    print("Standardizing DateTime formats...")
    
    # First convert both to datetime (if not already)
    eda_df['DateTime'] = pd.to_datetime(eda_df['DateTime'])
    ppg_df['DateTime'] = pd.to_datetime(ppg_df['DateTime'])
    
    # Remove timezone info from both
    if hasattr(eda_df['DateTime'].dt, 'tz'):
        eda_df['DateTime'] = eda_df['DateTime'].dt.tz_localize(None)
    if hasattr(ppg_df['DateTime'].dt, 'tz'):
        ppg_df['DateTime'] = ppg_df['DateTime'].dt.tz_localize(None)
    
    # Make sure DateTime is properly sorted
    eda_df = eda_df.sort_values('DateTime')
    ppg_df = ppg_df.sort_values('DateTime')
    
    # Debug: Print the first few timestamps
    print("\nFirst 3 timestamps in EDA data after standardization:")
    print(eda_df['DateTime'].head(3))
    print("\nFirst 3 timestamps in PPG data after standardization:")
    print(ppg_df['DateTime'].head(3))

    # Merge using asof join to find nearest timestamps
    print("Performing merge...")
    combined_df = pd.merge_asof(
        eda_df,
        ppg_df,
        on='DateTime',
        direction='nearest', 
        tolerance=pd.Timedelta('1s'),
        suffixes=('_eda', '_ppg')
    )

    # รวมคอลัมน์ที่ซ้ำกัน
    for _, _, new_col in duplicate_columns:
        eda_col = f"{new_col}_eda"
        ppg_col = f"{new_col}_ppg"
        
        if eda_col in combined_df.columns and ppg_col in combined_df.columns:
            # ใช้ค่าจาก EDA ก่อน ถ้าไม่มีค่าใน EDA ให้ใช้ค่าจาก PPG
            combined_df[new_col] = combined_df[eda_col].fillna(combined_df[ppg_col])
            # ลบคอลัมน์เดิม
            combined_df = combined_df.drop(columns=[eda_col, ppg_col])
            print(f"Combined columns: {eda_col} + {ppg_col} -> {new_col}")
    
    # Report on the results
    print(f"Successfully merged: {combined_df.shape[0]} rows, {combined_df.shape[1]} columns")
    
    if not combined_df.empty:
        print(f"Time range: {combined_df['DateTime'].min()} to {combined_df['DateTime'].max()}")
    
    return combined_df

def main():
    """Main function to merge EDA and PPG data"""
    # Load all data from directories
    print("\n=== Loading EDA data ===")
    eda_data = load_all_csv_from_dir(PROCESSED_PATH_EA)
    
    print("\n=== Loading PPG data ===")
    ppg_data = load_all_csv_from_dir(PROCESSED_PATH_PG)

    # Merge the data
    print("\n=== Merging datasets ===")
    combined_data = merge_eda_ppg(eda_data, ppg_data)

    if combined_data.empty:
        print("No data to save. Exiting.")
        return

    # Create output directory
    print("\n=== Saving combined data ===")
    COMBINED_PATH.mkdir(parents=True, exist_ok=True)

    # ทดสอบสิทธิ์การเขียน
    try:
        # สร้างไฟล์ชั่วคราวเพื่อทดสอบการเขียนไฟล์
        test_path = COMBINED_PATH / 'test_write_permission.txt'
        with open(test_path, 'w') as f:
            f.write('test')
        # ลบไฟล์ทดสอบ
        test_path.unlink()
        print(f"Write permission to {COMBINED_PATH} confirmed")
    except PermissionError:
        print(f"WARNING: No write permission to {COMBINED_PATH}")
        # ลองใช้ชื่อไฟล์และตำแหน่งอื่น
        output_path = Path(f'combined_eda_ppg_data.csv')
        print(f"Trying to save to current directory instead: {output_path}")
    except Exception as e:
        print(f"Error testing write permission: {str(e)}")

    try:
        # Save the combined data
        output_path = COMBINED_PATH / f'combined_eda_ppg_data.csv'
        print(f"Saving to: {output_path}")
        combined_data.to_csv(output_path, index=False, float_format="%.6f")
        print(f"Successfully saved merged data to {output_path}")
        print(f"   Rows: {combined_data.shape[0]}, Columns: {combined_data.shape[1]}")
    except PermissionError:
        # ถ้ายังไม่สามารถบันทึกได้ ลองบันทึกในไดเรกทอรีปัจจุบัน
        fallback_path = Path(f'combined_eda_ppg_data.csv')
        print(f"Permission denied. Saving to current directory instead: {fallback_path}")
        combined_data.to_csv(fallback_path, index=False, float_format="%.6f")
        print(f"Successfully saved merged data to {fallback_path}")
        print(f"   Rows: {combined_data.shape[0]}, Columns: {combined_data.shape[1]}")
    except Exception as e:
        print(f"Error saving data: {str(e)}")
        print("Trying one last method to save data...")
        try:
            # ลองบันทึกในไดเรกทอรีที่โปรแกรมทำงานอยู่
            desktop_path = Path.home() / "Desktop" / f'combined_eda_ppg_data.csv'
            print(f"Saving to desktop: {desktop_path}")
            combined_data.to_csv(desktop_path, index=False, float_format="%.6f")
            print(f"Successfully saved merged data to {desktop_path}")
        except Exception as e2:
            print(f"All saving attempts failed: {str(e2)}")

if __name__ == '__main__':
    main()
