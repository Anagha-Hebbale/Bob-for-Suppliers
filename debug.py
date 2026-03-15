import pandas as pd
import os

print("=" * 50)
print("DEBUGGING SUPPLY CHAIN DATA")
print("=" * 50)

# Check current directory
print("\n1. CURRENT DIRECTORY:")
print(f"   Path: {os.getcwd()}")
print(f"   Files: {[f for f in os.listdir('.') if f.endswith('.py')]}")

# Check data folder
print("\n2. DATA FOLDER:")
if os.path.exists('data'):
    print(f"   Found 'data' folder")
    csv_files = [f for f in os.listdir('data') if f.endswith('.csv')]
    print(f"   CSV files: {csv_files}")
    
    if csv_files:
        # Try to load first CSV
        first_file = csv_files[0]
        file_path = f"data/{first_file}"
        print(f"\n3. LOADING: {first_file}")
        try:
            df = pd.read_csv(file_path)
            print(f"   SUCCESS! Loaded {len(df)} rows, {len(df.columns)} columns")
            print(f"\n4. COLUMNS IN YOUR DATA:")
            for i, col in enumerate(df.columns, 1):
                print(f"   {i}. {col}")
            print(f"\n5. FIRST 3 ROWS:")
            print(df.head(3))
        except Exception as e:
            print(f"   ERROR: {e}")
    else:
        print("   No CSV files found in data folder")
else:
    print("   'data' folder not found!")

# Check current directory for CSVs
print("\n6. CSV FILES IN CURRENT DIRECTORY:")
current_csvs = [f for f in os.listdir('.') if f.endswith('.csv')]
if current_csvs:
    for csv in current_csvs:
        print(f"   - {csv}")
else:
    print("   No CSV files in current directory")

print("\n" + "=" * 50)