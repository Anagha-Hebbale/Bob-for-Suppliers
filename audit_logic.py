import pandas as pd
import os

# --- 1. SET UP PATHS ---
# This looks for your data folder outside the scripts folder
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
input_file = os.path.join(base_dir, 'data', 'DataCoSupplyChainDataset.csv')
output_file = os.path.join(base_dir, 'data', 'cleaned_data.csv')

try:
    # --- 2. LOAD & SAMPLE ---
    df = pd.read_csv(input_file, encoding='ISO-8859-1')

    # Use only 80% of the data
    df_80 = df.sample(frac=0.8, random_state=42).reset_index(drop=True)
    print(f"Successfully loaded {len(df_80)} rows (80% of original).")

    # --- 3. CREATE RISK COLUMNS ---

    # Target 1: Late Delivery (Already in dataset, but we ensure it's binary)
    # 1 = Late Risk, 0 = No Risk
    df_80['Late_delivery_risk'] = df_80['Late_delivery_risk'].astype(int)

    # Target 2: Absolute Loss (Profit < 0)
    df_80['Absolute_Loss'] = (df_80['Order Profit Per Order'] < 0).astype(int)

    # Target 3: Shipping Margin Loss (Profit < Shipping Cost)
    # We use 'Days for shipment (scheduled)' as a proxy for shipping expense
    df_80['Shipping_Margin_Loss'] = (df_80['Order Profit Per Order'] < df_80['Days for shipment (scheduled)']).astype(
        int)

    # --- 4. AUDIT REPORT ---
    print("\n--- DATA AUDIT REPORT ---")
    print(f"Total Late Orders: {df_80['Late_delivery_risk'].sum()}")
    print(f"Total Absolute Losses: {df_80['Absolute_Loss'].sum()}")
    print(f"Total Margin Losses: {df_80['Shipping_Margin_Loss'].sum()}")

    # --- 5. SAVE CLEANED DATA ---
    df_80.to_csv(output_file, index=False)
    print(f"\nSaved audited data to: {output_file}")

except FileNotFoundError:
    print(f"ERROR: Could not find the file at {input_file}")
    print("Make sure your CSV is in the 'data' folder and named exactly 'DataCoSupplyChainDataset.csv'")