import pandas as pd


# 1. Load dataset
def load_data():
    # latin1 encoding fixes the UnicodeDecodeError
    df = pd.read_csv("data/DataCoSupplyChainDataset.csv", encoding="latin1")
    return df


# 2. Clean dataset
def clean_data(df):

    # remove duplicate rows
    df = df.drop_duplicates()

    # convert order date column to datetime
    if 'order date (DateOrders)' in df.columns:
        df['order_date'] = pd.to_datetime(df['order date (DateOrders)'])

    # create useful feature: shipping delay
    if 'Days for shipping (real)' in df.columns and 'Days for shipment (scheduled)' in df.columns:
        df['shipping_delay'] = (
            df['Days for shipping (real)'] -
            df['Days for shipment (scheduled)']
        )

    # drop unnecessary columns (only if they exist)
    columns_to_drop = [
        'Customer Email',
        'Customer Password',
        'Product Description',
        'Order Zipcode'
    ]

    df = df.drop(columns=[col for col in columns_to_drop if col in df.columns])

    # fill missing numeric values with median
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

    # fill missing categorical values with "Unknown"
    categorical_cols = df.select_dtypes(include=['object']).columns
    df[categorical_cols] = df[categorical_cols].fillna("Unknown")

    return df


# 3. Save cleaned dataset
def save_data(df):
    df.to_csv("data/cleaned_data.csv", index=False)


# 4. Main execution
if __name__ == "__main__":

    print("Loading dataset...")
    df = load_data()

    print("Cleaning data...")
    df_clean = clean_data(df)

    print("Saving cleaned dataset...")
    save_data(df_clean)

    print("Preprocessing complete! File saved as data/cleaned_data.csv")