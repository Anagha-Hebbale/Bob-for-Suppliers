import pandas as pd


def load_data():
    df = pd.read_csv("data/cleaned_data.csv")
    return df


def create_features(df):

    # convert order_date again just to be safe
    df['order_date'] = pd.to_datetime(df['order_date'])

    # time-based features
    df['month'] = df['order_date'].dt.month
    df['year'] = df['order_date'].dt.year
    df['day_of_week'] = df['order_date'].dt.dayofweek

    # profit margin feature
    if 'Order Item Total' in df.columns and 'Order Profit Per Order' in df.columns:
        df['profit_margin'] = df['Order Profit Per Order'] / df['Order Item Total']

    # high discount flag
    if 'Order Item Discount' in df.columns:
        df['high_discount'] = df['Order Item Discount'] > 0.2

    return df


def save_data(df):
    df.to_csv("data/processed_data.csv", index=False)


if __name__ == "__main__":

    print("Loading cleaned dataset...")
    df = load_data()

    print("Creating features...")
    df = create_features(df)

    print("Saving processed dataset...")
    save_data(df)

    print("Feature engineering complete!")