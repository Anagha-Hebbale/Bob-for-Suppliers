import pandas as pd


def load_data():
    df = pd.read_csv("data/processed_data.csv")
    return df


def analyze_shipping_costs(df):

    print("\nAverage profit by shipping mode:\n")

    result = df.groupby("Shipping Mode")["Order Profit Per Order"].mean()

    print(result)

    print("\nAverage profit by shipping delay:\n")

    if "shipping_delay" in df.columns:
        delay_analysis = df.groupby("shipping_delay")["Order Profit Per Order"].mean()
        print(delay_analysis)


if __name__ == "__main__":

    df = load_data()

    analyze_shipping_costs(df)