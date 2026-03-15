import pandas as pd
import matplotlib.pyplot as plt


def load_data():

    df = pd.read_csv("data/processed_data.csv")

    return df


def seasonal_analysis(df):

    if "month" not in df.columns:
        print("Month column not found.")
        return

    demand = df.groupby("month")["Order Item Quantity"].sum()

    print("\nSeasonal demand pattern:\n")
    print(demand)

    plt.figure(figsize=(8,5))

    demand.plot(kind="bar")

    plt.title("Seasonal Demand Pattern")
    plt.xlabel("Month")
    plt.ylabel("Total Orders")

    plt.show()


if __name__ == "__main__":

    df = load_data()

    seasonal_analysis(df)