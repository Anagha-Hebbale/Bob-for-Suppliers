import pandas as pd
import joblib


# -----------------------------
# Load models
# -----------------------------
def load_models():

    late_model = joblib.load("models/late_delivery_model.pkl")
    profit_model = joblib.load("models/profit_prediction_model.pkl")
    return_model = joblib.load("models/return_risk_model.pkl")

    return late_model, profit_model, return_model


# -----------------------------
# Load dataset
# -----------------------------
def load_data():

    df = pd.read_csv("data/processed_data.csv")

    return df


# -----------------------------
# Prepare features
# -----------------------------
def prepare_features(df, late_model, profit_model):

    late_features = late_model.feature_names_in_
    profit_features = profit_model.feature_names_in_

    X_late = df[late_features]
    X_profit = df[profit_features]

    return X_late, X_profit


# -----------------------------
# Prepare return model features
# -----------------------------
def prepare_return_features(df):

    features = [
        "Order Item Quantity",
        "Order Item Discount",
        "Order Item Product Price"
    ]

    X_return = df[features]

    return X_return


# -----------------------------
# Generate predictions
# -----------------------------
def generate_predictions(df, late_model, profit_model, return_model, X_late, X_profit, X_return):

    df["predicted_late_delivery"] = late_model.predict(X_late)

    df["predicted_profit"] = profit_model.predict(X_profit)

    df["predicted_return_risk"] = return_model.predict(X_return)

    return df


# -----------------------------
# Shipping cost insights
# -----------------------------
def shipping_cost_insights(df):

    shipping_profit = df.groupby("Shipping Mode")["Order Profit Per Order"].mean()

    print("\nAverage Profit by Shipping Mode:")
    print(shipping_profit)

    return shipping_profit


# -----------------------------
# Seasonal demand insights
# -----------------------------
def seasonal_demand_insights(df):

    if "month" in df.columns:

        seasonal_demand = df.groupby("month")["Order Item Quantity"].sum()

        print("\nSeasonal Demand Pattern:")
        print(seasonal_demand)

        return seasonal_demand

    else:

        print("\nMonth column not found for seasonal analysis.")

        return None


# -----------------------------
# Generate recommendations
# -----------------------------
def generate_recommendations(df):

    recommendations = []

    for _, row in df.iterrows():

        rec = []

        if row["predicted_late_delivery"] == 1:
            rec.append("Use faster shipping")

        if row["predicted_profit"] < 0:
            rec.append("Adjust pricing or reduce discount")

        if row["predicted_return_risk"] == 1:
            rec.append("High return risk - review product listing or discount")

        if not rec:
            rec.append("Order operationally efficient")

        recommendations.append(" | ".join(rec))

    df["recommendation"] = recommendations

    return df


# -----------------------------
# Save results
# -----------------------------
def save_results(df):

    df.to_csv("data/final_operational_insights.csv", index=False)

    print("\nInsights saved to data/final_operational_insights.csv")


# -----------------------------
# MAIN PIPELINE
# -----------------------------
if __name__ == "__main__":

    print("Loading models...")
    late_model, profit_model, return_model = load_models()

    print("Loading dataset...")
    df = load_data()

    print("Preparing features...")
    X_late, X_profit = prepare_features(df, late_model, profit_model)

    X_return = prepare_return_features(df)

    print("Generating predictions...")
    df = generate_predictions(df, late_model, profit_model, return_model, X_late, X_profit, X_return)

    print("Analyzing shipping costs...")
    shipping_cost_insights(df)

    print("Analyzing seasonal demand...")
    seasonal_demand_insights(df)

    print("Generating recommendations...")
    df = generate_recommendations(df)

    print("Saving results...")
    save_results(df)

    print("\nSupply Chain Operational Insights Generated Successfully!")