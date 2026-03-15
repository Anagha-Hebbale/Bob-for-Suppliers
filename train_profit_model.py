import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import numpy as np


# 1. Load processed data
def load_data():
    df = pd.read_csv("data/processed_data.csv")
    return df


# 2. Prepare features and target
def prepare_data(df):

    target = "Order Profit Per Order"

    features = [
        "Order Item Quantity",
        "Order Item Product Price",
        "Order Item Discount",
        "shipping_delay",
        "Days for shipping (real)",
        "Days for shipment (scheduled)",
        "month",
        "day_of_week"
    ]

    # only keep columns that exist
    features = [f for f in features if f in df.columns]

    X = df[features]
    y = df[target]

    return X, y


# 3. Train model
def train_model(X_train, y_train):

    model = RandomForestRegressor(
        n_estimators=100,
        random_state=42
    )

    model.fit(X_train, y_train)

    return model


# 4. Evaluate model
def evaluate_model(model, X_test, y_test):

    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions)

    print("\nModel Performance")
    print("------------------")
    print("MAE :", mae)
    print("RMSE:", rmse)
    print("R²  :", r2)


# 5. Save model
def save_model(model):

    joblib.dump(model, "models/profit_prediction_model.pkl")

    print("\nModel saved to models/profit_prediction_model.pkl")


# 6. Main pipeline
if __name__ == "__main__":

    print("Loading processed data...")
    df = load_data()

    print("Preparing features...")
    X, y = prepare_data(df)

    print("Splitting dataset (80% train / 20% test)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    print("Training model...")
    model = train_model(X_train, y_train)

    print("Evaluating model...")
    evaluate_model(model, X_test, y_test)

    print("Saving model...")
    save_model(model)

    print("\nProfit prediction training complete!")