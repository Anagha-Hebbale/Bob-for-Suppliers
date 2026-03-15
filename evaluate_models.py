import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    mean_squared_error,
    r2_score
)


# Load models
def load_models():

    late_model = joblib.load("models/late_delivery_model.pkl")
    profit_model = joblib.load("models/profit_prediction_model.pkl")

    return late_model, profit_model


# Load dataset
def load_data():

    df = pd.read_csv("data/processed_data.csv")

    return df


# Evaluate Late Delivery Model
def evaluate_late_delivery(df, model):

    print("\nEvaluating Late Delivery Model...")

    target = "Late_delivery_risk"
    features = model.feature_names_in_

    X = df[features]
    y = df[target]

    # same 80/20 split used during training
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42
    )

    predictions = model.predict(X_test)

    acc = accuracy_score(y_test, predictions)

    print("\nAccuracy:", acc)

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, predictions))

    print("\nClassification Report:")
    print(classification_report(y_test, predictions))


# Evaluate Profit Prediction Model
def evaluate_profit_model(df, model):

    print("\nEvaluating Profit Prediction Model...")

    target = "Order Profit Per Order"
    features = model.feature_names_in_

    X = df[features]
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42
    )

    predictions = model.predict(X_test)

    rmse = mean_squared_error(y_test, predictions) ** 0.5
    r2 = r2_score(y_test, predictions)

    print("\nRMSE:", rmse)
    print("R² Score:", r2)


# Main
if __name__ == "__main__":

    print("Loading models...")
    late_model, profit_model = load_models()

    print("Loading dataset...")
    df = load_data()

    evaluate_late_delivery(df, late_model)

    evaluate_profit_model(df, profit_model)

    print("\nModel evaluation complete!")