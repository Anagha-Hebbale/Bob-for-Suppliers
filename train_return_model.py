import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


def load_data():
    df = pd.read_csv("data/processed_data.csv")
    return df


def prepare_data(df):

    df["Return_Risk"] = (df["Order Status"] == "RETURNED").astype(int)

    features = [
        "Order Item Quantity",
        "Order Item Discount",
        "Order Item Product Price"
    ]

    X = df[features]
    y = df["Return_Risk"]

    return X, y


def train_model(X, y):

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42
    )

    model = RandomForestClassifier()

    model.fit(X_train, y_train)

    preds = model.predict(X_test)

    acc = accuracy_score(y_test, preds)

    print("Return Risk Model Accuracy:", acc)

    return model


def save_model(model):

    joblib.dump(model, "models/return_risk_model.pkl")

    print("Model saved.")


if __name__ == "__main__":

    df = load_data()

    X, y = prepare_data(df)

    model = train_model(X, y)

    save_model(model)