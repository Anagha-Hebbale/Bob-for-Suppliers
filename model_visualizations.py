import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix


# Load models
def load_models():

    late_model = joblib.load("models/late_delivery_model.pkl")
    profit_model = joblib.load("models/profit_prediction_model.pkl")

    return late_model, profit_model


# Load dataset
def load_data():

    df = pd.read_csv("data/processed_data.csv")

    return df


# 1️⃣ Confusion Matrix
def plot_confusion_matrix(df, model):

    target = "Late_delivery_risk"
    features = model.feature_names_in_

    X = df[features]
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42
    )

    preds = model.predict(X_test)

    cm = confusion_matrix(y_test, preds)

    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")

    plt.title("Late Delivery Prediction - Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")

    plt.show()


# 2️⃣ Feature Importance
def plot_feature_importance(model):

    features = model.feature_names_in_
    importance = model.feature_importances_

    df_imp = pd.DataFrame({
        "feature": features,
        "importance": importance
    }).sort_values(by="importance", ascending=False)

    plt.figure(figsize=(8,6))

    sns.barplot(
        x="importance",
        y="feature",
        data=df_imp.head(10)
    )

    plt.title("Top Factors Influencing Late Delivery")
    plt.xlabel("Importance")
    plt.ylabel("Feature")

    plt.show()


# 3️⃣ Actual vs Predicted Profit
def plot_profit_predictions(df, model):

    target = "Order Profit Per Order"
    features = model.feature_names_in_

    X = df[features]
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42
    )

    preds = model.predict(X_test)

    plt.figure(figsize=(6,6))

    plt.scatter(y_test, preds, alpha=0.4)

    plt.xlabel("Actual Profit")
    plt.ylabel("Predicted Profit")

    plt.title("Actual vs Predicted Profit")

    plt.show()


# Main
if __name__ == "__main__":

    print("Loading models...")
    late_model, profit_model = load_models()

    print("Loading dataset...")
    df = load_data()

    print("Generating Confusion Matrix...")
    plot_confusion_matrix(df, late_model)

    print("Generating Feature Importance Chart...")
    plot_feature_importance(late_model)

    print("Generating Profit Prediction Chart...")
    plot_profit_predictions(df, profit_model)