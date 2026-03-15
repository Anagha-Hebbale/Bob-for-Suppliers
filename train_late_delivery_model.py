import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib


# 1. Load processed data
def load_data():
    df = pd.read_csv("data/processed_data.csv")
    return df


# 2. Prepare features and target
def prepare_data(df):

    target = "Late_delivery_risk"

    features = [
        "Days for shipping (real)",
        "Days for shipment (scheduled)",
        "shipping_delay",
        "Order Item Quantity",
        "Order Item Product Price",
        "Order Item Discount",
        "month",
        "day_of_week"
    ]

    # keep only features that actually exist in the dataset
    features = [f for f in features if f in df.columns]

    X = df[features]
    y = df[target]

    return X, y


# 3. Train model
def train_model(X_train, y_train):

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )

    model.fit(X_train, y_train)

    return model


# 4. Evaluate model
def evaluate_model(model, X_test, y_test):

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)

    print("\nModel Accuracy:", accuracy)

    print("\nClassification Report:\n")
    print(classification_report(y_test, predictions))


# 5. Save trained model
def save_model(model):

    joblib.dump(model, "models/late_delivery_model.pkl")

    print("\nModel saved to models/late_delivery_model.pkl")


# 6. Main pipeline
if __name__ == "__main__":

    print("Loading processed data...")
    df = load_data()

    print("Preparing features...")
    X, y = prepare_data(df)

    print("Splitting dataset...")
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

    print("\nTraining complete!")