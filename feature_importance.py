import joblib
import pandas as pd


# 1. Load trained models
def load_models():

    late_model = joblib.load("models/late_delivery_model.pkl")
    profit_model = joblib.load("models/profit_prediction_model.pkl")

    return late_model, profit_model


# 2. Get feature importance
def get_feature_importance(model):

    features = model.feature_names_in_
    importance = model.feature_importances_

    importance_df = pd.DataFrame({
        "feature": features,
        "importance": importance
    })

    importance_df = importance_df.sort_values(
        by="importance",
        ascending=False
    )

    return importance_df


# 3. Save results
def save_results(late_df, profit_df):

    late_df.to_csv("data/late_delivery_feature_importance.csv", index=False)
    profit_df.to_csv("data/profit_feature_importance.csv", index=False)

    print("\nFeature importance files saved in data/")


# 4. Main pipeline
if __name__ == "__main__":

    print("Loading models...")
    late_model, profit_model = load_models()

    print("Calculating feature importance...")

    late_importance = get_feature_importance(late_model)
    profit_importance = get_feature_importance(profit_model)

    print("\nTop factors affecting late delivery:\n")
    print(late_importance.head(10))

    print("\nTop factors affecting profit:\n")
    print(profit_importance.head(10))

    print("\nSaving results...")
    save_results(late_importance, profit_importance)

    print("\nFeature importance analysis complete!")