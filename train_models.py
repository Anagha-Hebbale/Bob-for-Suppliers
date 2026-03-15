import pandas as pd
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

# 1. SET UP PATHS
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_path = os.path.join(base_dir, 'data', 'cleaned_data.csv')
model_dir = os.path.join(base_dir, 'models')

# Ensure the models directory exists
os.makedirs(model_dir, exist_ok=True)

# 2. LOAD DATA
df = pd.read_csv(data_path)

# 3. PREPARE FEATURES & ENCODERS
# We need to save the Encoders so the Backend can use them later
features_to_encode = ['Type', 'Shipping Mode', 'Order Region']
other_features = ['Days for shipment (scheduled)', 'Product Price']
target_cols = {
    'late_delivery': 'Late_delivery_risk',
    'abs_loss': 'Absolute_Loss',
    'margin_loss': 'Shipping_Margin_Loss'
}

encoders = {}
X = df[features_to_encode + other_features].copy()

for col in features_to_encode:
    le = LabelEncoder()
    # Fill empty values with 'Unknown' to prevent crashes
    X[col] = le.fit_transform(X[col].astype(str))
    encoders[col] = le  # Store the "translator" for this column

# 4. TRAIN AND SAVE MODELS
print("🚀 Starting AI Training Phase...")

for name, target in target_cols.items():
    y = df[target]

    # Split: 80% to learn, 20% to test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize and Train
    # We use a Random Forest: a collection of decision trees
    model = RandomForestClassifier(n_estimators=100, max_depth=12, random_state=42)
    model.fit(X_train, y_train)

    # Evaluate
    acc = accuracy_score(y_test, model.predict(X_test))
    print(f"✅ Model: {name.upper()} | Accuracy: {acc:.2%}")

    # Save the individual model
    joblib.dump(model, os.path.join(model_dir, f'{name}_model.pkl'))

# 5. THE CRITICAL BACKEND STEP: Save the Encoders
# Without this file, the backend cannot understand text input
joblib.dump(encoders, os.path.join(model_dir, 'encoders.pkl'))

print(f"\n✨ TRAINING COMPLETE ✨")
print(f"All models and translators saved in: {model_dir}")