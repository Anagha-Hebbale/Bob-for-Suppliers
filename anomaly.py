#!/usr/bin/env python3
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

# Step 1: Load dataset
df = pd.read_csv("DataCoSupplyChainDataset.csv", encoding='latin1')

# Step 2: Feature engineering (Profit Ratio)
df['Profit Ratio'] = df['Order Profit Per Order'] / (df['Sales'] + 1)

# Step 3: Select features for anomaly detection
features = [
    'Order Item Discount',
    'Order Item Discount Rate',
    'Sales',
    'Order Profit Per Order',
    'Order Item Quantity',
    'Profit Ratio'
]

data = df[features].copy()

# Step 4: Handle missing values
data = data.dropna()

# Step 5: Initialize scaler
scaler = StandardScaler()

# Step 6: Train-test split (80/20)
train_data, test_data = train_test_split(
    data,
    test_size=0.2,
    random_state=42
)

# Step 7: Scale data
X_train = scaler.fit_transform(train_data)
X_test = scaler.transform(test_data)

print("Training size:", X_train.shape)
print("Testing size:", X_test.shape)

# Step 8: Train anomaly detection model
model = IsolationForest(
    n_estimators=100,
    contamination=0.02,
    random_state=42
)

model.fit(X_train)

# Step 9: Predict anomalies
predictions = model.predict(X_test)
scores = model.decision_function(X_test)

# Step 10: Attach predictions to real values
test_data = test_data.copy()
test_data['anomaly'] = predictions
test_data['anomaly_score'] = scores

# Step 11: Extract suspicious orders
anomalies = test_data[test_data['anomaly'] == -1]

print("Suspicious orders detected:", len(anomalies))
print(anomalies.head())

# Step 12: Save results
test_data.to_csv("test_anomaly_results.csv", index=False)
anomalies.to_csv("suspicious_orders.csv", index=False)

# Step 13: Create readable labels
test_data['anomaly_label'] = test_data['anomaly'].map({
    1: 'Normal',
    -1: 'Anomaly'
})

# Step 14: Visualization
plt.figure(figsize=(10,6))

sns.scatterplot(
    data=test_data,
    x='Sales',
    y='Order Profit Per Order',
    hue='anomaly_label',
    size='Order Item Discount Rate',  # dot size = discount percentage
    sizes=(20,200),
    palette={'Normal':'blue','Anomaly':'red'},
    alpha=0.6
)

plt.title("Supply Chain Order Anomaly Detection")
plt.xlabel("Sales")
plt.ylabel("Order Profit Per Order")
plt.legend(title="Order Type")
plt.show()