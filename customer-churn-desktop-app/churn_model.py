# ============================================
# Customer Churn Prediction - Model Training
# ============================================

import pandas as pd
import numpy as np
import pickle
import os
print("Current Working Directory:", os.getcwd())


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# ===============================
# Create model folder if not exist
# ===============================
if not os.path.exists("model"):
    os.makedirs("model")

# ===============================
# Load Dataset
# ===============================
df = pd.read_csv("data/Telco-Customer-Churn.csv")


print("Dataset Loaded Successfully")
print(df.head())

# ===============================
# Data Cleaning
# ===============================

# Convert TotalCharges to numeric
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

# Fill missing values
df["TotalCharges"].fillna(df["TotalCharges"].mean(), inplace=True)

# Drop customerID if exists
if "customerID" in df.columns:
    df.drop("customerID", axis=1, inplace=True)

# ===============================
# Encode Categorical Columns
# ===============================
label_encoders = {}

for col in df.select_dtypes(include="object").columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# ===============================
# Split Features & Target
# ===============================
X = df.drop("Churn", axis=1)
y = df["Churn"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ===============================
# Train Random Forest Model
# ===============================
model = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)

model.fit(X_train, y_train)

# ===============================
# Evaluate Model
# ===============================
y_pred = model.predict(X_test)

print("\nModel Evaluation:")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# ===============================
# Save Model & Encoders
# ===============================
pickle.dump(model, open("model/churn_model.pkl", "wb"))
pickle.dump(label_encoders, open("model/encoders.pkl", "wb"))

print("\nModel and encoders saved successfully!")

# After splitting features & target
feature_names = X.columns.tolist()

# Save feature names
pickle.dump(feature_names, open("model/feature_names.pkl", "wb"))
