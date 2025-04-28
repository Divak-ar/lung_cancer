import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import os

print("Loading dataset...")
# Load the data
lung_data = pd.read_csv("Dataset/survey lung cancer.csv")

# Preprocess the data
print("Preprocessing data...")
lung_data.GENDER = lung_data.GENDER.map({"M": 1, "F": 2})
lung_data.LUNG_CANCER = lung_data.LUNG_CANCER.map({"YES": 1, "NO": 2})

# Split the data
print("Splitting features and target...")
X = lung_data.iloc[:, 0:-1]
y = lung_data.iloc[:, -1]

# Save feature names
feature_names = X.columns.tolist()
with open('feature_names.txt', 'w') as f:
    f.write(','.join(feature_names))
print(f"Feature names saved to 'feature_names.txt': {', '.join(feature_names)}")

# Train the Random Forest model on the full dataset
print("Training Random Forest model on the entire dataset...")
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier.fit(X, y)

# Export the model
model_path = 'lung_cancer_model.pkl'
joblib.dump(rf_classifier, model_path)
print(f"Model successfully exported to '{model_path}'")

# Create a test prediction to verify the model works
print("\nVerifying model with a test prediction...")
# Sample data (you may adjust these values)
sample = {
    'GENDER': 1,  # Male
    'AGE': 65,
    'SMOKING': 1,  # Yes
    'YELLOW_FINGERS': 1,  # Yes
    'ANXIETY': 1,  # Yes
    'PEER_PRESSURE': 2,  # No
    'CHRONIC DISEASE': 1,  # Yes
    'FATIGUE': 1,  # Yes
    'ALLERGY': 2,  # No
    'WHEEZING': 1,  # Yes
    'ALCOHOL CONSUMING': 1,  # Yes
    'COUGHING': 1,  # Yes
    'SHORTNESS OF BREATH': 1,  # Yes
    'SWALLOWING DIFFICULTY': 1,  # Yes
    'CHEST PAIN': 1  # Yes
}

# Create DataFrame with single sample
sample_df = pd.DataFrame([sample], columns=feature_names)

# Make prediction
prediction = rf_classifier.predict(sample_df)[0]
probability = rf_classifier.predict_proba(sample_df)[0]

# Map prediction back to YES/NO
result = "YES" if prediction == 1 else "NO"
confidence = float(probability[0]) if result == "YES" else float(probability[1])

print(f"Sample prediction result: {result}")
print(f"Confidence: {round(confidence * 100, 2)}%")

print("\nModel export completed successfully. Ready for use in the Flask application.")