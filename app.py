from flask import Flask, request, jsonify, render_template
import pandas as pd
import numpy as np
import joblib
from flask_cors import CORS
import os

app = Flask(__name__)
CORS(app)

# Load the trained model
model_path = os.path.join(os.path.dirname(__file__), 'lung_cancer_model.pkl')
model = joblib.load(model_path)

# Load feature names
feature_names_path = os.path.join(os.path.dirname(__file__), 'feature_names.txt')
with open(feature_names_path, 'r') as f:
    feature_names = f.read().split(',')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get JSON data from request
        data = request.get_json(force=True)
        
        # Create a DataFrame with the expected format
        input_data = {}
        
        # Map GENDER from string to numeric
        if 'GENDER' in data:
            input_data['GENDER'] = 1 if data['GENDER'].upper() == 'M' else 2

        # Process all numeric fields
        numeric_fields = [
            'AGE', 'SMOKING', 'YELLOW_FINGERS', 'ANXIETY', 
            'PEER_PRESSURE', 'CHRONIC DISEASE', 'FATIGUE', 
            'ALLERGY', 'WHEEZING', 'ALCOHOL CONSUMING', 
            'COUGHING', 'SHORTNESS OF BREATH', 
            'SWALLOWING DIFFICULTY', 'CHEST PAIN'
        ]
        
        for field in numeric_fields:
            if field in data:
                input_data[field] = int(data[field])
            else:
                return jsonify({'error': f'Missing required field: {field}'}), 400
        
        # Create DataFrame with the correct column order
        df = pd.DataFrame([input_data], columns=feature_names)
        
        # Make prediction
        prediction = model.predict(df)[0]
        probability = model.predict_proba(df)[0]
        
        # Map prediction back to YES/NO
        result = "YES" if prediction == 1 else "NO"
        confidence = float(probability[0]) if result == "YES" else float(probability[1])
        
        # Return prediction
        return jsonify({
            'prediction': result,
            'confidence': round(confidence * 100, 2),
            'message': f"The model predicts: {result} ({round(confidence * 100, 2)}% confidence)"
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/feature-info', methods=['GET'])
def feature_info():
    """Returns information about the features needed for prediction"""
    feature_descriptions = {
        'GENDER': "Patient's gender (M for Male, F for Female)",
        'AGE': "Patient's age (numeric)",
        'SMOKING': "Smoking status (1 for Yes, 2 for No)",
        'YELLOW_FINGERS': "Presence of yellow fingers (1 for Yes, 2 for No)",
        'ANXIETY': "Presence of anxiety (1 for Yes, 2 for No)",
        'PEER_PRESSURE': "Presence of peer pressure (1 for Yes, 2 for No)",
        'CHRONIC DISEASE': "Presence of chronic disease (1 for Yes, 2 for No)",
        'FATIGUE': "Presence of fatigue (1 for Yes, 2 for No)",
        'ALLERGY': "Presence of allergies (1 for Yes, 2 for No)",
        'WHEEZING': "Presence of wheezing (1 for Yes, 2 for No)",
        'ALCOHOL CONSUMING': "Alcohol consumption (1 for Yes, 2 for No)",
        'COUGHING': "Presence of coughing (1 for Yes, 2 for No)",
        'SHORTNESS OF BREATH': "Shortness of breath (1 for Yes, 2 for No)",
        'SWALLOWING DIFFICULTY': "Difficulty swallowing (1 for Yes, 2 for No)",
        'CHEST PAIN': "Presence of chest pain (1 for Yes, 2 for No)"
    }
    
    return jsonify({
        'features': feature_descriptions,
        'notes': "For binary features, 1 typically means 'Yes' and 2 means 'No'"
    })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)