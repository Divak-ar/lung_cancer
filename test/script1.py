import requests
import json

# API endpoint
url = 'http://localhost:5000/predict'

# Sample patient data
sample_data = {
    'GENDER': 'M',
    'AGE': 60,
    'SMOKING': 1,
    'YELLOW_FINGERS': 1,
    'ANXIETY': 1,
    'PEER_PRESSURE': 2,
    'CHRONIC DISEASE': 1, 
    'FATIGUE': 2,
    'ALLERGY': 1,
    'WHEEZING': 2,
    'ALCOHOL CONSUMING': 1,
    'COUGHING': 2,
    'SHORTNESS OF BREATH': 2,
    'SWALLOWING DIFFICULTY': 1,
    'CHEST PAIN': 2
}

# Send POST request with JSON data
response = requests.post(url, json=sample_data)

# Check response
print("Status Code:", response.status_code)
print("Response:")
print(json.dumps(response.json(), indent=4))

# Try with a different patient
print("\nTrying with a different patient profile...")
sample_data2 = {
    'GENDER': 'F',
    'AGE': 45,
    'SMOKING': 2,
    'YELLOW_FINGERS': 2,
    'ANXIETY': 2,
    'PEER_PRESSURE': 2,
    'CHRONIC DISEASE': 2, 
    'FATIGUE': 1,
    'ALLERGY': 2,
    'WHEEZING': 2,
    'ALCOHOL CONSUMING': 2,
    'COUGHING': 1,
    'SHORTNESS OF BREATH': 1,
    'SWALLOWING DIFFICULTY': 2,
    'CHEST PAIN': 2
}

response = requests.post(url, json=sample_data2)
print("Status Code:", response.status_code)
print("Response:")
print(json.dumps(response.json(), indent=4))