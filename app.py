import os
import joblib
import pandas as pd
import logging
from flask import Flask, request, jsonify

app = Flask(__name__)

# Logging setup
logging.basicConfig(level=logging.INFO)

# Load model and scaler
try:
    model = joblib.load('logistic_regression_model.joblib')
    scaler = joblib.load('scaler.joblib')
    logging.info("Model and scaler loaded successfully.")
except Exception as e:
    logging.error(f"Error loading model/scaler: {e}")
    raise SystemExit("Failed to load model files.")

# Required input columns
REQUIRED_COLUMNS = [
    'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
    'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'
]

# Value ranges
RANGES = {
    'Pregnancies': (0, 20),
    'Glucose': (0, 300),
    'BloodPressure': (0, 200),
    'SkinThickness': (0, 100),
    'Insulin': (0, 900),
    'BMI': (0, 70),
    'DiabetesPedigreeFunction': (0, 3),
    'Age': (0, 120)
}


@app.route('/')
def home():
    return "Diabetes Prediction API Running"


@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok'})


@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()

        if not data:
            return jsonify({'error': 'No input data provided'}), 400

        # Check required fields
        for col in REQUIRED_COLUMNS:
            if col not in data:
                return jsonify({'error': f'Missing field: {col}'}), 400

        # Convert input
        try:
            df = pd.DataFrame([data])
            df = df[REQUIRED_COLUMNS].astype(float)
        except Exception:
            return jsonify({'error': 'All inputs must be numeric'}), 400

        # Validate ranges
        for col, (min_val, max_val) in RANGES.items():
            value = df[col][0]
            if not (min_val <= value <= max_val):
                return jsonify({
                    'error': f'{col} must be between {min_val} and {max_val}'
                }), 400

        # Scale
        scaled_data = scaler.transform(df)

        # Predict
        prediction = model.predict(scaled_data)[0]
        prediction_proba = model.predict_proba(scaled_data)[0]

        # Response
        result = {
            'prediction': int(prediction),
            'probabilities': {
                'no_diabetes': float(prediction_proba[0]),
                'diabetes': float(prediction_proba[1])
            }
        }

        return jsonify(result)

    except Exception as e:
        logging.error(f"Prediction error: {e}")
        return jsonify({'error': 'Internal server error'}), 500


# Important for deployment
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
