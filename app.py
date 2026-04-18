from flask import Flask, request, jsonify
import pandas as pd
import joblib

app = Flask(__name__)

# Load saved model files
model = joblib.load("models/anomaly_model.pkl")
scaler = joblib.load("models/scaler.pkl")
label_encoders = joblib.load("models/label_encoders.pkl")

# Home route
@app.route("/")
def home():
    return """
    <h2>Financial Transactions Anomaly Detection API</h2>
    <p>Use the <b>/predict</b> endpoint with POST method to check whether a transaction is normal or anomalous.</p>
    """

# Prediction route
@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        # Convert categorical input values using saved encoders
        transaction_type = label_encoders["transaction_type"].transform([data["transaction_type"]])[0]
        device_type = label_encoders["device_type"].transform([data["device_type"]])[0]
        location = label_encoders["location"].transform([data["location"]])[0]
        merchant_category = label_encoders["merchant_category"].transform([data["merchant_category"]])[0]

        # Create dataframe for model input
        input_data = pd.DataFrame({
            "transaction_type": [transaction_type],
            "device_type": [device_type],
            "location": [location],
            "merchant_category": [merchant_category],
            "hour_of_day": [data["hour_of_day"]],
            "transaction_amount": [data["transaction_amount"]],
            "account_balance_before": [data["account_balance_before"]],
            "balance_after_transaction": [data["balance_after_transaction"]],
            "transactions_last_24h": [data["transactions_last_24h"]],
            "is_international": [data["is_international"]]
        })

        # Scale the input
        input_scaled = scaler.transform(input_data)

        # Predict
        prediction = model.predict(input_scaled)[0]
        prediction_proba = model.predict_proba(input_scaled)[0]

        result = "Anomaly Transaction" if prediction == 1 else "Normal Transaction"

        return jsonify({
            "status": "success",
            "prediction": int(prediction),
            "result": result,
            "probability_normal": float(prediction_proba[0]),
            "probability_anomaly": float(prediction_proba[1])
        })

    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        })

if __name__ == "__main__":
    app.run(debug=True, port=5000)

    