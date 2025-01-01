from flask import Flask, request, jsonify
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

# Load the trained model
model = pickle.load(open('models/model.pkl', 'rb'))

@app.route('/')
def home():
    return "Welcome to the Customer Churn Prediction API!"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get data from POST request
        data = request.get_json()
        df = pd.DataFrame(data)

        # Preprocess data (scale features as needed)
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(df)

        # Make prediction
        prediction = model.predict(scaled_data)

        # Return prediction result
        return jsonify({
            'prediction': prediction.tolist()
        })

    except Exception as e:
        return jsonify({
            'error': str(e)
        })

if __name__ == '__main__':
    app.run(debug=True)
