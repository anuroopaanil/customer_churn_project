import joblib
import pandas as pd

def make_prediction(model_path, input_data):
    # Load trained model
    model = joblib.load(model_path)

    # Make prediction
    prediction = model.predict(input_data)
    return prediction

if __name__ == "__main__":
    # Example usage
    input_data = pd.DataFrame([[0.1, 0.2, 0.3, 0.4]])  # Replace with actual data
    prediction = make_prediction("scripts/model.pkl", input_data)
    print("Prediction:", prediction)
