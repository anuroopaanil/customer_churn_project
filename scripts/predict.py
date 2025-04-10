import joblib
import pandas as pd

def make_prediction(model_path, input_data):
    # Load trained model
    model = joblib.load(model_path)

    # Make prediction
    prediction = model.predict(input_data)
    return prediction

