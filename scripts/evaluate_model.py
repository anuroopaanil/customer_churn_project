import pandas as pd
from sklearn.metrics import accuracy_score
import pickle

def evaluate_model(input_path, model_path):
    # Load the data and model
    data = pd.read_csv(input_path)
    model = pickle.load(open(model_path, 'rb'))

    # Split into features (X) and target (y)
    X = data.drop('Churn', axis=1)
    y = data['Churn']

    # Make predictions
    y_pred = model.predict(X)

    # Calculate accuracy
    accuracy = accuracy_score(y, y_pred)
    print(f"Model Accuracy: {accuracy * 100:.2f}%")

if __name__ == "__main__":
    evaluate_model("data/processed_data.csv", "models/model.pkl")
