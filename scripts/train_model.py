import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pickle

def train_model(input_path, model_output_path):
    # Load the preprocessed data
    data = pd.read_csv(input_path)

    # Split into features (X) and target (y)
    X = data.drop('Churn', axis=1)  # Assuming 'Churn' is the target variable
    y = data['Churn']

    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train a model (Random Forest in this case)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Save the trained model
    pickle.dump(model, open(model_output_path, 'wb'))
    print(f"Model saved to {model_output_path}")

if __name__ == "__main__":
    train_model("data/processed_data.csv", "models/model.pkl")
