import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def preprocess_data(input_path, output_path):
    # Load the data
    data = pd.read_csv(input_path)

    # Preprocess (example: drop nulls, encode categorical columns, etc.)
    data = data.dropna()

    # Example feature scaling (standardizing the data)
    scaler = StandardScaler()
    data[['Feature1', 'Feature2', 'Feature3']] = scaler.fit_transform(data[['Feature1', 'Feature2', 'Feature3']])

    # Save processed data
    data.to_csv(output_path, index=False)

if __name__ == "__main__":
    preprocess_data("data/raw_data.csv", "data/processed_data.csv")
    print("Processed data saved to data/processed_data.csv")
