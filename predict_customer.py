import requests

# Example data to be passed for prediction
# Replace these values with real input data for prediction

data = {
    "Age": [30],               # Age feature
    "Tenure": [12],            # Tenure (Years as a customer)
    "Balance": [5000],         # Account balance
    "NumOfProducts": [2],      # Number of products owned
    "HasCrCard": [1],          # Whether customer has a credit card (1 = Yes, 0 = No)
    "IsActiveMember": [1],     # Whether customer is active (1 = Active, 0 = Inactive)
    "EstimatedSalary": [75000], # Salary of the customer
    "Gender": [1],             # 1 for Male, 0 for Female (after encoding)
    "Geography": [1],          # 0 = France, 1 = Spain, 2 = Germany (after encoding)
    "Feature10": [0],          # Replace with actual value for Feature10
    "Feature11": [1],          # Replace with actual value for Feature11
    "Feature12": [2],          # Replace with actual value for Feature12
    "Feature13": [1],          # Replace with actual value for Feature13
    "Feature14": [0],          # Replace with actual value for Feature14
    "Feature15": [1],          # Replace with actual value for Feature15
    "Feature16": [0],          # Replace with actual value for Feature16
    "Feature17": [1],          # Replace with actual value for Feature17
    "Feature18": [0],          # Replace with actual value for Feature18
    "Feature19": [1],          # Replace with actual value for Feature19
    "Feature20": [0],          # Replace with actual value for Feature20
}

# prediction request
response = requests.post("http://127.0.0.1:5000/predict", json=data)

#  response
print(response.json())

