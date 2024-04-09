import pandas as pd
import numpy as np

# Load the dataset

# Define the column names for the table and read in the dataset.
column_names = ['name', 'year', 'selling_price', 'km_driven', 'fuel', 'seller_type', 'transmission', 'owner']
df = pd.read_csv("chosen.csv", names=column_names, header=None)

# Transmission: One-hot encoding for the categorial data. Manual = 1, Automatic = 0.
df['transmission'] = (df['transmission'] != 'automatic').astype(int) 

# Fuel: one-hot encoding by creating a column, by comparing the fuel, getting the boolean value, and converting a binary 0, 1.
df['Diesel'] = (df['fuel'] == 'Diesel').astype(int) 
df['Petrol'] = (df['fuel'] == 'Petrol').astype(int)
df['CNG'] = (df['fuel'] == 'CNG').astype(int) 
df['LPG'] = (df['fuel'] == 'LPG').astype(int)
df['electric'] = (df['fuel'] == 'electric').astype(int)

# Seller type: one-hot encoding.
df['seller_individual'] = (df['seller_type'] == 'Individual').astype(int) 
df['seller_dealer'] = (df['seller_type'] == 'Dealer').astype(int) 
df['seller_trustmark'] = (df['seller_type'] == 'Trustmark Dealer').astype(int) 

# Owner: one-hot encoding.
df['first_owner'] = (df['owner'] == 'First Owner').astype(int) 
df['second_owner'] = (df['owner'] == 'Second Owner').astype(int) 
df['third_owner'] = (df['owner'] == 'Thrid Owner').astype(int) 
df['fourth_owner'] = (df['owner'] == 'Fourth & Above Owner').astype(int) 

# Remove the fuel, owner, etc..
df = df.drop(['fuel', 'owner', 'seller_type'], axis=1)

# Store the predictor and target variables.
X = df.drop(['name', 'selling_price'], axis=1)
y = df['selling_price']

# Standardize the features
X = (X - X.mean()) / X.std()

# Initialize weights
weights = np.zeros(X.shape[1])

# Set hyperparameters
learning_rate = 0.01
iterations = 1000

# Add a column of ones for the intercept term
X['intercept'] = 1

# Gradient Descent
for _ in range(iterations):
    # Compute predictions
    predictions = np.dot(X, weights)

    # Compute errors
    errors = predictions - y

    # Compute gradient
    gradient = np.dot(X.T, errors) / len(y)

    # Update weights
    weights -= learning_rate * gradient

# Print the weights
print("Weights:", weights)

# Make predictions
y_pred = np.dot(X, weights)

# Calculate Mean Squared Error
mse = np.mean((y_pred - y) ** 2)
print("Mean Squared Error:", mse)