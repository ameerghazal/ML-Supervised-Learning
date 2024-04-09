# Import the dataset and ski libraries to check our model.
import pandas as pd
import numpy as np
from matplotlib.pyplot import subplots

# Calculate the Mean Squared Error (MSE).
def MSE(df, yActual, yPredict):
  return (1 / len(df)) * sum((yActual - yPredict) ** 2)

def gradientDescent(training_examples, learning_rate, iterations):
  # Each training example is a pair of the form (x, t), where x is the vector of input values, and t is the target output value, learning rate (n).
  
  # Initialize each weight to 1 (given).
  weights = [1, 1]
  
  # Until termination is met, do the following:
  for _ in range(iterations):
    # Initialize each \delta w_{i} to zero.
    deltaWeights = [0,0]
    
    # For each (x, t) in training_examples
    for (inputVal, label) in training_examples:
      # Input the instance x to the unit and compute the output o.
      o = weights[0] + inputVal * (weights[1])
      
      # For each linear unit weight w_i, do
      for i in range(len(weights)):
        # \delta w_i = \delta w_i + learning_rate * (t - o)x_i
        if (i != 0): deltaWeights[i] += (learning_rate * (label - o) * inputVal)
        else: deltaWeights[i] += (learning_rate * (label - o) * 1) # x_0 = 1 in the equation.

    # For each linear unit weight wi, do
    for i in range(len(weights)):
      # w_i = w_i + \delta w_i
      weights[i] += deltaWeights[i]
      
    # Print weights after each iteration
    print(f"Iteration {_ + 1}: w0={weights[0]}, w1={weights[1]}")

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

# Print the first six.
print(df.head())
# print(df['seller_individual'])
# print(X, y)





