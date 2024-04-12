import pandas as pd
import numpy as np


# Calculate the Mean Squared Error (MSE).
def MSE(df, yActual, yPredict):
  return (1 / len(df)) * sum((yActual - yPredict) ** 2)


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

# Convert the selling price to american USD from rupee.
df['selling_price'] /= 83

# Ordinary Least Squares Linear Regression
class LinearRegressionOLS:
    def __init__(self):
        self.weights = None

    def f(self, X, y):
        # Add bias term
        X = np.insert(X.to_numpy(), 0, 1, axis=1)
        # Calculate weights using pseudoinverse
        self.weights = np.linalg.pinv(X.T.dot(X)).dot(X.T).dot(y.to_numpy())

    def p(self, X):
        # Add bias term
        X = np.insert(X.to_numpy(), 0, 1, axis=1)
        return X.dot(self.weights)

# Train the model
model = LinearRegressionOLS()
model.f(X, y)

# Make predictions
predictions = model.p(X)

# Print predictions
print("Predictions:", predictions)
print(f"Mean Squared Error (OLS OUR): {MSE(df, y, predictions )}")

# from sklearn.linear_model import LinearRegression
# lr = LinearRegression()
# lr.fit(X, y)
# y_pred_validation = lr.predict(X)
# mse_validation = MSE(df,y, y_pred_validation)
# print("Mean Squared Error (SKI) :", mse_validation)



# # Convert to numpy arrays for easier computation
# X = np.array(X)
# y = np.array(y)

# # Shuffle the data
# np.random.seed(42)  # Set random seed for reproducibility
# shuffle_index = np.random.permutation(len(df))
# X_shuffled, y_shuffled = X[shuffle_index], y[shuffle_index]

# # Calculate the sizes for train, validation, and test sets
# train_size = int(0.6 * len(df))
# val_size = int(0.2 * len(df))
# test_size = len(df) - train_size - val_size

# # Split the data into train, validation, and test sets
# X_train, y_train = X_shuffled[:train_size], y_shuffled[:train_size]
# X_val, y_val = X_shuffled[train_size:train_size+val_size], y_shuffled[train_size:train_size+val_size]
# X_test, y_test = X_shuffled[train_size+val_size:], y_shuffled[train_size+val_size:]

# # Add a column of ones for the intercept term
# # X_train = np.column_stack((np.ones(len(X_train)), X_train))
# # X_val = np.column_stack((np.ones(len(X_val)), X_val))
# # X_test = np.column_stack((np.ones(len(X_test)), X_test))

# # Calculate the pseudo-inverse of the training set features
# X_train_pseudo_inv = np.linalg.pinv(X_train)

# # Calculate the optimal weights for the training set
# w_train = np.dot(X_train_pseudo_inv, y_train)

# # Make predictions on the validation set
# y_val_pred = np.dot(X_val, w_train)

# # Calculate the mean squared error on the validation set
# mse_val = np.mean((y_val_pred - y_val) ** 2)
# print("Validation Mean Squared Error:", mse_val)

# # Optionally, make predictions on the test set
# y_test_pred = np.dot(X_test, w_train)

# # Calculate the mean squared error on the test set
# mse_test = np.mean((y_test_pred - y_test) ** 2)
# print("Test Mean Squared Error:", mse_test)


# Gradient desecent function.
def gradientDescent(X, y, learning_rate, iterations):
  # Each training example is a pair of the form (x, t), where x is the vector of input values, and t is the target output value, learning rate (n).
  
  # Initialize each weight, using a 1-D.
  weights = np.zeros(X.shape[1])
  
  # Until termination is met, do the following:
  for _ in range(iterations):
    # Initialize each \delta w_{i} to zero.
    deltaWeights = np.zeros(X.shape[1])
    
    # Compute the prediction data.
    predictions = np.dot(X, weights)

    # Loop for each linear unit weight.
    for i in range(len(weights)):
      # \delta w_i = \delta w_i + learning_rate * (t - o)x_i
      if (i != 0): deltaWeights += (learning_rate * (y - predictions) * X[:, i])
      else: deltaWeights += (learning_rate * (y - predictions) * 1) # x_0 = 1 in the equation.

    # For each linear unit weight wi, do
    for i in range(len(weights)):
      # w_i = w_i + \delta w_i
      weights[i] += deltaWeights[i]
