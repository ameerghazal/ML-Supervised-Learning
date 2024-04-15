# Import the dataset and ski libraries to check our model.
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl # Used for font.
from sklearn.model_selection import train_test_split # Used for testing.
from sklearn.neighbors import KNeighborsRegressor # Used for testing.
from sklearn.metrics import mean_squared_error, r2_score

# Calculate the Mean Squared Error (MSE).
def MSE(df, yActual, yPredict):
  return (1 / len(df)) * sum((yActual - yPredict) ** 2)

# Define the distance formula.
def distance(xi, xj):
  return np.sqrt(np.sum((xi - xj) ** 2))

# Define the K-Nearest-Neighbors Class
class KNN:
    # Initalize the k-field, based on the parameter passed in.
    def __init__(self, k=3):
        self.k = k
    
    # Fit the data, based on the specific X,y variables passed in.
    def fitData(self, X, y):
        self.training_X = X
        self.label_y = y

    # Predict the data, based on the training_x.
    def predictData(self, X):
      # For every row in X, we compute the nearestNeighbors. 
      predicted_labels = [self.nearestNeighbors(x) for x in X]

      # Return the array of predicted y-labels (selling_prices).
      return np.array(predicted_labels)
    
    # Nearest Neigbors, to compute the distances, sort, and mean.
    def nearestNeighbors(self, x):
      # Compute distances between x (passed in row) and all training_x examples (passed in dataset) in the passed in set.
      distances = [distance(x, training_x) for training_x in self.training_X]

      # Select the nearest terms by sorting and taking the k smallest.
      nearestNeighbors = np.argsort(distances)[:self.k] 

      # Sum, extract the labels of the KNN, and return the mean.
      return (sum(self.label_y[index] for index in nearestNeighbors)) / (self.k)

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

# Remove the fuel, owner, etc.
df = df.drop(['fuel', 'owner', 'seller_type'], axis=1)

# Convert the selling price to american USD from rupee.
df['selling_price'] /= 83

# Store the predictor and target variables.
X = df.drop(['name', 'selling_price'], axis=1)
y = df['selling_price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the KNN model.
knnModel = KNN(k=3)
knnModel.fitData(X_train.values, y_train.values)
y_pred = knnModel.predictData(X_test.values)

# Calculate and store the Mean Squared Error for this k value (Custom KNN)
mse_custom = mean_squared_error(y_test, y_pred)
print(mse_custom)

# Compare to Sklearn.
knn_sklearn = KNeighborsRegressor(n_neighbors=3)
knn_sklearn.fit(X_train, y_train)
y_pred_sklearn = knn_sklearn.predict(X_test)

# Calculate and store the Mean Squared Error for this k value (Scikit-learn KNN)
mse_sklearn = mean_squared_error(y_test, y_pred_sklearn)
print(mse_sklearn)

# Plotting actual vs predicted selling prices for custom KNN
plt.figure(figsize=(12, 6))
# Plotting for Residual plot test set.
mpl.rcParams.update({'font.size': 16})  # Set the font-size for plot.

# Plotting actual selling prices
plt.subplot(1, 2, 1)
plt.scatter(range(len(y_test)), y_test, color='blue', label='Actual')
plt.title('Actual Selling Prices')
plt.xlabel('Index')
plt.ylabel('Selling Price')
plt.legend()

# Plotting predicted selling prices
plt.subplot(1, 2, 2)
plt.scatter(range(len(y_pred)), y_pred, color='red', label='Predicted')
plt.title('Predicted Selling Prices')
plt.xlabel('Index')
plt.ylabel('Selling Price')
plt.legend()

plt.tight_layout()
plt.savefig("K_Nearest_Neighbors/KNN.png")
plt.show()


