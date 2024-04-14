import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsRegressor

# # Define the distance formula.
# def distance(xi, xj):
#     return np.sqrt(np.sum((xi - xj) ** 2))

# # Define the K-Nearest-Neighbors Algorithm
# def KNN(X_train, y_train, X_test, k):
#     predictions = []
#     for i in range(len(X_test)):
#         distances = []
#         for j in range(len(X_train)):
#             dist = distance(X_test[i], X_train[j])
#             distances.append((dist, y_train[j]))
#         distances.sort(key=lambda x: x[0])
#         neighbors = distances[:k]
#         prediction = np.mean([neighbor[1] for neighbor in neighbors])
#         predictions.append(prediction)
#     return predictions

# # Import the dataset.
# column_names = ['name', 'year', 'selling_price', 'km_driven', 'fuel', 'seller_type', 'transmission', 'owner']
# df = pd.read_csv("chosen.csv", names=column_names, header=None)

# # Encoding for the categorical data.
# df['transmission'] = (df['transmission'] != 'automatic').astype(int) 
# df['Diesel'] = (df['fuel'] == 'Diesel').astype(int) 
# df['Petrol'] = (df['fuel'] == 'Petrol').astype(int)
# df['CNG'] = (df['fuel'] == 'CNG').astype(int) 
# df['LPG'] = (df['fuel'] == 'LPG').astype(int)
# df['electric'] = (df['fuel'] == 'electric').astype(int)
# df['seller_individual'] = (df['seller_type'] == 'Individual').astype(int) 
# df['seller_dealer'] = (df['seller_type'] == 'Dealer').astype(int) 
# df['seller_trustmark'] = (df['seller_type'] == 'Trustmark Dealer').astype(int) 
# df['first_owner'] = (df['owner'] == 'First Owner').astype(int) 
# df['second_owner'] = (df['owner'] == 'Second Owner').astype(int) 
# df['third_owner'] = (df['owner'] == 'Thrid Owner').astype(int) 
# df['fourth_owner'] = (df['owner'] == 'Fourth & Above Owner').astype(int) 

# # Drop unnecessary columns.
# df = df.drop(['fuel', 'owner', 'seller_type'], axis=1)

# # Convert selling price to USD.
# df['selling_price'] /= 83

# # Store predictor and target variables.
# X = df.drop(['name', 'selling_price'], axis=1)
# y = df['selling_price']

# # Split the data into training and testing sets
# from sklearn.model_selection import train_test_split
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Convert dataframes to arrays
# X_train = X_train.values
# X_test = X_test.values
# y_train = y_train.values
# y_test = y_test.values

# # Set the value of k
# k = 3

# # Get predictions
# predictions = KNN(X_train, y_train, X_test, k)

# print(y_test)
# print(predictions)

# # Evaluate the model
# from sklearn.metrics import mean_squared_error
# mse = mean_squared_error(y_test, predictions)
# print("Mean Squared Error:", mse)












import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2)**2))

class KNN:
    def __init__(self, k=3):
        self.k = k

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        y_pred = [self._predict(x) for x in X]
        return np.array(y_pred)

    def _predict(self, x):
        distances = [euclidean_distance(x, x_train) for x_train in self.X_train]
        k_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = [self.y_train[i] for i in k_indices]
        return np.mean(k_nearest_labels)

# Define the column names for the table and read in the dataset.
column_names = ['name', 'year', 'selling_price', 'km_driven', 'fuel', 'seller_type', 'transmission', 'owner']
df = pd.read_csv("chosen.csv", names=column_names, header=None)

# Transmission: One-hot encoding for the categorical data. Manual = 1, Automatic = 0.
df['transmission'] = (df['transmission'] != 'automatic').astype(int) 

# Fuel: one-hot encoding.
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

# Convert the selling price to American USD from rupee.
df['selling_price'] /= 83

# Store the predictor and target variables.
X = df.drop(['name', 'selling_price'], axis=1)
y = df['selling_price']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define a range of k values to test
k_values = [1, 3, 5, 7, 9]

# Dictionary to store MSE for each k value
mse_custom_dict = {}
mse_sklearn_dict = {}
for k in k_values:
    # Instantiate and train the custom KNN model
    knn_custom = KNN(k=k)
    knn_custom.fit(X_train.values, y_train.values)
    y_pred_custom = knn_custom.predict(X_test.values)
    
    # Calculate and store the Mean Squared Error for this k value (Custom KNN)
    mse_custom = mean_squared_error(y_test, y_pred_custom)
    mse_custom_dict[k] = mse_custom

    # Instantiate and train the scikit-learn KNN model
    knn_sklearn = KNeighborsRegressor(n_neighbors=k)
    knn_sklearn.fit(X_train, y_train)
    y_pred_sklearn = knn_sklearn.predict(X_test)
    
    # Calculate and store the Mean Squared Error for this k value (Scikit-learn KNN)
    mse_sklearn = mean_squared_error(y_test, y_pred_sklearn)
    mse_sklearn_dict[k] = mse_sklearn

# Print the MSE for each k value (Custom KNN)
print("Custom K-nearest neighbor algorithm:")
for k, mse in mse_custom_dict.items():
    print(f"Mean Squared Error (k={k}): {mse}")

# Print the MSE for each k value (Scikit-learn KNN)
print("\nScikit-learn K-nearest neighbor algorithm:")
for k, mse in mse_sklearn_dict.items():
    print(f"Mean Squared Error (k={k}): {mse}")

# # Split the data into training, validation, and test sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# # Hyperparameter tuning for KNN
# best_k = None
# best_mse = float('inf')
# for k in range(1, 10):
#     knn = KNN(k=k)
#     knn.fit(X_train.values, y_train.values)
#     y_pred_val = knn.predict(X_val.values)
#     mse = mean_squared_error(y_val.values, y_pred_val)
#     if mse < best_mse:
#         best_mse = mse
#         best_k = k

# # Train the final model on the combined training and validation sets with the best K
# knn = KNN(k=best_k)
# knn.fit(np.concatenate([X_train.values, X_val.values]), np.concatenate([y_train.values, y_val.values]))
# y_pred_test = knn.predict(X_test.values)
# final_mse = mean_squared_error(y_test.values, y_pred_test)
# print(f"Best K: {best_k}, Final MSE on Test Set: {final_mse}")
