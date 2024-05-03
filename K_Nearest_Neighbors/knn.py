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
  
    # Define the distance formula.
    def distance(self, xi, xj):
      return np.sqrt(np.sum((xi - xj) ** 2))

    # Nearest Neigbors, to compute the distances, sort, and mean.
    def nearestNeighbors(self, x):
      # Compute distances between x (passed in row) and all training_x examples (passed in dataset) in the passed in set.
      distances = [self.distance(x, training_x) for training_x in self.training_X]

      # Select the nearest terms by sorting and taking the k smallest.
      nearestNeighbors = np.argsort(distances)[:self.k] 

      # Sum, extract the labels of the KNN, and return the mean.
      return (sum(self.label_y[index] for index in nearestNeighbors)) / (self.k)

    # Graph the function.
    def graphKNN(self, actual, pred, sklearn = None, double = False):
      # Determine which plot to use; if double is true, use the side by side comparsion.
      if (double) :
        # Plotting actual vs predicted selling prices for custom KNN
        plt.figure(figsize=(12, 6))
        # Plotting for Residual plot test set.
        mpl.rcParams.update({'font.size': 16})  # Set the font-size for plot.
        # Plotting actual selling prices
        plt.subplot(1, 2, 1)
        plt.scatter(range(len(actual)), actual, color='blue', label='Actual')
        plt.title('Actual Selling Prices')
        plt.xlabel('Index')
        plt.ylabel('Selling Price')
        plt.legend()

        # Plotting predicted selling prices
        plt.subplot(1, 2, 2)
        plt.scatter(range(len(pred)), pred, color='red', label='Predicted')
        plt.title('Predicted Selling Prices')
        plt.xlabel('Index')
        plt.ylabel('Selling Price')
        plt.legend()

        # Save the figure and show.
        plt.tight_layout()
        plt.savefig("K_Nearest_Neighbors/KNN.png")
        plt.show()

        return
      
      # Otherwise, use the entire comparsion of actual and predicted values.
      plt.figure(figsize=(10, 6))
      plt.plot(np.arange(len(actual)), np.sort(actual), color='blue', label='Actual Selling Price')
      plt.plot(np.arange(len(pred)), np.sort(pred), color='green', label='Predicted Selling Price (Custom KNN)')
      plt.plot(np.arange(len(sklearn)), np.sort(sklearn), color='purple', label='Predicted Selling Price (Sklearn KNN)')
      plt.xlabel('Data Point Index')
      # Set x-axis limits to emphasize the end part of the graph from index 500
      plt.xlim(500, len(pred))
      plt.ylabel('Selling Price')
      plt.title('Custom KNN vs Sklearn KNN: Predicted vs Actual (Test Set)')
      plt.legend()
      plt.savefig("K_Nearest_Neighbors/knnComparsion.png")
      plt.show()

      # Assuming y_test, y_pred, and y_pred_sklearn are your actual, predicted (custom), and predicted (Sklearn) selling prices respectively
      fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 6))

      # Plot entire data
      ax1.plot(np.arange(len(actual)), np.sort(actual), color='blue', label='Actual Selling Price')
      ax1.plot(np.arange(len(pred)), np.sort(pred), color='green', label='Predicted Selling Price (Custom KNN)')
      ax1.plot(np.arange(len(sklearn)), np.sort(sklearn), color='purple', label='Predicted Selling Price (Sklearn KNN)')
      ax1.set_xlabel('Data Point Index')
      ax1.set_ylabel('Selling Price')
      ax1.set_title('Custom KNN vs Sklearn KNN: Predicted vs Actual (Test Set)')
      ax1.legend()

      # Zoom in on section starting from index 800
      ax2.plot(np.arange(len(actual)), np.sort(actual), color='blue', label='Actual Selling Price')
      ax2.plot(np.arange(len(pred)), np.sort(pred), color='green', label='Predicted Selling Price (Custom KNN)')
      ax2.plot(np.arange(len(sklearn)), np.sort(sklearn), color='purple', label='Predicted Selling Price (Sklearn KNN)')
      ax2.set_xlabel('Data Point Index')
      ax2.set_ylabel('Selling Price')
      ax2.set_title('Zoomed In: Custom KNN vs Sklearn KNN: Predicted vs Actual (Test Set)')
      ax2.legend()

      # Set x-axis limits for the zoomed-in plot
      ax2.set_xlim(800, len(actual))

      plt.savefig("K_Nearest_Neighbors/knnComparisonZoomed.png")
      plt.show()

# Read in the one-hot encoded dataset.
column_names = ['name','year','selling_price','km_driven','transmission','Diesel','Petrol','CNG','LPG','electric','seller_individual','seller_dealer','seller_trustmark','first_owner','second_owner','third_owner','fourth_owner']
df = pd.read_csv("modified_chosen.csv", names=column_names, header=None)

# Store the predictor and target variables.
X = df.drop(['name', 'selling_price'], axis=1)
y = df['selling_price']

# Split the data into training, validation, and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=50)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=52)

# Hyperparameter tuning for KNN from 1 to 10.
best_k = None
best_mse = float('inf')
for k in range(1, 10):
    # Create the model, fit on training, predict on validation, and compute the mse.
    knn = KNN(k=k)
    knn.fitData(X_train.values, y_train.values)
    y_pred_val = knn.predictData(X_val.values)
    mse = mean_squared_error(y_val.values, y_pred_val)
    if mse < best_mse:
        best_mse = mse
        best_k = k

# Train the final model on the combined training and validation sets with the best K; compute the MSE.
knnModel = KNN(k=best_k)
knnModel.fitData(np.concatenate([X_train.values, X_val.values]), np.concatenate([y_train.values, y_val.values]))
y_pred = knnModel.predictData(X_test.values)
mse_custom = mean_squared_error(y_test, y_pred)
print(f"Best K: {best_k}, Final MSE on Test Set: {mse_custom}")

# Compare to Sklearn and compute the MSE.
knn_sklearn = KNeighborsRegressor(n_neighbors=best_k)
knn_sklearn.fit(np.concatenate([X_train.values, X_val.values]), np.concatenate([y_train.values, y_val.values]))
y_pred_sklearn = knn_sklearn.predict(X_test)
mse_sklearn = mean_squared_error(y_test, y_pred_sklearn)
print(f"Sklearn MSE {mse_custom}")

# Plot actual and predicted values
knnModel.graphKNN(y_test, y_pred, y_pred_sklearn)
knnModel.graphKNN(y_test, y_pred, double=True)

# At this point, we could reconcat the x-test, x-train, and x-valid datasets and publish the final fit.