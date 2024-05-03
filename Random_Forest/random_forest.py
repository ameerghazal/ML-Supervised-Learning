import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split # Used for testing.
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from dt import DecisionTreeRegression

# Calculate the Mean Squared Error (MSE).
def Final_MSE(df, yActual, yPredict):
  return (1 / len(df)) * sum((yActual - yPredict) ** 2)

# Random Forest Class.
class RandomForestRegression:
  # Constructor for values.
  def __init__(self, number_trees = 50, max_depth = None):
    self.number_trees = number_trees
    self.max_depth = max_depth
    self.trees = []

  # Fit the dataset.
  def fit(self, X, y): 
    # Loop for the number of trees passed in.
    for i in range(self.number_trees):
      # Create the tree.
      tree = DecisionTreeRegression(max_depth=self.max_depth)
      # Bootstrap based on the data.
      X_sampled, y_sampled = self.bootstrap(X,y)
      # Fit decision tree.
      tree.fit(X_sampled, y_sampled)
      # Add new tree to the list.
      self.trees.append(tree)

  # Bootstrapping method for sampling w/ replacement.
  def bootstrap(self, X, y):
    # Store the number of samples in x.
    samples = X.shape[0]
    # Sample w/ replacement based on the size of the attributes.
    sample_idxs = np.random.choice(samples, size = samples, replace=True)
    # Return the sampled data.
    return X.iloc[sample_idxs], y.iloc[sample_idxs]

  # Predict based on the specific data.
  def predict(self, X): 
    predictions = [tree.predict(X) for tree in self.trees]
    return np.mean(predictions, axis=0)

  # Graph tree.
  def graphTree(self, actual, rpred, dpred):
    # Plotting the results for both algorithms on the test set
    plt.figure(figsize=(10, 6))
    plt.plot(np.arange(len(actual)), np.sort(y_test), color='blue', label='Actual Selling Price (Test Set)')
    plt.plot(np.arange(len(rpred)), np.sort(test_pred), color='purple', label='Random Forest Predicted Selling Price (Test Set)')
    plt.plot(np.arange(len(dpred)), np.sort(dt_test_pred), color='green', label='Decision Tree Predicted Selling Price (Test Set)')
    plt.xlabel('Index')
    plt.ylabel('Selling Price')
    plt.title('Random Forest vs Decision Tree: Predicted vs Actual (Test Set)')
    plt.legend()
    plt.grid(True)
    plt.savefig("Random_Forest/comparsion.png")
    plt.show()

# Read in the one-hot encoded dataset.
column_names = ['name','year','selling_price','km_driven','transmission','Diesel','Petrol','CNG','LPG','electric','seller_individual','seller_dealer','seller_trustmark','first_owner','second_owner','third_owner','fourth_owner']
df = pd.read_csv("modified_chosen.csv", names=column_names, header=None)

# Store the predictor and target variables.
X = df.drop(['name', 'selling_price'], axis=1)
y = df['selling_price']

# Remove 20% for testing using scki-learning, in which 80% returned will be training data.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=50)

# Now, split the training dataframe into 75% training and 25% validation.
X_train, X_validation, y_train, y_validation = train_test_split(X_test, y_test, test_size=0.25, random_state=50)

# Test different hyperparameters for the number of trees and max-depths.
number_of_trees = [50, 100, 150]
max_depth_values = [5, 7, 10]
best_MSE = float('inf')
best_hyperparam = {'number_trees': 50, 'max_depth': 7}

# # Loop over the number of trees and max depths.
for number_trees in number_of_trees:
  for max_depth in max_depth_values:

    # Create and fit the model.
    randomForest = RandomForestRegression(number_trees=number_trees, max_depth=max_depth)
    randomForest.fit(X_train, y_train)

    # Make predictions on the validation set.
    pred = randomForest.predict(X_validation.values)

    # Calculate the MSE.
    mse_scratch = Final_MSE(X_validation, y_validation, pred)

    # Print the MSE for this combination of hyperparameters
    print(f"Number of Trees: {number_trees}, Max Depth: {max_depth}, Mean Squared Error (Validation Set): {mse_scratch}")

    # Update the best hyperparameters.
    if mse_scratch < best_MSE:
      best_MSE = mse_scratch
      best_hyperparam = {'number_trees': number_trees, 'max_depth': max_depth}
print(f"Best Hyperparameters: {best_hyperparam}, Best Mean Squared Error: {best_MSE}")

# Create, fit, and predict on random forest model with the best parameters on the test set.
randomForest = RandomForestRegression(number_trees=best_hyperparam['number_trees'], max_depth=best_hyperparam['max_depth'])
randomForest.fit(X_train, y_train)
test_pred = randomForest.predict(X_test.values)
MSE_test = Final_MSE(X_test, y_test, test_pred)
print(f"Mean Squared Error (Test Set) - Personal: {MSE_test}")

# Compare output to Sklearn.
from sklearn.ensemble import RandomForestRegressor

# Create, fit, and predict on sklearn model with the best parameters on the test set.
sklearn_rf = RandomForestRegressor(n_estimators=best_hyperparam['number_trees'], max_depth=best_hyperparam['max_depth'])
sklearn_rf.fit(X_train, y_train)
# sklearn_rf.fit(np.concatenate([X_train, X_validation]), np.concatenate([y_train, y_validation]))
sklearn_test_pred = sklearn_rf.predict(X_test.values)
sklearn_MSE_test = mean_squared_error(y_test, sklearn_test_pred)
print(f"Mean Squared Error (Test Set) - Scikit-learn: {sklearn_MSE_test}")

# Create and fit the DecisionTreeRegression model
dt = DecisionTreeRegression(max_depth=7)
dt.fit(X_train, y_train)
dt_test_pred = dt.predict(X_test.values)

# Plotting the results for both algorithms on the test set
randomForest.graphTree(y_test, test_pred, dt_test_pred)