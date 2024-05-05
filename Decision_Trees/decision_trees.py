import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split # Used for testing.
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Calculate the Mean Squared Error (MSE).
def Final_MSE(df, yActual, yPredict):
  return (1 / len(df)) * sum((yActual - yPredict) ** 2)

# Define the node class, which will be used for each node in the tree.
class Node:
  # Initalization function for the fields.
  def __init__(self, depth, max_depth = None):
    self.depth = depth
    self.max_depth = max_depth
    self.attribute_index = None
    self.threshold = None
    self.value = None
    self.left = None
    self.right = None

class DecisionTreeRegression:
  # Initalization function for the tree.
  def __init__(self, max_depth = None):
    self.max_depth = max_depth
    self.root = None

  # Fit the decision tree, based on the data and label.
  def fit(self, X, y):
    self.root = self.construct_tree(X.values, y.values, depth = 0)

  # Construct the tree function.
  def construct_tree(self, X, y, depth):
    # Base case: check if the tree is at the max depth or if the label has already been predicted for.
    if depth >= self.max_depth or len(np.unique(y)) == 1:
      leaf = Node(depth, self.max_depth) # Creates the final node.
      leaf.value = np.mean(y) # Average of the split-up predicted values to form a leaf.
      return leaf
    
    # Otherwise, find the best split, given the X (data) and y (label).
    attribute_index, threshold = self.best_split(X,y)

    # Check if the best split is un-defined - if so, compute the same leaf computation.
    if attribute_index is None:
      leaf = Node(depth, self.max_depth) # Creates the final node.
      leaf.value = np.mean(y) # Average of the split-up predicted values to form a leaf.
      return leaf
    
    # If we make it here, create a node, split the data, and create a left & right subtrees.
    node = Node(depth=depth, max_depth=self.max_depth)

    # Store the index of the attribute that is being tested on and the threshold to split the data.
    node.attribute_index, node.threshold = attribute_index, threshold

    # Split the data into left and right.
    X_left, X_right, y_left, y_right = self.normal_split(X, y, node.attribute_index, node.threshold)

    # Create the left and right subtrees and store them into their respective fields.
    node.left = self.construct_tree(X_left, y_left, depth + 1)
    node.right = self.construct_tree(X_right, y_right, depth + 1)

    # Return the node.
    return node

  # Determine the best split, based on the data.
  def best_split(self, X, y):
    # Store the rows, columns and init variables.
    rows, columns = X.shape
    best_attribute = None
    best_threshold = None
    best_MSE = float('inf') # Start at a very high MSE as we aim to decrease it.

    # Loop for each attribute (feature).
    for attr_index in range(columns):
      # Get the unique threshold based on the attribute.
      thresholds = np.unique(X[:, attr_index])

      # Loop for all these thresholds.
      for threshold in thresholds:
        # Split the data into left and right.
        X_left, X_right, y_left, y_right = self.normal_split(X, y, attr_index, threshold)

        # Calculate the MSE of the left and right.
        total_MSE = self.MSE(y_left, y_right)

        # Check the if the new MSE is lower (better) than the previous best MSE.
        if total_MSE < best_MSE:
          best_MSE = total_MSE # Reassign the best_mse.
          best_attribute = attr_index
          best_threshold = threshold
    
    # Return the best feature, with the least MSE, and the best spot to split for the threshold.
    return best_attribute, best_threshold

  # Split the dataset into left and right, based on the index and the threshold.
  def normal_split(self, X, y, attribute_index, threshold):
    # Grab all the X data's row, based on the specific attribute (column), split them, and return the relevant X, y (left and right) data.
    left_data = X[:, attribute_index] <= threshold
    right_data = X[:, attribute_index] > threshold
    return X[left_data], X[right_data], y[left_data], y[right_data]
  
  # Predict function based on the dataset. Returned is an array of predicted labels.
  def predict(self, X):
    # Init an empty predictions array to store the labels.
    pred = []

    # Loop for each sample inside the dataset passed in.
    for x in X:

      # Store the root of the current tree.
      node = self.root 
      
      # Loop until the node.left is empty.
      while node.left:
        # If less than threshold, go lower.
        if x[node.attribute_index] <= node.threshold:
          node = node.left
        else:
          node = node.right
      
      # Add lowest into the predictions list.
      pred.append(node.value)
    
    # Return the list.
    return np.array(pred)

  # Calculate the Mean Squared Error (MSE) for regression.
  def MSE(self, left_y, right_y):
    # Return a large value to avoid splitting further
    if len(left_y) == 0 or len(right_y) == 0:
      return float('inf')
    # Calculate the left and the right, and for y-hat, use the mean as the predictor.
    left_mse = np.mean((left_y - np.mean(left_y)) ** 2)
    right_mse = np.mean((right_y - np.mean(right_y)) ** 2)
    return (len(left_y) * left_mse + len(right_y) * right_mse) / (len(left_y) + len(right_y))

  # Graph the different MSE's
  def graph_MSE(self, mses, depths, best_depth_idx): 
    plt.figure(figsize=(10, 6))
    plt.plot(depths, mses, marker='o', label='Validation MSE')
    plt.xlabel('Max Depth')
    plt.ylabel('Mean Squared Error')
    plt.title('Validation MSE for Different Max Depths')
    plt.legend()
    plt.grid(True)
    plt.savefig("Decision_Trees/MSES.png")
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

# From here, we test for different max-depths to find the lowest MSE.
different_MSES = []
different_depths = []

for max_depth in range(1, 30):
  # Construct the tree
  decisionTree = DecisionTreeRegression(max_depth=max_depth)

  # Train the tree with the training data.
  decisionTree.fit(X_train, y_train)

  # Predict on the validation data.
  pred = decisionTree.predict(X_validation.values)

  # Calculate the MSE for your implementation
  mse_scratch = Final_MSE(X_validation, y_validation, pred)

  # Store the max_depth and corresponding MSE
  different_depths.append(max_depth)
  different_MSES.append(mse_scratch)

# Find the index of the lowest MSE
min_mse_idx = different_MSES.index(min(different_MSES))
best_depth = different_depths[min_mse_idx]

# Construct the decision tree, fit the data, make predictions on the data, and print out the predictions and the tree.
decisionTree = DecisionTreeRegression(max_depth=best_depth)
decisionTree.fit(X_train, y_train)
pred = decisionTree.predict(X_test.values)

# Calculate the MSE for your implementation
mse_scratch = Final_MSE(X_test, y_test, pred)
print("Your implementation MSE:", mse_scratch)
#print("Your implementation predictions:", pred)

# Graph the different MSE's
decisionTree.graph_MSE(different_MSES, different_depths, best_depth)

# Compare the output to Scikit-learn.
sklearn_decision_tree = DecisionTreeRegressor(max_depth=best_depth)
sklearn_decision_tree.fit(X_train, y_train)
sklearn_predictions = sklearn_decision_tree.predict(X_test)

# Calculate MSE for sklearn.
sklearn_mse = mean_squared_error(y_test, sklearn_predictions)
print("Scikit-learn MSE:", sklearn_mse)
#print("Scikit-learn predictions:", sklearn_predictions)

