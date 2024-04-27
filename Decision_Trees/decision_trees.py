import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split # Used for testing.
from sklearn.tree import DecisionTreeClassifier # 3

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
    self.root = self.construct_tree(X, y, depth = 0)

  # Construct the tree function.
  def construct_tree(self, X, y, depth):
    # Base case: check if the tree is at the max depth or if the label has already been predicted for.
    if depth == self.max_depth or len(np.unique(y)) == 1:
      leaf = Node(depth) # Creates the final node.
      leaf.value = np.mean(y) # Average of the split-up predicted values to form a leaf.
      return leaf
    
    # Otherwise, find the best split, given the X (data) and y (label).
    best_split = self.best_split(X,y)

    # Check if the best split is un-defined - if so, compute the same leaf computation.
    if best_split is None:
      leaf = Node(depth) # Creates the final node.
      leaf.value = np.mean(y) # Average of the split-up predicted values to form a leaf.
      return leaf
    
    # If we make it here, create a node, split the data, and create a left & right subtrees.
    node = Node(depth=depth, max_depth=self.max_depth)

    # Store the index of the attribute that is being tested on and the threshold to split the data.
    node.attribute_index, node.threshold = best_split

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
        total_MSE = self.MSE(y_left) + self.MSE(y_right)

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

  def predict(): return 0

  # Calculate the Mean Squared Error (MSE), [REGRESSION].
  def MSE(y):
    # For y-hat, use the mean as the predictor.
    return np.mean((y - np.mean(y)) ** 2)

  # Gini-index function for the tree, where the user passes in the left and right subtree [CLASSIFICATION].
  def gini_index(left_tree, right_tree, target_column):
    # Calculate the total size.
    size = len(left_tree) + len(right_tree)

    # Calculate the left and right gini by grabbing the values with the counts of them.
    left_gini = 1 - sum((left_tree[target_column].value_counts() / len(left_tree)) ** 2)
    right_gini = 1 - sum((right_tree[target_column].value_counts() / len(right_tree)) ** 2)

    # Add the following and compute the split ratio.
    return (len(left_tree) / size) * left_gini + (len(right_tree) / size) * right_gini

  # Print output decision tree.
  def print_tree(): return 0

  # Graph the decision tree.
  def graph_tree(): return 0


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