import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Node class to represent nodes in the decision tree
class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

# Decision tree class
class DecisionTree:
    def __init__(self, max_depth=None, min_samples_split=None):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.root = None

    def fit(self, X, y, bootstrap=False):
        if bootstrap:
            X, y = self._bootstrap(X, y)
        self.root = self._build_tree(X, y)

    def _bootstrap(self, X, y):
        n_samples = X.shape[0]
        idxs = np.random.choice(n_samples, n_samples, replace=True)
        return X.iloc[idxs], y.iloc[idxs]

    def _build_tree(self, X, y, depth=0):
        n_samples, n_features = X.shape
        if depth >= self.max_depth or n_samples < self.min_samples_split:
            return Node(value=np.mean(y))

        best_feature, best_threshold = self._find_best_split(X, y)
        left_indices = X.iloc[:, best_feature] < best_threshold
        right_indices = ~left_indices
        left_child = self._build_tree(X[left_indices], y[left_indices], depth + 1)
        right_child = self._build_tree(X[right_indices], y[right_indices], depth + 1)
        return Node(feature=best_feature, threshold=best_threshold, left=left_child, right=right_child)

    def _find_best_split(self, X, y):
        n_samples, n_features = X.shape
        best_mse = float('inf')
        best_feature = None
        best_threshold = None
        for feature in range(n_features):
            thresholds = np.unique(X.iloc[:, feature])
            for threshold in thresholds:
                left_indices = X.iloc[:, feature] < threshold
                right_indices = ~left_indices
                if np.sum(left_indices) == 0 or np.sum(right_indices) == 0:
                    continue
                mse = self._calculate_mse(y[left_indices], y[right_indices])
                if mse < best_mse:
                    best_mse = mse
                    best_feature = feature
                    best_threshold = threshold
        return best_feature, best_threshold

    def _calculate_mse(self, left_y, right_y):
        total_samples = len(left_y) + len(right_y)
        mse = (len(left_y) / total_samples) * np.var(left_y) + (len(right_y) / total_samples) * np.var(right_y)
        return mse

    def predict(self, X):
        return np.array([self._predict(inputs) for inputs in X])

    def _predict(self, inputs):
        node = self.root
        while node.left:
            if inputs[node.feature] < node.threshold:
                node = node.left
            else:
                node = node.right
        return node.value

# Random forest regression class
class RandomForestRegression:
    def __init__(self, n_trees=100, max_depth=None, min_samples_split=None, n_features=None):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.n_features = n_features
        self.trees = []

    def fit(self, X, y):
        for _ in range(self.n_trees):
            tree = DecisionTree(max_depth=self.max_depth, min_samples_split=self.min_samples_split)
            tree.fit(X, y, bootstrap=True)
            self.trees.append(tree)

    def predict(self, X):
        return np.mean([tree.predict(X) for tree in self.trees], axis=0)



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

# Usage
# Assuming X_train and y_train are your training data
rf = RandomForestRegression(n_trees=100, max_depth=2, min_samples_split=2, n_features=4)
rf.fit(X_train, y_train)
predictions = rf.predict(X_validation)
print(predictions)
