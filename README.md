# Machine Learning Supervised Learning: Car Price Prediction Model

- Proposed Domain: Building a model to predict car pricing, based on selling price, car type, year, miles driven, and more.
- Hypothesis: Utilizing the Supervised Learning regression models, we can accurately predict a car’s price based on its characteristics.
- ML that matters: The data can be used on Car Lots to predict used car prices.
- Dataset: Vehicle Data set on Kaggle https://www.kaggle.com/datasets/nehalbirla/vehicle-dataset-from-cardekho
- Total Size: 299 kB
- Number of Examples: 14,830
- Dimension of Instances (n): name, year, selling price, km driven, fuel, seller type, transmission, Owner
- Proposed Contributions: Ameer (5033) will implement Linear Regression and K-N-N., Namil (4033) will implement Decision Trees and Random Forest.

# One-hot Encoding

- Objective: Transform categorical data into a format suitable for machine learning models.
- Approach:
  - Transmission: Convert 'Manual' transmissions to 1 and 'Automatic' to 0.
  - Fuel: Create binary columns for each fuel type (Diesel, Petrol, CNG, LPG, Electric).
  - Seller Type: Create binary columns for each seller type (Individual, Dealer, Trustmark Dealer).
  - Owner: Create binary columns for each owner type (First Owner, Second Owner, Third Owner, Fourth & Above Owner).
- Implementation:
  - Use pandas to read the dataset and define column names.
  - Apply one-hot encoding using boolean comparisons to create new binary columns.
  - Drop original categorical columns (fuel, owner, seller_type).
  - Convert selling price from rupee to USD.
  - Save the modified DataFrame to a new CSV file for further analysis.
- Dataset: Utilizes the 'chosen.csv' dataset with columns for car details.
- Output: The modified dataset ('modified_chosen.csv') is used for all subsequent machine learning implementations.

# Methods

## Linear Regression

- Import Libraries: Import necessary libraries including pandas, numpy, matplotlib, and scikit-learn.
- Define LinearRegressionOLS Class:
  - Constructor: Initialize weights to None.
  - fitModel(X, y): Fit the model by adding a bias term to the input data and calculating weights using the pseudoinverse formula.
  - predictModel(X): Predict the output by adding a bias term to the input data and computing the dot product with the weights.
- Define Helper Functions:
  - MSE(df, yActual, yPredict): Calculate the Mean Squared Error.
  - RSquared(yActual, yPredict, variables, type): Calculate the R^2 value, with an option for adjusted R^2.
- Read Dataset: Read the one-hot encoded dataset from a CSV file.
- Prepare Data: Separate predictor and target variables. Split the data into training and testing sets.
- Train Model: Train the OLS regression model using the LinearRegressionOLS class.
- Test Model: Predict on the test set and calculate MSE, R^2, and adjusted R^2 using the trained model.
- Compare with sklearn: Use scikit-learn's LinearRegression to fit the model and predict on the test set. Calculate MSE and R^2 for comparison.
- Plot Residuals: Plot a residual plot for the test set to visualize the errors.
- Outlier Detection and Removal:
  - Calculate Z-scores: Calculate Z-scores for each data point.
  - Detect Outliers: Identify outliers based on a threshold (e.g., Z-score > 3).
  - Remove Outliers: Remove outliers from the dataset.
  - Retrain Model: Retrain the model without outliers and test its performance.

## K-Nearest-Neighbors

- Objective: Implement K-Nearest Neighbors (KNN) algorithm for predicting car prices based on key attributes.
- Data Preprocessing: One-hot encoding categorical data, splitting dataset into training, validation, and test sets.
- Algorithm Description:
  - Initialization: Set the number of neighbors (k) in the constructor.
  - Fit Data: Store the training data (X) and labels (y) in the class.
  - Predict Data:
    - For each new data point in X, calculate the distance to all training examples.
    - Select the k-nearest neighbors based on the smallest distances.
    - Predict the label (selling price) as the mean of the labels of the k-nearest neighbors.
  - Distance Formula: Euclidean distance is used to calculate the distance between two points.
- Mean Squared Error (MSE): Implement a function to calculate the MSE between actual and predicted labels.
- Graphing Functionality:
  - Plot actual vs. predicted selling prices for custom KNN.
  - Compare custom KNN with Sklearn KNN using side-by-side plots.
- Hyperparameter Tuning:
  - Iterate over k values from 1 to 10.
  - Train the model on the combined training and validation sets for each k.
  - Select the k with the lowest MSE on the validation set as the best k.
- Final Model Training:
  - Train the final model using the best k on the combined training and validation sets.
  - Compute the MSE on the test set for the final model.
- Comparison with Sklearn:
  - Train a Sklearn KNN model using the best k.
  - Compute the MSE on the test set for the Sklearn model.
- Graphical Comparison:
  - Plot actual vs. predicted selling prices for custom KNN and Sklearn KNN on the test set.

## Decision Trees

- Objective: Build a decision tree regression model to predict car prices based on key attributes.
- Data Preprocessing: Utilize one-hot encoding for categorical data and split the dataset into training, validation, and test sets.
- Node Representation: Define a Node class to represent each node in the decision tree, including attributes like depth, attribute index, threshold, value, left, and right nodes.
- Decision Tree Construction:
  - Start with the root node and recursively split the data based on the best attribute and threshold to minimize mean squared error (MSE).
  - Base case: Stop splitting if the tree reaches the maximum depth or if all labels in a node are the same.
  - Splitting Criteria: Find the best attribute and threshold by iterating over all features and thresholds, calculating the MSE for each split.
- Leaf Node Value: Compute the average of the labels in a leaf node as its predicted value.
- Prediction: Traverse the tree for each sample in the test dataset to predict its label based on the tree's structure.
- MSE Calculation: Define a function to calculate the MSE for a split, considering the MSE of both left and right nodes weighted by the number of samples.
- Hyperparameter Tuning: Test different max depths for the decision tree to find the depth that results in the lowest MSE on the validation set.
- Comparison with Scikit-learn: Validate the implementation by comparing the MSE of your decision tree with that of Scikit-learn's DecisionTreeRegressor.

## Random Forest

- Objective: Implement a Random Forest Regression model to predict car prices based on various attributes.
- Dataset: Utilize the Vehicle Dataset from Cardekho on Kaggle, containing 14,830 examples with attributes such as car name, year, selling price, kilometers driven, fuel type, seller type, transmission, and owner details.
- Preprocessing: One-hot encode categorical data (e.g., fuel type, seller type, transmission) to convert them into numerical format for model compatibility.
- Model Logic:
  - Define a RandomForestRegression class to handle the random forest regression process.
  - Constructor:
    - Accepts parameters for the number of trees (number_trees) and the maximum depth of each tree (max_depth).
    - Initializes an empty list trees to store the decision trees.
  - fit method:
    - Takes the training data (X, y) and fits a decision tree to a bootstrapped sample of the data for the specified number of trees.
    - Each tree is trained on a different bootstrap sample.
    - Adds each trained tree to the list trees.
  - bootstrap method:
    - Samples the data with replacement to create a bootstrap sample.
    - Returns the sampled data for training a decision tree.
  - predict method:
    - Takes new input data X and predicts the output by averaging the predictions from all the trees in the forest.
  - graphTree method:
    - Plots the actual selling price against the predicted selling prices from both the Random Forest and Decision Tree models on the test set.
    - Helps visualize the performance of the models.
- Testing and Validation:
  - Split the data into training, validation, and test sets.
  - Use a portion of the data for testing the model and ensure the model does not overfit or underfit.
- Hyperparameter Tuning:
  - Test different hyperparameters for the number of trees and maximum depths to find the combination that gives the lowest Mean Squared Error (MSE) on the validation set.
  - Use the best hyperparameters to train the final model.
- Comparison with scikit-learn:
  - Compare the performance of the implemented Random Forest Regression model with the scikit-learn's RandomForestRegressor on the test set.
- Conclusion:
  - Evaluate the model's performance on the test set using the MSE metric.
  - Discuss any observations or insights gained from the model's performance.
