# Import the dataset and ski libraries to check our model.
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl # Used for font.
from sklearn.model_selection import train_test_split # Used for testing.
from sklearn.linear_model import LinearRegression # Used for testing.
from sklearn.metrics import r2_score # Used for testing.
from sklearn.metrics import mean_squared_error

# Define the OLS Regression Algorithm from Scratch.
class LinearRegressionOLS:
    # Constructor for the weights, if not pre-assigned.
    def __init__(self, alpha = 0, li_ratio = 0):
        self.weights = None
        self.alpha = alpha
        self.li_ratio = li_ratio

    # Function used to fit the model, given the input data and the label to be predicted.
    def fitModel(self, X, y):
        # Add bias term
        X = np.insert(X.to_numpy(), 0, 1, axis=1)
        # Compute the number of features.
        I = np.identity(X.shape[1])
        # Lasso Penalty Function, where we multiply the reg paramaters and the number of features (1).
        lasso = self.alpha * self.li_ratio * np.ones(X.shape[1])
        # Ridge penalty function.
        ridge = self.alpha * I
        # Elastic penalty function, where we combine the above two equations, if non zero.
        elastic = ridge + (1 - self.li_ratio) * lasso
        # Calculate weights using pseudoinverse, taken module 7 slides, where elastic will converge to 0 if alpha and li_ratio are 0.
        self.weights = np.linalg.pinv(X.T.dot(X) + elastic).dot(X.T).dot(y.to_numpy())

    # Function used to predict the output, given the X (input data).
    def predictModel(self, X):
        # Add bias term
        X = np.insert(X.to_numpy(), 0, 1, axis=1)
        return X.dot(self.weights) # Compute h(x)

# Calculate the Mean Squared Error (MSE).
def MSE(df, yActual, yPredict):
  return (1 / len(df)) * sum((yActual - yPredict) ** 2)

# Calculate the r^2 value.
def RSquared(yActual, yPredict, variables, type = "multiple"):
    # Store the mean, and compute the RSS and TSS.
    yMean = np.mean(yActual)
    RSS = np.sum((yActual - yPredict) ** 2)
    TSS = np.sum((yActual - yMean) ** 2)

    # Determine which R^2 to return.
    if (type == "adjusted"):
        m = len(yActual) # Store the number of examples.
        return 1 - ((RSS / (m - variables - 1))) / ((TSS) / (m-1)) # R^2 adj = 1 - RSS/m-d-1 / TSS/m-1 (m = # examples, d = # variables in the model)
    
    # Otherwise, return multiple r^2
    return 1 - (RSS / TSS) # R^2 = 1 - RSS/TSS (multiple)

# Read in the one-hot encoded dataset.
column_names = ['name','year','selling_price','km_driven','transmission','Diesel','Petrol','CNG','LPG','electric','seller_individual','seller_dealer','seller_trustmark','first_owner','second_owner','third_owner','fourth_owner']
df = pd.read_csv("modified_chosen.csv", names=column_names, header=None)

# Store the predictor and target variables.
X = df.drop(['name', 'selling_price'], axis=1)
y = df['selling_price']

# Remove 20% for testing using scki-learning, in which 80% returned will be training data.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=50)

# Train the OLS regression model.
LrModel = LinearRegressionOLS()
LrModel.fitModel(X_train, y_train)

# Fit the model with the concat data and predict on the test set.
y_test_predictions = LrModel.predictModel(X_test)

# Calculate the MSE, R^2 on the test data.
MSE_test = MSE(df=X_test, yActual=y_test, yPredict=y_test_predictions)
multipleRSquared_test = RSquared(yActual=y_test, yPredict=y_test_predictions, variables=X.shape[1])
adjRSquared_test = RSquared(yActual=y_test, yPredict=y_test_predictions, variables=X.shape[1], type="adjusted")

# Print out both the results.
print(f"Mean Squared Error (Test Set): {MSE_test}")
print(f"R^2 (Test Set): {multipleRSquared_test}")
print(f"Adjusted R^2 (Test Set): {adjRSquared_test}")

# Compare output to the sklearning function.
LrSkl = LinearRegression()
LrSkl.fit(X_train, y_train) # fit the model.
y_ski_test_pred = LrSkl.predict(X_test) # Preds. based on test.
print(f"Sklearn Mean Squared Error (Test): { mean_squared_error(y_test, y_ski_test_pred)}")
print(f"Sklearn r^2 (Test): {r2_score(y_test, y_ski_test_pred)}")

# Plotting the regression models
residuals_test = y_test - y_test_predictions
plt.scatter(y_test_predictions, residuals_test, color='blue', label="Residuals")
plt.axhline(y=0, color='red', linestyle='--', label='Zero Residual Line')
plt.xlabel('Predicted Selling Price')
plt.ylabel('Residuals')
plt.title('Residual Plot (Test Set)')
plt.legend()
plt.tight_layout()
plt.savefig("Linear_Regression/ResidualPlot.png")
plt.show()


# Novelty: add LASSO and Ridge regression regularization parameters.
LrModel = LinearRegressionOLS(alpha=0.01, li_ratio=1)  # Setting l1_ratio=1 for Lasso
LrModel.fitModel(X_train, y_train)

# Fit the model with the concat data and predict on the test set.
y_test_predictions = LrModel.predictModel(X_test)

# Calculate the MSE, R^2 on the test data.
MSE_test = MSE(df=X_test, yActual=y_test, yPredict=y_test_predictions)
multipleRSquared_test = RSquared(yActual=y_test, yPredict=y_test_predictions, variables=X.shape[1])
adjRSquared_test = RSquared(yActual=y_test, yPredict=y_test_predictions, variables=X.shape[1], type="adjusted")

# Print out both the results.
print(f"Lasso Mean Squared Error (Test Set): {MSE_test}")
print(f"R^2 (Test Set): {multipleRSquared_test}")
print(f"Adjusted R^2 (Test Set): {adjRSquared_test}")


# Novelty: search the dataset for outliers (z-score calculation, greater than three) (boxplot (?)), remove outliers, and refit the data and test to see if there are improvements.
from scipy import stats

# Calculate the Z-scores within the dataset.
z_scores = stats.zscore(df.drop(['name', 'selling_price'], axis=1))

# Define a threshold for outlier detection (+- 3)
threshold = 3

# Find the outliers within the dataset (+- 3 are considered outliers as they are significantly far from the mean and may skew the results).
outliers = (np.abs(z_scores) > threshold).any(axis=1)

# Remove outliers
df_no_outliers = df[~outliers]

# Print the number of outliers removed
num_outliers_removed = outliers.sum()
print(f"Number of outliers removed: {num_outliers_removed}")

# Split the data into X and y again, based on the no outliers.
X_no_outliers = df_no_outliers.drop(['name', 'selling_price'], axis=1)
y_no_outliers = df_no_outliers['selling_price']

# Split the data into training and test sets again
X_train_no_outliers, X_test_no_outliers, y_train_no_outliers, y_test_no_outliers = train_test_split(X_no_outliers, y_no_outliers, test_size=0.2, random_state=50)

# Create and train the linear regression model without outliers
lr_no_outliers = LinearRegressionOLS()
lr_no_outliers.fitModel(X_train_no_outliers, y_train_no_outliers)

# Make predictions on the test set without outliers
y_pred_no_outliers = lr_no_outliers.predictModel(X_test_no_outliers)

# Evaluate the model without outliers
mse_no_outliers = mean_squared_error(y_test_no_outliers, y_pred_no_outliers)
r2_no_outliers = r2_score(y_test_no_outliers, y_pred_no_outliers)

print(f"Mean Squared Error without outliers: {mse_no_outliers}")
print(f"R^2 Score without outliers: {r2_no_outliers}")