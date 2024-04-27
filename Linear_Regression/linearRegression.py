# Import the dataset and ski libraries to check our model.
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl # Used for font.
from sklearn.model_selection import train_test_split # Used for testing.
from sklearn.linear_model import LinearRegression # Used for testing.
from sklearn.metrics import r2_score # Used for testing.


# Define the OLS Regression Algorithm from Scratch.
class LinearRegressionOLS:
    # Constructor for the weights, if not pre-assigned.
    def __init__(self):
        self.weights = None

    # Function used to fit the model, given the input data and the label to be predicted.
    def fitModel(self, X, y):
        # Add bias term
        X = np.insert(X.to_numpy(), 0, 1, axis=1)
        # Calculate weights using pseudoinverse, taken module 7 slides.
        self.weights = np.linalg.pinv(X.T.dot(X)).dot(X.T).dot(y.to_numpy())

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

# Now, split the training dataframe into 75% training and 25% validation.
X_train, X_validation, y_train, y_validation = train_test_split(X_test, y_test, test_size=0.25, random_state=50)

# Train the OLS regression model.
LrModel = LinearRegressionOLS()
LrModel.fitModel(X_train, y_train)

# Make predictions on the validation data.
y_validation_predictions = LrModel.predictModel(X_validation)

# Calculate the MSE, R^2 on the validation data.
MSE_validation = MSE(df=df, yActual=y_validation, yPredict=y_validation_predictions)
multipleRSquared_validation = RSquared(yActual=y_validation, yPredict=y_validation_predictions, variables=X.shape[1])
adjRSquared_validation = RSquared(yActual=y_validation, yPredict=y_validation_predictions, variables=X.shape[1], type="adjusted")

# Make predictions on the test daa.
y_test_predictions = LrModel.predictModel(X_test)

# Calculate the MSE, R^2 on the test data.
MSE_test = MSE(df=df, yActual=y_test, yPredict=y_test_predictions)
multipleRSquared_test = RSquared(yActual=y_test, yPredict=y_test_predictions, variables=X.shape[1])
adjRSquared_test = RSquared(yActual=y_test, yPredict=y_test_predictions, variables=X.shape[1], type="adjusted")

# Print out both the results.
print(f"Mean Squared Error (Validation Set): {MSE_validation}")
print(f"Mean Squared Error (Test Set): {MSE_test}")
print(f"R^2 (Validation Set): {multipleRSquared_validation}")
print(f"Adjusted R^2 (Validation Set): {adjRSquared_validation}")
print(f"R^2 (Test Set): {multipleRSquared_test}")
print(f"Adjusted R^2 (Test Set): {adjRSquared_test}")

# Compare output to the sklearning function.
LrSkl = LinearRegression()
LrSkl.fit(X, y) # fit the model.
y_ski_valid_pred = LrSkl.predict(X_validation) # Preds. based on valid.
y_ski_test_pred = LrSkl.predict(X_test) # Preds. based on test.
print(f"Sklearn Mean Squared Error (Validation): {MSE(df=df, yActual=y_validation, yPredict=y_ski_valid_pred)}")
print(f"Sklearn Mean Squared Error (Test): { MSE(df=df, yActual=y_test, yPredict=y_ski_test_pred)}")
print(f"Sklearn r^2 (Validation): {r2_score(y_validation, y_ski_valid_pred)}")
print(f"Sklearn r^2 (Test): {r2_score(y_test, y_ski_test_pred)}")


# TODO: ASSESS GOOD FIT OF LINEAR MODEL WITH RSS, R^2, PLOTTING, ETC.


# Plotting the regression models
# fig, ax = plt.subplots(1, 2, figsize=(15, 5))

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

# # Plotting for OLS Regression Model for test set.
# ax[0].scatter(y_test, y_test_predictions, color='blue')
# ax[0].plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--')
# ax[0].set_xlabel('Actual Selling Price')
# ax[0].set_ylabel('Predicted Selling Price')
# ax[0].set_title('OLS Regression Model')
# ax[0].legend(['Predicted', 'Actual'])

# # Plotting for Residual plot test set.
# residuals_test = y_test - y_test_predictions
# ax[1].scatter(y_test_predictions, residuals_test, color='blue', label="Residuals")
# ax[1].axhline(y=0, color='red', linestyle='--', label='Zero Residual Line')
# ax[1].set_xlabel('Predicted Selling Price')
# ax[1].set_ylabel('Residuals')
# ax[1].set_title('Residual Plot (Test Set)')
# ax[1].legend()

# plt.tight_layout()
# plt.savefig("Linear_Regression/LRPlots.png")
# plt.show()
