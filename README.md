# Machine Learning Supervised Learning: Car Price Prediction Model

- Proposed Domain: Building a model to predict car pricing, based on selling price, car type, year, miles driven, and more. 
- Hypothesis: Utilizing the Supervised Learning regression models, we can accurately predict a car’s price based on its characteristics.
- ML that matters: The data can be used on Car Lots to predict used car prices. 
- Dataset: Vehicle Data set on Kaggle https://www.kaggle.com/datasets/nehalbirla/vehicle-dataset-from-cardekho
- Total Size: 299 kB
- Number of Examples: 14,830
- Dimension of Instances (n): name, year, selling price, km driven, fuel, seller type, transmission, Owner
- Proposed Contributions: Ameer (5033) will implement Linear Regression and K-N-N., Namil (4033) will implement Decision Trees and Random Forest.

## Checkpoint

- Aim to get two methods done by checkpoint (LR, KNN).
- Check the book examples for LR, KNN.
- Watch videos on how to apply these.
- Test results with sci-kit and chat.
- Draft up checkpoint in overleaf. Keep it simple, 1 page, similar to RL.
- Overfiting / underfitting.
- Testing data with validity sets; check the sample report.
- Process: Finish HW5, Do the algoritms, plot the algorithms side-by-side, write up checkpoint.

## Novelty
- Data pre-processing, one-hot enconding data that is categorical (HW5).
- Validation and tester sets for the report (HW5).
- Ski-learning comparison for model checking. 
- Also, if we have non-complete data it is a novelty.
- The types of methods we use that are not covered could be noveltys.

## K-Nearest-Neighbors

- Logic: given our dataset, we break it down into training examples (instances). From there, we have the relevant data for the specific training examples and we would predict the label (output) based on our car data. In our case, the output label is the car price, while the input is car data from the dataset.
- We can add weights and learning rates to test other options.
- May need to split into validation sets (?).
- Use HW-4 for pseduo-code implementation.
- Euclidean-distance (or other distance formulas). 

## Linear Regression

- Logic: Predict a response variable based on a predictor variable(s). Furthermore, it is used to predict the value of a dependent variables based on the values of one or more independent variables, in which we have options of simple linear regression (SLR) and multiple linear regression (MLR).
- Task: Predict the label y on an unknown instance x by using a linear combination of the features/attributes of x (aim to minimize the true risk).
- LR with Gradient Descent (RSS equation), with program in HW-4.
- Explore full vs. stochastic gradient descent, in which the pseduo code is in the book.

## Decision Trees

## Random Forest
