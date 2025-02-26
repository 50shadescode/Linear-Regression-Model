# Linear-Regression-Model
Simple Linear Regression in Python

Overview

This project demonstrates how to perform Simple Linear Regression using Python's scikit-learn library. The model is trained to predict a dependent variable (y) based on an independent variable (X).

Prerequisites

Ensure you have the following dependencies installed:

pip install numpy matplotlib scikit-learn

Dataset

The dataset consists of a simple array:

X: Independent variable (features)

y: Dependent variable (target values)

X = np.array([1, 2, 3, 4, 5]).reshape(-1, 1)
y = np.array([2, 3, 5, 6, 8])

Steps to Implement Simple Linear Regression

1. Import Necessary Libraries

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

numpy: For numerical operations

matplotlib.pyplot: For data visualization

sklearn.linear_model.LinearRegression: To create a linear regression model

sklearn.metrics.r2_score: To evaluate model performance

2. Prepare the Data

X = np.array([1, 2, 3, 4, 5]).reshape(-1, 1)
y = np.array([2, 3, 5, 6, 8])

X is reshaped into a 2D array as required by sklearn

y contains the target values

3. Train the Model

model = LinearRegression()
model.fit(X, y)

LinearRegression(): Initializes the regression model

.fit(X, y): Trains the model using the least squares method

4. Make Predictions

y_pred = model.predict(X)

Uses the trained model to predict y values based on X

5. Evaluate the Model

r2 = r2_score(y, y_pred)
print(f"R-squared: {r2}")

Calculates R-squared, which measures how well the model explains the data.

6. Visualize the Results

plt.scatter(X, y, color='blue', label='Actual data')
plt.plot(X, y_pred, color='red', label='Regression Line')
plt.xlabel("X (Independent Variable)")
plt.ylabel("y (Dependent Variable)")
plt.legend()
plt.show()

Blue dots: Original data points

Red line: Predicted regression line

Output

R-squared score is printed in the console.

Scatter plot with regression line is displayed.

Conclusion

This simple linear regression model provides insights into the relationship between X and y. The model can be further improved by evaluating residuals and handling data preprocessing techniques.

Next Steps

Extend the model to handle multiple linear regression

Perform residual analysis

Implement cross-validation for better performance assessment
