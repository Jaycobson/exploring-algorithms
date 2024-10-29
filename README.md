# Linear Regression from Scratch 🧮

This project implements **Linear Regression from scratch** using **Python and NumPy**. The goal is to understand how linear regression works at a mathematical level by implementing it without relying on high-level machine learning libraries like `scikit-learn`.  

---

## 📋 Table of Contents

1. Overview
2. Features
3. Mathematics Behind Linear Regression
4. Code Explanation
5. Example Usage 


---

## 🧐 Overview

Linear Regression is a **simple but powerful algorithm** used to predict continuous values. This project demonstrates how to:

- Compute the **optimal coefficients** for the linear model using the **normal equation**.  
- **Make predictions** on new data.  
- Evaluate the model using the **R-squared metric**.  

The code provides a minimalistic, clean implementation of **multiple linear regression** from scratch.

---

## ✨ Features

- **Linear regression model** with coefficient estimation via **closed-form solution** (Normal Equation).  
- **Prediction function** to generate predictions for new data points.  
- **R-squared metric** to evaluate model performance.  
- No external ML libraries used – only **NumPy**!

---

## 🧑‍🏫 Mathematics Behind Linear Regression

The **linear regression model** attempts to fit a linear equation to the given data:

$$ y = \theta_0 + \theta_1 x_1 + \theta_2 x_2 + \ldots + \theta_n x_n $$

Where:
- y is the target value  
- $ \theta_0\ $ is the intercept  
- \(\theta_1, \theta_2, \ldots\) are the coefficients  

Using the **normal equation**, the optimal coefficients can be computed as:

\[
\hat{\theta} = (X^T X)^{-1} X^T y
\]

---

## 🚀 How to Use

### Prerequisites
- Python 3.x
- NumPy library (`pip install numpy`)

class LinearRegression:
    def __init__(self):
        self.intercept = None
        self.coef = None
        
__init__: Initializes the intercept and coefficients to None.



def fit(self, X, y):
    ones = np.ones((len(X), 1))
    X = np.concatenate((ones, X), axis=1)
    XT = X.T
    XTX = XT.dot(X)
    XTX_inv = np.linalg.inv(XTX)
    XTy = XT.dot(y)
    self.coef = XTX_inv.dot(XTy)
    
fit: This method calculates the coefficients using the normal equation. A column of ones is added to the input data to represent the intercept term.


def predict(self, X):
    ones = np.ones((len(X), 1))
    X = np.concatenate((ones, X), axis=1)
    return X.dot(self.coef)

    
predict: Predicts target values for new input data based on the learned coefficients.


def rsquared(self, X, y):
    ypred = self.predict(X)
    ss_total = np.sum((y - np.mean(y))**2)
    ss_residual = np.sum((y - ypred)**2)
    return 1 - (ss_residual / ss_total)

    
rsquared: Computes the R-squared value to evaluate how well the model fits the data.

💻 Example Usage
import numpy as np
from linear_regression import LinearRegression

# Initialize the model
lr = LinearRegression()

# Training data (X: features, y: target)
X = np.array([[2, 2, 3], [1, 3, 4], [4, 2, 5]])
y = np.array([3, 7, 5])

# Fit the model to the data
lr.fit(X, y)

# Make predictions on new data
pred = lr.predict([[3, 5, 3]])
print(f"Prediction: {pred}")

# Evaluate the model with R-squared
r2 = lr.rsquared(X, y)
print(f"R-squared: {r2}")
