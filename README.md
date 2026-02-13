# Multiple Linear Regression with 3D Visualization

This repository contains an implementation of **Multiple Linear Regression** using Scikit-Learn. Unlike Simple Linear Regression which fits a line in 2D space, this project fits a **hyperplane** (specifically a plane) in 3D space to predict a target variable based on two independent features.

## 📄 Project Overview

The objective of this notebook is to visualize how Linear Regression expands to higher dimensions. It uses interactive 3D plotting to demonstrate how the model finds the "best fit plane" to minimize error between actual data points and predicted values.

Key highlights:
* **Synthetic Data Generation:** Creating a custom dataset with 2 input features and noise.
* **3D Interactive Plotting:** Using `Plotly` to visualize the relationship between features and the target.
* **Hyperplane Visualization:** Drawing the regression surface (plane) over the scattered data points.

## 📊 Dataset

The dataset is generated programmatically using Scikit-Learn's `make_regression` module:
* **Samples:** 500
* **Features (Inputs):** 2 (`feature1`, `feature2`)
* **Target (Output):** 1 (`target`)
* **Noise:** 50 (Standard deviation of gaussian noise)

## 🧮 Mathematical Model

In Multiple Linear Regression with two independent variables, the equation becomes:

$$y = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + \epsilon$$

Where:
* $y$ is the target variable.
* $x_1, x_2$ are the input features.
* $\beta_0$ is the intercept (bias).
* $\beta_1, \beta_2$ are the coefficients (weights) for each feature.

## 🛠️ Implementation Details

### 1. Model Training
The project uses the standard `LinearRegression` class from Scikit-Learn:
```python
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(X_train, y_train)
