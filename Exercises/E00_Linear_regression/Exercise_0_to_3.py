# Linear regression exercises 0 - 3
# x  - called minutes per month
# y  - SEK per month

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# a)


def simulate_variables(samples):
    np.random.seed(42)
    phone_minutes_x = np.abs(np.random.normal(loc=100, scale=100, size=samples))
    phone_error = np.random.normal(loc=0, scale=50, size=samples)
    phone_cost_y = 2 * phone_minutes_x + 25 + phone_error
    return phone_minutes_x, phone_error, phone_cost_y


phone_minutes_x, phone_error, phone_cost_y = simulate_variables(400)


def plot_data(x_value, y_value):
    fig, ax = plt.subplots(dpi=100)
    sns.scatterplot(x=x_value, y=y_value, label="400 simulated values")
    ax.set(xlabel="Called minutes", ylabel="SEK per month")


plot_data(phone_minutes_x, phone_cost_y)

# b)
phone_all_np = np.vstack((phone_minutes_x, phone_cost_y)).T
phone_outliers_removed = phone_all_np[(phone_all_np[:,0] <= 300) & (phone_all_np[:, 1] >= 0)]
# phone_outliers_removed from:
# https://stackoverflow.com/questions/41898000/python-select-row-in-numpy-array-where-multiple-conditions-are-met
# answered Jan 27 '17 at 16:06 by  kennytm

print("Length after removed outliers:", int(phone_outliers_removed.size/2))

phone_true_y = 2 * phone_outliers_removed[:,0] + 25
phone_all_true = np.vstack((phone_outliers_removed[:,0], phone_true_y))


def plot_scatter_line(x1, y1, x2, y2):
    fig, ax = plt.subplots(dpi=100)
    sns.scatterplot(x=x1, y=y1)
    sns.lineplot(x=x2, y=y2, color='r', linewidth = 3.5, linestyle='--',
                 label="Parameter line after removing outliers")
    ax.set(xlabel="Called minutes", ylabel="SEK per month")


plot_scatter_line(phone_outliers_removed.T[0, :], phone_outliers_removed.T[1, :], phone_all_true[0,:],
                  phone_all_true[1,:])


def least_square(x1, y1):
    """We can rewrite the line equation as y = Ap, where A = [[x 1]] and p = [[m], [c]]."""
    A = np.vstack([x1, np.ones(len(x1))]).T
    mm, cc = np.linalg.lstsq(A, y1, rcond=None)[0]
    plt.plot(x1, mm * x1 + cc, 'k', label='Fitted line least-squares solution')


x1 = phone_outliers_removed[:,0]
y1 = (phone_outliers_removed[:,1]).reshape(len(phone_outliers_removed[:,0]))
least_square(x1,y1)

# collect to one DataFrame
phone_all_df = pd.DataFrame(data=[phone_minutes_x, phone_cost_y, phone_outliers_removed[:,0], phone_true_y]).T\
    .rename(columns={0: "minutes", 1: "cost", 2: "true_minutes", 3: "true_cost"})


# Exercise 1. Train|test split
# a)
phone_outliers_removed = phone_all_df.drop(["minutes", "cost"], axis="columns").dropna()


def train_test_split(df, train_fraction=0.7):
    train_len = int(len(df) * train_fraction)
    train = df.sample(n=train_len, random_state=42, replace=False)
    test = df.sample(n=(len(df) - train_len), random_state=42, replace=False)

    return train, test


train, test = train_test_split(phone_outliers_removed, train_fraction=0.7)

# b)
print(f"Length Xy_train = {len(train)}")
print(f"Length Xy_test = {len(test)}\n")
print(f"Sum of X_train + y_train = {len(train) +len(test)}")


def least_square(x1, y1):
    """We can rewrite the line equation as y = Ap, where A = [[x 1]] and p = [[m], [c]]."""
    A = np.vstack([x1, np.ones(len(x1))]).T
    mm, cc = np.linalg.lstsq(A, y1, rcond=None)[0]
    plt.plot(x1, mm * x1 + cc, 'k', label='Fitted line least-squares solution')


# 2. Simple linear regression with ordinary normal equation - least square (*)
np_train = np.array(train)
x = np_train[:, 0]
y = np_train[:, 1]
least_square(x, y)

# Simple linear regression
# t avrundat= beta_0 + beta_1*x


def prediction(X, y):
    beta_1, beta_0 = np.polyfit(X, y, deg=1)
    #print(f"Intercept {beta_0:.3f}")
    #print(f"Slope {beta_1:.3f}")
    return beta_0, beta_1


beta_0, beta_1 = prediction(np_train[:, 0], np_train[:, 1])


def y_hat(x):
    return beta_0 + beta_1 * x


equation = y_hat(np_train[:, 0])

""" 
3. Prediction and evaluation (*)
"""
np_test = np.array(test)
x = np_test[:, 0]
y = np_test[:, 1]

beta_0, beta_1 = prediction(np_test[:, 0], np_test[:, 1])
equation = y_hat(np_test[:, 0])

plt.plot(x, equation, 'b', linewidth=3.5, linestyle='--', label='Test data') # m * x + c
plt.legend()
plt.show()

# Evaluation
# MAE - Mean Absolut errror
# MSE - Mean squared error (sqyúare of original unit, puished outliers more than MAE, somewhat hard to interpret)
# RMSE - Root mean square error (original unit, punishes oultiners more than MAE, easy to interpret)
X_train, y_train = train.drop("true_cost", axis=1), train["true_cost"]
X_test, y_test = test.drop("true_cost", axis=1), test["true_cost"]

m = len(y_test)
y_hat = equation
MAE = 1/m * np.sum(np.abs(y_test - y_hat))
MSE = 1/m * np.sum((y_test - y_hat)**2)
RMSE = np.sqrt(MSE)

print(MAE, MSE, RMSE)
# b) Mean absolute error on testing data: 36.97 kr Mean squared error on testing data: 2374 kr^2 Root mean squared error on testing data: 48.72 k
# OBS Stämmer inte med min kod! Väldigt små skillnader iom numpys least square

