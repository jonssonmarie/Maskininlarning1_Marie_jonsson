# 5. Multiple linear regression (*)

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from Exercise_4 import simulate_x_variables


# Exercise 5
def plot_rmse(data):
    sns.lineplot(data=data, x="samples", y="RMSE")
    #sns.set(xlabel="simulated samples", ylabel="RMSE")
    plt.show()


def remove_outliers(df):
    df_outliers_rem = df[(df["x1"] < 300) & (df["x3"] < 4) & (df["y"] > 0)]
    return df_outliers_rem


def train_test_split(df, train_fraction=0.8):
    train_len = int(len(df) * train_fraction)
    train = df.sample(n=train_len, random_state=42, replace=False)
    test = df.sample(n=(len(df) - train_len), random_state=42, replace=False)
    return train, test


RMSE_data = []
lst = [10, 1000, 10000, 100000, 1000000]
for i in lst:
    df = simulate_x_variables(i)
    df_outliers_rem = remove_outliers(df)

    train, test = train_test_split(df_outliers_rem, train_fraction=0.8)

    X, y = train.drop("y", axis="columns"), train["y"]

    # OLS normal equation/closed from equation
    regression_fit = lambda X, y: np.linalg.inv(X.T @ X) @ X.T @ y

    beta_hat = regression_fit(X,y)

    predict = lambda x, beta: np.dot(x, beta)   # dot product in linear algebra

    test_sample = test.drop("y", axis="columns")
    y_hat = predict(test_sample, beta_hat)

    X_train, y_train = train.drop("y", axis=1), train["y"]
    X_test, y_test = test.drop("y", axis=1), test["y"]

    # Prediction this uses OLS normal equation
    beta_hat = regression_fit(X_train, y_train)

    predict = lambda X, weights: X @ weights
    y_hat = predict(X_test.to_numpy(), beta_hat.to_numpy().reshape(4,1))

    beta_hat.to_numpy().reshape(4, 1)

    # Evaluation
    # MAE - Mean Absolut error
    # MSE - Mean squared error (square of original unit, punished outliers more than MAE, somewhat hard to interpret)
    # RMSE - Root mean square error (original unit, punishes outliers more than MAE, easy to interpret)

    m = len(y_test)
    y_hat = np.reshape(y_hat, m)
    MAE = 1/m * np.sum(np.abs(y_test - y_hat))
    MSE = 1/m * np.sum((y_test - y_hat)**2)
    RMSE = np.sqrt(MSE)
    rmse_input = [i, RMSE]
    RMSE_data.append(rmse_input)

    #print(f"MAE - Mean Absolut error = \n {MAE}")
    #print(f"MSE - Mean squared error = \n {MSE}")
    print(f"RMSE - Root mean square error =\n {RMSE}")
    print("Loop =", i, "\n")

print(RMSE_data)
RMSE_data = pd.DataFrame(RMSE_data, columns=["samples", "RMSE"])
plot_rmse(RMSE_data)