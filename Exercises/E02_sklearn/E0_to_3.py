import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import PolynomialFeatures

"""
0. EDA - Exploratory Data Analysis
"""
df = sns.load_dataset("mpg", cache=True, data_home=None)


def initial_analyse(data):
    """
    :param data: DataFrame
    :return: print
    """
    print("info():\n", data.info(), "\n")
    print("describe():\n", data.describe(), "\n")
    print("value_counts():\n", data.value_counts(), "\n")
    print("head():\n", data.head(), "\n")
    print("tail():\n", data.tail(), "\n")
    print("columns:\n", data.columns, "\n")
    print("index:\n", data.index, "\n")


initial_analyse(df)


def drop_nan(data):
    """
    :param data: DataFrame
    :return: DataFrame without rows with any NaN
    """
    data_new = data.dropna(axis=0, how="any")
    return data_new


df_update = drop_nan(df)


def scatter_plot(data):
    """
    :param data: DataFrame
    :return: None
    """
    fig, ax = plt.subplots(2, 3, dpi=100, figsize=(16, 8))

    ax[0, 0].scatter(x=data["mpg"], y=data["horsepower"])
    ax[0, 0].set(xlabel="mpg", ylabel="horsepower")

    ax[0, 1].scatter(x=data["mpg"], y=data["cylinders"])
    ax[0, 1].set(xlabel="mpg", ylabel="cylinders")

    ax[0, 2].scatter(x=data["weight"], y=data["mpg"])
    ax[0, 2].set(xlabel="Weight", ylabel="mpg")

    ax[1, 0].scatter(x=data["displacement"], y=data["mpg"])     # displacement = cylindervolym
    ax[1, 0].set(xlabel="displacement", ylabel="mpg")

    ax[1, 1].scatter(x=data["model_year"], y=data["mpg"])
    ax[1, 1].set(xlabel="model_year", ylabel="mpg")

    ax[1, 2].scatter(x=data["acceleration"], y=data["mpg"])
    ax[1, 2].set(xlabel="acceleration", ylabel="mpg")

    plt.tight_layout()
    plt.show()


scatter_plot(df_update)


df_update = df_update.drop(columns=["name", "origin"])

""" 
1. Train|test split (*)
"""


def create_X_y(data, y_name):
    """
    :param data: DataFrame
    :param y_name: string
    :return: DataFrame, Series
    """
    X, y = data.drop([y_name], axis=1), data[y_name]
    return X, y


X, y = create_X_y(df_update, "mpg")


def train_split(para_X, para_y):
    """
    Train test split
    :param para_X: np.array
    :param para_y: np.array
    :return: np.array, np.array, np.array, np.array
    """
    X_train, X_test, y_train, y_test = train_test_split(para_X, para_y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test


X_train, X_test, y_train, y_test = train_split(X, y)


"""
2. Function for evaluation (*)
"""

# we use normalization here
# instantiate an object from the class MinMaxScaler
scaler = MinMaxScaler()


def feature_scaling(X_training, X_testing):
    """
    :param X_training: np.array
    :param X_testing: np.array
    :return: np.array, np.array
    """
    # transform both X_train and X_test
    scaler.fit(X_training)
    scaled_X_train = scaler.transform(X_training)
    scaled_X_test = scaler.transform(X_testing)
    return scaled_X_train, scaled_X_test


scaled_X_train, scaled_X_test = feature_scaling(X_train, X_test)


def create_SVD(scaled_X_training, y_training, scaled_X_testing):
    """
    :param scaled_X_training: np.array
    :param y_training: np.array
    :param scaled_X_testing: np.array
    :return: np.array
    """
    model_SVD = LinearRegression()
    model_SVD.fit(scaled_X_training, y_training)  # fit - Compute the minimum and maximum to be used for later scaling.
    y_pred_SVD = model_SVD.predict(scaled_X_testing)
    # weights
    print("\nSVD Linear Regression")
    print(f"Weights (beta_hats) : {model_SVD.coef_}")
    print(f"Intercept: {model_SVD.intercept_}")
    return y_pred_SVD


def create_SGD(scaled_X_training, y_training, scaled_X_testing):
    """ Stochastic gradient descent (SGD)
    :param scaled_X_training: np.array
    :param y_training: np.array
    :param scaled_X_testing: np.array
    :return: np.array
    """
    # note that SGD requires features to be scaled
    model_SGD = SGDRegressor(loss="squared_error", learning_rate="invscaling", max_iter=100000)
    model_SGD.fit(scaled_X_training, y_training)
    y_pred_SGD = model_SGD.predict(scaled_X_testing)
    print("\nSGD Stochastic gradient descent")
    print(f"Weights (beta_hats) {model_SGD.coef_}")
    print(f"Intercept {model_SGD.intercept_}")
    return y_pred_SGD


def create_SVD_scaled(scaled_X_training, y_training, scaled_X_testing):
    """
    :param scaled_X_training: np.array
    :param y_training: np.array
    :param scaled_X_testing: np.array
    :return: np.array
    """
    model_SVD = LinearRegression()
    model_SVD.fit(scaled_X_training, y_training)  # fit - Compute the minimum and maximum to be used for later scaling.
    y_pred_SVD_scale = model_SVD.predict(scaled_X_testing)
    # weights
    print("\nSVD scaled Linear Regression")
    print(f"Weights (beta_hats) : {model_SVD.coef_}")
    print(f"Intercept: {model_SVD.intercept_}")
    return y_pred_SVD_scale


"""
The fit method is calculating the mean and variance of each of the features present in our data. 
SVD - Singular Value Decomposition that is used for calculating pseudoinverse in OLS normal equation
"""

evalation_df = []


def evaluation(y_testing, y_predication):
    """
    MAE - Mean Absolut error
    MSE - Mean squared error (square of original unit)
    RMSE - Root mean square error (original unit)
    :param y_testing: np.array
    :param y_predication: np.array
    :return: None
    """
    mae = mean_absolute_error(y_testing, y_predication)
    mse = mean_squared_error(y_testing, y_predication)
    rmse = np.sqrt(mse)
    tot = [mae, mse, rmse]
    evalation_df.append(tot)


'''
3. Compare models (*) 
'''


def poly_lin_regression(X_training, y_training, grad):
    """
    :param X_training: np.array
    :param y_training: np.array
    :param grad: int
    :return: np.array of predicted values
    """
    polynomial_features = PolynomialFeatures(degree=grad, include_bias=False)  # skapar med degree = n  = > x^n
    poly_X = polynomial_features.fit_transform(X_training)  # bara fit_transform på X

    model = LinearRegression()
    model.fit(poly_X, y_training)  # tränar genom att köra .fit
    print(f"\nPolynomial linear regression {grad}:")
    #print(f"Weights (beta_hats) : {model.coef_}")
    print(f"Intercept: {model.intercept_}")
    y_pred = model.predict(poly_X)
    return y_pred


y_pred = create_SVD(scaled_X_train, y_train, scaled_X_test)
evaluation(y_test, y_pred)

y_pred_scale = create_SVD_scaled(scaled_X_train, y_train, scaled_X_test)
evaluation(y_test, y_pred_scale)

y_pred_SGD = create_SGD(scaled_X_train, y_train, scaled_X_test)
evaluation(y_test, y_pred_SGD)

# 3. Compare models (*)
scaled_X_train2 = scaler.transform(X_train)

y_pred_poly_1 = poly_lin_regression(scaled_X_train2, y_train, 1)
evaluation(y_train, y_pred_poly_1)

y_pred_poly_2 = poly_lin_regression(scaled_X_train2, y_train, 2)
evaluation(y_train, y_pred_poly_2)

y_pred_poly_3 = poly_lin_regression(scaled_X_train2, y_train, 3)
evaluation(y_train, y_pred_poly_3)

eval_df = pd.DataFrame(evalation_df, columns=["MAE", "MSE", "RMSE"],
                       index=["SVD", "SVD_scaled", "SGD", "Poly 1", "Poly 2", "Poly 3"]).round(decimals=2)

print("\nEvaluation metrics per model\n", eval_df)
