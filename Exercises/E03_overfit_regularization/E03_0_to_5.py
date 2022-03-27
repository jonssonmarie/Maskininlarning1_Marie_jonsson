"""
Overfit and regularization exercises
"""
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV  # CV = cross-validation
from sklearn.linear_model import LassoCV
from sklearn.linear_model import ElasticNetCV


"""
0. Tips data EDA (*)
"""
# load data via seaborn
data = sns.load_dataset("tips", cache=True, data_home=None)


def initial_analyse(df):
    """
    :param df: DataFrame
    :return: print
    """
    print("info():\n", df.info(), "\n")
    print("describe():\n", df.describe(), "\n")
    # print("value_counts():\n", df.value_counts(), "\n")
    # print("head():\n", df.head(), "\n")
    # print("tail():\n", df.tail(), "\n")
    print("columns:\n", df.columns, "\n")
    print("index:\n", df.index, "\n")


initial_analyse(data)


def get_quantitative_data(df):
    """
    * https://medium.com/analytics-vidhya/how-to-visualize-pandas-descriptive-statistics-functions-480c3f2ea87c
    :param df: DataFrame
    :return: DataFrame with numeric data
    """
    describe_df = df.describe(include=['int64', 'float64'])   # *
    describe_df.reset_index(inplace=True)
    # Remove "count"
    describe_df = describe_df[describe_df["index"] != "count"]
    return describe_df


def plot_statistics(df):
    """
    Plot statistic returned from describe_qualitative_data
    :param df: DataFrame
    :return: None
    """
    plt.figure(figsize=(15, 8))
    plt.subplot(1, 3, 1)
    sns.barplot(x=df["index"], y=df["total_bill"], data=df)

    plt.subplot(1, 3, 2)
    sns.barplot(x=df["index"], y=df["tip"], data=df)

    plt.subplot(1, 3, 3)
    sns.barplot(x=df["index"], y=df["size"], data=df)
    plt.show()


df_describe = get_quantitative_data(data)
plot_statistics(df_describe)


def plot_qualitative_data(df):
    """
    Plot statistic on qualitative data
    :param df:
    :return:
    """
    plt.figure(figsize=(15, 8))
    plt.subplot(1, 3, 1)
    sns.scatterplot(data=df, x="total_bill", y="tip", hue="sex")
    plt.title("Relationship Total Bill vs Tip, split by Gender")

    plt.subplot(1, 3, 2)
    sns.scatterplot(data=df, x="total_bill", y="tip", hue="smoker")
    plt.title("Relationship Total Bill vs Tip, split by Smoker")

    plt.subplot(1, 3, 3)
    sns.scatterplot(data=df, x="size", y="tip", hue="day")
    plt.title("Relationship Total Bill vs Tip, split by Day")

    plt.show()


plot_qualitative_data(data)


def plot_box(df):
    """
    :param df: DataFrame
    :return: None
    """
    # sort data after day in the week
    thursday = df[df['day'] == 'Thur']
    friday = df[df['day'] == 'Fri']
    saturday = df[df['day'] == 'Sat']
    sunday = df[df['day'] == 'Sun']

    plt.figure(figsize=(15, 8))
    plt.subplot(1, 4, 1)
    sns.boxplot(data=thursday, x="sex", y="tip", hue="smoker")
    plt.title("Thursday")

    plt.subplot(1, 4, 2)
    sns.boxplot(data=friday, x="sex", y="tip", hue="smoker")
    plt.title("Friday")

    plt.subplot(1, 4, 3)
    sns.boxplot(data=saturday, x="sex", y="tip", hue="smoker")
    plt.title("Saturday")

    plt.subplot(1, 4, 4)
    sns.boxplot(data=sunday, x="sex", y="tip", hue="smoker")
    plt.title("Sunday")

    plt.show()


plot_box(data)


def pie_plot(df):
    """
    Pie plot of gender
    :param df: DataFrame
    :return: None
    """
    df.groupby('sex').size().plot(kind='pie', autopct='%.1f%%')
    plt.ylabel("Gender", size=20)
    plt.show()


pie_plot(data)


def data_preparation(df, y_name):
    """
    Data preparation
    :param df: DataFrame with data
    :param y_name: string, column name to set as y
    :return: DataFrame, Series
    """
    df = df.drop(columns=["sex", "day", "time", "smoker"], axis=1)
    X, y = df.drop([y_name], axis=1), df[y_name]
    return X, y


X, y = data_preparation(data, "tip")


""" 
1. Train|test split (*)
"""


def train_splitting(X_features, y_feature):
    """
    Split data to train and test for both X and y
    :param X_features: DataFrame
    :param y_feature: DataFame y
    :return: DataFrame, DataFrame, Series, Series
    """
    X_train, X_test, y_train, y_test = train_test_split(X_features, y_feature, test_size=0.3, random_state=42)
    return X_train, X_test, y_train, y_test


X_train, X_test, y_train, y_test = train_splitting(X, y)


"""
2. Feature standardization (*)
"""


def x_standardised(x_train, x_test):

    x_train_standardised = (x_train - x_train.mean()) / x_train.std()
    x_test_standardised = (x_test - x_train.mean()) / x_train.std()

    print("X_train total_bill std", x_train_standardised["total_bill"].std(),
          "\nX_train size std", x_train_standardised["size"].std())
    print("X_train total_bill mean", x_train_standardised["total_bill"].mean(),
          "\nX_train size mean", x_train_standardised["size"].mean())
    print("X_test total_bill std", x_test_standardised["total_bill"].std(),
          "\nX_test size std", x_test_standardised["size"].std())
    print("X_test total_bill mean", x_test_standardised["total_bill"].mean(),
          "\nX_test size mean", x_test_standardised["size"].mean())
    # they are ok!


x_standardised(X_train, X_test)


scaler = StandardScaler()


def feature_scaling(X_training, X_testing):
    """
    .fit - Compute the mean and std to be used for later scaling
    .transform - Perform standardization by centering and scaling.
    :param X_training: np.array
    :param X_testing: np.array
    :return: np.array, np.array
    """
    scaler.fit(X_training)
    # transform both X_train and X_test
    scaled_X_train = scaler.transform(X_training)
    scaled_X_test = scaler.transform(X_testing)
    return scaled_X_train, scaled_X_test


evaluation_df = []


def evaluation(y_testing, y_predication):
    """
    Calculate RMSE and append RMSE to a list
    MSE - Mean squared error  =  sum((x - x_bar)^2) / n, sum 1 to n-1
    RMSE - Root mean square error =  sqrt(MSE)
    :param y_testing: np.array
    :param y_predication: np.array
    :return: None
    """
    mse = mean_squared_error(y_testing, y_predication)
    rmse = np.sqrt(mse)
    evaluation_df.append(rmse)


def statistics(y_testing, y_predict):
    """
    MAE - Mean Absolut error =  sum(abs(x - x_bar)) / n, sum 1 to n-1
    MSE - Mean squared error  =  sum((x - x_bar)^2) / n, sum 1 to n-1
    RMSE - Root mean square error =  sqrt(MSE)
    :param y_testing:
    :param y_predict:
    :return: print MSE, MAE, RMSE
    """
    MSE = mean_squared_error(y_testing, y_predict)
    MAE = mean_absolute_error(y_testing, y_predict)
    RMSE = np.sqrt(MSE)

    print(f"MSE: {MSE:.3f}, MAE: {MAE:.3f}, RMSE: {RMSE:.3f}\n")


"""
3. Polynomial features (*)
"""


def polynomial_features(x_train, x_test):
    """
    Generate a new feature matrix consisting of all polynomial combinations of the features with degree
    less than or equal to the specified degree.
    :param x_train: DataFrame
    :param x_test:DataFrame
    :return: print
    """
    model_polynomial = PolynomialFeatures(degree=2, include_bias=False)
    # include_bias set to False since a column with 1 will be set later
    
    poly_features_X_train = model_polynomial.fit_transform(x_train)
    print("\nshape after fit_transform on X_train: ", poly_features_X_train.shape)

    poly_features_X_train = model_polynomial.transform(x_train)
    print("shape after transform on X_train: ", poly_features_X_train.shape)
    
    poly_features_X_test = model_polynomial.transform(x_test)
    print("shape after transform on X_test: ", poly_features_X_test.shape)

    test = model_polynomial.fit_transform(x_test)

    # test if equal:
    print("poly_features_X_test == test).sum() : ", np.sum((poly_features_X_test == test)))
    print("poly_features_X_test.size:", (poly_features_X_test.size),"\ntest.size: ", test.size,"\n")



polynomial_features(X_train, X_test)

"""
4. Polynomial regression (*)
"""

rmse_test = []
rmse_train = []


def polynomial_regression(x_train, x_test, y_training, y_testing):
    """
    PolynomialFeatures: Generate a new feature matrix consisting of all polynomial combinations of
    the features with degree less than or equal to the specified degree.
    .fit_transform - Fit to data, then transform it. Combine .fit and .transform
    .fit - Compute the mean and std to be used for later scaling
    .transform - Perform standardization by centering and scaling.
    :param x_train: DataFrame
    :param x_test: DataFrame
    :param y_training: Series
    :param y_testing: Series
    :return: np.array, np.array
    """
    for n in range(1, 5):
        model_poly_regression = PolynomialFeatures(degree=n, include_bias=False)

        poly_features_X_train = model_poly_regression.fit_transform(x_train)
        poly_features_X_test = model_poly_regression.transform(x_test)

        model = LinearRegression()
        model.fit(poly_features_X_train, y_training)

        y_pred_test = model.predict(poly_features_X_test)
        y_pred_train = model.predict(poly_features_X_train)

        # evaluation calculate RMSE for each loop for y_testing/predicted and y_training/predicted to a DataFrame

        evaluation(y_testing, y_pred_test)
        evaluation(y_training, y_pred_train)

        # rmse collected for plot
        rmse_test.append(np.sqrt(mean_squared_error(y_testing, y_pred_test)))
        rmse_train.append(np.sqrt(mean_squared_error(y_training, y_pred_train)))

        if n == 4:
            return poly_features_X_train, poly_features_X_test


poly_features_X_train, poly_features_X_test = polynomial_regression(X_train, X_test, y_train, y_test)


def line_plot(rmse_all):
    """
    Plot RMSE during the polynomial_regression with degree 1 to 4
    :param rmse_all: list
    :return: None
    """
    plt.figure(figsize=(15, 8))
    sns.lineplot(data=rmse_all)
    plt.title("Training loss and test loss")
    plt.show()


rmse_poly_1to_4 = pd.DataFrame([rmse_train, rmse_test]).T
rmse_poly_1to_4.columns = ["rmse_train", "rmse_test"]
line_plot(rmse_poly_1to_4)

# initiate DataFrame
eval_df = pd.DataFrame(evaluation_df, columns=["RMSE"],
                       index=["Linear Regression 1 test", "Linear Regression 1 train", "Linear Regression 2 test",
                              "Linear Regression 2 train", "Linear Regression 3 test", "Linear Regression 3 train",
                              "Linear Regression 4 test", "Linear Regression 4 train"]).round(decimals=2)

print("statistics:\n", eval_df)


"""
5. Regularization methods (*)
"""


def ridge_regression(x_train, x_testing, y_training, y_testing):
    """
    Ridge regression with built-in cross-validation (CV).
    .fit - Compute the mean and std to be used for later scaling
    scoring="neg_mean_squared_error" : if you want to use it to tune your models, or cross_validate using the
    utilities present in Scikit, use 'neg_mean_squared_error'
    :param x_train: DataFrame
    :param x_testing: DataFrame
    :param y_training: Series
    :param y_testing: Series
    :return: Print
    """
    model_ridgeCV = RidgeCV(alphas=[0.01, 0.1, 0.5, 1, 5, 10], scoring="neg_mean_squared_error")
    model_ridgeCV.fit(x_train, y_training)
    y_pred = model_ridgeCV.predict(x_testing)

    # print statistics statistics MSE, MAE, RMSE
    print("\nRidge CV regression: ")
    statistics(y_testing, y_pred)

    print("model_ridgeCV.alpha_:", model_ridgeCV.alpha_)
    print("model_ridgeCV.coef_", model_ridgeCV.coef_)


ridge_regression(poly_features_X_train, poly_features_X_test, y_train, y_test)


def lasso_regression(poly_X_train, poly_X_test, y_training, y_testing):
    """
    Lasso linear model with iterative fitting along a regularization path.
    The best model is selected by cross-validation.
    cross-validation capabilities to automatically select the best hyper-parameters
    :param poly_X_train:
    :param poly_X_test:
    :param y_training: Series
    :param y_testing: Series
    :return: Print
    """
    model_lassoCV = LassoCV(eps=0.001, n_alphas=100, max_iter=10000, cv=5)  # cv = k - inclination
    model_lassoCV.fit(poly_X_train, y_training)
    y_pred = model_lassoCV.predict(poly_X_test)

    # print statistics MSE, MAE, RMSE
    print("\nLasso CV regression: ")
    statistics(y_testing, y_pred)
    print(f"Chosen alpha (penalty term) = {model_lassoCV.alpha_}")
    # we notice that many coefficients have been set to 0 using Lasso it has selected some features for us
    print("model_lassoCV.coef_", model_lassoCV.coef_)


lasso_regression(poly_features_X_train, poly_features_X_test, y_train, y_test)


def elastic_net(poly_X_train, poly_X_test, y_training, y_testing):
    """
    Elastic Net CV model with iterative fitting along a regularization path.
    The best model is selected by cross-validation.
    cross-validation capabilities to automatically select the best hyper-parameters
    Elastic-net is useful when there are multiple features that are correlated with one another.
    Lasso is likely to pick one of these at random, while elastic-net is likely to pick both.
    A practical advantage of trading-off between Lasso and Ridge is that it allows Elastic-Net to
    inherit some of Ridge’s stability under rotation.
    :param poly_X_train:
    :param poly_X_test:
    :param y_training: Series
    :param y_testing: Series
    :return: Print
    """
    # l1_ratio is alpha in the theory
    model_elastic = ElasticNetCV(l1_ratio=[.001, .01, .05, .1, .5, .9, .95, 1], eps=0.001, n_alphas=100, max_iter=10000)
    model_elastic.fit(poly_X_train, y_training)
    y_pred = model_elastic.predict(poly_X_test)

    # print statistics statistics MSE, MAE, RMSE
    print("\nElastic Net CV regression: ")
    statistics(y_testing, y_pred)

    print(f"L1 ratio: {model_elastic.l1_ratio_}")
    print(f"alpha {model_elastic.alpha_}")


elastic_net(poly_features_X_train, poly_features_X_test, y_train, y_test)
