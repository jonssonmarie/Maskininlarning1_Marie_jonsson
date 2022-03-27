"""
Logistic regression exercises
"""
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import random
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix, accuracy_score, ConfusionMatrixDisplay, classification_report
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score
import sys
import os

sys.path.append("..")
path = r"../../assets"
os.chdir(path)
sys.path.insert(0, r'assets/')
parent1 = os.path.dirname(__file__)  # föräldermapp för denna filen
root = os.path.dirname(parent1) # föräldermapp för föräldern
# lägger till denna i en lista som Python söker efter när man importerar moduler
sys.path.append(root)
from assets import initial_analyse

""" 
0. Iris flower dataset (*)
"""
# Load data
data = sns.load_dataset("iris", cache=True, data_home=None)

# a ) Initial analyse
initial_analyse.initial_analyse(data)


# b)
df_dummies = pd.get_dummies(data, columns=["species"], drop_first=False)  # columns=["species"]


def plot_classes(df):
    """
    a subset of features to plot, we check if it is possible to separate the classes
    :param df: DataFrame
    :return: None
    """
    feature_plot = sns.pairplot(data=df[["sepal_width", "sepal_length", "petal_length", "petal_width", "species"]],
                                hue="species", corner=True, palette="tab10")
    feature_plot.fig.suptitle("Check if it is possible to separate the classes", y=0.99)  # y= some height<1
    plt.show()


plot_classes(data)


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


def scatter_plot(df):
    """
    Scatter plot flower species as total
    :param df: DataFrame
    :return: None
    """
    fig, ax = plt.subplots(1, 2, dpi=100, figsize=(16, 8))
    plt.suptitle("Variance analyse on Spieces Sepal/petal - width/length")
    ax[0].scatter(x=df["species"], y=df["sepal_width"])
    ax[0].set(xlabel="species", ylabel="sepal_width")

    ax[1].scatter(x=df["species"], y=df["petal_length"])
    ax[1].set(xlabel="species", ylabel="petal_length")

    plt.tight_layout()
    plt.show()


scatter_plot(data)


def plot_stat_per_flower(df):
    """
    Scatter plot per flower
    :param df: DataFrame
    :return: None
    """
    fig, ax = plt.subplots(1, 3, dpi=100, figsize=(16, 8))
    plt.suptitle("Varaince analyse between species length vs width")
    data1 = df[df["species_versicolor"] == 1]
    ax[0].scatter(x=data1["petal_length"], y=data1["petal_width"])
    ax[0].set(xlabel="petal_length", ylabel="petal_width")
    ax[0].set_title("species_versicolor")

    data2 = df[df["species_virginica"] == 1]
    ax[1].scatter(x=data2["petal_length"], y=data2["petal_width"])
    ax[1].set(xlabel="petal_length", ylabel="petal_width")
    ax[1].set_title("species_virginica")

    data3 = df[df["species_setosa"] == 1]
    ax[2].scatter(x=data3["petal_length"], y=data3["petal_width"])
    ax[2].set(xlabel="petal_length", ylabel="petal_width")
    ax[2].set_title("species_setosa")

    plt.tight_layout()
    plt.show()


plot_stat_per_flower(df_dummies)


def plot_statistics(df):
    """
    Plot statistic returned from describe_qualitative_data
    :param df: DataFrame
    :return: None
    """
    plt.figure(figsize=(15, 8))
    plt.suptitle("Statistics from df.describe()")
    plt.subplot(1, 4, 1)
    sns.barplot(x=df["index"], y=df["sepal_length"], data=df)

    plt.subplot(1, 4, 2)
    sns.barplot(x=df["index"], y=df["sepal_width"], data=df)

    plt.subplot(1, 4, 3)
    sns.barplot(x=df["index"], y=df["petal_length"], data=df)

    plt.subplot(1, 4, 4)
    sns.barplot(x=df["index"], y=df["petal_width"], data=df)

    plt.show()


df_describe = get_quantitative_data(df_dummies)
plot_statistics(df_describe)


# d )
def plot_box(df):
    """
    Plot statistic returned from describe_qualitative_data
    :param df: DataFrame
    :return: None
    """
    plt.figure(figsize=(15, 8))
    plt.suptitle("Speices")
    plt.subplot(1, 4, 1)
    sns.boxplot(x=df["species"], y=df["sepal_length"], data=df)

    plt.subplot(1, 4, 2)
    sns.boxplot(x=df["species"], y=df["sepal_width"], data=df)

    plt.subplot(1, 4, 3)
    sns.boxplot(x=df["species"], y=df["petal_length"], data=df)

    plt.subplot(1, 4, 4)
    sns.boxplot(x=df["species"], y=df["petal_width"], data=df)

    plt.show()


plot_box(data)


# d )
def plot_heatmap(df):
    """
    The closer the value is to 1 between two features, the more positively linear relationships is between them.
    The closer the value is to -1 the more negatively linear relationships is between them.
    :param df: DataFrame
    :return: None
    """
    plt.figure(figsize=(15, 8))
    plt.title("Heatmap")
    sns.heatmap(df.corr(), annot=True)
    plt.show()


plot_heatmap(data)


# mapping data to get all species in one column
mapping = {"setosa": 0, "versicolor": 1, 'virginica': 2}
data = data.replace(to_replace={"species":mapping})


def find_outlier(df):
    """
    :param df: DataFrame
    :return: DataFrame with all rows with outliers
    """

    outlier_index = pd.DataFrame(columns=df.columns)

    for variable in df.columns:
        q1 = df[variable].quantile(0.25)
        q3 = df[variable].quantile(0.75)
        iqr = q3 - q1

        lower_threshold = q1 - 1.5 * iqr
        high_threshold = q3 + 1.5 * iqr

        outlier_index = pd.concat([outlier_index, df[(df[variable] < lower_threshold) |
                                                     (df[variable] > high_threshold)]], axis=0)

    return outlier_index


setosa = find_outlier(data[data["species"] == 0].drop("species", axis=1))
versicolor = find_outlier(data[data["species"] == 1].drop("species", axis=1))
virginica = find_outlier(data[data["species"] == 2].drop("species", axis=1))
outliers_to_remove = pd.concat([setosa, versicolor, virginica], axis=0)


def remove_outliers(df, df_outliers):
    """
    remove outliers from df
    :param df: DataFrame
    :param df_outliers: DataFrame
    :return: DataFrame, df without outliers
    """
    df_new = df.drop(df_outliers.index)
    return df_new


data_wo_outliers = remove_outliers(data, outliers_to_remove)


def data_preparation(df, y_name):
    """
    Data preparation
    :param df: DataFrame with data
    :param y_name: string, column name to set as y
    :return: DataFrame, Series
    """
    X, y = df.drop([y_name], axis=1), df[y_name]
    return X, y


X, y = data_preparation(data_wo_outliers, "species")


def train_split(x_matrix, y_series):
    """
    :param x_matrix: DataFrame
    :param y_series: Series
    :return: np.array, np,array, np.array, np.array
    """
    X_train, X_test, y_train, y_test = train_test_split(x_matrix, y_series, test_size=0.33, random_state=42)
    return X_train, X_test, y_train, y_test


X_train, X_test, y_train, y_test = train_split(X, y)


scaler = StandardScaler()


def x_parameters_scaling(x_training, x_testing):
    """
    :param x_training: np.array
    :param x_testing: np.array
    :return: np.array, np.array
    """
    scaler.fit(x_training)
    x_train_stand = scaler.transform(x_training)
    x_test_stand = scaler.transform(x_testing)
    return x_train_stand, x_test_stand


x_train_stand, x_test_stand = x_parameters_scaling(X_train, X_test)


def test_different_solvers():
    """
    Uses GridSearchCv to test different solvers
    :return: Print
    """
    # Reference: https://machinelearningmastery.com/hyperparameters-for-classification-machine-learning-algorithms/
    # Logistic regression does not really have any critical hyperparameters to tune.
    model = LogisticRegression()

    # Values to Try
    # Test different solvers (algorithm to use in the optimization problem).
    solvers = ["newton-cg", "lbfgs", "liblinear"]
    # l2 is the only penalty that works for all of these solvers (it is possible to try different kinds of penalties,
    # however it will result in warnings).
    penalty = ["l2"]
    # Inverse of regularization strength, smaller values specify stronger regularization.
    c_values = [100, 10, 1.0, 0.1, 0.01]

    # Grid Search
    grid = dict(solver=solvers, penalty=penalty, C=c_values)  # Creates a dictionary with the values to try
    # Exhaustive search over specified parameter values for an estimator.
    grid_search = GridSearchCV(estimator=model, param_grid=grid)
    grid_result = grid_search.fit(x_train_stand, y_train)  # Run the Grid Search Cross Validation on the train data.

    # Print Results
    # Best Score - Mean cross-validated score of the best_estimator
    print(f"Best Score: {grid_result.best_score_:.4f} using {grid_result.best_params_}")
    print()

    # All results
    means = grid_result.cv_results_["mean_test_score"]
    stds = grid_result.cv_results_["std_test_score"]
    params = grid_result.cv_results_["params"]
    for mean, stdev, param in zip(means, stds, params):
        print(f"Mean: {mean:.3f}, Std: {stdev:.3f}, using the params: {param}")     # Prints the mean values


test_different_solvers()


def logistic_regression(x_train_std, x_test_std, y_training, y_testing):
    """

    :param x_train_std: np.array
    :param x_test_std: np.array
    :param y_training: np.array
    :param y_testing: np.array
    :return: None
    """
    # Reference: https://machinelearningmastery.com/repeated-k-fold-cross-validation-with-python/
    cv = KFold(n_splits=10, random_state=1, shuffle=True)

    # create model
    model = LogisticRegression(penalty="l2", solver="newton-cg", C=10)
    model.fit(x_train_stand, y_train)
    y_predict = model.predict(x_test_std)

    # evaluate model
    scores = cross_val_score(model, x_train_std, y_training, scoring='accuracy', cv=cv, n_jobs=-1)
    print('\nAccuracy: %.3f (%.3f)' % (np.mean(scores), np.std(scores)))

    # report performance
    print("\nclassification_report\n", classification_report(y_testing, y_predict))

    # plot confusion matrix
    cm = confusion_matrix(y_testing, y_predict)
    ConfusionMatrixDisplay(cm).plot()
    plt.show()


logistic_regression(x_train_stand, x_test_stand, y_train, y_test)


"""
4.  k -folded cross-validation for evaluation (**)
"""


def remove_rows(df):
    """
    Used by check_statistic_manually to remove two rows randomly to fit shape
    :param df: DataFrame
    :return: DataFrame with two rows randomly removed, index reset
    """
    for i in range(0,2):
        n = random.randint(0, 136)
        df = df.reset_index(drop=True)
        df = df.drop(df.index[[n]], axis=0)
        i += 1
    df = df.reset_index(drop=True)
    return df


def check_statistic_manually():
    """
    :return: Print
    """

    statistic_df = []

    for i in range(0, 10):
        df = remove_rows(data_wo_outliers)
        X, y = data_preparation(df, "species")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
        X_train_val, X_test_val, y_train_val, y_test_val = train_test_split(X_train, y_train, test_size=0.33, random_state=42)
        x_train_stand, x_test_stand = x_parameters_scaling(X_train_val, X_test_val)

        # create model
        model = LogisticRegression(penalty="l2", solver="newton-cg", C=10)
        model.fit(x_train_stand, y_train_val)
        y_predict = model.predict(x_test_stand)

        # evaluate model
        acc = accuracy_score(y_test_val, y_predict)
        f1 = f1_score(y_test_val, y_predict, labels=None, average='macro')
        rec = recall_score(y_test_val, y_predict, average='macro')
        prec = precision_score(y_test_val, y_predict, average='macro')
        stat_data = [acc, f1, rec, prec]
        statistic_df.append(stat_data)
        i += 1
    statistic_df = pd.DataFrame(statistic_df, columns=["accuracy","F1_score", "Recall", "Precision"])

    acc_mean = np.mean(statistic_df["accuracy"])
    f1_mean = np.mean(statistic_df["F1_score"])
    rec_mean = np.mean(statistic_df["Recall"])
    prec_mean = np.mean(statistic_df["Precision"])

    acc_std = np.std(statistic_df["accuracy"])
    f1_std = np.std(statistic_df["F1_score"])
    rec_std = np.std(statistic_df["Recall"])
    prec_std = np.std(statistic_df["Precision"])

    print(f"acc_mean: {acc_mean}      acc_std:{acc_std}")
    print(f"f1_mean: {f1_mean}       f1_std: {f1_std}")
    print(f"rec_mean: {rec_mean}      red_std: {rec_std}")
    print(f"prec_mean: {prec_mean}     prec_std {prec_std}")

    return statistic_df


print("\n",check_statistic_manually())
