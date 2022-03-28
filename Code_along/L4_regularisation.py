import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import RidgeCV # ridge regression with cross-validation
from sklearn.metrics import SCORERS
from sklearn.linear_model import LassoCV
from sklearn.linear_model import ElasticNetCV


# Data preparation
df = pd.read_csv("../Data/Advertising.csv", index_col=0)
X, y = df.drop("sales", axis=1), df["sales"]

# in exercise 2 Polynomial regression you've found the elbow in degree 4, as the error increases after that
# however to be safe and we assume that the model shouldn't have too many interactions between different features,
# I will choose 3
# please try with 4 and see how your evaluation score differs
model_polynomial = PolynomialFeatures(3, include_bias=False)
poly_features = model_polynomial.fit_transform(X)
print(poly_features.shape)

# important to not forget
X_train, X_test, y_train, y_test = train_test_split(poly_features, y, test_size=0.33, random_state=42)

# from 3 features we've featured engineered 34
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

"""
Feature standardization
Remove sample mean and divide by sample standard deviation
LASSO, Ridge and Elasticnet regression that we'll use later require that the data is scaled.
"""
scaler = StandardScaler()
scaled_X_train = scaler.fit_transform(X_train)  # Sparat medelvärdet och standardavvikelsen för X_train
scaled_X_test = scaler.transform(X_test)  # transform med hjälp av tränongsdatans (medelvärde, std avv)
# för vi vill inte ha medelvärde, std avv för test utan använder X_trains medelvärde, std avv sen på X_test.

print(f"Scaled X_train mean {scaled_X_train.mean():.2f}, std {scaled_X_train.std():.2f}")
print(f"Scaled X_test mean {scaled_X_test.mean():.2f}, std {scaled_X_test.std():.2f}")

"""
Regularization techniques
Bild
Problem with overfitting was discussed in previous lecture. 
When model is too complex, data noisy and dataset is too small the model picks up patterns in the noise. 
The output of a linear regression is the weighted sum: 
    y = theta_0 + theta_1 * x_1 + theta_2 * x_2 + .. + Theta_n * x_n
    , where the weights  represents the importance of the feature. 
 Want to constrain the weights associated with noise, through regularization. 
 We do this by adding a regularization term to the cost function used in training the model. 
 Note that the cost function for evaluation now will differ from the training.
 """

""" 
Ridge regression
bild
"""


def ridge_regression(X_train, X_test, y, penalty=0):
    # alpha = 0 should give linear regression
    # note that alhpa is same as lambda in theory, i.e. penalty term. sklearn has chosen alpha to generalize their API
    model_ridge = Ridge(alpha=penalty)
    model_ridge.fit(X_train, y)
    y_pred = model_ridge.predict(X_test)
    return y_pred


y_pred = ridge_regression(scaled_X_train, scaled_X_test, y_train, 0)
# penalty är cost funktionen och den är 0 då blir det OLS

MSE = mean_squared_error(y_test, y_pred)
MAE = mean_absolute_error(y_test, y_pred)
RMSE = np.sqrt(MSE)

print(MSE, MAE, RMSE)

# check with linear regression -> RMSE very similar!
model_linear = LinearRegression()
model_linear.fit(scaled_X_train, y_train)
y_pred = model_linear.predict(scaled_X_test)
MSE = mean_squared_error(y_test, y_pred)
MAE = mean_absolute_error(y_test, y_pred)
RMSE = np.sqrt(MSE)

print(MSE, MAE, RMSE)  # samma värden som ridge_regression iom penalty = 0 i ridge_regression


# Lasso regression
model_lasso = Lasso(alpha=.1)
model_lasso.fit(scaled_X_train, y_train)
y_pred = model_lasso.predict(scaled_X_test)

MSE = mean_squared_error(y_test, y_pred)
MAE = mean_absolute_error(y_test, y_pred)
RMSE = np.sqrt(MSE)

print(MSE, MAE, RMSE)

# dags att kaliberera alpha och lambda hyperparametrarna
"""k-fold cross-validation
One strategy to choose the best hyperparameter alpha is to take the training part of the data and
shuffle dataset randomly
split into k groups
for each group -> take one test, the rest training -> fit the model -> predict on test -> get evaluation metric
take the mean of the evaluation metrics
choose the parameters and train on the entire training dataset
Repeat this process for each alpha, to see which yielded lowest RMSE. 

k-fold cross-validation:
- good for smaller datasets
- fair evaluation, as a mean of the evaluation metric for all k groups is calculated
- expensive to compute as it requires k+1 times of training
"""

# Ridge regression
# print(SCORERS.keys())
# negative because sklearn uses convention of higher return values are better
# alpha same as lambada in theory - penalty term
model_ridgeCV = RidgeCV(alphas=[.0001, .001, .01, .1, .5, 1, 5, .7, .9, 1, 5, 10], scoring="neg_mean_squared_error")
model_ridgeCV.fit(scaled_X_train, y_train)
print("model_ridgeCV.alpha_:", model_ridgeCV.alpha_)  # best alpha is 0.1 av de som vi stoppat in.
# Finjustering kan göras, text .12, .15
# it seems that linear regression outperformed ridge regression in this case
# however it could depend on the distribution of the train|test data, so using alpha = 0.1 is more robust here
y_pred = model_ridgeCV.predict(scaled_X_test)

MSE = mean_squared_error(y_test, y_pred)
MAE = mean_absolute_error(y_test, y_pred)
RMSE = np.sqrt(MSE)

print(MSE, MAE, RMSE)
print("model_ridgeCV.coef_", model_ridgeCV.coef_)

# Lasso regression
# it is trying 100 different alphas along regularization path epsilon
model_lassoCV = LassoCV(eps=0.001, n_alphas=100, max_iter=1e4, cv=5)  # cv = k alltså lutningen
model_lassoCV.fit(scaled_X_train, y_train)
print(f"Chosen alpha (penalty term) = {model_lassoCV.alpha_}")

y_pred = model_lassoCV.predict(scaled_X_test)

MSE = mean_squared_error(y_test, y_pred)
MAE = mean_absolute_error(y_test, y_pred)
RMSE = np.sqrt(MSE)

print(MSE, MAE, RMSE)


# we notice that many coefficients have been set to 0 using Lasso
# it has selected some features for us
print("model_lassoCV.coef_", model_lassoCV.coef_)  # många features den sätter till 0 = slänger dem
# går detta att kombinera med Rigde Regression? => Elastic net

"""
Elastic net
Elastic net is a combination of both Ridge l2-regularization and Lasso l1-regularization. 
The cost function to be minimized for elastic net is:
bild
"""
# note that alpha here is lambda in the theory
# l1_ratio is alpha in the theory
model_elastic = ElasticNetCV(l1_ratio=[.1, .5, .7, .9, .99, 1], eps=0.001, n_alphas=100, max_iter=10000)
# max_iter= 1000 är för lite
model_elastic.fit(scaled_X_train, y_train)

print(f"L1 ratio: {model_elastic.l1_ratio_}")   # this would remove ridge and pick Lasso regression entirely
print(f"alpha {model_elastic.alpha_}")

y_pred = model_elastic.predict(scaled_X_test)

MSE = mean_squared_error(y_test, y_pred)
MAE = mean_absolute_error(y_test, y_pred)
RMSE = np.sqrt(MSE)

print(MSE, MAE, RMSE)
# Elastic Net är mer robust
# note that the result is same for Lasso regression which is expected
