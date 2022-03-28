# Scikit -learn

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

df = pd.read_csv("../Data/Advertising.csv", index_col =0)

print(f" Number of features {df.shape[1] -1}") # subtract one as price_unit_area is the label and not
print(f"{df.shape[0]} samples")

X, y = df.drop("sales", axis=1), df["sales"]

print(X.shape, y.shape)

""" 
Scikit-learn recipe
(Förenklat)
Steps
1. train| test split or train|validation|test split
2. Scale dataset
- many algorithms require scaling, some don't  
   OLS kräver inte skalning, gradient descent kräver skalning. 
   Skalning betyder text normalisering av datan, en och samma skala, alltså en och samma enhet
- which type och scaling method to use?
- scale training data using training data, sale test data using training data (undvika data leakeage)
3. Fit algorithm to training data
4. predict on test data 
5. Evaluation metrics on test data
"""

# Train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

"""Feature scaling
Two popular scaling techniques are normalization and feature standardization

Normalization (min-max feature scaling)
bild Feature_scaling
Compute min and max from training data and use on training and test data

Feature standardization (standard score scaling)  score = z värdet i Normalfördelningen N(0,1)
bild Feature_scaling
my och sigma computed from training data

OBS sample x-bar (inte my) och s inte sigma ska det vara i formlerna
"""

# we use normalization here
# instantiate an object from the class MinMaxScaler
scaler = MinMaxScaler()
print("type(scalar)",type(scaler))

# do scaler.fit on X_train - NOT on X-test
# fit - compute the minimum and maximum to be used for later scaling.
scaler.fit(X_train)     # use the training data to fit the scaler
print(scaler.data_max_, scaler.data_min_)

# transform both X_train and X_test
scaled_X_train = scaler.transform(X_train)
scaled_X_test = scaler.transform(X_test)

print(f"Min value in X_train {scaled_X_train.min():.2f} ≤ scaled_X_train ≤ {scaled_X_train.max():.2f}")
print(f"{scaled_X_test.min():.2f} ≤ scaled_X_test ≤ {scaled_X_test.max():.2f}") # natural that it isn't [0,1] since we fit to training data

# training data scaled to 0-1 (std normalfördelning)
print(f"Min value in X_train {scaled_X_train.min():.2f}")
print(f"Min value in X_train {scaled_X_train.max():.2f}")
print(f"Min value in X_testn {scaled_X_test.min():.2f}")    # ok värdena skiljt från 0 och 1
print(f"Min value in X_test {scaled_X_test.max():.2f}")     # ok värdena skiljt från 0 och 1 då vi anvönde X-train max och X-test max är inte samma
# För man räknar ut skalningsfaktorn med x_train. Är då något värde större i x_test än största värdet i x_train blir det större än 1.
# .fit är gjord på X_train inte X-test
# Algorithm - linear regression

# SVD - Singular Value Decomposition that is used for calculating pseudoinverse in OLS normal equation
# instantiate and object

model_SVD = LinearRegression()
model_SVD.fit(scaled_X_train, y_train)

# weights
print(f"Weights (beta_hats) : {model_SVD.coef_}")
print(f"Intercept: {model_SVD.intercept_}")


# Stochastic gradient descent (SGD)
# note that SGD requires features to be scaled
model_SGD = SGDRegressor(loss = "squared_error", learning_rate="invscaling", max_iter=100000)
model_SGD.fit(scaled_X_train, y_train)
print(f"Weights (beta_hats) {model_SGD.coef_}")
print(f"Intercept {model_SGD.intercept_}")  # utan scalning , gradient blir extremt stort eller liten


# Manual test
# sanity check
# We test predict one sample to manually do a reasonability check
test_sample_features = scaled_X_test[0].reshape(1,-1)
test_sample_target = y_test.values[0]
# notera inte skalat targets bara features
print(X_test.iloc[0])  # visar oskalade data

# uses the weights and intercept from the fitting
# ett case, fler kan testas
print(model_SGD.predict(test_sample_features)[0], model_SVD.predict(test_sample_features)[0], test_sample_target)

# Evaluation
# first predict on our test data
y_pred_SVD = model_SVD.predict(scaled_X_test)
y_pred_SGD = model_SGD.predict(scaled_X_test)

mae_SVD = mean_absolute_error(y_test, y_pred_SVD)
mae_SGD = mean_absolute_error(y_test, y_pred_SGD)

mse_SVD = mean_squared_error(y_test, y_pred_SVD)
mse_SGD = mean_squared_error(y_test, y_pred_SGD)

rmse_SVD = np.sqrt(mse_SVD)
rmse_SGD = np.sqrt(mse_SGD)

print(f"SVD: MAE {mae_SVD:.2f}, MSE {mse_SVD:.2f}, RMSE {rmse_SVD:.2f}")
print(f"SGD: MAE {mae_SGD:.2f}, MSE {mse_SGD:.2f}, RMSE {rmse_SGD:.2f}")

"""
SVD: 1.51, MSE: 3.80, RMSE: 1.95
SGD: 58170544253.09, MSE: 5043167221419103748096.00, RMSE: 71015260482.65  - skalning har missats någonstans
"""