import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Polynomial regress, underfitting, overfitting

""" 
Bild Simulate_data
"""

# Simulate data
# Simulate from a 2nd order polynomial with Gaussian noise i.e.

samples = 100
# kolonnmatris/radvektor
X = np.random.uniform(-3, 3, (samples, 1))
f = lambda x: x**2 + 0.5 * x + 3

y = f(X) + np.random.randn(samples, 1)

plt.plot(X, y, '.')

x = np.linspace(-3, 3, samples)
plt.plot(x, f(x), '.')
plt.show()

"""
Polynomial regression
Polynomial regression fits a polynomial of order  to model the relationship between independent variable 
and dependent variable . 
The polynomial regression model in general:
            bild

It is linear in terms of the unknown parameters, and can be expressed in matrix form and solved using 
OLS normal equation as we did for multiple linear regression. In fact polynomial regression is a special case of 
multiple linear regression.
"""

# PolynomialFeatures creates feature matrix to represent the polynomial combinations
polynomial_features = PolynomialFeatures(degree=2, include_bias=False)
# include_bias = True skapar en kolonn med ettor det vil, vi inte har för vi använder LinearRegression
# från sklearn somm lägger in en kolonn med ettor skapar med degree = 2  = > x^2
# include_bias är intercept i sklearn LinearRegression
poly_X = polynomial_features.fit_transform(X)  # bara fit_transform på X
print(poly_X.shape)
print(poly_X[:3])
# The fit_transform method is calculating the mean and variance of each of the features present in our data.
# The transform() method is transforming all the features using the respective mean and variance.

model = LinearRegression()
model.fit(poly_X, y)  # tränar genom att köra .fit
print(model.coef_)
print(model.intercept_)

x = np.linspace(-3, 3, samples).reshape(samples, 1)
x_poly_features = polynomial_features.transform(x)   # transformerar från 1 dimensionell till 2 dimensionell
y_pred = model.predict(x_poly_features)
print(x.shape)  # bra ger 100 st

print(y_pred.shape, x.shape)

plt.plot(x, y_pred, label='Model')

plt.show()

"""
Underfitting
Underfitting is when a model is too simple to represent the data accurately.
"""
model = LinearRegression()
model.fit(X,y)
plt.plot(x, model.predict(x.reshape(-1,1)), label= "Underfitting, model too simple")
plt.show()


"""
Overfitting
Model too complicated, and fitted too much to the data. 
Complicated model (high variance) risk to fit to noise in training data, which make them generalize worse. 
Overfitting usually occurs when there is too small traning set, and/or it is not representative for testing data.
"""

poly_model_30 = PolynomialFeatures(30, include_bias=False)
X_features = poly_model_30.fit_transform(X)
print(X_features.shape)

model = LinearRegression()
model.fit(X_features, y)

x = np.linspace(-3,3, samples)
x_poly_features = poly_model_30.transform(x.reshape(-1,1))
pred = model.predict(x_poly_features)

plt.plot(x, pred)
#ax.set(title="Overfitting, model too complex", ylim=[1,15]);

# The model captures more points in training data but can't generalize to test data

"""Choose a model
Note that we are not always able to plot the data and its corresponding graphs as our data might be high dimensional. 
In order to choose correct model we can use a cost/loss function to keep track of the error for 
different models (different degrees of polynomial).
"""


# we increase number of simulated samples
samples = 10000
X = np.random.uniform(-3,3, (samples,1))
f = lambda x: 2*x**3 + x**2 + .5*x + 3 # change model to 3rd degree polynomial
y = f(X)+np.random.randn(samples,1)

print(f"X.shape: {X.shape}, y.shape: {y.shape}")

"""
Train|Validation|Test split
We split our data set into

training set
validation set
testing set
Reason for this split is to use the validation data for choosing the degree of the polynomial (a hyperparameter)
"""

# use train_test_split twice to obtain train|val|test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.5, random_state=42)

print(X_train.shape, y_train.shape, X_val.shape, y_val.shape, X_test.shape, y_test.shape)

"""Fit model and predict"""

from sklearn.metrics import mean_squared_error

RMSE_val = []

for degree in range(1, 100):
    model_poly = PolynomialFeatures(degree,
                                    include_bias=False)  # bias False as LinearRegression has intercept by default
    train_features = model_poly.fit_transform(X_train)
    val_features = model_poly.transform(X_val)
    model_lin_reg = LinearRegression()

    model_lin_reg.fit(train_features, y_train)

    y_pred_val = model_lin_reg.predict(val_features)

    RMSE_val.append(np.sqrt(mean_squared_error(y_val, y_pred_val)))


plt.plot(range(1,10), RMSE_val[:9],'--o', label = "Validation")
plt.show()
#ax.set(xlabel = "Degree", ylabel = "RMSE", title = "RMSE on validation data for different degrees of polynomial")
# we see that from degreen 3 the error is low, and it doesn't change much when going higher degrees
# hence we would choose degree 3 here, which corresponds to our simulated polynomial degree.
# this type of plot is called elbow plot.
# now we could move on to train the model using degree 3 and then predict on testing data

# try much higher degree polynomials and see that error actually increases
# note that this might happen much faster for real data
plt.plot(range(50), RMSE_val[:50],'--.', label = "Validation")
plt.show()


"""
Bias-Variance Trade-off
A models generalization error is bias + variance + irreducible error

- bias - difference between average prediction and correct value.
    - high bias, pay little attention to data, 
    - oversimplifies and underfits.
- variance - spread of our data
    - many degrees of freedom -> high variance -> overfit to data
- irreducible error - due to noise of data, only way to decrease this is to clean the data itself.
    
Higher model complexity -> higher variance, lower bias

Lower model complexity -> lower variance, higher bias

The goal is to choose a model that is complex enough not to underfit, but not too complex to overfit. 
Need to find a balance between bias and variance.
"""

