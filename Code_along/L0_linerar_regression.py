import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
# import ipykernel bara för jupyter
import black

# Linear regression code along
# Dataset from ISLR - Introduction to Statistical learning with R

df = pd.read_csv("Data/Advertising.csv", index_col=0)

print(df.head())

print(df.describe().T )

# viktigt att kålla koll på dimensionerna

fig, ax = plt.subplots(1, 3, figsize=(12, 3), dpi=100)

for i, feature in enumerate(df.columns[:-1]):
    sns.scatterplot(data = df, x = feature, y = "sales", ax = ax[i])  # sales här, Kokchun har Sales i sitt dataset
    ax[i].set(xlabel = "Spending", title = f"{feature} spendings")
    #plt.show()

sns.pairplot(df, corner = True, height = 2)


# Simple linear regression

# t avrundat= beta_0 + beta_1*x

X , y = df["TV"], df["sales"]
beta_1, beta_0 = np.polyfit(X, y, deg=1)
print(f"Intercept {beta_0:.3f}")
print(f"Slope {beta_1:.3f}")


def y_hat(x):
    return beta_0 + beta_1 * x

spend = np.linspace(0,300)

sns.scatterplot(data=df, x = "TV", y = "sales")
sns.lineplot(x = spend, y = y_hat(spend), color="red")

sns.regplot(x= X , y= y)

# Multiple linear regression

X, y = df.drop("sales", axis= "columns"), df["sales"]
X.insert (0, "Intercept", 1)
print(X.head(), y.head())


# OLS normal equation/cloesed from equatioin

regression_fit = lambda X, y : np.linalg.inv(X.T @ X) @ X.T @ y 

beta_hat = regression_fit(X,y)
print("beta_hat: \n",beta_hat)


predict = lambda x, beta: np.dot(x, beta)  # viktigt att kålla koll på dimensionerna
# dot product in linear algebra !

# don't do this in reality, here we test for sanity
test_sample = [1, 230.1, 37.8, 69.2]

y_hat = predict(test_sample, beta_hat)

print(f"y_hat:  {y_hat:.2f}")
print(f"True vallue for this sample: {y.iloc[0]}")


# Train | test split
# split 70% training, 30% test
train_fraction = int(len(df) * 0.7)
print(f"{train_fraction} samples for training data")
print(f"{len(df)-train_fraction} samples for test data")

train = df.sample(n=train_fraction, random_state=42, replace=False)
test = df.drop(train.index)
print(f"{train.index.isin(test.index).sum()} data from test in training")

x_train, y_train = train.drop("sales", axis=1), train["sales"]
#print(x_train)
#print(y_train)

X_train, y_train =  train.drop("sales", axis=1), train["sales"]
X_test, y_test = test.drop("sales", axis=1), test["sales"]

X_train.insert(0, "Intercept", 1)
X_test.insert(0, "Intercept", 1)

# Prediction
#  this useds OLS normal equation

beta_hat = regression_fit(X_train, y_train)
print(beta_hat)
predict = lambda X, weights: X @ weights
y_hat = predict(X_test.to_numpy(), beta_hat.to_numpy().reshape(4,1))
print(y_hat[:5], y_test[:5])

print(y_hat.shape)
print(X_test.shape, beta_hat.shape)

beta_hat.to_numpy().reshape(4,1) # .shape


# Evaluation
# MAE - Mean Absolut errror
# MSE - Mean squared error (sqyúare of original unit, puished outliers more than MAE, somewhat hard to interpret)
# RMSE - Root mean square error (original unit, punishes oultiners more than MAE, easy to interpret)

m = len(y_test)
y_hat = np.reshape(y_hat, m)

MAE = 1/m * np.sum(np.abs(y_test-y_hat))
MSE = 1/m * np.sum((y_test-y_hat)**2)
RMSE = np.sqrt(MSE)

print(MAE, MSE, RMSE)

#plt.show()