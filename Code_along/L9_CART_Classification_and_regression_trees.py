"""
CART - Classification and regression trees
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay


df_hit = pd.read_csv("../Data/Hitters.csv", delimiter=',')
print(df_hit.head(3))
print(df_hit.info())

df_hit.dropna(inplace=True)

sns.scatterplot(data=df_hit, x="Years", y="Hits", hue="Salary")
plt.show()
"""
Decision tree regression
The goal is to stratify or segment the players into several regions. In decision tree for regression, 
the algorithm creates a tree to minimize the RSS (residual sum of squares). 
The tree-building process uses recursive binary splitting, a top-down greedy approach to divide the 
predictor space into branches. For example the baseball dataset with years and hits we could have a 
split into the following regions: see picture
"""
model_tree_reg = DecisionTreeRegressor(max_depth=2)

X, y = df_hit[["Years", "Hits"]], df_hit["Salary"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(X_train.shape, X_test.shape)

sns.scatterplot(data=X_train, x="Years", y="Hits", color="darkgreen")

y_test.hist()

model_tree_reg.fit(X_train, y_train)
print(model_tree_reg.feature_importances_, model_tree_reg.feature_names_in_)

fig, ax = plt.figure(figsize=(16,8), dpi=100), plt.axes()
tree.plot_tree(model_tree_reg, filled=True, ax=ax, feature_names=list(X.columns), impurity=False, rounded=True)

# TODO: make an exercise for students to extract decision thresholds and draw decision boundary
print(tree.export_text(model_tree_reg, feature_names = list(X_train.columns)))

y_pred = model_tree_reg.predict(X_test)
print(mean_absolute_error(y_test, y_pred), np.sqrt(mean_squared_error(y_test, y_pred)))

"""
Decision tree classification
As in the regression case, the tree is grown through recursive binary splitting that minimizes a loss function locally. 
However RSS can't be used as loss function in classification. Instead Gini impurity or cross-entropy can be used.

Gini impurity measures a nodes purity, with a small value showing that most of the observations come from one class:
see picture
"""

df_default = pd.read_csv("../data/Default.csv")
# gör en one-hot encoding - Behövs dock inte då Regression trees klarar y/n
df_default_encoded = pd.get_dummies(df_default, drop_first=True)
print(df_default_encoded.head(3))

X, y = df_default_encoded.drop("default_Yes", axis=1), df_default_encoded["default_Yes"]  # student_Yes
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(X_train.shape, X_test.shape)

clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

print(classification_report(y_test, y_pred))
cm = confusion_matrix(y_test, y_pred)
ConfusionMatrixDisplay(cm).plot()
plt.show()
# remember how logistic regression scored on this dataset

# slutsats. läs statlearning slide 8:
# Unfortunately, trees generally do not have the same level of  predictive accuracy as some of the other regression and
# classification approaches seen in this book.
