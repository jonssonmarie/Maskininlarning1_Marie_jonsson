"""
GridSearchCV and Pipeline
"""

from sklearn.datasets import load_wine
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, ConfusionMatrixDisplay, confusion_matrix


wine = load_wine()
print(wine.keys())


df = pd.DataFrame(wine.data, columns=wine.feature_names)
df = pd.concat([df, pd.DataFrame(wine.target, columns=["wine_class"])], axis = 1)
print(df.head())
print(wine.target)

# Train|test split
X, y = df, wine.target


# use the same test set as in previous lecture
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)
# för att förstöra lite, mindre train data
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

"""
Pipeline & GridSearchCV
Pipeline - sequentially apply list of transforms and a final estimator. All intermediate steps have to implement 
fit and transform while last step only needs to implement fit.

GridSearchCV - exhaustive search for specified parameter values for an estimator. 
In this case we use the pipeline as estimator.
It does cross-validation to find the specified parameter values.

Note that there are other ways to search for parameter values.
"""

scaler = StandardScaler()

# pipeline with StandardScaler and KNN
pipe_KNN = Pipeline([("scaler", StandardScaler()), ("knn", KNeighborsClassifier())])

# pipeline with StandardScaler and LogisticRegression
pipe_log_reg = Pipeline([("scaler", StandardScaler()),
            ("logistic", LogisticRegression(solver="saga", multi_class="ovr", max_iter=10000, penalty="elasticnet"))])


param_grid_KNN = {"knn__n_neighbors": list(range(1, 50))}
# utnyttjar transformatorn och ropar på den samt parameternamn med dubbla understreck. Så ändrar vi på parametrarna
param_grid_logistic = {"logistic__l1_ratio": np.linspace(0, 1, 20)}    # l1_ratio = np.linspace(0, 1, 20)

# kolla klassbalans, för att kolla om recall eller accuracy för scoring. Här verkar det vara ganska jämnt
sns.countplot(y)

classifier_KNN = GridSearchCV(estimator=pipe_KNN, param_grid=param_grid_KNN, cv=5, scoring="accuracy", verbose=1)
classifier_logistic = GridSearchCV(estimator=pipe_log_reg, param_grid=param_grid_logistic, cv=5, scoring="accuracy", verbose=1)
# verbose skriver ut hur långt man kommit i arbetet

# it will scale the data to X_train using StandardScaler
classifier_KNN.fit(X_train, y_train)
classifier_logistic.fit(X_train, y_train)
print(classifier_KNN)
print(classifier_logistic)

print(classifier_KNN.best_estimator_.get_params())   # kollar vilka parametrar man får, baserat på vad vi lagt in
print(classifier_logistic.best_estimator_.get_params())  # kolla in l1_ratio, det är den som är den bästa enligt classifier_logistic

# prediction på X_test - KNN
y_pred = classifier_KNN.predict(X_test)
print(classification_report(y_test, y_pred))

# confusion matrix  KNN
cm = confusion_matrix(y_test, y_pred)
ConfusionMatrixDisplay(cm).plot()

# prediction på X_test - Logstic
y_pred = classifier_logistic.predict(X_test)
print(classification_report(y_test, y_pred))

# confusion matrix - Logistic
cm = confusion_matrix(y_test, y_pred)
ConfusionMatrixDisplay(cm).plot()

""" 
'Deploy' model  (simulerar production därav 'Deploy') 
- Train on all data using the parameter that we found through GridSearchCV
- save the model
- load the model and do infernce on new data
"""

"""
Labb 
Var inte ledsna om recommender blir dålig, den i verkligheten är komplex och oftast deeplearning. Vi ska testa på detta.
Kolla youtube video och följ efter för att skapa enkel kod för labben
Använd den stora ml-latest.zip (size:265 MB)  Här krävs att man bearbetar datan och analyserar
För frågor i markdown vid sidan av om pythonscript
"""