"""
K Nearest Neighbours, KNN
minns pichu, pickachu med euklidisk distance, här mäter vi ett annat avstånd
"""
from sklearn.datasets import load_wine
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report, accuracy_score
# from sklearn.neighbors import NearestNeighbors  # kommer i labben
import tensorflow as tf

# Wine data EDA

wine = load_wine()
print(wine.keys())
print(wine.feature_names)
print(wine.target_names)

df = pd.DataFrame(wine.data, columns = wine.feature_names)
df = pd.concat([df, pd.DataFrame(wine.target, columns = ["wine_class"])], axis = 1)
"""
alternativt läs in hela så här:
df = load_iris(as_frame = True)
df = df.frame
"""
print(df.head())
print(df.info())
print(df["wine_class"].value_counts())
print(wine.target)

# a subset of features to plot, we check if it is possible to separate the classes
sns.pairplot(data=df[["alcohol", "malic_acid","total_phenols", "ash" ,"wine_class"]], hue="wine_class", corner=True, palette="tab10")
plt.show()

sns.heatmap(df.corr(), annot=False, vmin= -1, vmax=1)
plt.show()

print(df.corr())

"""
KNN Classification
KNN or k-nearest neighbours is a supervised machine learning algorithm that can be used for both regression or 
classification. It calculates the distance between a test data point and all training data, find k training 
points nearest to the test data. Then it does majority voting to classify that test point to majority of the 
class of the training data points that are closest. For regression instead it takes an average of those 
k points that are closest.

In KNN it is absolute necessity to do feature scaling as the distance calculated using a distance metric 
can be very wrong if the features are in different scales.
"""

X, y = df.drop("wine_class", axis = "columns"), df["wine_class"]
# want to have somewhat larger test set for evaluation metrics later
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)  # 0.5 i lectures

scaler = MinMaxScaler()
scaled_X_train = scaler.fit_transform(X_train)
scaled_X_test = scaler.transform(X_test)   # applicerar N(0,1) från X_train på X-test och då blir den inte exakt 0 och 1

model_KNN = KNeighborsClassifier(n_neighbors=1)  # default på p=2 är euklidic metric och det är det vi använder här
# lägg märke till utfallet vid test_size=0.5 och n_neighbors=1 och jmft med 0.33 och 5
model_KNN.fit(scaled_X_train, y_train)

y_pred = model_KNN.predict(scaled_X_test)
print(y_pred)


cm = confusion_matrix(y_test, y_pred)
ConfusionMatrixDisplay(cm).plot()
plt.show()

print(classification_report(y_test, y_pred))

"""
Elbow plot to choose k
Note that this is actually cheating as we use the testing data for hyperparameter tuning. 
This gives us data leakage. A correct way to do it here is to split the data set into train|validation|test sets, 
perform elbow plot on validation data. Choose a k and do the test on testing data. As our data set is so small, a 
better way would be to use cross-validation which I will show in the next lecture. Here I will do the cheating 
method to show the elbow plot
"""

error = 1 - accuracy_score(y_test, y_pred)

error_list = []

for k in range(1,50):
    model_KNN = KNeighborsClassifier(n_neighbors=k)
    model_KNN.fit(scaled_X_train, y_train)
    y_pred = model_KNN.predict(scaled_X_test)
    error_list.append(1 - accuracy_score(y_test, y_pred))

fig, ax = plt.figure(), plt.axes()
ax.plot(range(1, len(error_list) + 1), error_list, ".-")
ax.set(title="Elbow plot", xlabel="k neightbours", ylabel="Error")
# ett fundamentalt fel är att det är gjort på test_data
# lågt k större chans för overfit,  högt k större chans för underfit

# we choose k = 11 here  , något mellan 10 -20 pga hur det ser ut
model_KNN = KNeighborsClassifier(11)
model_KNN.fit(scaled_X_train, y_train)
y_pred = model_KNN.predict(scaled_X_test)


print(classification_report(y_test, y_pred))
cm = confusion_matrix(y_test, y_pred)
ConfusionMatrixDisplay(cm).plot()
plt.show()

print(classification_report(y_test, y_pred))
