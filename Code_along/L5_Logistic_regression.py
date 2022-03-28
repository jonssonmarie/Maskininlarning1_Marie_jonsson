# Lecture notes - Logistic regression
"""
pd.get-dummies() byter ut kvalitativ data till numerisk data

"""
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score, ConfusionMatrixDisplay, classification_report

# EDA
df = pd.read_csv("../Data/Default.csv")

"""print("df.head\n", df.head())
print("df.info\n", df.info())
print("df.describe\n", df.describe())

print("value count default\n", df["default"].value_counts())
print("value count student\n", df["student"].value_counts())"""

sns.scatterplot(data=df, x="balance", y="income", hue="default")
# we see that the data is very closely packed, which will be hard to train a good model for finding default
# also the data is highly imbalanced

fig, axes = plt.subplots(1, 2, dpi=100, figsize=(10,4))

for ax, col in zip(axes, ["balance", "income"]):
    sns.boxplot(data=df, x="default", y=col, ax=ax)
    plt.show()

"""One-hot encoding
The features default and student are categorical. 
In order to make calculations on those variables, we need to represent them using dummy variables. 
This is called one-hot encoding.
"""

df = pd.get_dummies(df, columns=["default", "student"], drop_first=True)
print(df.head())


"""
Logistic function
The response variable that we want to predict is qualitative or categorical. 
We want to model the conditional probability of default given the balance
"""

logistic_function = lambda beta_0, beta_1, x: np.exp(beta_0 + beta_1*x)/(1+np.exp(beta_0+beta_1*x))
x = np.linspace(-10,10)

beta_0, beta_1 = 5, 1


plt.plot(x, logistic_function(beta_0, beta_1, x))
plt.title("Logistic function")
plt.show()

# uppgiften är att estimera beta_0 och beta_1 s.a. logistiska funktionen ger korrekta klasser till target variabeln
# values are constrained to 0 and 1, S-shaped curve

# Logistic regression
# Odds, se bild
X, y = df.drop("default_Yes", axis=1), df["default_Yes"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
print(X_train.shape, X_test.shape)

scaler = StandardScaler()
scaled_X_train = scaler.fit_transform(X_train)
scaled_X_test = scaler.transform(X_test)

# dubbelkollar här !
print(scaled_X_train.mean(), scaled_X_test.mean())  # bra ligger nära 0
print(scaled_X_train.std(), scaled_X_test.std())   # bra ligger nära 1

model = LogisticRegression(penalty="none")
model.fit(scaled_X_train, y_train)
print("LogisticRegression",model.coef_, model.intercept_)

# falskt test dataset
test_sample = pd.DataFrame({"balance": [1500, 1500], "income": [
                           40000, 40000], "student_Yes": [1, 0]})

print(test_sample) # only difference is that one is student and the other is not

scaled_test_sample = scaler.transform(test_sample)

# first column is -1 label i.e. not default, second column is label 1 i.e. default
print(model.predict_proba(scaled_test_sample), "\n")
# we see that being student increases the change of getting default


# gör en predict först!
y_pred_probability = model.predict_proba(scaled_X_test)
print(y_pred_probability[:5], "\n", y_test[:3], "\n")

"""
Evaluation metrics for classification
Confusion matrix
"""
# systematisk prediction här
y_pred = model.predict(scaled_X_test)
acc = accuracy_score(y_test, y_pred)
# note that the accuracy is very high, which is because of the accuracy paradox as the data is highly imbalanced.
# this means that we could make a model that always classifies negative on default, and achieve very high accuracy
print(f"Accuracy: {acc:.3f}")

cm = confusion_matrix(y_test, y_pred, labels=model.classes_)

disp = ConfusionMatrixDisplay(cm, display_labels=model.classes_)
ConfusionMatrixDisplay(cm).plot()
plt.show()

# metrics
# rec = TP / (TP + FN)
print(classification_report(y_test, y_pred))
# recall and f1 score is very bad here for label 1 indicating that the model having a lot of FN,
# which means it tends to classify someone that should default as not default
# this can also be seen in the confusion matrix

# manual calculation of the same as classification_report
rec = 33 / (33 + 77)
prec = 33 / (33 + 13)
print(f"Recall: {rec}")
print(f"Precision: {prec}")

"""
accuracy = (3178+33)/(3178+12+77+33)
recall = 33/(33+77)
precisiion = 33/(33+12)

f1 = 2*recall*precisiion/(recall+precisiion)
print(accuracy, recall, precisiion, f1)
"""


