from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

df = pd.read_csv("../Data/Heart.csv", index_col=0)
df.head()
df.info()

df.dropna(inplace=True)
# AHD - Arterial Heart Decease
sns.countplot(data=df, x="AHD")

df_dummies = pd.get_dummies(df, drop_first=True)
X, y = df_dummies.drop("AHD_Yes", axis=1), df_dummies["AHD_Yes"]


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Random forest classification, behöver inte scala datan här, det görs i Random Forest
model_rf = RandomForestClassifier(random_state=42)
model_rf.fit(X_train, y_train)

y_pred = model_rf.predict(X_test)
print("sannolihket tillhöra \nförsta klassen | andra klassen: \n", y_pred)

param_grid = {"n_estimators": [50, 100, 150, 200, 300],
              "criterion": ["gini", "entropy"],
              "max_features": ["auto","sqrt", "log2"]}
clf = GridSearchCV(RandomForestClassifier(), param_grid,
                   cv=5, verbose=1, scoring="recall")
# recall då vi hellre vill ha falska positiva då man kan undersöka dem vidare och upptäcka det.
clf.fit(X_train, y_train)
print(clf.best_params_)

"""
Evaluation
- default
- hyperparameters tuned
"""
y_pred_tuned = clf.predict(X_test)


def evaluate_classification(y_test, y_pred):
    print(classification_report(y_test, y_pred))
    cm = confusion_matrix(y_test, y_pred)
    ConfusionMatrixDisplay(cm, display_labels=["No", "Yes"]).plot()


evaluate_classification(y_test, y_pred)

# recall = TP / (TP+FN)

# model_rf.feature_importances_: ju högre upp i trädet jus större importens har den, detta är andelar av "viktighet"
# model_rf.feature_importances_.sum() är nära 1, det är bra, tänk statistik
print(model_rf.feature_importances_,"\n" ,model_rf.feature_importances_.sum())

feature_importance = pd.DataFrame([X.columns, model_rf.feature_importances_]).T
feature_importance.columns = ["Feature", "Importance"]
feature_importance.sort_values(by="Importance", ascending=False, inplace=True)
sns.barplot(data = feature_importance, x = "Importance", y = "Feature")

plt.show()


