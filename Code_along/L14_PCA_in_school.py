from sklearn.datasets import load_breast_cancer
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

raw_data = load_breast_cancer()
X = raw_data.data
y = raw_data.target

df = pd.DataFrame(X, columns=raw_data.feature_names)
df.head()

# PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
print("X_train shape", X_train.shape)

scaler = StandardScaler()
scaled_X_train = scaler.fit_transform(X_train)

pca = PCA(n_components = X.shape[1])
pca_transformations = pca.fit_transform(scaled_X_train)
print("PCA transformations", pca_transformations.shape)

proportion_variance_explained = pca.explained_variance_ratio_
pve_cum_sum = np.cumsum(proportion_variance_explained)

fig1, ax = plt.figure(), plt.axes()
ax.plot(range(1, len(pve_cum_sum)+1), pve_cum_sum, 'o--')
ax.set(title = "Proportion variance explained (PVE) elbow plot", ylabel = "PVE", xlabel = "Number of PC")

# Visualisation
fig2, ax = plt.figure(), plt.axes()
ax.scatter(pca_transformations[:,0], pca_transformations[:,1], c = y_train)
ax.set(title = "First 2 PC that data has transformed into", xlabel = "1st PC", ylabel = "2nd PC")

print(proportion_variance_explained[0], proportion_variance_explained[1])

# PCA in a pipeline

from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

def evaluate_model(model, title):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print(classification_report(y_test, y_pred))
    cm = confusion_matrix(y_test, y_pred)
    ConfusionMatrixDisplay(cm).plot()
    plt.title(title)

from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

pipe_with_pca = Pipeline([
    ("scaling", StandardScaler()),
    ("dimension_reduction", PCA(10)),
    ("svm", LinearSVC(max_iter=10000))
])

pipe_no_pca = Pipeline([
    ("scaling", StandardScaler()),
    ("svm", LinearSVC(max_iter=10000))
])

evaluate_model(pipe_no_pca, "Without PCA")
evaluate_model(pipe_with_pca, "With PCA")
