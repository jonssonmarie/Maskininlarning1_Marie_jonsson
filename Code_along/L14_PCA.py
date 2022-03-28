# Principal Component Analysis (PCA)



from sklearn.datasets import load_breast_cancer
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np



raw_data = load_breast_cancer()
X = raw_data.data
y = raw_data.target

df = pd.DataFrame(X, columns = raw_data.feature_names)
print(df)

"""
Curse of dimensionality
Two points picked randomly in a unit square has lower distance than two points picked randomly in a unit cube as there 
is more space in higher dimensions. This reasoning applies to higher dimensional hypercubes. This means that a new 
data point has higher probability of being far away from training dataset in higher dimensional dataset, i.e. the 
model has higher chance of overfitting.

one approach is to get more data points for the training set (not always possible)
another approach is to reduce the number of dimensions
"""
"""
Principal component analysis (PCA)
PCA is an unsupervised learning technique for reducing dimensionality of the data to a lower dimensional 
representation of that dataset. For example a dataset with computer specifications, the hardware specifications 
are more important than the color of the computer when it comes to predicting its price.

Mathematically we do this by finding a linear combination of feature variables with maximal variance and mutually 
uncorrelated. For feature variables  we get 1st principal component  as:

, where the loading vector or principal component vector 
 is normalized. The objective is hence to find the principal component vectors. These vectors are eigenvectors 
 of the covariance matrix of the feature matrix. The variance explained is represented by the eigenvalues. 
 We want to find principal components that explain most of the variance of the original dataset in order not 
 to lose too much information when reducing dimensions.

when we have computed the eigenvectors, we sort them by eigenvalues and pick a number  dimensions to project our 
data onto.
the number  can be obtained through a knee plot of the proportional variance explained
by reducing number of dimensions we also decreases the amount of data that the model needs to process, and can 
speed up computations very important is that the data is scaled with feature standardization i.e. 0 mean and unit variance
"""

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=42)

print(f"X_train shape: {X_train.shape}")
# very important to feature standardize for PCA to work properly, it has to have 0 mean and unit variance
scaler = StandardScaler()
scaled_X_train = scaler.fit_transform(X_train)

pca = PCA(n_components=X.shape[1])
pca_transformations = pca.fit_transform(scaled_X_train)
print(f"PCA transformed shape {pca_transformations.shape}")

proportion_variance_explained = np.cumsum(pca.explained_variance_ratio_)


fig, ax = plt.figure(), plt.axes()

ax.plot(range(1, len(proportion_variance_explained)+1),
        proportion_variance_explained, 'o--')

ax.set(title="Proportion variance explained (PVE) by principal components",
       xlabel="Number of principal components", ylabel="PVE");

# here we can pick around 6-10 PCs

"""
Visualize first 2 PCs
visualize the first 2 principal components
with the help of PCA we can visualize high dimensional data
"""
fig, ax = plt.figure(), plt.axes()
ax.scatter(pca_transformations[:,0], pca_transformations[:,1], c = y_train, cmap = "viridis")
ax.set(title = "Data transformed into first 2 princiap components", xlabel = "PC1", ylabel = "PC2")

# explained variance ratio by the first two principal components
# note that although it explains only about 0.63 of all the variance we can see that it is somewhat linearly separable
print(pca.explained_variance_ratio_[0], pca.explained_variance_ratio_[1])

"""
PCA in pipeline
PCA can be used in pipeline where data is first scaled followed by PCA and afterwards a classifier or regressor for regression problem
some features are correlated and explains very little of the variance, and hence can be represented by lower dimensional principal components
"""

# helper function
def evaluate_model(model, title = ""):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print(classification_report(y_test, y_pred))
    cm = confusion_matrix(y_test, y_pred)
    ConfusionMatrixDisplay(cm, display_labels=["No", "Yes"]).plot()
    plt.title(title)

from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report

# we use SVM with linear kernel as we saw that it is linearly separable with the first two PCs
pipe_with_pca = Pipeline([("scaling", StandardScaler()),
                 ("reduce_dimensions", PCA(10)),
                 ("svm", LinearSVC(max_iter = 10000))])

pipe_no_pca = Pipeline([("scaling", StandardScaler()),
                 ("svm", LinearSVC(max_iter = 10000))])

evaluate_model(pipe_no_pca, "Without PCA")
evaluate_model(pipe_with_pca, "With PCA")



raw_data = load_breast_cancer()
X = raw_data.data
y = raw_data.target


# PCA
"""
endast fit_transform på X_train för att kolla antal PC
"""

# proportion_varaince_explained = sum(pca.explanined_variance_ratio_= ska bli 1
# pve_sum_cum = np.cumsum(proportion_varaince_explained)

# plot Visualization
# det syns att det är linjärt separerbart

# linjär model - support Vector machine


#evaluate_model
# model-predict på X_test utan skalning då den ska in i pipeline där det görs
# model är i detta fal pipe_namn


