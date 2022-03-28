"""
Support Vector Machine (SVM)
"""
from utils import plot_svm_margins
from sklearn.svm import SVC
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans

# we simulate data points
blobs = make_blobs([300, 300], 2, centers=[(-2,-2),(2,2)],random_state=42)[0]  # [0]  tar bort tuple och ger x, y
# make_blobs ger klasserna också, men här använder vi inte dessa för att öva på klasser
plt.scatter(blobs[:,0], blobs[:,1])
plt.title("Blobs")
plt.show()

# a problem is that we don't have any labels, and SVM is a supervised learning algorithm

""" 
Clustering
 - K-means algorithm
 - unsupervised learning algorithm

Generate labels
we can generate labels by using an unsupervised learning algorithm
KMeans with two clusters
"""
kmean = KMeans(2)
kmean.fit(blobs)
blobs = np.c_[blobs, kmean.predict(blobs)]  # np.c_ är concat i numpy, här får vi x och y
df = pd.DataFrame(blobs, columns=["X1", "X2", "label"])    # till dp för plot av någon anledning, tror scatterplot även tar np
sns.scatterplot(data=df, x="X1", y="X2", hue="label")
plt.title("X, y")

X, y = df.drop("label", axis=1).to_numpy(), df["label"].to_numpy()   # visar hur man gör tillbaka till np, som finns ovanför

"""
Support Vector Machines (SVM)
We start with the history from maximum margin classifier to support vector classifier to support vector machine.
"""

"""
Maximum Margin Classifier (MMC)
If the data is perfectly linearly separable with a hyperplane, then there exists infinite amount of hyperplanes that 
satisfies this as there are infinite amount of small tilts, rotations to a hyperplane. 
However the optimal hyperplane is the hyperplane in which it is farthest from the training observations is 
called maximumm margin classifier. In maximum margin classifier:

- the closest points from each class to the hyperplane are called support vectors, the margin is constructed with 
- the help of these support vectors
- there are no misclassifications in training data, the data are in the correct side of the line
- the variance is high and bias low, as it doesn't allow any misclassifications in training data. 
    This can cause overfitting.
- sensitive to individual observation, as one training point could be a different support vector, 
    which causes hyperplane to be very different
- the magnitude of the distance of observation to hyperplane is a measure of confidence that the observation is 
    correctly classified
    
Mathematically MMC can be constructed as: see picture
"""
"""
Soft Margin Classifier or Support Vector Classifier (SVC)
allows for some misclassifications, which gives greater robustness to individual observations
better classification of most training observations
Mathematically
    see picture
"""
"""
Support vector machines (SVM)
There are many cases where the data is not linearly separable with a hyperplane, and thus SVC won't work. 
A solution is to enlarge the feature space as in polynomial linear regression. 
In SVM we enlarge the feature space using kernels to accomodate non-linear boundaries between classes. 
With some linear algebra it can be shown that an observation can be classified as:  see picture
"""


# Linear kernal
fig, ax = plt.subplots(1, 3, dpi=100, figsize=(12, 4))

# C is a regularization parameter inversely proportional to C in theory
for i, C in enumerate([10, .1, .01], 1):  # C ges av enumerate här
    plt.subplot(1, 3, i)
    classifier = SVC(kernel="linear", C=C)  # de streckade linjerna är för att konstruera hyperplanet, hela linjen= hyperplanet är den som bestämmer klass
    plot_svm_margins(classifier, X, y)
    ax[i-1].set(title = f"C = {C}, linear kernel")

# when C = 10 here, we basically have a maximum margin classifier
# with smaller C we allow for margin to be "softer" allowing more observations over the margins and more support vectors


# RBR kernel
fig, ax = plt.subplots(1, 3, dpi=100, figsize=(12, 4))

# C is a regularization parameter inversely proportional to C in theory
for i, C in enumerate([10, 1, .1], 1):
    plt.subplot(1, 3, i)
    plot_svm_margins(SVC(kernel="rbf", C=C), X, y)
    ax[i-1].set(title=f"Radial basis function kernel with C={C}")
    plt.show()

# Polynomial kernel
fig, ax = plt.subplots(1, 3, dpi=100, figsize=(12, 4))

# C is a regularization parameter inversely proportional to C in theory
for i, C in enumerate([10, 1, .1], 1):
    plt.subplot(1, 3, i)
    plot_svm_margins(SVC(kernel="poly", C=C), X, y)
    ax[i-1].set(title=f"Polynomial kernel function with C={C}")


# Sigmoid Kernal
fig, ax = plt.subplots(1, 3, dpi=100, figsize=(12, 4))

# C is a regularization parameter inversely proportional to C in theory
for i, C in enumerate([10, 1, .1], 1):
    plt.subplot(1, 3, i)
    plot_svm_margins(SVC(kernel="sigmoid", C=C), X, y)
    ax[i-1].set(title=f"Sigmoid kernel function with C={C}")

plt.show()
