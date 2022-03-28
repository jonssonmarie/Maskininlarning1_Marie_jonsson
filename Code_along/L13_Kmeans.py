from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# Lektion k-means clustering
# illustrerar teorin här

# skapar data med make_blops för att kunna illustrera teorin, den skapar även labels, så om unsupervised - släng labels
X, y = make_blobs(500, centers=4, random_state=42, cluster_std=1) # ändra tcluster_std till 2 så får du närmare punkter
# testa och se med 0.5 också även högre/lägre

# , columns = ["x1", "x2", "label"])
df = pd.DataFrame([X[:, 0], X[:, 1], y]).T

df.columns = ["x1", "x2", "label"]
df["label"] = df.label.astype(int)

# plot data
sns.scatterplot(data=df, x="x1", y="x2", hue="label", palette="tab10")
plt.title("Original data")

# we don't have labels in unsupervised learning
# in this simulation we drop the label, but for real world data there is no label in beginning
X = df.drop("label", axis=1)
print(X.head())

"""
Feature scaling
need to scale dataset with either feature standardization or normalization
in unsupervised, as there is no label, we use the whole dataset in scaling
we don't divide into training and test dataset, instead we use the whole dataset
"""
scaler = StandardScaler()
scaled_X = scaler.fit_transform(X)
print(scaled_X[:5])
print(scaled_X.mean(), scaled_X.std())

# plot Scaled X with feature standardization
sns.scatterplot(x=scaled_X[:, 0], y=scaled_X[:, 1])
plt.title("Scaled X with feature standardization")

"""
k-means clustering
- k-means clustering is an unsupervised learning algorithm, which means that there are no labels
  1. k number of clusters are chosen
  2. k points are randomly selected as cluster centers
  3. the nearest points to each cluster center are classified as that cluster
  4. the center of the cluster is recalculated
  5. repeat 3 and 4 until convergence
note that nearest points are defined by some distance metric

Choose k
- plot an elbow plot of sum of squared distances (inertia in sklearn) and find the an inflexion point to choose 
  , i.e. the point with significant lower rate of change than before (note that this might be hard to find exact)
- domain skills, it's important to understand your dataset to find an adequate  and also equally important to be 
  able to know what the clusters represent
- note that it is hard to find correct number of clusters, and it is here the art and domain skills become more 
  important
"""

clusters = np.arange(1, 10)
sum_squared_distances = [KMeans(k).fit(scaled_X).inertia_ for k in clusters]

# plot Elbow plot to find k
fig, ax = plt.figure(), plt.axes()
ax.plot(clusters, sum_squared_distances, '--o')
ax.set(title="Elbow plot to find k", xlabel="Number of clusters k",
       ylabel="SSD = Sum of squared distances to cluster centers")


# note here that it is very hard to pick 3 or 4 clusters as the clusters are close to each other

SSD_differences = pd.Series(sum_squared_distances).diff()[1:] # efter 4 är det inte stor förändring och
                                                              # man ska kolla på signifikanta skillnader
SSD_differences.index = clusters[:-1]
print(SSD_differences)

"""
Silhouette score
Note that it's usually not possible to plot the clusters, instead the silhouette score in combination with 
elbow plot can help in determining clusters.
se bild
"""
# plot Silhouette plot
fig1, ax = plt.figure(), plt.axes()
kmeans = [KMeans(k).fit(scaled_X) for k in clusters]
silhouette_list = [silhouette_score(scaled_X, kmean.labels_) for kmean in kmeans[1:]]
ax.plot(clusters[1:], silhouette_list, "o-")
ax.set(title="Silhouette plot", xlabel="k clusters", ylabel="Silhouette score")

"""
Visualization
note that we don't have the luxury to visualize real world data as dimensions usually are much higher than 2
"""
kmeans = [KMeans(n_clusters=k).fit(scaled_X) for k in clusters]
df_plot = pd.DataFrame(scaled_X, columns=["x1", "x2"])
number_plots = round(len(clusters)/2)
print("se varför avrunding", len(clusters)/2)

# plot {i+1} clusters
fig2, axes = plt.subplots(2, number_plots, figsize=(16, 8))

for i, ax in enumerate(axes.flatten()):
    cluster_centers = kmeans[i].cluster_centers_  # notera annan index i KMeans, börjar på 1
    df_plot["label"] = kmeans[i].labels_

    sns.scatterplot(data=df_plot, x="x1", y="x2", hue="label", ax=ax, palette="tab10")

    ax.scatter(cluster_centers[:, 0], cluster_centers[:, 1], s = 200, marker = '*', color="black", label="centroid")
    ax.legend([],[], frameon = False)
    ax.set(title = f"{i+1} clusters")
plt.show()
