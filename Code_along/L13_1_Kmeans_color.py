"""Image processing
We use OpenCV in Python for image processing. Install it through

pipenv install opencv-python
This library has existed for very long time and the documentation looks very old, but it is very useful for various
kinds of image processing"""

import matplotlib.pyplot as plt
import cv2
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
import numpy as np


# blue green red
img_BGR = cv2.imread("../Data/blomma.png")
img_BGR2 = img_BGR[:,:, [2,1,0]]
plt.imshow(img_BGR)  # vad har hänt! Inverterad, den läser in värden i BGR ordning. Man kan inte läsa in och konverta samtidigt
plt.imshow(img_BGR2)

# convert BGR to RGB
flower = cv2.cvtColor(img_BGR, cv2.COLOR_BGR2RGB)
plt.imshow(flower)

# läser in kaninen Bella
img_BGR_rabbit = cv2.imread("../data/bella2.jpeg")
rabbit = img_BGR_rabbit[:,:,[2,1,0]]
img = cv2.cvtColor(img_BGR_rabbit, cv2.COLOR_BGR2RGB)

# make image smaller
rabbit = cv2.resize(img, (int(img.shape[1]*.4), int(img.shape[0]*.4)))

plt.imshow(rabbit)

"""
Color quantization
Reduce number of colors in an image by replacing with cluster center. 
These cluster centers can be computed with k-means algorithm.

useful for simple color segmentation
useful when a display only can show a small number of colors
can be used for compression
First we flatten the each of the 3 matrices (R, G, B) to one dimensional vector for each color channel resulting in:

R	G	B
R1	G1	B1
R2	G2	B2
...	...	...
RN	GN	BN
Using k-means with k as the number of colors, we find the cluster centers and give that color to every point in that 
cluster. Finally reshape it back to original shape.
"""



# note that the wildcard -1 calculates row*columns
X = flower.reshape(-1, 3)

# need to normalize data to 0-1
scaler = MinMaxScaler()

scaled_X = scaler.fit_transform(X)

kmean = KMeans(2)
kmean.fit(scaled_X)

print("Cluster centers:",kmean.cluster_centers_)
print("Cluster labels:", kmean.labels_)

# 1 picks second cluster center, all 0 picks first cluster center
quantized_color_space = kmean.cluster_centers_[kmean.labels_]

quantized = quantized_color_space.reshape(flower.shape)
plt.imshow(quantized)


# segmentation
mask = (quantized[:,:,1] >.75)
plt.imshow(mask, cmap = "gray")

segmented_flower = (mask[:, :, None]*flower)
plt.imshow(segmented_flower)
plt.axis("off")

# Reduce colors

X = rabbit.reshape(-1, 3)
scaled_X = scaler.fit_transform(X)

# for example a device can only show 10 discrete colors

kmean = KMeans(10)
kmean.fit(scaled_X)
quantized_color_space = kmean.cluster_centers_[kmean.labels_]

quantized = quantized_color_space.reshape(rabbit.shape)
plt.imshow(quantized)
