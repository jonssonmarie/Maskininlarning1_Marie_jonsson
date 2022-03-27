"""
K-Nearest Neighbor classification (KNN) exercises
These are introductory exercises in Machine learning with focus in KNN, but also an introductory exercise in
computer vision.
"""
"""
0. MNIST data (*)
"""
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import random
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report, accuracy_score
import cv2


# import data
(X_train_3d, Y_train), (X_test_3d, Y_test) = keras.datasets.mnist.load_data()

"""
print(X_train.shape, Y_train.shape, X_test.shape, Y_test.shape)
(60000, 28, 28) (60000,) (10000, 28, 28) (10000,)
X_train = 60000 bilder med 28x28 pixlar, med högre värde på siffran ju större pixelintensitet)
Y_train = 60000 pixlar a´ 1 array/vektor
X_test = 10000 bilder med 28x28 pixlar, med högre värde på siffran ju större pixelintensitet
Y_test = 10000 pixlar a´ 1 array/vektor
"""
print(X_train_3d.max(), X_train_3d.min())


# plot 20 randomly chosen numbers
fig, ax = plt.subplots(2, 10, figsize=(20, 6))
ax = ax.flatten()

for i in range(20):
    num_random = random.randint(0, 59999)
    pic_image = X_train_3d[num_random][:]
    ax[i].imshow(pic_image, cmap="gray_r")
    ax[i].set_xlabel(Y_train[i])

plt.show()


X_train_2d_1 = np.reshape(X_train_3d, (60000, -1))    # or reshape(-1, 28 * 28)
X_test_2d = np.reshape(X_test_3d, (10000, -1))


"""
1. Train|test|validation split (*)
"""


def train_split(x_matrix, y_series):
    """
    :param x_matrix: ndarray
    :param y_series: np.ndarray
    :return: ndarray, ndarray, ndarray, ndarray
    """
    X_train, X_test, y_train, y_test = train_test_split(x_matrix, y_series, test_size=0.16, random_state=42)
    return X_train, X_test, y_train, y_test


X_train_2d, X_2d_val, y_train, y_val = train_split(X_train_2d_1, Y_train)

"""
2. Hyperparameter tuning (*)
"""
error = []


def calculate_error(x_training_2d, y_training, x_validation, y_validation):
    """
    :param x_training_2d: ndarray
    :param y_training: ndarray
    :param x_validation: ndarray
    :param y_validation: ndarray
    :return: None
    """
    for k in range(1, 10):
        model_KNN = KNeighborsClassifier(n_neighbors=k)
        model_KNN.fit(x_training_2d, y_training)
        y_pred = model_KNN.predict(x_validation)
        error.append(1 - accuracy_score(y_validation, y_pred))


calculate_error(X_train_2d, y_train, X_2d_val, y_val)


def plot_error(error_lst):
    """
    :param error_lst: lst
    :return: None
    """
    fig, ax = plt.figure(), plt.axes()
    ax.plot(range(1, len(error_lst) + 1), error_lst, ".-")
    ax.set(title="Elbow plot", xlabel="k neightbours", ylabel="Error")
    plt.show()
    # I choose k = 1 see elbow_plot.png


plot_error(error)


"""
3. Train and predict
  a) Do a classification report and based on the report, can you figure out which number 
  had highest proportions of false negatives?
  False negative in this case means that the true label is true but the model predicted not. 
  Answer : label 8

  b) Plot a confusion matrix, does this confirm your answer in a? 
  Answer: No, label 5 (860)

  c) Compute the number of misclassifications for each number. 
  Which number had most misclassifications, do you have any suggestions on why this would be the case?
  Answer: 8 or 5, it was 8
"""


def train_split_3d_to_2d(X_training_3d, X_testing_3d):
    """
    :param X_training_3d: ndarray
    :param X_testing_3d: ndarray
    :return: DataFrame
    """
    x_train = X_training_3d.reshape(-1, 28 * 28)
    x_test = X_testing_3d.reshape(-1, 28 * 28)

    print(x_train.shape, x_test.shape)
    return x_train, x_test


x_train, x_test = train_split_3d_to_2d(X_train_3d, X_test_3d)
model_KNN = KNeighborsClassifier(1)
model_KNN.fit(x_train, Y_train)
y_pred = model_KNN.predict(x_test)


def KNeighborsClassifier_model( Y_testing):
    """
    :param Y_testing: ndarray
    :return: None, print classification report
    """
    print(classification_report(Y_testing, y_pred))
    cm = confusion_matrix(Y_testing, y_pred)
    ConfusionMatrixDisplay(cm).plot()
    plt.show()

    for i, label in enumerate(cm):
        print(f"True label {i}, algorithm misclassified {sum(label) - label[i]} of those")

    print(classification_report(Y_test, y_pred))


KNeighborsClassifier_model(Y_test)

"""
True label 0, algorithm misclassified 7 of those
True label 1, algorithm misclassified 6 of those
True label 2, algorithm misclassified 40 of those
True label 3, algorithm misclassified 40 of those
True label 4, algorithm misclassified 38 of those
True label 5, algorithm misclassified 32 of those
True label 6, algorithm misclassified 14 of those
True label 7, algorithm misclassified 36 of those
True label 8, algorithm misclassified 54 of those
True label 9, algorithm misclassified 42 of those

k = 1 was right to choose
"""
"""
The precision is the ratio tp / (tp + fp) where tp - true positives and fp - false positives. 
The precision is intuitively the ability of the classifier not to label as positive, a sample that is negative.

The recall is the ratio tp / (tp + fn) where tp - true positives and fn - false negatives. 
The recall is intuitively the ability of the classifier to find all the positive samples.

The F-beta score can be interpreted as a weighted harmonic mean of the precision and recall, 
where an F-beta score reaches its best value at 1 and worst score at 0.

The F-beta score weights recall more than precision by a factor of beta. beta == 1.0 means recall and precision 
are equally important.

                precision   recall   f1-score  support

           0       0.98      0.99      0.99       980
           1       0.97      0.99      0.98      1135
           2       0.98      0.96      0.97      1032
           3       0.96      0.96      0.96      1010
           4       0.97      0.96      0.97       982
           5       0.95      0.96      0.96       892
           6       0.98      0.99      0.98       958
           7       0.96      0.96      0.96      1028
           8       0.98      0.94      0.96       974
           9       0.96      0.96      0.96      1009

    accuracy                           0.97     10000
   macro avg       0.97      0.97      0.97     10000
weighted avg       0.97      0.97      0.97     10000
"""


"""
4. Predict your own handwriting (**)
"""
file_names = ["ett", "tre", "nio", "fem", "tva", "fyra", "sex", "sju", "atta", "sju_", "noll"]
real_numbers = [1, 3, 9, 5, 2, 4, 6, 7, 8, 7, 0]

for name in file_names:
    image = cv2.imread(f"Numbers/{name}.jpg", cv2.IMREAD_GRAYSCALE)
    plt.imshow(image, cmap="gray_r")
    #plt.show()


for file_name, real_number in zip(file_names, real_numbers):

    image = cv2.imread(f"Numbers/{file_name}.jpg", cv2.IMREAD_GRAYSCALE)
    # Read it as grayscale (this drops two of the dimensions that coloured pictures have and leaves one pixel channel)
    image = cv2.resize(image, (28, 28))  # Resize to 28*28 pixels
    image = np.asarray(image)  # Saves the pixel intensity values as an np.array

    image = image.reshape((28 * 28,))
    image = image.reshape(1, -1)

    predicted_number = model_KNN.predict(image)
    print(f"Real Number: {real_number} Predicted Number: {predicted_number[0]}")


# it sucks to get the right number with my style dosn't matter if white or black background or pen
