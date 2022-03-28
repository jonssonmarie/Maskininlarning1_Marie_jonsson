# Naive Bayes code along
# Spam/ham classification

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.naive_bayes import MultinomialNB, ComplementNB, BernoulliNB
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.pipeline import Pipeline


df = pd.read_csv("../Data/spam.csv", encoding="latin-1")
#print(df.head())

# check how many rows have values in the Unnamed columns
np.sum(df[df.columns[2:]].notna())

print(df.shape)
# read some of the Unnamed that have values
#df.loc[df["Unnamed: 2"].notna()].iloc[:5]


# as it is very few rows, we remove those columns
df_no_NaN = df.dropna(axis=1)
df_no_NaN.columns = ["class", "content"]
df_no_NaN.info()

# check balance of spam/ham - this is an unbalanced dataset
sns.countplot(data = df_no_NaN, x = "class")


df = pd.get_dummies(df_no_NaN, columns = ["class"], drop_first=True)
X, y = df["content"], df["class_spam"]
X.head()

# TF-IDF vector
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

tfidf_vectorizer = TfidfVectorizer(stop_words="english")
X_tfidf = tfidf_vectorizer.fit_transform(X)
print(repr(X_tfidf))
print(X_tfidf.todense())
print(f"Min value: {X_tfidf.min()}, max value: {X_tfidf.max()}")

X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.33, random_state=42)
print(X_train.shape, X_test.shape)

fig, ax = plt.subplots(1,2, dpi = 100, figsize = (8,3))
sns.countplot(x = y_train, ax = ax[0])
sns.countplot(x = y_test, ax = ax[1])

"""
Naive Bayes
family of probability classifiers based on Bayes theorem
they have strong naive independent assumption between features
for text it means that the each word is assumed to be independent of other words (bag of words model)
se bild
"""
naive_bayes = dict(multinomial=MultinomialNB(),
                   bernoulli=BernoulliNB(),
                   complement=ComplementNB())


def evaluate_model(model):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print(classification_report(y_test, y_pred))
    cm = confusion_matrix(y_test, y_pred)
    ConfusionMatrixDisplay(cm, display_labels=["Ham", "Spam"]).plot()
    plt.show()


evaluate_model(naive_bayes["multinomial"])
evaluate_model(naive_bayes["bernoulli"])
evaluate_model(naive_bayes["complement"])
# precision = TP/(TP+FP)  finns på wikipedia bla
# recall = TP/(TP+FN)
# high precision -> low FP -> low hams in spam box
# high recall -> low FN -> low spams in inbox
# want to have high precision as we don't want to have any hams in spam box
# can accept somewhat lower recall as it's not the end of the world if
# we get spams in inbox. Ideally both precision and recall are high.


# Trying other models also
evaluate_model(LinearSVC())
evaluate_model(RandomForestClassifier())

"""
Combining models
- plurality vote
- the class with most votes wins
"""
vote_clf = VotingClassifier([("rf", RandomForestClassifier()),
                             ("svc", LinearSVC()),
                             ("complement", ComplementNB()),
                             ("bernoulli", BernoulliNB()),
                             ("multinomial", MultinomialNB())], voting="hard")
# använd udda antal modeller pga röstning - 2 ja 2 nej
# kan inte köra mjuk då alla klasser ger inte propability

evaluate_model(vote_clf)
# we see that a combination of models improves the performance

# Deployment
pipe = Pipeline([("tfidf", TfidfVectorizer(stop_words="english")), ("vote", vote_clf)])

# fits on all data
pipe.fit(X, y)  # tränar på alla data vi har då vi tränat på train innan
print(pipe.predict(["Come and collect your $1000 Bitcoins!"]))
# 0 is ham and 1 is spam
print(pipe.predict(["You will win, $1000 for free"]))
print(pipe.predict(["You get $500"]))

"""
print resultat:
# 0 is ham and 1 is spam
[0]
[1]
[1]
"""

