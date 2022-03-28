import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer


"""
Lecture notes - NLP intro - text feature extraction

This is the lecture note for NLP intro, more on text feature extraction and preprocessing 
will be given in the NLP section of the deep learning course. 

Note that this lecture note gives a brief introduction to NLP with focus on text extraction. 
I encourage you to read further about text text feature extraction.
"""

"""
Term frequency
term frequency $tf(t,d)$ - relative frequency of term $t$ in document $d$, i.e. how frequent a term occurs in a document
se bild
"""
review1 = "I LOVE this book about love"
review2 = "No this book was okay"

all_words = [text.lower().split() for text in [review1, review2]]
print(all_words)

# flattens 2D list to 1D list
all_words = [word for text in all_words for word in text]
print(f"Flattened all words: {all_words}")

# alternativ till nestlad list comprehension:
all_words2 = all_words[0:] + all_words[1:]
print("all_words2", all_words2)
# alt 2
all_words3 = review1 + ' ' + review2
print("all_words3", all_words3)


# removes all copies, but sets don't have any particular ordering
unique_words = set(all_words)
print(f"Unique words: {unique_words}")

# dictionary of all words
vocabulary = {word: index for index, word in enumerate(unique_words)}
print(vocabulary)


def term_frequency_vectorizer(document, vocabulary):
    term_frequency = np.zeros(len(vocabulary))

    for word in document.lower().split():
        index = vocabulary[word]
        term_frequency[index] += 1

    return term_frequency


# note that we consider the raw count itself and not divide by total number of terms in the document
# this is another way to define the term frequency and more simplistic to ease understanding
review1_term_freq = term_frequency_vectorizer(review1, vocabulary)
review2_term_freq = term_frequency_vectorizer(review2, vocabulary)

# review1 = "I LOVE this book about love"
# review2 = "No this book was okay"
print(review1)
print(review1_term_freq)
print(review2)
print(review2_term_freq)

# skapa bag of words
bag_of_words = pd.DataFrame([review1_term_freq, review2_term_freq],
                            columns=vocabulary.keys(), dtype="int16")
print(bag_of_words)


"""
Feature extraction with sklearn
- CountVectorizer - creates a bag of words model
- TfidfTransformer - transforms it using TF-IDF
- TfidfVectorizer - does CountVectorizer and TfidfTransformer
"""
count_vectorizer = CountVectorizer()
bag_of_words_sparse = count_vectorizer.fit_transform([review1, review2])
bag_of_words_sparse.todense(), count_vectorizer.get_feature_names_out()
# note that it ignores one letter words such as I

bag_of_words = pd.DataFrame(bag_of_words_sparse.todense(), columns=count_vectorizer.get_feature_names_out())
print("bag_of_words:\n",bag_of_words)

"""
TF-IDF 
- Term frequency - inverse document frequency
- TF-IDF is a way to represent how important a word is across a corpus of documents. Basically it is a vector with numeric weights on each word, where higher weights is put on rarer terms.
see bild
"""

tfidf_transformer = TfidfTransformer()
tfidf = tfidf_transformer.fit_transform(bag_of_words_sparse)
print(tfidf)

print(tfidf.todense())

# creates tfidf vector in one go
tfidf_vectorizer = TfidfVectorizer()
tfidf_words = tfidf_vectorizer.fit_transform([review1, review2]).todense()
print(tfidf_words)

tfidf_vectorizer.get_feature_names()
print(review1)
print(review2)

pd.DataFrame(tfidf_words, columns = tfidf_vectorizer.get_feature_names_out())
