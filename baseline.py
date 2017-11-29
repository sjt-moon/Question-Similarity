import pandas as pd
pd.options.mode.chained_assignment = None
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import NMF
import scipy as sp
import numpy as np

from pipeline import *

# This achieves 63.45365% accuracy
# correct predictions: 501978
# wrong answers: 289116

def cosine_similarity_model(X_train, y_train, X_test, y_test):
    '''There is no need to train anything. Use all of the data.'''
    print('calculating cosine similarity')

    correct, wrong = 0, 0
    i = 0
    sz = X_train.shape[0]
    while (i+1<sz):
        if (i%(int(sz/10))==0):
            print("{}% training set completed".format(i*100/sz))
        sim = cosine_similarity(X_train[i], X_train[i+1])[0][0]
        pred = 1 if sim>=0.5 else 0
        if pred==y_train[int(i/2)]:
            correct += 1
        else:
            wrong += 1
        i += 1
    i = 0
    sz = X_test.shape[0]
    while (i+1<sz):
        if (i%(int(sz/10))==0):
            print("{}% test set completed".format(i*100/sz))
        sim = cosine_similarity(X_test[i], X_test[i+1])[0][0]
        pred = 1 if sim>=0.5 else 0
        if pred==y_test[int(i/2)]:
            correct += 1
        else:
            wrong += 1
        i += 1
    print("#correct predictions:\t{}\n#wrong answers:\t{}\naccuracy:\t{}".format(correct, wrong, correct/(correct+wrong)))

def main():
    xtr, ytr, xte, yte = get_tfidf_vectors()
    cosine_similarity_model(xtr, ytr, xte, yte)

#main()
    
    
