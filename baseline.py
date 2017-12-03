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

def cosine_similarity_model(X, y):
    '''There is no need to train anything. Use all of the data.
    
    Note that len(X) = 2*len(y)

    @para
    X: sparse matrices, tfdif matrices
    y: dataframe
    '''
    assert sp.sparse.issparse(X)
    assert X.shape[0] == 2*len(y)
    print('calculating cosine similarity')

    correct, wrong = 0, 0
    i = 0
    sz = X.shape[0]
    while (i<sz):
        if (i%(int(sz/10))==0):
            print("{}% training set completed".format(i*100/sz))
        sim = cosine_similarity(np.array(X[i].todense())[0], np.array(X[i+1].todense())[0])[0][0]
        pred = 1 if sim>=0.5 else 0
        if pred==int(y.loc[int(i/2)]):
            correct += 1
        else:
            wrong += 1
        i += 2
    print("#correct predictions:\t{}\n#wrong answers:\t{}\naccuracy:\t{}".format(correct, wrong, correct/(correct+wrong)))

def main():
    xtr, ytr, xva, yva, xte, yte = get_tfidf_vectors()
    cosine_similarity_model(xte, yte)

#main()
    
    
