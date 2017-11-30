import pandas as pd
pd.options.mode.chained_assignment = None
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import NMF
import scipy as sp
import numpy as np

def get_tfidf_vectors(filename="train_x_tab.csv", split_ratio = 0.75):
    '''Return a sparce tf-idf matrix M.
    
    M has #questions of rows and #words of columns

    @return
    first 4 are train/test dataset
    tf-idf model
    '''
    print('loading the data...')

    # split dataset into train/test sets
    X = pd.read_csv(filename, sep='\t', names=['qid','question'])
    sz_train_set = int(split_ratio * len(X))
    if sz_train_set%2!=0:
        sz_train_set += 1
    train_questions, test_questions = X['question'][:sz_train_set], X['question'][sz_train_set:]

    # train tfidf with training set
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_train_questions = tfidf_vectorizer.fit_transform(train_questions.values.astype('U'))

    # transform test set with tfidf model
    tfidf_test_questions = tfidf_vectorizer.transform(test_questions.values.astype('U'))

    # split labels
    # 2 adjacent rows for X relates to 1 row in labels (y)
    Y = []
    with open('train_y.txt') as fr:
        for line in fr.readlines():
            Y.append(int(line.strip().split(' ')[-1]))
    y_train, y_test = Y[:int(sz_train_set/2)], Y[int(sz_train_set/2):]

    return tfidf_train_questions, y_train, tfidf_test_questions, y_test, tfidf_vectorizer

def generate_sentence_topics(X_train, X_test, k):
    '''Matrix factorization into k topics.

    @return
    W_train, lower dimensional representation for questions
    W_test
    nmf_model

    @explain
    Use NMF (Non-negative matrix factorization), since elements in the 
        question-word matrix should always be non-negative and it's more
        faster and requires less memory.
    Matrix (Q, T), Q = #question while T = #terms 
    Matrix = W x H, W -> (Q, k), H -> (k, T)
    '''
    nmf_model = NMF(n_components=k, init='random', random_state=0)
    W_train = nmf_model.fit_transform(X_train)
    print('reconstruction err: {}'.format(nmf_model.reconstruction_err_))
    W_test = nmf_model.transform(X_test)
    return W_train, W_test, nmf_model

def to_dataframe(X_train, X_test, filename_tr='nmf_tr.csv', filename_te='nmf_te.csv'):
    '''Tranform sparse matrix to dataframe.'''
    train_sz = X_train.shape[0]
    test_sz = X_test.shape[0]
    is_sparce = sp.sparse.issparse(X_train)

    # write to csv
    i = 0
    with open(filename_tr,'w') as ftr:
        while (i<train_sz):
            if (i%(int(train_sz/10))==0):
                print('{}% complete on training set'.format(i*100/train_sz))
            if is_sparce:
                data = [str(d) for d in list(np.array(X_train[i].todense())[0])]
            else:
                data = [str(d) for d in list(np.array(X_train[i]))]
            line = str(i+1) + '\t'.join(data) + '\n'
            ftr.write(line)
            i += 1

    with open(filename_te,'w') as fte:
        while (i<train_sz+test_sz):
            if ((i-train_sz)%(int(test_sz/10))==0):
                print('{}% complete on test set'.format((i-train_sz)*100/test_sz))
            if is_sparce:
                data = [str(d) for d in list(np.array(X_test[i-train_sz].todense())[0])]
            else:
                data = [str(d) for d in list(np.array(X_test[i-train_sz]))]
            line = str(i+1) + '\t'.join(data) + '\n'
            fte.write(line)
            i += 1
    print('Now get dataframes with pd.read_csv for {} and {}'.format(filename_tr, filename_te))

