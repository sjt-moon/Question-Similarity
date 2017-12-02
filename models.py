import numpy as np
import importlib
import pandas as pd
import pipeline
import baseline
import re

from sklearn.linear_model import LogisticRegression
from sklearn import svm
import matplotlib.pyplot as plt

# get raw text dataframe, dataframe
raw_text = pd.read_csv('./train_x_tab.csv',sep='\t',names=['qid','question'])
xtr, ytr, xva, yva, xte, yte = pipeline.split_dataset()

# get tfidf matrix & model, array
tf_tr, tf_va, tf_te, tfidf = pipeline.get_tfidf_vectors(xtr,xva,xte)

# word2vec, dataframe
word2vec = pipeline.get_word2vec()

# raw text features, array
feat_tr, feat_va, feat_te = pipeline.gen_raw_text_features(xtr,xva,xte,tfidf,word2vec)

# sentence topics, k = 30, array
w_tr, w_va, w_te, nmf = pipeline.generate_sentence_topics(tf_tr,tf_va,tf_te,30)

# models

# LR
def LR(x_train,y_train,x_valid,y_valid,x_test,y_test,penalty=['l1','l2'],C=[100,1,0.01],tol=1e-4):
    '''Find the best LR model
    
    @para
    x & y: dataframe, pairwise features, i.e. len(x) == len(y)

    @return
    best lr model
    '''
    best_accuracy = -9999
    best_model = None
    for p in penalty:
        for c in C:
            clf_lr = LogisticRegression(C=c, penalty=p, tol=tol)
            clf_lr.fit(x_train, y_train)
            score = clf_lr.score(x_valid, y_valid)
            if not best_model or score>best_score:
                best_model = clf_lr
                best_accuracy = score
    print('Best LR model is:')
    print(best_model.get_params())
    print('Best score on validation set: {}'.format(best_accuracy))
    print('Accuracy on test set: {}'.format(best_model.score(x_test, y_test)))
    return best_model

def SVM(x_train,y_train,x_valid,y_valid,x_test,y_test,C=[100,1,0.01]):
    '''Find the best LR model
    
    @para
    x & y: dataframe, pairwise features, i.e. len(x) == len(y)

    @return
    best lr model
    '''
    best_accuracy = -9999
    best_model = None
    for c in C:
        clf = svm.SVC(C=c)
        clf.fit(x_train, y_train)
        score = clf.score(x_valid, y_valid)
        if not best_model or score>best_accuracy:
            best_model = clf
            best_accuracy = score
    print('Best SVM model is:')
    print(best_model.get_params())
    print('Best score on validation set: {}'.format(best_accuracy))
    print('Accuracy on test set: {}'.format(best_model.score(x_test, y_test)))
    return best_model
