import numpy as np
import importlib
import pandas as pd
import pipeline
import baseline
import re

from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
import matplotlib.pyplot as plt


import warnings
warnings.filterwarnings("ignore")

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
            if not best_model or score>best_accuracy:
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

# Desicion trees
def DT(x_train,y_train,x_valid,y_valid,x_test,y_test):
    clf = tree.DecisionTreeClassifier()
    clf.fit(x_train, y_train)
    score = clf.score(x_valid, y_valid)
    print('Score on validation set: {}'.format(score))
    print('Accuracy on test set: {}'.format(clf.score(x_test, y_test)))
    return clf

# kNN
def kNN(x_train,y_train,x_valid,y_valid,x_test,y_test,n_neighbors=[5,10,20,50]):
    best_accuracy = -9999
    best_model = None
    for k in n_neighbors:
        clf = KNeighborsClassifier(n_neighbors=k)
        clf.fit(x_train, y_train)
        score = clf.score(x_valid, y_valid)
        if not best_model or score>best_accuracy:
            best_model = clf
            best_accuracy = score
    print('Best kNN model is:')
    print(best_model.get_params())
    print('Best score on validation set: {}'.format(best_accuracy))
    print('Accuracy on test set: {}'.format(best_model.score(x_test, y_test)))
    return best_model

# Random forest
def RF(x_train,y_train,x_valid,y_valid,x_test,y_test,n_estimators=[30,40,50,60,70,80,100]):
    best_accuracy = -9999
    best_model = None
    for k in n_estimators:
        clf = RandomForestClassifier(n_estimators=k)
        clf.fit(x_train, y_train)
        score = clf.score(x_valid, y_valid)
        if not best_model or score>best_accuracy:
            best_model = clf
            best_accuracy = score
    #print('Best kNN model is of {} estimators'.format(best_model.estimators_))
    print('Feature importance')
    print(best_model.feature_importances_)
    print('Best score on validation set: {}'.format(best_accuracy))
    print('Accuracy on test set: {}'.format(best_model.score(x_test, y_test)))
    return best_model

# GBDT
def GBDT(x_train,y_train,x_valid,y_valid,x_test,y_test,n_estimators=[100,200,300],depths=[1,3,5],lrs=[0.1,0.5,1.0]):
    best_accuracy = -9999
    best_model = None
    for k in n_estimators:
        for d in depths:
            for lr in lrs:
                clf = GradientBoostingClassifier(n_estimators=100,learning_rate=lr,max_depth=d)
                clf.fit(x_train, y_train)
                score = clf.score(x_valid, y_valid)
                if not best_model or score>best_accuracy:
                    best_model = clf
                    best_accuracy = score
    #print('Best GBDT model is of {} estimators'.format(best_model.estimators_))
    print('Feature importance')
    print(best_model.feature_importances_)
    print('Best score on validation set: {}'.format(best_accuracy))
    print('Accuracy on test set: {}'.format(best_model.score(x_test, y_test)))
    return best_model

# Adaboost
def Adaboost(x_train,y_train,x_valid,y_valid,x_test,y_test,n_estimators=[100,200,300],lrs=[0.1,0.5,1.0]):
    best_accuracy = -9999
    best_model = None
    for k in n_estimators:
        for lr in lrs:
            clf = AdaBoostClassifier(n_estimators=k, learning_rate=lr)
            clf.fit(x_train, y_train)
            score = clf.score(x_valid, y_valid)
            if not best_model or score>best_accuracy:
                best_model = clf
                best_accuracy = score
    #print('Best GBDT model is of {} estimators'.format(best_model.estimators_))
    print('Feature importance')
    print(best_model.feature_importances_)
    print('Best score on validation set: {}'.format(best_accuracy))
    print('Accuracy on test set: {}'.format(best_model.score(x_test, y_test)))
    return best_model




