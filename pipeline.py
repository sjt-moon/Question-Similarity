import pandas as pd
pd.options.mode.chained_assignment = None
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import NMF
import scipy as sp
import numpy as np

def split_dataset(fx="train_x_tab.csv", fy="train_y_tab.csv", ratios = [0.6,0.2,0.2]):
    assert len(ratios)==3 and sum(ratios)==1

    X = pd.read_csv(fx, sep='\t', names=['qid','question'])
    y = pd.read_csv(fy, sep='\t', names=['q1','q2','label'])
    assert len(X) == 2 * len(y)

    valid_start_index = int(ratios[0] * len(y))
    test_start_index = int(ratios[1] * len(y)) + valid_start_index
    print(valid_start_index, test_start_index)

    x_tr,x_va,x_te = X[:2*valid_start_index],X[2*valid_start_index:2*test_start_index],X[2*test_start_index:]
    y_tr, y_va, y_te = y[:valid_start_index],y[valid_start_index:test_start_index],y[test_start_index:]

    # change row indices, start from 0
    x_va.index = range(len(x_va))
    y_va.index = range(len(y_va))
    x_te.index = range(len(x_te))
    y_te.index = range(len(y_te))
    
    assert len(x_tr) == 2 * len(y_tr)
    assert len(x_va) == 2 * len(y_va)
    assert len(x_te) == 2 * len(y_te)

    return x_tr, y_tr, x_va, y_va, x_te, y_te

def gen_raw_feat(X, tfidf, word2vec, filename):
    '''generate PAIRWISE features from raw text. (length would be half of before)
    
    @para
    X: dataframe
    col_names: name for each feature, easy to_dataframe

    @return
    since it's too large, write to csv and return none

    @features (for each question)
    #chars 
    #terms
    #all terms in 2 questions
    #common terms for the 2 questions
    #intersection ratios for 2 questions, i.e. how many common terms in each question
    product of sentence2vec
    '''
    assert 'question' in X.columns

    #word2vec = get_word2vec()
    #print('word2vec ready!')

    # average terms in a question is 8.81, 75% questions have less than 10 terms
    # we maintain at most first max_terms_per_question terms
    # max_terms_per_question = 20

    #feat = []
    with open(filename,'w') as fw:

        col_names = ['diff_num_chars','diff_num_terms','common_ratio','sen_product',]
        fw.write('\t'.join(col_names)+'\n')

        sz = len(X)
        print(sz)
        i = 0
        while i<sz:
            if i%(int(sz/100))==0:
                print('{}% complete'.format(int(i*100/sz)))

            q1, q2 = X['question'][i], X['question'][i+1]
            try:
                q1 = set([w for w in q1.split()]) if len(q1)>0 else set()
            except:
                print("Err at {}:\t{}".format(i,q1))
                q1 = set()
            try:
                q2 = set([w for w in q2.split()]) if len(q2)>0 else set()
            except:
                print("Err at {}:\t{}".format(i,q2))
                q2 = set()

            common = q1.intersection(q2)
            union = q1.union(q2)
    
            num_chars_q1 = sum(len(w) for w in q1)
            num_chars_q2 = sum(len(w) for w in q2)
            diff_num_chars = abs(num_chars_q1 - num_chars_q2)

            num_terms_q1, num_terms_q2 = len(q1), len(q2)
            diff_num_terms = abs(num_terms_q1 - num_terms_q2)

            common_sz, union_sz = len(common), len(union)
            common_ratio = 1.0 * common_sz / union_sz

            # sentence2vec 
            v1 = sentence2vec(q1, tfidf, word2vec)
            v2 = sentence2vec(q2, tfidf, word2vec)
            sen_product = v1.dot(v2)

            line=[diff_num_chars, diff_num_terms, common_ratio, sen_product]
            line = [str(ele) for ele in line]
            #feat.append(line)
            fw.write('\t'.join(line)+'\n')

            i += 2
    return

def get_word2vec():
    '''Map term to word2vec -> dataframe'''
    word2vec = pd.DataFrame(np.load('./embeddings.npy'))
    
    keys_embeddings = []
    with open('./keys_embeddings.txt') as fr:
        for line in fr.readlines():
            keys_embeddings.append(line.strip().lower())
    word2vec['word'] = keys_embeddings
    word2vec.set_index('word')
    return word2vec

def get_word2vec_question(word2vec, question, max_terms_per_question):
    '''get word2vec for each term of the question

    @para
    word2vec: dataframe
    question: set of terms
    max_terms_per_question: int
    
    @return
    vec: max_terms_per_question x 300 matrix, 
        each row is word2vec for the corresponding word in question
    '''
    assert 'word' in word2vec.columns

    vec = []
    words = str(question).split()
    k = 0
    for term in question:
        if k>=max_terms_per_question:
            break
        word_vec = word2vec.loc[word2vec['word']==term].loc[:, word2vec.columns[:300]].values[0]
        vec.append(word_vec)
        k += 1
    while k<max_terms_per_question:
        vec.append(np.zeros(300))
        k += 1
    return np.array(vec)

def sentence2vec(question, tfidf, word2vec):
    '''get sentence2vec for each question. same length with word2vec
    
    @para
    question: set of terms
    tfidf: trained model to get weights for each term in each question
    '''
    assert 'word' in word2vec.columns

    sentence = [' '.join(question),]
    tf_vec = np.array(tfidf.transform(sentence).todense())[0]
    denominator = sum(tf_vec)
    vec = np.zeros(300)

    for term in question:
        if term not in tfidf.vocabulary_:
            continue
        index = tfidf.vocabulary_[term]
        word_vec = word2vec.loc[word2vec['word']==term].loc[:, word2vec.columns[:300]].values[0]
        #if len(word_vec)<=0:
        #    continue
        vec += tf_vec[index] * word_vec / denominator
    return vec

def get_tfidf_vectors(x_train, x_valid, x_test):
    '''Return a sparce tf-idf matrix M.
    
    M has #questions of rows and #words of columns

    @return
    first 3 are train/valid/test dataset
    tf-idf model
    col_names: feature name for each column
    '''
    assert 'qid' in x_train.columns
    assert 'question' in x_train.columns

    q_train, q_valid, q_test = x_train['question'], x_valid['question'], x_test['question']

    # train tfidf with training set
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_train = tfidf_vectorizer.fit_transform(q_train.values.astype('U'))

    # transform valid/test set with tfidf model
    tfidf_valid = tfidf_vectorizer.transform(q_valid.values.astype('U'))
    tfidf_test = tfidf_vectorizer.transform(q_test.values.astype('U'))

    # get column names, i.e. word for each col
    # k: word, v: index
    col_names = [(k,v) for k,v in tfidf_vectorizer.vocabulary_.items()]
    col_names = sorted(col_names, key=lambda x:x[1])
    col_names = [w for w,i in col_names]

    return tfidf_train, tfidf_valid, tfidf_test, tfidf_vectorizer, col_names

def generate_sentence_topics(X_train, X_valid, X_test, k):
    '''Matrix factorization into k topics.

    @return
    W_train, lower dimensional representation for questions
    W_valid
    W_test
    nmf_model
    col_names

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
    W_valid = nmf_model.transform(X_valid)
    W_test = nmf_model.transform(X_test)

    col_names = ['topic_'+str(i) for i in range(k)]

    return W_train, W_valid, W_test, nmf_model, col_names

def to_pairwise_cosine(x,col_name=['cos_sim']):
    '''x[i], x[i+1] -> cosine(x[i], x[i+1])
    
    @para
    x: sparse matrix or np.array
    '''
    import warnings
    warnings.filterwarnings("ignore")

    is_sparse = sp.sparse.issparse(X)
    
    ans = []
    i = 0
    sz = x.shape[0]
    while i<sz:
        if i%int(sz/10)==0:
            print('{}%'.format(i*100/sz))
        if is_sparse:
            v1 = np.array(x[i].todense())[0]
            v1 = np.array(x[i].todense())[0]
            sim = cosine_similarity(v1, v2)[0][0]
        else:
            sim = cosine_similarity(x[i], x[i+1])[0][0]
        ans.append(sim)
        i += 2
    return pd.DataFrame(np.array(ans), columns=col_name)

def to_dataframe(X, col_names, filename=None):
    assert col_names and len(col_names)>0
    if sp.sparse.issparse(X):
        if not filename:
            print('This is a sparse matrix, directly convert might cause memory error')
            print('Please provide file name to write to disk')
            return
        _sparse_to_dataframe_(X,col_names,filename)
        return
    df = pd.DataFrame(np.array(X), columns=col_names)
    print('Now you get a dataframe! congrats')
    return df

def _sparse_to_dataframe_(X, col_names, filename='nmf_tr.csv'):
    '''Tranform sparse matrix X to dataframe.
    
    @format
    first col: question id, start from 1 (same as source)
    next K cols: K features
    '''
    sz = X.shape[0]
    assert sp.sparse.issparse(X)

    # write to csv
    i = 0
    with open(filename,'w') as fw:
        first_line = 'id\t' + '\t'.join(col_names) + '\n'
        fw.write(first_line)
        while (i<sz):
            if (i%(int(sz/10))==0):
                print('{}% complete on training set'.format(i*100/sz))
            data = [str(d) for d in list(np.array(X[i].todense())[0])]
            line = str(i+1) + '\t' + '\t'.join(data) + '\n'
            fw.write(line)
            i += 1
    print('Now get dataframes with pd.read_csv for {}'.format(filename))

def main():
    # This is an example for work flow
    # all data are dataframes except tfidf matrix

    # get raw text dataframe
    raw_text = pd.read_csv('./train_x_tab.csv',sep='\t',names=['qid','question'])
    print('size: ',len(raw_text))
    raw_text.head()

    xtr, ytr, xva, yva, xte, yte = pipeline.split_dataset()

    # get tfidf matrix & model
    # sparse matrix!
    tf_tr, tf_va, tf_te, tfidf, words = pipeline.get_tfidf_vectors(xtr,xva,xte)

    word2vec = pipeline.get_word2vec()

    # get features
    # write to file
    pipeline.gen_raw_feat(xtr,tfidf,word2vec,'features.txt')
    
    # get lower dimensional representations for questions
    # topics k=30
    w_tr, w_va, w_te, nmf, nmf_cols = pipeline.generate_sentence_topics(tf_tr,tf_va,tf_te,30)




