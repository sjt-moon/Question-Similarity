import pandas as pd
pd.options.mode.chained_assignment = None
import numpy as np
import time
from nltk.tokenize import RegexpTokenizer
from nltk.stem.porter import *
import re

def isEnglish(s):
    s = str(s)
    try:
        s.encode(encoding='utf-8').decode('ascii')
    except UnicodeDecodeError:
        return False
    else:
        return True

def to_csv(filename = "train_x2.txt"):
    '''Combine each question as a str, rather than seperated words.
    '''
    with open(filename) as fr:
        with open("train_x_tab.csv",'w') as fw:
            with open("train_y_tab.csv",'w') as fy:
                qid = 1
                i = 0
                for line in fr.readlines():
                    if i%10000==0:
                        print("{}w pieces of data complete...".format(i/10000))
                    line = line.strip().split('_')
                    q1, q2, label = line[0], line[1], line[2]
                    fw.write(str(qid)+'\t'+q1+'\n')
                    qid += 1
                    fw.write(str(qid)+'\t'+q2+'\n')
                    qid += 1
                    fy.write(str(qid-2)+'\t'+str(qid-1)+'\t'+label+'\n')
                    i += 1

def process(write_file_name='train_x2.txt', filename='./train.csv/train.csv', is_training=True):
    '''Separate words by whitespace.

    @format
    question_id words
    '''
    print('still loading......')
    train = pd.read_csv(filename)

    print('start working!')
    start_time = time.time()
    #tokenizer = RegexpTokenizer(r'\w+')
    stemmer = PorterStemmer()
    err_encoding = 0

    with open(write_file_name, 'w') as fw:
        sz = train.shape[0]
        for i in range(sz):
            if i*10%sz==0:
                print("{}% complete...".format(i*100/sz))
            if not isEnglish(train['question1'][i]):
                err_encoding += 1
                continue
            if not isEnglish(train['question2'][i]):
                err_encoding += 1
                continue
            reg = re.compile('[^A-Za-z0-9]')
            
            try:
                #qid_1 = train['qid1'][i]
                q1 = reg.split(str(train['question1'][i]).lower())
                #q1 = re.sub('[^A-Za-z ]','',str(train['question1'][i]))
                q1 = [word for word in q1 if len(word)>0]
                line_q1 = ' '.join(q1) + '_'
            
                #qid_2 = train['qid2'][i]
                q2 = reg.split(str(train['question2'][i]).lower())
                #q2 = re.sub('[^A-Za-z ]','',str(train['question2'][i]))
                q2 = [word for word in q2 if len(word)>0]
                line_q2 = ' '.join(q2)

                if is_training:
                    line = line_q1 + line_q2 + '_' + str(train['is_duplicate'][i]) + '\n'
                else:
                    line = line_q1 + line_q2 + '\n'
                fw.write(line)
            except:
                print(train['qid1'][i])
                print(train['qid2'][i])
    print("{} queries correctly processed.\n{} queries failed for encoding.".format(len(train)-err_encoding, err_encoding))
    print("{} seconds taken".format(time.time()-start_time))
    print("{} pieces of data processed per second".format(len(train)/(time.time()-start_time)))

# main

#process()
#process('test_x2.txt', './test.csv/test.csv', False)
to_csv()
