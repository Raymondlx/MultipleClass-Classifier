# coding: utf-8
'''
Demo for three-class version of svc classifer
Chinese Language Text
Raymond
'''
import time
from sklearn.cross_validation import train_test_split
from gensim.models.word2vec import Word2Vec
from sklearn.preprocessing import scale
from sklearn import metrics
from sklearn import svm
from sklearn.linear_model import SGDClassifier
import numpy as np
from sklearn.preprocessing import label_binarize
from sklearn import svm
from sklearn import tree
from sklearn.multiclass import OneVsRestClassifier # Belongs to SVC


def calculate_result(actual, pred):

    m_accuracy = metrics.accuracy_score(actual,pred)
    # m_precision = metrics.precision_score(actual, pred, 'macro')
    # m_recall = metrics.recall_score(actual, pred, 'macro')
    m_f1 = metrics.f1_score(actual, pred, average='macro')

    # return m_accuracy, m_precision, m_recall, m_f1

    return m_accuracy, m_f1

def loadinfor():
    # load from local
    with open('./Corpus/qqnews_ent_200','r') as infile:
        test1 = infile.readlines()

    with open('./Corpus/qqnews_finance_200','r') as infile:
        test2 = infile.readlines()

    with open('./Corpus/qqnews_sport_200', 'r') as infile:
        test3 = infile.readlines()

    return test1, test2, test3

def numpyProcess(t1,t2,t3,number,test_size):


    tag_ent = ['ent' for i in range(number)]
    tag_finance = ['finance' for i in range(number)]
    tag_sport = ['sport' for i in range(number)]

    y = np.concatenate((tag_ent,tag_finance,tag_sport))
    # y = label_binarize(y, classes=[0, 1, 2])
    x_train, x_test, y_train, y_test = train_test_split(np.concatenate((t1,t2,t3)), y, test_size=test_size)
    count1 = 0
    count2 = 0
    for i in x_train:
        count1+=1
    for n in x_test:
        count2+=2
    print count1,count2

    return x_train, x_test, y_train, y_test

def cleanText(corpus):

    corpus = [z.lower().replace('\n','').split() for z in corpus]
    return corpus
def covertTag(tag):

    tag = [z for z in tag]
    return tag

def builVocab(dim,x_train):

    w2v = Word2Vec(size=dim, min_count=1)# test
    w2v.build_vocab(x_train)
    return w2v

def buildWordVector(w2v,text,size):
    # get the average value of each word
    vec = np.zeros(size).reshape((1, size))
    count = 0.
    for word in text:
        try:
            vec += w2v[word].reshape((1, size))
            count += 1.
        except KeyError:
            continue
    if count != 0:
        vec /= count
    return vec

def scaleVec(w2v,trainSet,dim):
    train_vecs = np.concatenate([buildWordVector(w2v,z, dim) for z in trainSet])
    train_vecs = scale(train_vecs)
    # [[0.25148279 - 0.3889378   0.63863381..., 0.16895451  1.58589705
    #  0.05162426]] as a sentence contains much words
    return train_vecs


def OVRClassifier(train_vecs,test_vecs,y_test,y_train):

    start = time.clock()

    lr = OneVsRestClassifier(estimator=svm.SVC(random_state=0))

    # map trained vectors with its tags
    lr.fit(train_vecs, y_train)

    predict =lr.predict(test_vecs)

    end = time.clock()

    Time = end - start

    m_accuracy,  m_f1 = calculate_result(y_test,predict)

    return m_accuracy,  m_f1, Time

def DecisionTreeClassifier(train_vecs,test_vecs,y_test,y_train):
    start = time.clock()

    lr = tree.DecisionTreeClassifier()

    lr.fit(train_vecs,y_train)

    predict = lr.predict(test_vecs)

    end = time.clock()

    Time = end - start

    m_accuracy,  m_f1 = calculate_result(y_test, predict)

    return m_accuracy,  m_f1, Time

def SGDclassifier(train_vecs,test_vecs,y_test,y_train):

    start = time.clock()

    lr = SGDClassifier(loss='log', penalty='l1')

    lr.fit(train_vecs, y_train)


    predict = lr.predict(test_vecs)

    end = time.clock()

    Time = end - start

    m_accuracy, m_f1 = calculate_result(y_test, predict)

    return m_accuracy, m_f1, Time

def SVMclassifier(train_vecs,test_vecs,y_test,y_train):

    start = time.clock()

    lr = svm.SVC()

    lr.fit(train_vecs, y_train)

    predict = lr.predict(test_vecs)

    end = time.clock()

    Time = end - start

    m_accuracy, m_f1 = calculate_result(y_test, predict)

    return m_accuracy,m_f1, Time



if __name__ == "__main__":
    # load infor
    test_size = 0.2
    t1, t2, t3 = loadinfor()
    start = time.clock()
    text_train, text_test, tag_train, tag_test = numpyProcess(t1, t2, t3, 200,test_size)
    #print text_train
    print "-------------------------------------------"
    #print tag_train
    print "-------------------------------------------"
     # print text_test
     # print tag_test

    # cleanText
    text_train = cleanText(text_train)
    text_test = cleanText(text_test)


    # build vocab
    vocab = builVocab(300,text_train)

    # training data and scale
    vocab.train(text_train)
    train_vec = scaleVec(vocab,text_train,300)
    vocab.train(text_test)
    test_vec = scaleVec(vocab,text_test,300)



    # add-on
    # svcClassifier(train_vec, test_vec, tag_test, tag_train)
    # DecisionTreeClassifier(train_vec, test_vec, tag_test, tag_train)
    # SGDclassifier(train_vec, test_vec, tag_test, tag_train)


    with open('./Corpus/qqnews_ent_200', 'r') as infile:
        test1 = [infile.readline()]




    end = time.clock()
    print "The running time is: %f s" %(end-start)







    # print text_train
    # print tag_train
    # print text_test
    # print tag_test

    # ---------------------------------------------
    # # load file and clean it
    # t1, t2 = loadinfor()
    # x_train, x_test, y_train, y_test = numpyProcess(t1,t2)
    # x_train = cleanText(x_train)
    # x_test = cleanText(x_test)
    # # build vocab
    # vocab = builVocab(300,x_train)
    # # training data
    # vocab.train(x_train)
    # vocab.train(x_test)
    # # scale data
    # train_vec = scaleVec(vocab,x_train,300)
    # test_vec = scaleVec(vocab,x_test,300)
    # # add on machine-learning algorithm
    # sgdClassifier(test_vec,test_vec,y_test,y_train)

    # a = ['1','2']
    # b = ['3','4']
    # c = np.concatenate((a,b))
    # print c
    # print vocab
    # print x_train
    # print x_test
    # print y_train
    # print y_test