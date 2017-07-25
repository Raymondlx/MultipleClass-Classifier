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
import numpy as np
from sklearn.preprocessing import label_binarize
from sklearn import svm
from sklearn.multiclass import OneVsRestClassifier # Belongs to SVC



def loadinfor():
    # load from local
    with open('./Corpus/qqnews_ent_2000','r') as infile:
        test1 = infile.readlines()

    with open('./Corpus/qqnews_finance_2000','r') as infile:
        test2 = infile.readlines()

    with open('./Corpus/qqnews_sport_2000', 'r') as infile:
        test3 = infile.readlines()

    return test1, test2, test3

def numpyProcess(t1,t2,t3,number):


    tag_ent = ['ent' for i in range(number)]
    tag_finance = ['finance' for i in range(number)]
    tag_sport = ['sport' for i in range(number)]

    y = np.concatenate((tag_ent,tag_finance,tag_sport))
    # y = label_binarize(y, classes=[0, 1, 2])
    x_train, x_test, y_train, y_test = train_test_split(np.concatenate((t1,t2,t3)), y, test_size=0.5)

    # # use 1 and 0 to build the array,[1,1,...,0,0]
    # y = np.concatenate((np.ones(number_pos),np.zeros(number_neg)))
    # # use tran_split to randomly select Train and Test
    # x_train, x_test, y_train, y_test = train_test_split(np.concatenate((pos, neg)), y, test_size=0.5)
    # print x_train
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

def calculate_result(actual,pred):
    m_precision = metrics.precision_score(actual,pred)
    m_recall = metrics.recall_score(actual,pred)
    print 'Predict info:'
    print 'Precision:{0:.3f}'.format(m_precision)
    print 'Recall:{0:0.3f}'.format(m_recall)
    print 'F1-score:{0:.3f}'.format(metrics.f1_score(actual,pred))

def sgdClassifier(train_vecs,test_vecs,y_test,y_train):
    lr = OneVsRestClassifier(estimator=svm.SVC(random_state=0))

    # map trained vectors with its tags
    lr.fit(train_vecs, y_train)
    # predict untagged context
    print '-------------------------------------'
    print 'Prediction Result'
    print '-------------------------------------'
    # print text_test

    pred = lr.predict(test_vecs)
    # pred2 = np.array(pred)
    # output = {i:[pred[i],text_test[i]] for i in range(len(pred))}

   # print y_test
   # print pred
    # print output

    print 'SGDClassifier Test Accuracy: %.2f' % lr.score(test_vecs, y_test)
    # calculate_result(y_test,pred)


if __name__ == "__main__":
    # load infor
    t1, t2, t3 = loadinfor()
    start = time.clock()
    text_train, text_test, tag_train, tag_test = numpyProcess(t1, t2, t3, 2000)
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
    sgdClassifier(train_vec, test_vec, tag_test, tag_train)

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