import time

from gensim.models.word2vec import Word2Vec
from sklearn.preprocessing import scale
import numpy as np
from sklearn.preprocessing import label_binarize
from sklearn import svm
from sklearn import tree
from sklearn.linear_model import SGDClassifier
from sklearn.multiclass import OneVsRestClassifier


def cleanText(corpus):

    corpus = [z.lower().replace('\n','').split() for z in corpus]
    return corpus
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

def OneVsRest(train_vec,train_tag,input_vec):

    start = time.clock()

    lr = OneVsRestClassifier(estimator=svm.SVC(random_state=0))

    lr.fit(train_vec,train_tag)

    pred = lr.predict(input_vec)

    end = time.clock()

    print '-------------------------------------'
    print 'OneVsRest Prediction Result'
    print '-------------------------------------'

    print 'Class:'+str(pred)
    print 'Time:'+str(end-start)+'s'

def SVC(train_vec,train_tag,input_vec):

    start = time.clock()

    lr = svm.SVC()

    lr.fit(train_vec,train_tag)

    pred = lr.predict(input_vec)

    end = time.clock()

    print '-------------------------------------'
    print 'SVC Prediction Result'
    print '-------------------------------------'

    print 'Class:'+str(pred)
    print 'Time:'+str(end-start)+'s'

def Tree(train_vec,train_tag,input_vec):

    start = time.clock()

    lr = tree.DecisionTreeClassifier()

    lr.fit(train_vec,train_tag)

    pred = lr.predict(input_vec)

    end = time.clock()

    print '-------------------------------------'
    print 'Tree Prediction Result'
    print '-------------------------------------'

    print 'Class:'+str(pred)
    print 'Time:'+str(end-start)+'s'

def SGD(train_vec,train_tag,input_vec):

    start = time.clock()

    lr = SGDClassifier(loss='log', penalty='l1')

    lr.fit(train_vec,train_tag)

    pred = lr.predict(input_vec)

    end = time.clock()

    print '-------------------------------------'
    print 'SGD Prediction Result'
    print '-------------------------------------'

    print 'Class:'+str(pred)
    print 'Time:'+str(end-start)+'s'

def TrainSetLoad():

    with open('./Corpus/qqnews_ent_2000','r') as infile:
        test1 = infile.readlines()

    with open('./Corpus/qqnews_finance_2000','r') as infile:
        test2 = infile.readlines()

    with open('./Corpus/qqnews_sport_2000', 'r') as infile:
        test3 = infile.readlines()

    return test1, test2, test3


def TrainingProcess(t1, t2 ,t3, number):

    tag_ent = ['ent' for i in range(number)]
    tag_finance = ['finance' for i in range(number)]
    tag_sport = ['sport' for i in range(number)]

    # Numpy work
    x_train_text =np.concatenate((t1, t2, t3))
    y_train_tag = np.concatenate((tag_ent, tag_finance, tag_sport))

    # clean Text
    x_train_clean_text = cleanText(x_train_text)

    # build vocab
    vocab = builVocab(300,x_train_clean_text)

    # get vector and scale
    vocab.train(x_train_clean_text)
    x_train_vec = scaleVec(vocab,x_train_clean_text,300)

    return x_train_vec, y_train_tag,vocab


def PredictInput(input):

    # get trained vector
    t1, t2, t3 = TrainSetLoad()
    train_vec, train_tag, vocab = TrainingProcess(t1,t2,t3,2000)

    # clean Text
    x_input_clean = cleanText(input)

    # get vector and scale
    vocab.train(x_input_clean)
    x_input_vec = scaleVec(vocab,x_input_clean,300)

    # predict the class
    # SVC(train_vec, train_tag, x_input_vec)
    #
    # OneVsRest(train_vec, train_tag, x_input_vec)
    #
    # Tree(train_vec, train_tag, x_input_vec)

    i = 0
    while i<100:
        SGD(train_vec, train_tag, x_input_vec)
        i +=1


if __name__ == "__main__":

    with open('./Corpus/qqnews_sport_2000', 'r') as infile:
        article = [infile.readline()]

    PredictInput(article)

