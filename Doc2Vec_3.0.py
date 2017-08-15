# coding: utf-8
import multiprocessing
from gensim.models import Doc2Vec
from sklearn import metrics
import gensim
import time
LabeledSentence = gensim.models.doc2vec.TaggedDocument
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import SGDClassifier
import numpy as np
from random import shuffle


def loadinfor():
    # load from local
    with open('./Corpus/qqnews_ent_1000','r') as infile:
        test1 = infile.readlines()

    with open('./Corpus/qqnews_finance_1000','r') as infile:
        test2 = infile.readlines()

    with open('./Corpus/qqnews_sport_1000', 'r') as infile:
        test3 = infile.readlines()

    return test1, test2, test3

def numpyProcess(t1,t2,t3,number,test_size):


    tag_ent = ['ent' for i in range(number)]
    tag_finance = ['finance' for i in range(number)]
    tag_sport = ['sport' for i in range(number)]

    y = np.concatenate((tag_ent,tag_finance,tag_sport))
    x_train, x_test, y_train, y_test = train_test_split(np.concatenate((t1,t2,t3)), y, test_size=test_size)

    return x_train, x_test, y_train, y_test

# Do some very minor text preprocessing
def cleanText(corpus):

    corpus = [z.lower().replace('\n','').split() for z in corpus]
    return corpus

def labelizeReviews(reviews, label_type):
    labelized = []
    for i,v in enumerate(reviews):
        label = '%s_%s'%(label_type,i)
        for article in v:
            labelized.append(LabeledSentence(words = article.split(),tags = [label]))
    with open('./'+str(label_type),'a') as rawData:
        for i in labelized:
            rawData.write(''.join(i[0])+'(---)'+''.join(i[1])+'\n')
    return labelized

def getVecs_train(model, tag, dim):
    index = []
    current=''
    previous=''
    count = 0
    vecs = []
    for z in tag:
        current = z
        if current != previous and (current not in index):
            vecs.append(np.array(model.docvecs[z]).reshape((1, dim)))
            index.append(current)
            count += 1
        previous = z
    # print '****',count
    return np.concatenate(vecs)

def getVecs_test(model, test, dim):
    vecs = [np.array(model.infer_vector(z)).reshape((1,dim)) for z in test]
    return np.concatenate(vecs)

def load_model(path_model,path_raw, x_test, dim):

    print 'Start loading model...'
    start = time.clock()
    tag = []
    count2 = 0
    model = Doc2Vec.load(path_model)
    # model = path_model
    with open (path_raw) as raw_data:
        for line in raw_data:
            line = line.strip('\n').split('(---)')
            tag.append(line[1])
    for i in tag:
        count2 += 1
    # print count2,'--------------'

    train_vec = getVecs_train(model, tag, dim)
    test_vec = getVecs_test(model, x_test,dim)
    end = time.clock()

    print 'Getting vecs in '+str(end-start)+' s...'
    return train_vec, test_vec


def build_model(x_train, dim):
    cores = multiprocessing.cpu_count()
    print cores
    start = time.clock()
    # dm-model
    model = Doc2Vec(size=dim, window=10, min_count=5, workers=cores, alpha=0.025, min_alpha=0.025, negative=5)

    model.build_vocab(x_train)

    # to get a better result by training more
    for epoch in range(10):
        # random improves
        begin = time.clock()
        print 'Start processing shuffling for '+str(epoch+1)+' times...'
        shuffle(x_train)
        model.train(x_train,total_examples=len(x_train),epochs=1)
        model.alpha -= 0.002  # decrease the learning rate
        model.min_alpha = model.alpha  # fix the learning rate, no decay
        finish = time.clock()
        print 'Finish processing epoch '+str(epoch+1)+' in '+str(finish-begin)+' s ...'

    model.save('./model3.0')

    end = time.clock()

    print 'Model Successfully saved...'
    print 'Building Model in '+str(end-start)+'s...'

def calculate_result(actual, pred):

    m_accuracy = metrics.accuracy_score(actual,pred)
    m_f1 = metrics.f1_score(actual, pred, average='macro')

    return m_accuracy, m_f1

def SGDclassifier(train_vecs,test_vecs,y_test,y_train):

    start = time.clock()

    lr = SGDClassifier(loss='log', penalty='l1')

    lr.fit(train_vecs, y_train)

    predict = lr.predict(test_vecs)

    end = time.clock()

    Time = end - start

    m_accuracy, m_f1 = calculate_result(y_test, predict)

    print m_accuracy, m_f1, Time

if __name__ == "__main__":
    article_number_each = 1000
    test_size = 0.2
    dim = 300
    # load data
    t1, t2, t3 = loadinfor()

    # get train_set and test_set
    x_train, x_test, y_train, y_test = numpyProcess(t1, t2, t3, article_number_each, test_size)

    # clean Text
    x_train = cleanText(x_train)
    x_test = cleanText(x_test)

    # Tag on Text
    x_train = labelizeReviews(x_train, 'TRAIN')

    # build model
    model = build_model(x_train, dim)

    # load model
    train_vec, test_vec = load_model('./model3.0', './TRAIN', x_test, dim)

    # classifier
    SGDclassifier(train_vec, test_vec, y_test, y_train)

