# coding: utf-8
from gensim.models import Doc2Vec
import gensim
import time
LabeledSentence = gensim.models.doc2vec.TaggedDocument
from sklearn.cross_validation import train_test_split
import numpy as np

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
            rawData.write(''.join(i[0])+'/'+''.join(i[1])+'\n')
    return labelized



def buildModel(size,x_train,x_test):
    model1 = Doc2Vec(size=size, window=10, min_count=5, workers=4,alpha=0.025, min_alpha=0.025,negative=5)
    all_Text = x_train+x_test
    model1.build_vocab(all_Text)

    for epoch in range(1):
        model1.train(x_train,total_examples=model1.corpus_count,epochs=model1.iter)
        model1.alpha -= 0.002  # decrease the learning rate
        model1.min_alpha = model1.alpha  # fix the learning rate, no decay

    for epoch in range(1):
        model1.train(x_test,total_examples=model1.corpus_count,epochs=model1.iter)
        model1.alpha -= 0.002  # decrease the learning rate
        model1.min_alpha = model1.alpha  # fix the learning rate, no decay

    # model1.train(x_train,total_examples=model1.corpus_count,epochs=model1.iter)

    model1.save('./model.doc2vec')

def load_raw(tag, sim, path):

    raw = []
    words = ''
    with open(path) as raw_data:
        for line in raw_data.readlines():
            line = line.strip('\n').split('/')
            raw.append(line)

    for article in raw:
        # print article
        if article[1] == tag:
            words = words + article[0]

    print words, sim




def similar(path, tag):

    model2 = Doc2Vec.load(path)

    test_text = ['《', '经济', '赚钱' '》', '危机' '金融', '复活', '天天', '地产']

    test_vec = model2.infer_vector(test_text)

    sims = model2.docvecs.most_similar([test_vec],topn=10)

    for tag, sim in sims:

        load_raw(tag, sim, './TRAIN')





if __name__ == "__main__":


    start = time.clock()
    t1, t2, t3 = loadinfor()
    x_train, x_test, y_train, y_test = numpyProcess(t1, t2, t3, 1000, 0.2)

    x_train = cleanText(x_train)
    x_test = cleanText(x_test)
    #
    # for i in x_train:
    #     print i
    # label the text
    x_train = labelizeReviews(x_train, 'TRAIN')
    x_test = labelizeReviews(x_test, 'TEST')


    # # build model and get vectors
    # size = 400
    # buildModel(size, x_train, x_test)
    end = time.clock()

    print 'Finished in '+ str(end-start)+'s'

    similar('./model.doc2vec','TRAIN_22')


