
import TripleClassifer
# import JsonLoad
import time
import os


# Load the data by different size
# def AutoSeg(path,max):
#
#     number = 200
#     while number <= max:
#         JsonLoad.LoadMulti(path,number)
#         number += 200

def AutoAnalysisData(vec_dim,test_size,range):

    number = 200

    # load the files with different size
    while number <= range:

        with open('./Corpus/qqnews_ent_' + str(number), 'r') as infile:
            test1 = infile.readlines()
        with open('./Corpus/qqnews_finance_' + str(number), 'r') as infile:
            test2 = infile.readlines()
        with open('./Corpus/qqnews_sport_' + str(number), 'r') as infile:
            test3 = infile.readlines()

        # recording time
        start = time.clock()

        # Tag on labels
        text_train, text_test, tag_train, tag_test = TripleClassifer.numpyProcess(test1, test2, test3, number, test_size)

        # cleanText
        text_train = TripleClassifer.cleanText(text_train)
        text_test = TripleClassifer.cleanText(text_test)

        # build vocab
        vocab = TripleClassifer.builVocab(vec_dim, text_train)

        # training data and scale
        vocab.train(text_train,total_examples=vocab.corpus_count)
        train_vec = TripleClassifer.scaleVec(vocab, text_train, vec_dim)
        vocab.train(text_test,total_examples=vocab.corpus_count)
        test_vec = TripleClassifer.scaleVec(vocab, text_test, vec_dim)

        # apply svc classier
        result = TripleClassifer.sgdClassifier(train_vec, test_vec, tag_test, tag_train)

        # ends here
        end = time.clock()

        with open('./Diary/SVC/DataRange_'+str(range)+'_TestSize_'+str(test_size)+'_Dim_'+str(vec_dim),'a') as output:
            output.write('-------------------------------------------------\n')
            output.write('Data Amount:'+str(number)+'\n')
            output.write('Test Size:'+str(test_size)+'\n')
            output.write('Vector Dim:'+str(vec_dim)+'\n')
            output.write('SVC Test Accuracy:'+str(result)+'\n')
            output.write('Time Efficiency:'+str('%.3f'%(end-start))+'s\n')
            output.write('-------------------------------------------------\n\n')

        number += 200

def AutoAnalysisDataAvrg(classifier,vec_dim,test_size,range,averageNum):
    number = 200

    # load the files with different size
    while number <= range:

        count = 1
        sumAccuracy = 0
        # sumPrecision = 0
        # sumRecall = 0
        sumF1 = 0
        sumTime = 0
        sumTrainTime = 0
        while count <= averageNum:
           with open('./Corpus/qqnews_ent_' + str(number), 'r') as infile:
            test1 = infile.readlines()
           with open('./Corpus/qqnews_finance_' + str(number), 'r') as infile:
            test2 = infile.readlines()
           with open('./Corpus/qqnews_sport_' + str(number), 'r') as infile:
            test3 = infile.readlines()

        # recording time
           startTrain = time.clock()

        # Tag on labels
           text_train, text_test, tag_train, tag_test = TripleClassifer.numpyProcess(test1, test2, test3, number,
                                                                                  test_size)

        # cleanText
           text_train = TripleClassifer.cleanText(text_train)
           text_test = TripleClassifer.cleanText(text_test)

        # build vocab
           vocab = TripleClassifer.builVocab(vec_dim, text_train)

        # training data and scale (epochs = vocab.iter)
           vocab.train(text_train,total_examples=vocab.corpus_count)
           train_vec = TripleClassifer.scaleVec(vocab, text_train, vec_dim)
           vocab.train(text_test,total_examples=vocab.corpus_count)
           test_vec = TripleClassifer.scaleVec(vocab, text_test, vec_dim)

           endTrain = time.clock()-startTrain

           sumTrainTime += endTrain

        # apply svc/tree classier
           if classifier == 'ovr':
               result_accuracy, result_f1, result_time = \
                   TripleClassifer.OVRClassifier(train_vec, test_vec, tag_test, tag_train)
               sumAccuracy += result_accuracy
               # sumPrecision += result_precision
               # sumRecall += result_recall
               sumF1 += result_f1
               sumTime += result_time

           if classifier == 'tree':
               result_accuracy, result_f1, result_time = \
                   TripleClassifer.DecisionTreeClassifier(train_vec, test_vec, tag_test, tag_train)
               sumAccuracy += result_accuracy
               # sumPrecision += result_precision
               # sumRecall += result_recall
               sumF1 += result_f1
               sumTime += result_time

           if classifier == 'svm':
               result_accuracy, result_f1, result_time = \
                   TripleClassifer.SVMclassifier(train_vec, test_vec, tag_test, tag_train)
               sumAccuracy += result_accuracy
               # sumPrecision += result_precision
               # sumRecall += result_recall
               sumF1 += result_f1
               sumTime += result_time

           if classifier == 'sgd':
               result_accuracy,result_f1, result_time = \
                   TripleClassifer.SGDclassifier(train_vec, test_vec, tag_test, tag_train)
               sumAccuracy += result_accuracy
               # sumPrecision += result_precision
               # sumRecall += result_recall
               sumF1 += result_f1
               sumTime += result_time

        # ends here
           count += 1

        # calculate the average value
        newPath = os.path.join('./Diary/', str(classifier))
        if not os.path.isdir(newPath):
            os.mkdir(newPath)
        with open('./Diary/'+str(classifier)+'/DataRange_' + str(range) + '_TestSize_' + str(test_size) + '_Dim_' + str(vec_dim),
                  'a') as output:
            output.write('-------------------------------------------------\n')
            output.write('Data Amount:' + str(number) + '\n')
            output.write('Test Size:' + str(test_size) + '\n')
            output.write('Vector Dim:' + str(vec_dim) + '\n')
            output.write('Accuracy:' + str('%.3f' %(sumAccuracy/averageNum)) + '\n')
            # output.write('Precision Score:' + str('%.3f' %(sumPrecision / averageNum)) + '\n')
            # output.write('Recall Score:' + str('%.3f' %(sumRecall / averageNum)) + '\n')
            output.write('F1 Score:' + str('%.3f' %(sumF1 / averageNum)) + '\n')
            output.write('Classifier Time Efficiency:' + str('%.3f' % (sumTime/averageNum)) + 's\n')
            output.write('Word2Vec Time Efficiency:' + str('%.3f' % (sumTrainTime/averageNum)) + 's\n')
            output.write('-------------------------------------------------\n\n')

        number += 200


def AutoAnalysisTestSize(classifier,vec_dim,test_range,data_range,averageNum):

    start = time.clock()

    testSize = 0.1

    while testSize < test_range:
        print 'Start Processing ' + str(classifier) + ' with testSize ' + str(testSize) + '............'
        AutoAnalysisDataAvrg(classifier,vec_dim,testSize,data_range,averageNum)
        print 'Finish Processing ' + str(classifier) + ' with testSize ' + str(testSize) + '............'
        testSize += 0.1

    end = time.clock()

    print '**Finished Processing All'+str(classifier)+'..............'
    print '**Using'+str('%.3f' %(end))+'s\n'



if __name__ == "__main__":

    path = './qqnews_ent'
    # max = 2000
    # AutoSeg(path,max)

    # AutoAnalysisData(300,0.4,2000)
    # AutoAnalysisDataAvrg('tree',300,0.4,2000,5)
    # AutoAnalysisDataAvrg('sgd',300,0.4,2000,5)
    # AutoAnalysisDataAvrg('svm',300,0.4,2000,5)
    # AutoAnalysisTestSize(300,0.5,2000,5)


    # For Cloud Server
    AutoAnalysisTestSize('sgd', 300, 0.6, 2000, 10)
    AutoAnalysisTestSize('ovr', 300, 0.6, 2000, 10)
    AutoAnalysisTestSize('tree', 300, 0.6, 2000, 10)
    AutoAnalysisTestSize('svm', 300, 0.6, 2000, 10)



