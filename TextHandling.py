import os

def multiText(path):

    pathDir = os.listdir(path)
    for allDir in pathDir:
        child = os.path.join('%s/%s' % (path, allDir))
        print child

def getLines(path):

    number = 0
    pathDir = os.listdir(path)
    for doc in pathDir:
        with open(path+'/'+doc) as file:
            for line in file:
                number += line.count('\n')
        print 'In '+doc+', there are', number, 'lines'
        number = 0

# read text from a directory, and do some modifications
def segment(path):

    pathDir =os.listdir(path)
    for doc in pathDir:
        with open(path+'/'+doc,'r') as file:
            for line in file:
                line = line.replace('.', '\n')
                with open(path+'/'+doc + '.temp', 'a') as writeIn:
                        writeIn.write(line)

def shapeData(path,size):

    counter = 0
    pathDir = os.listdir(path)
    for doc in pathDir:
        with open(path + '/' + doc, 'r') as file:
            for line in file:
                if counter <=size :
                    with open(path + '/' + doc + '.clean', 'a') as writeIn:
                        writeIn.write(line)
                        counter += 1
            counter += 1


path = './data'
# segment(path)
# shapeData(path, 40)
getLines(path)


# path = './color'
# print getLines(path)
# segment(path)
# print getLines(path+'.clean')