# encoding=utf-8
import json
# by doing this, could write in Chinese words
import sys
reload(sys)
sys.setdefaultencoding('utf-8')
import os
import jieba

# Loading all the files under the same path
def LoadMulti(path,max):

    pathDir = os.listdir(path)
    for doc in pathDir:
        LoadSingle(path,doc,max)

def LoadSingle(path,doc,max):

    count = 0
    with open(path+'/'+doc,'r') as jsonFile:
        # each line is a object for json
        for line in jsonFile.readlines():
            data = json.loads(line)
            if count<max :
                if SegWord(data["Content"],path,doc,max) == True:
                    count += 1

            # print data["Content"]

def SegWord(sentence,path,doc,max):
    if sentence != '\n' and sentence.strip()!="":
        seg_list = jieba.cut(sentence)
        list =  str("\t".join(seg_list))
        newPath = os.path.join('./','Corpus')
        if not os.path.isdir(newPath):
            os.mkdir(newPath)
        with open('./Corpus/'+doc.strip('.json')+'_'+str(max),'a') as OutputFile:
            OutputFile.write(list+'\n')
        return True
    else:
        return False

    # seg_list = jieba.cut("我来到北京清华大学", cut_all=True)
    # print "Full Mode:", "\t".join(seg_list)  # 全模式
    # seg_list = jieba.cut("我来到北京清华大学", cut_all=False)
    # print "Default Mode:", "/ ".join(seg_list)  # 精确模式
    # seg_list = jieba.cut("他来到了网易杭研大厦")  # 默认是精确模式
    # print ", ".join(seg_list)
    # seg_list = jieba.cut_for_search("小明硕士毕业于中国科学院计算所，后在日本京都大学深造")  # 搜索引擎模式
    # print ", ".join(seg_list)

if __name__ == "__main__":
    path = './qqnews_ent'
    max = 1000
    LoadMulti(path,max)
    # SegWord()
