# encoding=utf-8
import json
# by doing this, could write in Chinese words
import sys
reload(sys)
sys.setdefaultencoding('utf-8')
import os
import jieba

def Load(path):

    with open(path+'/qqnews_finance.json','r') as jsonFile:
        # each line is a object for json
        for line in jsonFile.readlines():
            data = json.loads(line)
            SegWord(data["Content"],path)

            # print data["Content"]

def SegWord(sentence,path):
    seg_list = jieba.cut(sentence)
    list =  str("\t".join(seg_list))

    with open(path+'/qqnews_finance','a') as OutputFile:
         OutputFile.write(list)

    # seg_list = jieba.cut("我来到北京清华大学", cut_all=True)
    # print "Full Mode:", "\t".join(seg_list)  # 全模式
    # seg_list = jieba.cut("我来到北京清华大学", cut_all=False)
    # print "Default Mode:", "/ ".join(seg_list)  # 精确模式
    # seg_list = jieba.cut("他来到了网易杭研大厦")  # 默认是精确模式
    # print ", ".join(seg_list)
    # seg_list = jieba.cut_for_search("小明硕士毕业于中国科学院计算所，后在日本京都大学深造")  # 搜索引擎模式
    # print ", ".join(seg_list)

if __name__ == "__main__":
    path = './qqnews_ent_finance_sports/qqnews_ent_finance_sports'
    Load(path)
    # SegWord()
