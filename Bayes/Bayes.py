# coding=utf-8
#from numpy import *
import numpy as np

def loadDataSet():
    postingList=[['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                 ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                 ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                 ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                 ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                 ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    classVec = [0,1,0,1,0,1]    #1 is abusive, 0 not
    return postingList,classVec

"""
把多个句子向量组成的矩阵，合并为词语set
"""
def createVocabList(dataSet):
    vocabSet = set([])
    for document in dataSet:
        vocabSet = vocabSet | set(document)
    return list(vocabSet)

"""
vocabList：['cute', 'love', 'help', 'garbage', 'quit', 'I', 'problems', 'is', 'park', 'stop', 'flea', 'dalmation', 'licks', 'food', 'not', 'him', 'buying', 'posting', 'has', 'worthless', 'ate', 'to', 'maybe', 'please', 'dog', 'how', 'stupid', 'so', 'take', 'mr', 'steak', 'my']
inputSet：词汇向量：['my', 'dog', 'has', 'flea', 'problems', 'help', 'please']
返回：给定词vocabList在给定文档inputSet中是否出现。出现则该位置标为1，否则为0。若某词不在给定词中，超出范围则打印报错语句。
[0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1]
"""
def setOfWords2Vec (vocabList, inputSet):
    returnVec = [0] * len(vocabList)
    for word in inputSet:
        if word in vocabList:
            # returnVec[vocabList.index(word)] = 1  词集模型，在一篇文档中出现就记一次。
            returnVec[vocabList.index(word)] += 1   #词袋模型，在一篇文档中出现几次记几次。

    return returnVec

"""
trainMatrix: 某个给定词汇表例如：[help，good]在n片文档中的是否出现矩阵：[[1,0],[0,1]]
trainCategory：每篇文档类别构成的向量。 [0, 0, 1, 0, 1...]
bayes: p(ci | w) = p(w | ci) * p(ci) / p(w)
"""
def trainNB0(trainMatrix, trainCategory):
    numTrainDocs = len(trainMatrix)
    numWords = len(trainMatrix[0])  #numWords 词汇表词汇个数

    pAbusive = sum(trainCategory)/float(numTrainDocs)   #trainCategory由0，1构成，加总即是1个个数，除以文档总数 = 侮辱性文档的概率

    p0Num = np.ones(numWords);    p1Num = np.ones(numWords) #和词汇表同个数列表，分别代表词汇表中各词出现在0，1两类文档中的次数。缺省设置为出现一次，避免出现概率为0的情况：拉普拉斯修正法
    p0Denom = 2.0;  p1Denom = 2.0   #0，1两类文档中词汇总数

    for i in range(numTrainDocs):
        if trainCategory[i] == 1:   #侮辱性文档
            p1Num += trainMatrix[i] #p1Num = [0,0]开始和每篇文档的出现次数向量叠加
            p1Denom += sum(trainMatrix[i])  #1两类文档中词汇数加总
        else:
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])

    p1Vect = np.log(p1Num / p1Denom)   #词汇表在1类文档中的出现概率。用log 把0.0x 的概率数转换为 -2.xxx的数，避免下溢出
    p0Vect = np.log(p0Num / p0Denom)

    return p0Vect, p1Vect, pAbusive

"""
vec2Classify：通过setOfWords2Vec的词向量。numpy 的数组。
p0Vec: 0类文档的概率向量 array([-2.56494936, -2.56494936, -2.56494936, ....])
p1Vec：1类文档的概率向量 array([-3.04452244, -3.04452244, -3.04452244, ....])
pClass1：1类文档的概率。
bayes: p(ci | w) = p(w | ci) * p(ci) / p(w)
"""
def classifyNB(vec2Classify, p0Vec, p1Vec, pClass1):
    """
    根据 p(ci | w) = p(w | ci) * p(ci) / p(w)；2者比较，分母相同，比较分子即可
    比较项：p(w | ci) * p(ci)

    根据：len(f(x)) 和 f(x)正相关，可取分子的len进行比较即可
    比较项：log(p(w | ci) * p(ci))

    根据：len(a*b) = len(a) + len(b)
    比较项：log(p(w | ci)）+  log（p(ci))；
    其中：log（p(ci)) = np.log(pClass1)；
    其中：log(p(w | ci)）= log（p(w1 | ci) * p(w2 | ci) * ... * p(wn | ci)）

    根据：len(a*b) = len(a) + len(b)
    比较项：log(p(w | ci) = log（p(w1 | ci)) + log（p(w2 | ci)) + ... log（p(wn | ci)) 即 trainNB0 的返回p0V， p1V

    vec2Classify 是一个 array（[1, 0, 1, 0]）；1表示出现
    vec2Classify * p1Vec 是2个向量对应元素相乘。因此不出现的词概率被置为0，把出现词的概率相加。

    最终：p1 = sum(vec2Classify * p1Vec) + np.log(pClass1)

    p(ci | w) = p(w | ci) * p(ci) / p(w) = p(ci)/p(w) * 连乘积（i = 1 -> d）p(wi|Ci)
    h(x) = argmax P(c|x) = argmax P(c) * 连乘积（i = 1 -> d）P(xi|Ci)
    """
    p1 = sum(vec2Classify * p1Vec) + np.log(pClass1)
    p0 = sum(vec2Classify * p0Vec) + np.log(1.0 - pClass1)


    if p1 > p0:
        return 1
    else:
        return 0

def testingNB():
    listOPosts, listClasses = loadDataSet()
    myVocabList = createVocabList((listOPosts))

    trainMat = [];

    for postinDoc in listOPosts:
        trainMat.append(setOfWords2Vec(myVocabList, postinDoc))

    p0V, p1V, pAb = trainNB0(trainMat, listClasses)

    #test1
    testEntry = ['love', 'my', 'dalmation']
    thisDoc = np.array(setOfWords2Vec(myVocabList, testEntry))
    print testEntry, 'classified as: ', classifyNB(thisDoc, p0V, p1V, pAb)

    # test2
    testEntry = ['stupid', 'garbage']
    thisDoc = np.array(setOfWords2Vec(myVocabList, testEntry))
    print testEntry, 'classified as: ', classifyNB(thisDoc, p0V, p1V, pAb)













