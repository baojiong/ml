# coding=utf-8
import Bayes as bayes
import numpy as np

"""
把文本拆成单词向量
"""
def textParse(bigString):
    import re
    listOfToken = re.split(r'\W*', bigString)
    return [tok.lower() for tok in listOfToken if len(tok) > 2]

def spamTest():
    docList = []; classList = []; fullText = []

    for i in range(1, 26):
        wordList = textParse(open('email/spam/%d.txt' % i).read())
        docList.append(wordList)
        fullText.extend(docList)
        classList.append(1)

        wordList = textParse(open('email/ham/%d.txt' % i).read())
        docList.append(wordList)
        fullText.extend(docList)
        classList.append(0)

    vocabList = bayes.createVocabList(docList)

    """
    trainingSet =  [1, 49]
    生成10个50以内的随机数，加入testSet
    从trainingSet中删掉这些数。
    结果就是把【1...49]，1分为2，10个作为 testSet， 其他作为 trainingSet
    trainingSet = [0, 1, 2, 4, 5, 6, 8, 9, 11, 12, 13, 14, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 32, 33, 35, 38, 39, 41, 43, 44, 45, 46, 47, 48, 49]
    testSet = [36, 3, 40, 31, 10, 42, 7, 37, 15, 34]
    """
    trainingSet = range(50); testSet = []
    for i in range(10):
        randIndex = int(np.random.uniform(0, len(trainingSet)))
        testSet.append(trainingSet[randIndex])
        del(trainingSet[randIndex])

    trainMat = []; trainClasses = []
    for docIndex in trainingSet:
        trainMat.append(bayes.setOfWords2Vec(vocabList, docList[docIndex]))
        trainClasses.append(classList[docIndex])

    p0V, p1V, pSpam = bayes.trainNB0(np.array(trainMat), np.array(trainClasses))
    errorCount = 0

    for docIndex in testSet:
        wordVector = bayes.setOfWords2Vec(vocabList, docList[docIndex])
        if bayes.classifyNB(np.array(wordVector), p0V, p1V, pSpam) != classList[docIndex]:
            errorCount += 1

    print 'the error rate is: ', float(errorCount)/len(testSet)