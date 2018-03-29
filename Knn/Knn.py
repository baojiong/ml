# coding=utf-8
from numpy import *
import operator
import matplotlib.pyplot as plt
from os import listdir
import os


def createDataSet():
    group = array([[1.0, 1.1], [1.0, 1.1], [0, 0], [0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels

def classify0 (inX, dataSet, labels, k):
    dataSetSize = dataSet.shape[0]
    diffMat = tile(inX, (dataSetSize, 1)) - dataSet
    sqDiffMat = diffMat ** 2
    sqDistance = sqDiffMat.sum(axis = 1)
    distance = sqDistance ** 0.5

    sortedDistIndicies = distance.argsort()

    classCount = {}

    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1

    sortedClassCount = sorted(classCount.iteritems(), key = operator.itemgetter(1), reverse = True)
    return sortedClassCount[0][0]

def file2matrix(filename):
    """
    从文件中读入训练数据，并存储为矩阵 文件格式：单行：1.0   23  56.1    A
    """
    fr = open(filename)
    numberOfLine = len(fr.readlines())

    returnMat = zeros((numberOfLine, 3))
    classLabelVector = []

    fr = open(filename)
    index = 0

    for line in fr.readlines():
        line = line.strip()
        listFromLine = line.split('\t')     #[1.0, 23, 56.1, A]
        returnMat[index, :] = listFromLine[0:3]
        classLabelVector.append(int(listFromLine[-1]))
        index += 1

    return returnMat, classLabelVector

def autoNorm(dataSet):
    """
    训练数据归一化
    norm(x) = (x - min)/(max - min)
    """
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals

    normDataSet = zeros(shape(dataSet))
    m = dataSet.shape[0]
    normDataSet = dataSet - tile(minVals, (m, 1))
    normDataSet = normDataSet/tile(ranges, (m, 1))

    return normDataSet, ranges, minVals #返回 归一化矩阵；范围列表；最小数列表；

def datingClassTest(filename):
    hoRatio = 0.10
    datingDataMat, datingLabels = file2matrix(filename)
    normMat, ranges, minVals = autoNorm(datingDataMat)
    m = normMat.shape[0]
    numTestVecs = int(m*hoRatio)

    errorCount = 0.0

    for i in range(numTestVecs):
        classifierResult = classify0(normMat[i, :], normMat[numTestVecs:m, :], datingLabels[numTestVecs:m], 3)
        print ("the classifier came back with: %d, the real answer is: %d, result is: %s" %(classifierResult,
                                                                                           datingLabels[i],
                                                                                           classifierResult == datingLabels[i]))
        if classifierResult != datingLabels[i]:
            errorCount += 1.0

    print ("the total error rate is %f" %(errorCount/float(numTestVecs)))
    print (errorCount)

def figureOut(filename):
    fig = plt.figure()
    ax = fig.add_subplot(111)

    datingDataMat, datingLables = file2matrix(filename)

    ax.scatter(datingDataMat[:, 1], datingDataMat[:, 2], 15.0*array(datingLables), 15.0*array(datingLables))
    plt.show()

def img2Victor(filename):
    returnVect = zeros((1, 1024))

    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVect[0, 32*i+j] = int(lineStr[j])

    return returnVect

def handwritingClassTest():
    hwLabels = []
    trainingFileList = listdir('Knn/trainingDigits')

    m = len(trainingFileList)

    trainingMat = zeros((m, 1024))

    for i in range(m):
        fileNameStr = trainingFileList[i]
        label = fileNameStr.split('.')[0]
        label = label.split('_')[0]
        hwLabels.append(label)

        trainingMat[i,:] = img2Victor('Knn/trainingDigits/%s' % fileNameStr)

    testFileList = listdir('Knn/testDigits')
    mTest = len(testFileList)

    errorCount = 0

    for i in range(mTest):
        testFileNameStr = testFileList[i]
        testLabel = testFileNameStr.split('.')[0]
        testLabel = testLabel.split('_')[0]

        testVictor = img2Victor('Knn/testDigits/%s' % testFileList[i])
        result = classify0(testVictor, trainingMat, hwLabels, 3)
        print("FileName = %s, Classify came back %s, the real result is %s. result is: %s" % (testFileNameStr, result, testLabel, result == testLabel))

        if result != testLabel:
            errorCount += 1.0

    print("Total number of error is %d" % errorCount)
    print("Error rate is %f" % (errorCount/float(mTest)))

if __name__== "__main__":
    handwritingClassTest()