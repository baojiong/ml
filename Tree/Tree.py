# coding=utf-8
from math import log
import operator

def createDataSet():
    dataSet = [[1, 1, 'yes'],
               [1, 1, 'yes'],
               [1, 0, 'no'],
               [0, 1, 'no'],
               [0, 1, 'no']]

    labels = ['no surfacing', 'flippers']
    return dataSet, labels

"""
主成分分析
对类别list [A,A,B,A,B,B,B] 进行投票：{A：3，B：4 ...}
返回字典和第一个类别名
"""
def majorityCnt1(classList):
    classCount = {}
    for vote in classList:
        if vote not in classCount.keys():
            classCount[vote] = 0
        classCount[vote] += 1

    """
        if vote not in classCounts.keys():
            classCounts[vote] = 0
        classCounts[vote] += 1

        等价于：
        classCounts[vote] = classCounts.get(vote, 0) + 1
    """

    sortedClassCount = sorted(classCount.iteritems(), key = operator.itemgetter(1), reverse = True)

    """
    对类别list 进行投票：{A：3，B：4 ...}
    返回第一个键值对的键名即得票最高的类别
    """
    return sortedClassCount[0][0], classCount

"""
 计算一个DataSet的香浓熵 = -Sigma p(k)log2p(k), 所有分类的熵加总
"""
def calcShannonEnt(dataSet):
    numEntries = len(dataSet)

    classList = [example[-1] for example in dataSet]
    majorClass, classCounts  = majorityCnt1(classList)

    shannonEnt = 0.0

    for key in classCounts:
        prob = float(classCounts[key]) / numEntries
        shannonEnt -= prob * log(prob, 2)

    return shannonEnt

"""
 按照DataSet中向量的第axis维按 axis = value 划分向量，并把axis维限量去除。
 用于遍历按照一个特征值，所有的取值划分后的各个DataSet，计算总的香农熵和信息增益。
"""
def splitDataSet(dataSet, axis, value):
    retDataSet = []
    for featVec in dataSet:
        if featVec[axis] == value:
            """
            reduceFeatVec = 每个向量 从 0 到 axis 个元素 + axis 到最末尾 个元素，即符合条件的向量都被移除了 axis 位置的特征值。
            """
            reduceFeatVec = featVec[:axis]
            reduceFeatVec.extend(featVec[axis + 1:])
            retDataSet.append(reduceFeatVec)
    return retDataSet

"""
 按照DataSet中向量的第axis划分后的信息增益，未搞清楚划分后熵的计算，书里采用2。
"""
def calcInfoGain(dataSet, axis):
    baseEnt = calcShannonEnt(dataSet)

    featList = [example[axis] for example in dataSet]
    valueSet = set(featList)

    newEnt = 0.0
    for value in valueSet:
        newEnt += calcShannonEnt(splitDataSet(dataSet, axis, value))

    return baseEnt - newEnt

def calcInfoGain2(dataSet, axis):
    baseEnt = calcShannonEnt(dataSet)

    featList = [example[axis] for example in dataSet]
    valueSet = set(featList)

    newEnt = 0.0
    for value in valueSet:
        subDataSet = splitDataSet(dataSet, axis, value)
        prob = len(subDataSet) / float(len(dataSet))
        newEnt += prob * log(prob, 2)

    return baseEnt - newEnt

def chooseBestFeatureToSplit(dataSet):
    numFeatures = len(dataSet[0]) - 1
    baseEntropy = calcShannonEnt(dataSet)
    bestInfoGain = 0.0
    bestFeature = -1

    for i in range(numFeatures):
        """
        dataSet = [[1,2],[3,4],[5,6]]
        [example[0] for example in dataSet]
        
        [1, 3, 5]
        """
        featList = [example[i] for example in dataSet]
        uniqueVales = set(featList)
        newEntropy = 0.0

        """
        实际上的划分，需要按某个特征值，按所有可取值进行划分，把子集的熵加总。划分要进行到某个子集所有向量的类别一致，则成为一个叶子。
        Value 必须是离散值。
        """
        for value in uniqueVales:
            subDataSet = splitDataSet(dataSet, i, value)
            prob = len(subDataSet)/float(len(dataSet))
            newEntropy += prob * log(prob, 2)
            print newEntropy

        infoGain = baseEntropy - newEntropy
        print "baseEnt = %f, newEnt = %f, infoGain = %f" % (baseEntropy, newEntropy, infoGain)

        if infoGain > bestInfoGain:
            bestInfoGain = infoGain
            bestFeature = i

    return bestFeature

"""
主成分分析
对类别list [A,A,B,A,B,B,B] 进行投票：{A：3，B：4 ...}
返回第一个键值对的键名即得票最高的类别
"""
def majorityCnt(classList):
    classCount = {}
    for vote in classList:
        if vote not in classCount.keys():
            classCount[vote] = 0
        classCount[vote] += 1

    sortedClassCount = sorted(classCount.iteritems(), key = operator.itemgetter(1), reverse = True)

    """
    对类别list 进行投票：{A：3，B：4 ...}
    返回第一个键值对的键名即得票最高的类别
    """
    return sortedClassCount[0][0]


def createTree(dataSet, labels):
    classList = [example[-1] for example in dataSet]

    """
    终止条件
    """
    if classList.count(classList[0]) == len(classList): #count - 计算第一个元素个数是否等于全部个数，若等于表示类别已经完全一致
        return classList[0] #返回类别值
    if len(dataSet[0]) == 1: #特征已全部抽取分裂完毕，只剩类别列了，但仍不能完全划分成仅包含唯一类别的分类
        return majorityCnt(classList) #在此情况下，挑选投票数最高的类别返回

    bestFeat = chooseBestFeatureToSplit(dataSet)
    bestFeatLabel = labels[bestFeat]

    myTree = {bestFeatLabel:{}} #任意一棵子树的根
    del(labels[bestFeat])
    featValues = [example[bestFeat] for example in dataSet] #本次分裂特征列
    uniqueVals = set(featValues)

    for value in uniqueVals:
        subLabels = labels[:]
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet, bestFeat, value), subLabels)

    return myTree

def classify(inputTree, featLabels, testVec):   # {'no surfacing': {0: 'no', 1: {'flippers': {0: {'head': {0: 'no', 1: 'yes'}}, 1: 'no'}}}};
                                                # ['no surfacing', 'flippers', 'head']
                                                # [1, 1, 0],

    firstStr = inputTree.keys()[0]  # 'no surfacing'
    secondDict = inputTree[firstStr]  # {0: 'no', 1: {'flippers': {0: {'head': {0: 'no', 1: 'yes'}}, 1: 'no'}}}

    featIndex = featLabels.index(firstStr)  # List.index(obj): 返回根节点的index

    for key in secondDict.keys():  # 遍历树的生长取值，首次是[0, 1]
        if testVec[featIndex] == key:
            if type(secondDict[key]).__name__ == 'dict':  # 基线条件：如果该分支是棵树的话，递归调用。
                classLabel = classify(secondDict[key], featLabels, testVec)
            else:  # 停止条件：如果该分支不是树的话，返回结果：该键的值
                classLabel = secondDict[key]

    return classLabel


def storeTree(inputTree, filename):
    import pickle
    fw = open(filename, 'w')
    pickle.dump(inputTree, fw)
    fw.close()

def grabTree(filename):
    import pickle
    fr = open(filename)
    return pickle.load(fr)


