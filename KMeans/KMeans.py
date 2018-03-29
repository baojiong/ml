# coding=utf-8
import numpy as np

def loadDataSet(fileName):
    dataMat = []

    fr = open(fileName)
    for line in fr.readlines():
        curLine = line.strip().split('\t')
        fltLine = map(float, curLine)
        dataMat.append(fltLine)

    return dataMat

def distEclud(vecA, vecB):
    return np.sqrt(np.sum(np.power(vecA - vecB, 2)))

def randCent(dataSet, k):
    n = np.shape(dataSet)[1]
    centroids = np.mat(np.zeros((k, n)))    #k个点。n维向量可以看成一个n维的点，一个向量就是一个点。

    """
    k个随机质心要在边界范围内。这可以通过找到数据每一维的最大和最小值来完成
    通过取0-1.0之间的随机数，通过 最小值 + （0-1.0之间的随机数）* 取值范围
    取值范围 = 最大值 - 最小值
    """
    for j in range(n):
        minJ = min(dataSet[:, j])
        rangeJ = float(max(dataSet[:, j]) - minJ)
        centroids[:, j] = minJ + rangeJ * np.random.rand(k, 1)

    return centroids

def kMeans(dataSet, k, distMeans = distEclud, createCent = randCent):
    m = np.shape(dataSet)[0]
    """
    创建一个矩阵，存储簇分配结果。
    第一列存质心下标，第二列存误差值，误差值指当前点到质心距离，后面会用该误差来评价聚类效果
    """
    clusterAssment = np.mat(np.zeros((m,2)))
    centroids = createCent(dataSet, k)
    clusterChanged = True

    while clusterChanged:
        clusterChanged = False

        for i in range(m):  #遍历所有点
            minDist = np.inf; minIndex = -1 #假设 最短距离为 无限大（inf）， 最近点的下标为 -1

            for j in range(k): #遍历所有质心
                distJI = distMeans(centroids[j,:], dataSet[i, :]) #计算j和i的距离，计算每个质心到同一点的距离
                if distJI < minDist:
                    minDist = distJI; minIndex = j  #记录最小的质心

            if clusterAssment[i, 0] != minIndex:    clusterChanged = True   #如果质心改变，则需要再重新计算质心后再次迭代。

            clusterAssment[i, :] = minIndex, minDist ** 2   #记录质心位置 和 误差 = 距离平方，加总后的平方和 用于误差衡量聚类质量

        print centroids

        """
        重新计算质心
        """
        for cent in range(k):   #遍历每一个质心
            """
            nonzeros(a)返回数组a中值不为零的元素的下标，它的返回值是一个长度为a.ndim(数组a的轴数)的元组
            >>> b2 = np.array([[True, False, True], [True, False, False]])
            >>> np.nonzero(b2)
            (array([0, 0, 1]), array([0, 2, 0]))
            """
            ptsInClust = dataSet[np.nonzero(clusterAssment[:, 0].A == cent)[0]] #get all the point in this cluster
            centroids[cent, :] = np.mean(ptsInClust, axis = 0)

    return centroids, clusterAssment

"""
2分K均值法

将所有点看成一个簇
当簇数目小于k时
对于每一个簇
    计算总误差
    在给定的簇上面进行k-均值聚类（k=2）
    计算将该簇一分为二之后的总误差
选择使得总误差最小的簇进行划分
"""
def bikmeans(dataSet, k, distMeas = distEclud):
    m = np.shape(dataSet)[0]
    clusterAssment = np.mat(np.zeros((m, 2))) #一列记录质心，一列记录误差

    centroid0 = np.mean(dataSet, axis = 0).tolist()[0]  #获取数据集每一列数据的均值，组成一个长为列数的列表，实际上就是计算每一维的均值。

    centList = [centroid0]

    for j in range(m):  #遍历每个点
        clusterAssment[j, 1] =  distMeas(np.mat(centroid0), dataSet[j, :]) ** 2   #j点到质心的平方距离，记录在clusterAssment 第二列

    while len(centList) < k:  #当族数小于K
        lowestSSE = np.inf

        for i in range(len(centList)):
            # 通过数组过滤筛选出属于第i类的数据集合
            ptsInCurrCluster = dataSet[np.nonzero(clusterAssment[:, 0].A == i)[0], :]
            # 将该族划分成2个簇
            centroidMat, splitClustAss = kMeans(ptsInCurrCluster, 2, distMeas)

            # 计算误差平方和
            sseSplit = sum(splitClustAss[:,1])

            # 计算数据集中不属于该类的数据的误差平方和
            sseNotSplit = sum(clusterAssment[np.nonzero(clusterAssment[:, 0].A != i)[0], 1])
            print "sseSplit, and notSplit: ", sseSplit, sseNotSplit

            if(sseSplit + sseNotSplit) < lowestSSE:
                bestCentToSplit = i
                # 第i类划分后得到质心向量
                bestNewCents = centroidMat
                # 复制第i类中数据点的聚类结果即误差值
                bestClustAss = splitClustAss.copy()
                # 将划分第i类后的总误差作为当前最小误差
                lowestSSE = sseSplit + sseNotSplit

            # 数组过滤筛选出本次2-均值聚类划分后类编号为1数据点，将这些数据点类编号变为
            # 当前类个数+1，作为新的一个聚类
            bestClustAss[np.nonzero(bestClustAss[:, 0].A == 1)[0], 0] = len(centList)
            # 同理，将划分数据集中类编号为0的数据点的类编号仍置为被划分的类编号，使类编号
            #  连续不出现空缺
            bestClustAss[np.nonzero(bestClustAss[:, 0].A == 0)[0], 0] = bestCentToSplit
            # 打印本次执行2-均值聚类算法的类
            print 'the bestCentToSplit is:' % bestCentToSplit
            # 打印被划分的类的数据个数
            print 'the len of bestClustAss is:' % (len(bestClustAss))
            # 更新质心列表中的变化后的质心向量
            centList[bestCentToSplit] = bestNewCents[0, :]
            # 添加新的类的质心向量
            centList.append(bestNewCents[1, :])
            # 更新clusterAssment列表中参与2-均值聚类数据点变化后的分类编号，及数据该类的误差平方
            clusterAssment[np.nonzero(clusterAssment[:, 0].A == bestCentToSplit)[0], :] = bestClustAss
            # 返回聚类结果
            return np.mat(centList), clusterAssment





def biKmeans (dataSet, k):
    m = np.shape(dataSet)[0]
    clusterAssment = np.mat(np.zeros((m, 2)))

    cent0 = np.mean(dataSet, axis = 0).toList()[0]
    centList = [cent0]

    for j in range(m):
        clusterAssment[j, 1] = distMeas(np.mat(cent0), dataSet[j, :]) ** 2


    while len(centList) < k:

        lowestSSE = sum(clusterAssment[:, 1])
        lowestCentIndex = -1;

        for i in range(len(centList)):
            currCluster = 按i取簇
            tempClusterAssment, tempcentList = kMeans(currCluster, 2)

            newSSE = sum(tempClusterAssment[:, 1])

            if newSSE < lowestSSE:
                lowestSSE = newSSE
                lowestCentIndex = i

        currCluster = 按lowestCentIndex取簇
        tempClusterAssment, tempcentList = kMeans(currCluster, 2)

        for j in range(m):
            if clusterAssment[j, 0] == lowestCentIndex:
                del(clusterAssment[j])
                clusterAssment.append(tempClusterAssment)

        for i in range(len(centList)):
            centList.remove(centList[lowestCentIndex])
            centList.append()


