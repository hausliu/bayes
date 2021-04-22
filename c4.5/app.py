import math
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


# information entropy 计算信息熵
def ent(dataset):
    labels = {}
    for featureVector in dataset:
        if featureVector[-1] not in labels:
            labels[featureVector[-1]] = 1
        else:
            labels[featureVector[-1]] += 1
    lenOfDataset = len(dataset)
    valueOfEnt = 0
    for key in labels:
        p = labels[key] / lenOfDataset
        valueOfEnt += p * math.log2(p)
    valueOfEnt = -valueOfEnt
    return valueOfEnt


def loadNBA():
    df = pd.read_csv('./NBAData.csv', encoding='gbk')
    df.dropna(inplace=True)  # 删除有缺失的行
    # df.info()            #查看删除后的information

    # 下面三行将第一列label置换到最后一列，方便后续处理
    labelSerise = df.iloc[:, 0]
    df.drop(df.columns[0], axis=1, inplace=True)  # dataframe删除列，按列序号删除需要使用df.columes[c1,c2,...], axis置为1
    df.insert(df.shape[1], 'Pos(位置)', labelSerise)

    # 处理类似于SF-PG的多位置标签
    for i in range(df.shape[0]):
        label = df.iloc[i, -1]
        if label.find('-') >= 0:
            df.iloc[i, -1] = df.iloc[i, -1][:label.find('-')]
    return df


# 利用信息熵和信息增益进行分类，对于每个属性选择信息增益最大的分割点（2分割），然后将数据二值化为0和1，同时存储分割点信息
# 离散化数据集

# 将数据集按照numOfColumn列的某个分割点进行2划分，返回两个子集
def splitByPoint(dataset, point, numOfColumn):
    subset1 = []
    subset2 = []
    for featureVector in dataset:
        if featureVector[numOfColumn] < point:
            subset1.append(featureVector)
        else:
            subset2.append(featureVector)
    return subset1, subset2


# 获取分割点分割后对应的信息增益
def gainOfPoint(dataset, subset1, subset2):
    gain = ent(dataset) - (ent(subset1) * (len(subset1) / len(dataset))) - (
            ent(subset2) * (len(subset2) / len(dataset)))
    return gain


# 获取每个属性的最优分割点（信息增益最大的分割点）
def getPoints(dataset):
    breakPoints = {}
    for numOfColumn in range(len(dataset[-1]) - 1):

        column = dataset[:, numOfColumn]

        # print('length of column',numOfColumn,':',len(column))
        # print('length of seted column',numOfColumn,':',len(set(column)))
        # print(sorted(set(column)),'\n')

        gainValue = 0  # 初始化该属性的信息增益
        column = sorted(set(column))
        for point in column:
            subset1, subset2 = splitByPoint(dataset, point, numOfColumn)
            if gainOfPoint(dataset, subset1, subset2) > gainValue:
                gainValue = gainOfPoint(dataset, subset1, subset2)
                breakPoints[numOfColumn] = point
    return breakPoints


# 通过points字典存储的最优分割点对dataset进行离散化（大于分割点的值为1，小于分割点的值为0）
def discretize(dataset, columnNames):
    points = getPoints(dataset)
    print('breakPoints:', points)
    for featureVector in dataset:
        for i in range(len(featureVector) - 1):
            if featureVector[i] < points[i]:
                featureVector[i] = 0
            else:
                featureVector[i] = 1
    breakPoints = {}
    q = 0
    for x in columnNames[:-1]:
        breakPoints[x] = points[q]
        q += 1

    return dataset, breakPoints


# dataset = loadNBA().values
# columnNames = list(loadNBA())
# print(discretize(dataset,columnNames))


# 上述代码块实现连续数据的离散化处理，找到每个属性的最优分割点，即信息增益最大的点，将每个属性列的值离散化为0和1


# 获取dataset中信息增益最大的属性,返回属性的列号
def getBestFeature(dataset):
    bestGainRate = 0
    bestFeature = 0
    # 对于每一个属性，求信息增益
    for numOfColumn in range(len(dataset[0]) - 1):
        column = dataset[:, numOfColumn]
        column = sorted(set(column))
        currentGain = ent(dataset)
        for value in column:
            # 获取属性值为value的subset
            subset = []
            for featureVector in dataset:
                if featureVector[numOfColumn] == value:
                    subset.append(featureVector)
            subEnt = (len(subset) / len(dataset)) * ent(subset)
            currentGain -= subEnt
        if (currentGain / ent(dataset)) > bestGainRate:
            bestGainRate = currentGain
            bestFeature = numOfColumn
    return bestFeature


# test
# dataset = loadNBA().values
# dataset,breakPoints = discretize(dataset)
# dataset = pd.read_csv('./test2.csv').values
# print('bsetFeature is column',getBestFeature(dataset))

# 按照属性和属性值划分数据集，同时删除axis列
def splitDataset(dataSet, num, value):
    retDataSet = []
    # featVec = []
    for featVec in dataSet:
        if featVec[num] == value:
            # reducedFeatVec = featVec[:axis]
            # reducedFeatVec.extend(featVec[axis + 1:])
            # reducedFeatVec = np.hstack(reducedFeatVec,featVec[axis + 1:])
            retDataSet.append(np.delete(featVec, num, axis=0))
    return np.array(retDataSet)


# test
# dataset = pd.read_csv('./test.csv').values
# print(splitDataset(dataset,0,1))

def createTree(dataset, columnLabels):
    labels = dataset[:, -1]
    if len(set(labels)) == 1:
        # print(' 标签中无不同类：return ',labels[0])
        return labels[0]
    if len(dataset[0]) == 1:
        maxLabel = labels[0]
        maxLabelTimes = 1
        for label in set(labels):
            if sum(labels == label) > maxLabelTimes:
                maxLabel = label
                maxLabelTimes = sum(labels == label)
        return maxLabel
    bestFeature = getBestFeature(dataset)  # 获取最优划分属性的列号
    bestFeatureLabel = columnLabels[bestFeature]  # 获取最优分割属性名称
    tree = {bestFeatureLabel: {}}
    # del (columnLabels[bestFeature])                     #删除已经选过的属性名称
    columnLabels = np.delete(columnLabels, bestFeature, axis=0)
    # dataset = np.delete(dataset,bestFeature,axis = 1)   #删除dataset中已经选过的列
    attributeValues = set(dataset[:, bestFeature])
    for value in attributeValues:
        # subColumnLabels = columnLabels
        subColumnLabels = columnLabels

        subDataset = splitDataset(dataset, bestFeature, value)
        # 将01标签转化为其代表的分割点数据，使决策树含义更明确
        if value == 0:
            value = '<' + str(breakPoints[bestFeatureLabel])
        else:
            value = '>' + str(breakPoints[bestFeatureLabel])
        tree[bestFeatureLabel][value] = createTree(subDataset, subColumnLabels)
    return tree


# 实现对离散化前后数据的比较，即0对应<breakpoint,1对应>breakpoint
def isEqual(v, s):
    if str(v) == '0':
        if s[0] == '<':
            return 1
        else:
            return 0
    else:
        if s[0] == '>':
            return 1
        else:
            return 0


# 递归实现 预测单个向量的标签，输入为离散化后的0，1向量，输出为预测的label
def predict(featureVector, tree):
    t = {'2P%': 0, 'FT%': 1, 'TRB': 2, 'AST': 3, 'STL': 4, 'PTS': 5}
    # 2P%,FT%,TRB,AST,STL,PTS
    for key in tree:
        if type(tree[key]) == type({1: 2}):
            for s in tree[key]:
                if isEqual(featureVector[t[key]], s):
                    if tree[key][s] in ['C', 'SF', 'PG', 'SG', 'PF']:
                        return tree[key][s]
                    else:
                        return predict(featureVector, tree[key][s])
        else:
            return tree[key]


# 获取整个数据测试准确率
def getAccuracy(dataset, myTree):
    r = 0
    w = 0
    for x in dataset:
        if x[-1] == predict(x, myTree):
            r += 1
        else:
            w += 1
    return r / (r + w)


if __name__ == '__main__':
    dataset = loadNBA().values
    columnLabels = list(loadNBA())
    dataset, breakPoints = discretize(dataset, columnLabels)
    # dataset = pd.read_csv('./test.csv').values
    # print(columnLabels,dataset)
    myTree = createTree(dataset, columnLabels)
    print(myTree)
    for x in range(10):
        print('real:', dataset[x][-1], '| predict:', predict(dataset[x], myTree))
    print(getAccuracy(dataset, myTree))

    dataset = loadNBA().values
    columnLabels = list(loadNBA())
    dataset, breakPoints = discretize(dataset, columnLabels)
    x_train, x_test = train_test_split(dataset, test_size=0.3, random_state=2)

    myTree = createTree(x_train, columnLabels)
    print(getAccuracy(x_test, myTree))
