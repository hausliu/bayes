import pandas as pd
import numpy as np
import copy  # 深度拷贝需要引入copy模块


def load():
    df = pd.read_csv('./newData.csv', encoding='gbk')
    df.dropna(inplace=True)  # 删除有缺失的行
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


# ### 以上内容为数据处理

def initializeCenter(k, data):
    center = {}
    for i in range(k):
        center[i] = data[i][:-1]
    return center


# center = initializeCenter(5,data)
# print(center)


def fit(k, data, tolerance=0.000001, maxTimes=9999999):
    centers = initializeCenter(k, data)
    clf = {}
    count = 0
    # 在maxTimes次数内循环
    for i in range(maxTimes):
        count += 1

        # 初始化分类器，共k类，每类对应一个list存放该类的instance
        for i in range(k):
            clf[i] = []

        # 对于每一个instance，计算和每个center的欧式距离，求出最短距离的center，并放入该类对应的list中
        for instance in data:
            distances = []
            for centerKey in centers:  # 此处为遍历字典的key
                distances.append(np.linalg.norm(instance[:-1] - centers[centerKey]))  # 计算和每个center的距离添加到ditances列表中
            # print(distances)
            indexOfMinDistance = distances.index(min(distances))
            clf[indexOfMinDistance].append(instance)

        # 计算新的中心点，和旧中心点作比较，判断是否退出循环
        oldCenters = centers.copy()
        for key in clf:
            m = np.array(clf[key])
            m = np.delete(m, len(m[0]) - 1, axis=1)  # m为删除均值后的一个类的所有instance
            centers[key] = np.average(m, axis=0)  # 计算每一列的均值，即新的center

        # 输出中间过程结果
        print("第{0}次迭代".format(count))

        # 判断是否达到结束循环条件
        flag = 1
        for key in centers:
            if np.linalg.norm(centers[key] - oldCenters[key]) != 0:
                flag = 0
        if flag == 1:
            return clf

    return clf  # clf 为返回的类别和对应item的数组


# statistic 将clf中每条item的label提取出来
def statistic(clf):
    res = {}
    for key in clf:
        m = np.array(clf[key])
        clf[key] = m[:, -1]
        res[key] = clf[key].flatten()
    return res


# analyse 为分析每个类各种标签的数量标签和比例
def analyse(res):
    statistic = {}
    for key in res:
        if key not in statistic:
            statistic[key] = {}
        for l in res[key]:
            if l not in statistic[key]:
                statistic[key][l] = 0
            else:
                statistic[key][l] += 1
    # 为结果排序输出
    sortedStatistic = {}
    for key in statistic:
        sortedStatistic[key] = {}
        for subkey in sorted(statistic[key]):
            sortedStatistic[key][subkey] = statistic[key][subkey]

    for key in sortedStatistic:
        sortedStatistic[key]['sum'] = 0
        for subkey in sortedStatistic[key]:
            sortedStatistic[key]['sum'] += sortedStatistic[key][subkey]
        sortedStatistic[key]['sum'] = int(sortedStatistic[key]['sum'] / 2)

    percentageStatistic = copy.deepcopy(sortedStatistic)
    for key in percentageStatistic:
        for subkey in percentageStatistic[key]:
            percentageStatistic[key][subkey] = round(percentageStatistic[key][subkey] / percentageStatistic[key]['sum'],
                                                     2)
        del percentageStatistic[key]['sum']

    return sortedStatistic, percentageStatistic  # {0:{'PG':20,'SF':39},1:{'PG':20,'SF':39},,,}


if __name__ == '__main__':
    df = load()
    dataWithLable = df.values  # dataframe 转 numpy
    clf = fit(5, dataWithLable, tolerance=0.005, maxTimes=99)
    res = statistic(clf)
    print('数量统计分析：')
    s, p = analyse(res)
    for key in s:
        print('类别' + str(key), ':', s[key])
    print('百分比统计分析：')
    for key in p:
        print('类别' + str(key), ':', p[key])
    print('')
