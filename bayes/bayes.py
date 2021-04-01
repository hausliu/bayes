import csv
import random
import math


def loadCsv(filename):
    lines = csv.reader(open(filename, "r"))
    dataset = list(lines)
    for i in range(len(dataset)):
        for j in range(len(dataset[i]) - 1):
            dataset[i][j] = float(dataset[i][j])
    return dataset


def splitDataset(dataset, splitRatio):
    trainSize = int(len(dataset) * splitRatio)
    trainSet = []
    copy = list(dataset)
    while len(trainSet) < trainSize:
        index = random.randrange(len(copy))
        trainSet.append(copy.pop(index))
    return [trainSet, copy]


def separateByClass(dataset):
    separated = {}
    for i in range(len(dataset)):
        vector = dataset[i]
        if (vector[-1] not in separated):
            separated[vector[-1]] = []
        separated[vector[-1]].append(vector)
    return separated


def mean(numbers):
    return sum(numbers) / float(len(numbers))


def stdev(numbers):
    avg = mean(numbers)
    variance = sum([pow(x - avg, 2) for x in numbers]) / float(len(numbers) - 1)
    return math.sqrt(variance)


def summarize(dataset):
    summaries = [(mean(attribute), stdev(attribute)) for attribute in zip(*dataset)]
    del summaries[-1]
    return summaries


def summarizeByClass(dataset):
    separated = separateByClass(dataset)
    summaries = {}
    for classValue, instances in separated.items():
        summaries[classValue] = summarize(instances)
    return summaries


def calculateProbability(x, mean, stdev):
    exponent = math.exp(-(math.pow(x - mean, 2) / (2 * math.pow(stdev, 2))))
    return (1 / (math.sqrt(2 * math.pi) * stdev)) * exponent


def calculateClassProbabilities(summaries, inputVector):
    probabilities = {}
    for classValue, classSummaries in summaries.items():
        probabilities[classValue] = 1
        for i in range(len(classSummaries)):
            mean, stdev = classSummaries[i]
            x = inputVector[i]
            probabilities[classValue] *= calculateProbability(x, mean, stdev)
    return probabilities


def predict(summaries, inputVector):
    probabilities = calculateClassProbabilities(summaries, inputVector)
    bestLabel, bestProb = None, -1
    for classValue, probability in probabilities.items():
        if bestLabel is None or probability > bestProb:
            bestProb = probability
            bestLabel = classValue
    return bestLabel


def getPredictions(summaries, testSet):
    predictions = []
    for i in range(len(testSet)):
        result = predict(summaries, testSet[i])
        predictions.append(result)
    return predictions


def getAccuracy(testSet, predictions):
    correct = 0
    for i in range(len(testSet)):
        if testSet[i][-1] == predictions[i]:
            correct += 1
    return (correct / float(len(testSet))) * 100.0


def main():
    filename = 'data.csv'
    splitRatio = 0.8
    dataset = loadCsv(filename)
    # 分割训练集和测试集
    # trainingSet, testSet = splitDataset(dataset, splitRatio)
    trainingSet = dataset
    testSet = [[0.52, 0.8, 1.0, 0.0, 0.0, 5.0, 'SF'], [0.33, 0.5, 0.0, 0.0, 0.0, 1.0, 'PF'],
               [0.47, 0.7, 2.0, 4.0, 1.0, 8.0, 'PG']]
    print('Split {0} rows into train={1} and test={2} rows'.format(len(dataset), len(trainingSet), len(testSet)))
    # dataset按标签类别进行分类，separated格式如下：{'c':[[样本1],[样本2],[样本3]],'SF':[[。。],[。。],[。。]]}
    # 对separated字典计算每一类对应每列的均值和标准差
    separated = {}
    for i in range(len(dataset)):
        vector = dataset[i]
        if (vector[-1] not in separated):
            separated[vector[-1]] = []
        separated[vector[-1]].append(vector)

    y = []
    for x in testSet:
        y.append(x[-1])

    separatedWithoutLable = separated.copy()
    for key in separatedWithoutLable:
        for l in separatedWithoutLable[key]:
            l.pop()

    # 计算统计量（statistics）均值（mean value）和标准差（standard deviation）
    statistics = {}
    for classValue, instances in separatedWithoutLable.items():  # dict_items([('C', [[1, 2, 3], [4, 5, 6], [7, 8, 9]]), ('D', [[3, 2, 1], [4, 3, 2], [5, 4, 3]])])
        # listOfStatistics = [(sum(instances))]
        statistics[classValue] = [(mean(a), stdev(a)) for a in zip(*instances)]

    predictions = []
    for i in range(len(testSet)):
        res = predict(statistics, testSet[i])
        predictions.append(res)

    count = 0
    for i in range(len(testSet)):
        # print(y[i], predictions[i])
        if y[i] == predictions[i]:
            count += 1
    accuracy = count / float(len(testSet))

    # predictions = getPredictions(summaries, testSet)
    # accuracy = getAccuracy(testSet, predictions)
    print('Accuracy: {0}%'.format(accuracy))


if __name__ == '__main__':
    main()
