
from matplotlib.font_manager import FontProperties
import matplotlib.pyplot as plt
from math import log
import operator
import pickle


def createDataSet():
    dataSet = [[0, 0, 0, 0, 'no'],
               [0, 0, 0, 1, 'no'],
               [0, 1, 0, 1, 'yes'],
               [0, 1, 1, 0, 'yes'],
               [0, 0, 0, 0, 'no'],
               [1, 0, 0, 0, 'no'],
               [1, 0, 0, 1, 'no'],
               [1, 1, 1, 1, 'yes'],
               [1, 0, 1, 2, 'yes'],
               [1, 0, 1, 2, 'yes'],
               [2, 0, 1, 2, 'yes'],
               [2, 0, 1, 1, 'yes'],
               [2, 1, 0, 1, 'yes'],
               [2, 1, 0, 2, 'yes'],
               [2, 0, 0, 0, 'no']]

    labels = ['F1-AGE', 'F2-WORK', 'F3-HOME', 'F4-LOAN']
    return dataSet, labels


def createTree(dataset, labels, featureLabels):
    classlist = [example[-1] for example in dataset]
    if classlist.count(classlist[0]) == len(classlist):
        return classlist[0]
    if len(dataset[0]) == 1:
        return majorityCnt(classlist)

    bestFeature = chooseBestFeatureToSplit(dataset)
    bestFeatureLabel = labels[bestFeature]
    featureLabels.append(bestFeatureLabel)
    myTree = {bestFeatureLabel: {}}
    del labels[bestFeature]
    featureValues = [example[bestFeature] for example in dataset]
    uniqueValues = set(featureValues)
    for value in uniqueValues:
        sublabels = labels[:]
        myTree[bestFeatureLabel][value] = createTree(
            splitDataSet(dataset, bestFeature, value), sublabels, featureLabels)

    return myTree


def majorityCnt(classlist):
    classCount = {}
    for vote in classlist:
        if vote not in classCount.keys():
            classCount[vote] = 0
        classCount[vote] += 1

    sortedclassCount = sorted(
        classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedclassCount[0][0]


def chooseBestFeatureToSplit(dataset):
    numFeatures = len(dataset[0]) - 1
    baseEntropy = calcEntropy(dataset)
    bestInfoGain = 0
    bestFeature = -1
    for i in range(numFeatures):
        featureList = [example[i] for example in dataset]
        uniqueValues = set(featureList)
        newEntropy = 0
        for val in uniqueValues:
            subDataSet = splitDataSet(dataset, i, val)
            prob = len(subDataSet) / float(len(dataset))
            newEntropy += prob * calcEntropy(subDataSet)
        infoGain = baseEntropy - newEntropy
        if (infoGain > bestInfoGain):
            bestInfoGain = infoGain
            bestFeature = i
    return bestFeature


def splitDataSet(dataset, axis, val):
    retDataSet = []
    for featureVec in dataset:
        if featureVec[axis] == val:
            reducedFeatureVec = featureVec[:axis]
            reducedFeatureVec.extend(featureVec[axis+1:])
            retDataSet.append(reducedFeatureVec)
    return retDataSet


def calcEntropy(dataset):
    numExamples = len(dataset)
    labelCount = {}
    for featureVec in dataset:
        currentLabel = featureVec[-1]
        if currentLabel not in labelCount.keys():
            labelCount[currentLabel] = 0
        labelCount[currentLabel] += 1

    Entropy = 0
    for key in labelCount:
        prop = float(labelCount[key]) / numExamples
        Entropy -= prop * log(prop, 2)
    return Entropy


if __name__ == '__main__':
    dataset, labels = createDataSet()
    featureLabels = []
    myTree = createTree(dataset, labels, featureLabels)

