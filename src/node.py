# -*- coding: utf-8 -*-
""" core class of ID3 Trees """
"""
Author: Jennifer Cao
Email: jennifer.cao@wisc.edu
"""
import numpy as np
import copy
import math

def log2(x):
    return math.log(x) / math.log(2)

class TreeNode(object):
    def __init__(self, dataset, attribute, parentClass = None):
        """
        initialize a tree node
        :@ param dataset: train or test data (initialize)
        :@ param attribute: a list of attribute (name, type) mapping
        :@ param datasize: number of instances.
        :@ param featureSize: number of attributes.
        """
        self.dataSet = dataset
        self.attribute = attribute
        self.uniformDataSet, self.uniformAttribute = self._uniformDataSet(dataset, attribute)
        self.featureSize = len(attribute) - 1
        self.dataSize = len(dataset)
        # which attribute to split, and contains datasets according to which attribute value
        self.splitFeatureName = '-'
        self.splitFeatureValue = '-'
        self.children = []
        self.classOutput, self.classNum, self.dataDistribute = self.getClass(parentClass)

    def __str__(self):
        return "{} {} [{}]".format(self.splitFeatureName,self.splitFeatureValue, self.dataDistribute)

    def _calcThreshold(self, featureSample):
        featureLabelDict = {}
        for i in range(len(featureSample)):
            val, label = featureSample[i]
            if val not in featureLabelDict:
                featureLabelDict[val] = set()
            featureLabelDict[val].add(label)

        featureValues = sorted(np.unique(np.array(featureSample)[:,0]).tolist(),
                key = lambda x:float(x))

        bestThreshold = None
        bestEntropy = None
        for i in range(len(featureValues) - 1):
            val1, val2 = featureValues[i], featureValues[i + 1]
            if len(featureLabelDict[val1] | featureLabelDict[val2]) != 1:
                candidateThreshold = (float(val1) + float(val2)) / 2
                labels = {'low':{}, 'high':{}}
                for sample in featureSample:
                    label = sample[1]
                    if float(sample[0]) <= candidateThreshold:
                        if label not in labels['low']:
                            labels['low'][label] = 0
                        labels['low'][label] += 1
                    else:
                        if label not in labels['high']:
                            labels['high'][label] = 0
                        labels['high'][label] += 1
                entropy = 0.0
                for key in labels:
                    subEntropy = 0
                    for label in labels[key]:
                        prob = float(labels[key][label]) / sum(labels[key].values())
                        subEntropy -= log2(prob) * prob
                    entropy += (subEntropy * sum(labels[key].values()) / len(featureSample))
                if bestEntropy == None or entropy < bestEntropy:
                    bestEntropy = entropy
                    bestThreshold = candidateThreshold
        if bestThreshold == None:
            bestThreshold = np.mean(np.array([float(k) for k in featureValues[:2]]))
        return bestThreshold

    def _uniformDataSet(self, dataset, attribute):
        """ uniform dataset and attribute. Private function """
        data = np.array(dataset)
        if len(data) == 0:
            return [], attribute
        newAttribute = []
        for i in range(len(attribute)):
            if isinstance(attribute[i][1], list) == False:
                threshold = self._calcThreshold(np.array(data)[:,(i,-1)].tolist())
                newAttribute.append((attribute[i][0], ['<= %.6f' % threshold, '> %.6f' % threshold]))
            else:
                newAttribute.append(attribute[i])

        newDataSet = copy.deepcopy(dataset)
        for instance in newDataSet:
            for i in range(len(instance) - 1):
                if newAttribute[i][1][0].startswith('<=') or newAttribute[i][1][0].startswith('>'):
                    threshold = float(newAttribute[i][1][0].split('<= ')[1])
                    if instance[i] <= threshold:
                        instance[i] = newAttribute[i][1][0]
                    else:
                        instance[i] = newAttribute[i][1][1]
        return newDataSet, newAttribute

    def getClass(self, parentClass):
        """
        vote for the major class
        :@ rparam: the class of this Tree Node
        """
        # If the number of training instances that reach a leaf node is 0, return class of its parent.
        if  self.dataSize == 0:
            return parentClass, 0, '0 0'
        classes, countClass = np.unique(np.array(self.uniformDataSet)[:,-1],
                return_counts= True)

        classDistribute = dict(zip(classes, countClass))
        predictClass = classes[np.argmax(countClass)]

        # training instances reaching a leaf are equally represented
        if np.unique(countClass).shape[0] == 1 and countClass.shape[0] != 1:
            predictClass = parentClass

        return predictClass, len(classes), ' '.join([str(classDistribute.get(key, '0')) for key in self.attribute[-1][1]])
