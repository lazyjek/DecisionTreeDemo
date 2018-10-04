# -*- coding: utf-8 -*-
""" core class of ID3 Trees """
"""
Author: Jennifer Cao
Email: jennifer.cao@wisc.edu
"""
import copy
from node import TreeNode, log2
class DecisionTree(object):
    def __init__(self, root):
        """
        initialize a decision tree
        """
        self.root = root

    def getEntropy(self, node, col = -1):
        """
        calculate entropy of a tree node
        :@ param node: tree node.
        :@ param col: index of the feature that are going to calculate entropy
        :@ rparam entropy: total entropy for a single feature
        """
        labels = {}
        if col < 0:
            # get basic entropy for all instances.
            for instance in node.uniformDataSet:
                label = instance[-1]
                if label not in labels:
                    labels[label] = 0
                labels[label] += 1
            entropy = 0
            for label in labels:
                prob = float(labels[label]) / node.dataSize
                entropy -= prob * log2(prob)
            return entropy
        else:
            labels = {key:{} for key in node.uniformAttribute[col][1]}
            for instance in node.uniformDataSet:
                label = instance[-1]
                feature = instance[col]
                if label not in labels[feature]:
                    labels[feature][label] = 0
                labels[feature][label] += 1
            # get entropy for unique feature
            entropy = 0
            for key in labels:
                subEntropy = 0
                for label in labels[key]:
                    prob = float(labels[key][label]) / sum(labels[key].values())
                    subEntropy -= prob * log2(prob)
                entropy += (subEntropy * sum(labels[key].values()) / node.dataSize)
            return entropy

    def chooseSplitFeature(self, node):
            """
            choose best split feature for a node. compared entropy
            of a single feature and the whole base entropy.
            :@ param node: input tree node
            :@ rparam: index of the best feature.
            """
            bestInformationGain = 0.0
            baseEntropy = self.getEntropy(node)
            bestFeature = -1
            for i in range(node.featureSize):
                curEntropy = self.getEntropy(node, i)
                informationGain = baseEntropy - curEntropy
                if informationGain > bestInformationGain:
                    bestInformationGain = informationGain
                    bestFeature = i
            return bestFeature

    def splitTree(self, node):
        """
        according to best split feature, split datasets into multiple parts
        :@ param node: input Tree Node
        :@ rparam children of input node
        """
        children = []
        bestFeature = self.chooseSplitFeature(node)
        if bestFeature < 0:
            return children
        for key in node.uniformAttribute[bestFeature][1]:
            newDataSet = []
            if key.startswith('<='):
                newDataSet = [instance for instance in node.dataSet
                        if instance[bestFeature] <= float(key.split('<= ')[1])]
            elif key.startswith('>'):
                newDataSet = [instance for instance in node.dataSet
                        if instance[bestFeature] > float(key.split('> ')[1])]
            else:
                newDataSet = [instance for instance in node.dataSet if instance[bestFeature] == key]
            newNode = TreeNode(newDataSet, copy.deepcopy(node.attribute),
                    copy.deepcopy(node.classOutput))
            newNode.splitFeatureName = node.attribute[bestFeature][0]
            newNode.splitFeatureValue = key
            children.append(newNode)
        return children

    def createTree(self, root, m = 4):
        """
        recursively create decision tree.
        stopping criteria:
            #1 there are fewer than m training instances reaching the node
            #2 all of the training instances reaching the node belong to the same class
            #3 no feature has positive information gain
            #4 there are no more remaining candidate splits at the node
        :@ param root: Root of the decision tree
        :@ param m: minimum instances in a leaf
        :@ rparam
        """
        if root == None:
            return root
        if root.dataSize < m or root.classNum == 1 or \
                -1 == self.chooseSplitFeature(root) or root.featureSize == 0:
                    return root
        root.children = self.splitTree(root)
        for child in root.children:
            child = self.createTree(child, m)
        return root

    def printTree(self, root, depth):
        """
        visualize tree split cariteria recursively.
        you can start from root,0
        :@ param root: current node to be printed.
        :@ param depth: the level of the printed node
        """
        indent = ('|' + '\t') * (depth - 1)
        if root == None:
            return
        if root.children == []:
            print "{}{} {}".format(indent, root, root.classOutput)
            return
        if root.splitFeatureName != '-':
            print "{}{}".format(indent, root)
        for child in root.children:
            self.printTree(child, depth + 1)
        return

    def predict(self, root, instance):
        """
        predict class for a single instance.
        :@ param root: root of a trained decision tree.
        :@ param instance: a single instance of input dataset
        :@ rparam: predict class
        """
        if root.children == []:
            return root.classOutput
        attributeNameIndexes = dict([(root.uniformAttribute[i][0],
            i) for i in range(len(root.uniformAttribute))])
        for child in root.children:
            splitValue = child.splitFeatureValue
            curValue = instance[attributeNameIndexes[child.splitFeatureName]]
            if splitValue.startswith('<=') and curValue <= float(splitValue.split('<=')[1]):
                break
            elif splitValue.startswith('>') and curValue <= float(splitValue.split('>')[1]):
                break
            elif splitValue == curValue:
                break
            else:
                continue
        return self.predict(child, instance)

def utEntropy():
    """
    unit test for function [getEntropy]
    """
    from data_provider import data_provider
    attribute, dataset = data_provider('../test.arff')
    root = TreeNode(dataset, attribute)
    curTree = DecisionTree(root)
    try:
        assert('%.3f' % curTree.getEntropy(root) == '0.940')
        assert('%.3f' % (curTree.getEntropy(root) - curTree.getEntropy(root, 0)) == '0.152')
        print '[getEntropy] TEST PASS'
    except AssertionError:
        print '[getEntropy] TEST FAILED'

def utSplitFeature():
    """
    unit test for function [chooseSplitFeature]
    """
    from data_provider import data_provider
    attribute, dataset = data_provider('../test.arff')
    root = TreeNode(dataset, attribute)
    curTree = DecisionTree(root)
    bestFeature = curTree.chooseSplitFeature(root)
    try:
        assert(bestFeature == 0)
        print '[chooseSplitFeature] TEST PASS'
    except AssertionError:
        print '[chooseSplitFeature] TEST FAILED'

def utTreeSplit():
    """
    unit test for function [splitTree]
    """
    from data_provider import data_provider
    attribute, dataset = data_provider('../test.arff')
    root = TreeNode(dataset, attribute)
    curTree = DecisionTree(root)
    children = curTree.splitTree(root)
    try:
        assert('{} {}'.format(children[0], children[0].classOutput) == 'Humidity high [4 3] negative')
        assert('{} {}'.format(children[1], children[1].classOutput) == 'Humidity normal [1 6] positive')
        print '[splitTree] TEST PASS'
    except AssertionError:
        print '[splitTree] TEST FAILED'

def utCreateTree():
    """
    unit test for function [createTree]
    examine the tree structure
    compared graph with:
        http://pages.cs.wisc.edu/~yliang/cs760_fall18/homework/hw2/diabetes/m=4.txt
    """
    from data_provider import data_provider
    attribute, dataset = data_provider('../diabetes_train.arff')
    root = TreeNode(dataset, attribute)
    curTree = DecisionTree(root)
    curTree.createTree(root, 4)
    curTree.printTree(root, 0)
    print '---------------- please compare this graph with the url ------------------'
    print 'http://pages.cs.wisc.edu/~yliang/cs760_fall18/homework/hw2/diabetes/m=4.txt'

def utPredict():
    """
    unit test for function [predict]
    testfiles:
        # trainset: diabetes_train.arff
        # testset: diabetes_test.arff
    """
    from data_provider import data_provider
    attribute, dataset = data_provider('../diabetes_train.arff')
    attribute, testset = data_provider('../diabetes_test.arff')
    root = TreeNode(dataset, attribute)
    curTree = DecisionTree(root)
    curTree.createTree(root, 4)
    try:
        assert(curTree.predict(root, testset[0]) == 'positive')
        assert(curTree.predict(root, testset[22]) == 'positive')
        assert(curTree.predict(root, testset[52]) == 'positive')
        assert(curTree.predict(root, testset[3]) == 'negative')
        assert(curTree.predict(root, testset[78]) == 'negative')
        assert(curTree.predict(root, testset[99]) == 'negative')
        print '[predict] TEST PASS'
    except AssertionError:
        print '[predict] TEST FAILED'

if __name__ == '__main__':
    utCreateTree()
    utEntropy()
    utTreeSplit()
    utSplitFeature()
    utPredict()
