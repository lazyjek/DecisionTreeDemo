import matplotlib.pyplot as plt
from data_provider import data_provider
from node import TreeNode
from tree import DecisionTree
import random

def selectSample(trainset, rate):
    newTrainset = [[] for i in range(10)]
    mergeCount = rate / 0.1
    randomRange = 10
    if rate < 0.1:
        mergeCount = 1
        randomRange = int(1.0/rate)
    grids = [(count, count + mergeCount) for count in range(10)]
    for instance in trainset:
        rNum = random.randint(0, randomRange)
        for i in range(len(grids)):
            grid = grids[i]
            if grid[0] <= rNum < grid[1]:
                newTrainset[i].append(instance)
    return newTrainset

def part2():
    """randomly choose 5%, 10%, 20%, 50%, 100% samples to train, and choose 10 sets each time"""
    plt.figure()
    for trainFileName, testFileName, key in [('../diabetes_train.arff',
        '../diabetes_test.arff', 'diabetes'), ('../heart_train.arff','../heart_test.arff', 'heart')]:
        attribute, trainset = data_provider(trainFileName)
        testAttribute, testset = data_provider(testFileName)
        m = 4
        avgPoints = []
        maxPoints = []
        minPoints = []
        for rate in (0.05, 0.1, 0.2, 0.5, 1):
            accuracys = []
            for newTrainset in selectSample(trainset, rate):
                root = TreeNode(newTrainset, attribute)
                curTree = DecisionTree(root)
                curTree.createTree(root, m)
                trueSamples = 0
                falseSamples = 0
                for instance in testset:
                    if curTree.predict(root, instance) == instance[-1]:
                        trueSamples += 1
                    else:
                        falseSamples += 1
                accuracys.append(float(trueSamples) / (trueSamples + falseSamples))
            accuracy = float(sum(accuracys)) / len(accuracys)
            avgPoints.append([int(rate*100), accuracy])
            maxPoints.append([int(rate*100), max(accuracys)])
            minPoints.append([int(rate*100), min(accuracys)])

        mapping = {'diabetes':1, 'heart':2}
        ax = plt.subplot(1,2,mapping[key])
        ax.set_xlim(0, 105)
        ax.set_ylim(0.45,0.9)
        ax.set_ylabel('accuracy')
        ax.set_title(key)
        ax.plot([x[0] for x in avgPoints], [x[1] for x in avgPoints], label = 'average')
        ax.plot([x[0] for x in maxPoints], [x[1] for x in maxPoints], label = 'maximum')
        ax.plot([x[0] for x in minPoints], [x[1] for x in minPoints], label = 'minimum')
        ax.legend()
    plt.xlabel('dataset sample percentage')
    plt.savefig('../part2.pdf')

def part3():
    points = {}
    plt.figure()
    for trainFileName, testFileName, key in [('../diabetes_train.arff',
        '../diabetes_test.arff', 'diabetes'), ('../heart_train.arff','../heart_test.arff', 'heart')]:
        attribute, trainset = data_provider(trainFileName)
        testAttribute, testset = data_provider(testFileName)
        root = TreeNode(trainset, attribute)
        curTree = DecisionTree(root)

        points = []
        for m in (2,5,10,20):
            curTree.createTree(root, m)
            trueSamples = 0
            falseSamples = 0
            for instance in testset:
                if curTree.predict(root, instance) == instance[-1]:
                    trueSamples += 1
                else:
                    falseSamples += 1
            points.append([m, float(trueSamples) / (trueSamples + falseSamples)])

        mapping = {'diabetes':1, 'heart':2}
        for x,y in points:
            ax = plt.subplot(2,1,mapping[key])
            ax.set_xlim(0, 22)
            ax.set_ylim(0.6,0.8)
            ax.set_ylabel('accuracy')
            ax.set_title(key)
            plt.annotate('%.3f'%y, xy = (x-0.02,y+0.02))
            plt.annotate('m=%d'%x, xy = (x-0.02, y-0.07))
            ax.plot(x,y,'o-')

    plt.xlabel('tree number m')
    plt.savefig('../part3.pdf')

if __name__ == '__main__':
    part2()
    #part3()
