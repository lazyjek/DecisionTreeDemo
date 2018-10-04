import sys
from data_provider import data_provider
from tree import DecisionTree
from node import TreeNode

if __name__ == '__main__':
    try: assert(len(sys.argv) >= 4)
    except AssertionError:
        print >> sys.stderr, "[ERROR] you should provide at least 3 inputs!"
        sys.exit()
    trainFileName = sys.argv[1]
    testFileName = sys.argv[2]
    try: m = int(sys.argv[3])
    except: print >> sys.stderr, "[ERROR] [m] should be in integer!"; sys.exit()

    attribute, trainset = data_provider(trainFileName)
    testAttribute, testset = data_provider(testFileName)
    try: assert(testAttribute == attribute)
    except AssertionError:
        print >> sys.stderr, "[ERROR] pls check the attributes of test data."
        sys.exit()

    # train
    root = TreeNode(trainset, attribute)
    curTree = DecisionTree(root)
    curTree.createTree(root, m)
    curTree.printTree(root, 0)

    # test
    print '<Predictions for the Test Set Instances>'
    index = 1
    for instance in testset:
        print '{}: Actual: {} Predicted: {}'.format(index,
                instance[-1],
                curTree.predict(root, instance))
        index += 1


