# Name: Tianying Chu
# Andrew ID: tianying

# Program #2: Decision Tree Learner

import sys
import numpy as np

def GiniImpurity(data):
    classes = list(set(data[:, -1]))
    if len(classes) != 1:
        class0 = data[: , -1] == classes[0]
        class1 = data[: , -1] == classes[1]
        G = 1 - np.square(class0.mean()) - np.square(class1.mean())
        return G
    else:
        return 0

def GiniGain(data, splitIndex):
    attrs = list(set(data[:, splitIndex]))
    if len(attrs) != 1:
        GiniRoot = GiniImpurity(data)
        
        subSet0 = data[data[:, splitIndex] == attrs[0]]
        Gini0 = GiniImpurity(subSet0)
        Percent0 = (data[:, splitIndex] == attrs[0]).mean()
        
        subSet1 = data[data[:, splitIndex] == attrs[1]]
        Gini1 = GiniImpurity(subSet1)
        Percent1 = (data[:, splitIndex] == attrs[1]).mean()
        
        Gain = GiniRoot - Percent0 * Gini0 - Percent1 * Gini1
    else:
        Gain = 0
    return Gain
    
def BestAttr(data):
    Gain = dict()
    for splitIndex in range(data.shape[1] - 1):
        Gain[splitIndex] = GiniGain(data, splitIndex)
    if Gain[max(Gain, key = Gain.get)] > 0:
        Best = max(Gain, key = Gain.get)
    else:
        Best = None
    return Best

class Node:
    def __init__(self, key, depth):
        self.left = None
        self.right = None
        self.val = key
        self.depth = depth
        self.leftleaf = None
        self.rightleaf = None
        self.leftbranch = None
        self.rightbranch = None
   
def majorityVote(data):
    classes = list(set(data[: , -1]))
    if len(classes) != 1:
        class0 = data[: , -1] == classes[0]
        class1 = data[: , -1] == classes[1]
        class0Percent = np.square(class0.mean())
        class1Percent = np.square(class1.mean())
        if class0Percent == class1Percent:
            return max(classes)
        elif class0Percent > class1Percent:
            return classes[0]
        else:
            return classes[1]
    else:
        return list(set(data[: , -1]))[0]

def train_tree(node, data):
    if len(list(set(data[:, node.val]))) != 1:
        node.leftbranch = list(set(data[:, node.val]))[0]
        subset0 = data[data[:, node.val] == node.leftbranch]
            
        node.rightbranch = list(set(data[:, node.val]))[1]
        subset1 = data[data[:, node.val] == node.rightbranch]
    
        if node.depth < maxDepth and node.depth < data.shape[1] - 1:
            if BestAttr(subset0) != None:
                node.left = Node(BestAttr(subset0), node.depth + 1)
                train_tree(node.left, subset0)
            else:
                node.leftleaf = majorityVote(subset0)
                            
            if BestAttr(subset1) != None:
                node.right = Node(BestAttr(subset1), node.depth + 1)
                train_tree(node.right, subset1)
            else:
                node.rightleaf = majorityVote(subset1)
        
        else:
            node.leftleaf = majorityVote(subset0)
            node.rightleaf = majorityVote(subset1)
        
    else:
        node.leftbranch = list(set(data[:, node.val]))[0]
        node.leftleaf = majorityVote(data)
    
    return node
    
def h(x, node):
    if node.left != None and x[node.val] == node.leftbranch:
        return h(x, node.left)
    elif node.right != None and x[node.val] == node.rightbranch:
        return h(x, node.right)
    elif node.left == None and x[node.val] == node.leftbranch:
        return node.leftleaf
    elif node.right == None and x[node.val] == node.rightbranch:
        return node.rightleaf
 
def printTree(node, data, attrName):
    classes = list(set(data[: , -1]))
    
    if node != None:
        if node.depth == 1:
            class0num = (data[: , -1] == classes[0]).sum()
            class1num = (data[: , -1] == classes[1]).sum()
            print('[%d %s/%d %s]' % (class0num, classes[0], class1num, classes[1]))
        
        if len(classes) != 1:
            subset0 = data[data[:, node.val] == node.leftbranch]
            class0left = (subset0[: , -1] == classes[0]).sum()
            class1left = (subset0[: , -1] == classes[1]).sum()
            print('%s %s = %s: [%d %s/%d %s]'
                  % ('|' * node.depth, attrName[node.val], node.leftbranch,
                     class0left, classes[0], class1left, classes[1]))
            printTree(node.left, subset0, attrName)
            
            subset1 = data[data[:, node.val] == node.rightbranch]
            class0right = (subset1[: , -1] == classes[0]).sum()
            class1right = (subset1[: , -1] == classes[1]).sum()
            print('%s %s = %s: [%d %s/%d %s]'
                  % ('|' * node.depth, attrName[node.val], node.rightbranch,
                     class0right, classes[0], class1right, classes[1]))
            printTree(node.right, subset1, attrName)

def main():
    trainData = np.genfromtxt(trainInput, dtype = np.str)
    testData = np.genfromtxt(testInput, dtype = np.str)
    
    root = Node(BestAttr(trainData[1: ,]), 1)
    tree = train_tree(root, trainData[1: ,])
    
    errorTrain = 0
    countTrain = 0
    errorTest = 0
    countTest = 0
    
    with open(trainOut, 'w') as f:
        for line in trainData[1:, ]:
            y = h(line, tree)
            #print(y)
            f.write(y + '\n')
            if y != line[-1]:
                errorTrain += 1
            countTrain += 1
    
    with open(testOut, 'w') as f:
        for line in testData[1:, ]:
            y = h(line, tree)
            #print(y)
            f.write(y + '\n')
            if y != line[-1]:
                errorTest += 1
            countTest += 1
    
    rateTrain = errorTrain / countTrain
    rateTest = errorTest / countTest
    
    with open(metricsOut, 'w') as f:
        f.write('error(train): ' + str(rateTrain) + '\nerror(test): ' + str(rateTest))
    
    printTree(tree, trainData[1: ,], trainData[0])

if __name__ == '__main__':
    
    '''
    trainInput = 'politicians_train.tsv'
    testInput = 'politicians_test.tsv'
    maxDepth = 4
    trainOut = 'train.labels'
    testOut = 'test.labels'
    metricsOut = 'metrics.txt'
    
    '''
    trainInput = sys.argv[1]
    testInput = sys.argv[2]
    maxDepth = int(sys.argv[3])
    trainOut = sys.argv[4]
    testOut = sys.argv[5]
    metricsOut = sys.argv[6]
    
    
    main()
