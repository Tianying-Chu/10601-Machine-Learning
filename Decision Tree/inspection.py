# Name: Tianying Chu
# Andrew ID: tianying

# Program #1: Inspecting the Data

import sys
import numpy as np

def GiniImpurity(data):
    classes = list(set(data[: , -1]))
    class1 = data[: , -1] == classes[0]
    class2 = data[: , -1] == classes[1]
    G = 1 - np.square(class1.mean()) - np.square(class2.mean())
    return G

def majorityVote(data):
    classes = list(set(data[: , -1]))
    class1 = data[: , -1] == classes[0]
    class2 = data[: , -1] == classes[1]
    class1Percent = np.square(class1.mean())
    class2Percent = np.square(class2.mean())
    if class1Percent > class2Percent:
        return classes[0]
    else:
        return classes[1]

def errorRate(data, label):
    errorBool = data[: , -1] != label
    errorRate = errorBool.mean()
    return errorRate

def main():
    data = np.genfromtxt(Input, dtype = np.str)
    Gini = GiniImpurity(data[1: ,])
    mostLabel = majorityVote(data[1: ,])
    error = errorRate(data[1: ,], mostLabel)
    
    with open(Output, 'w') as f:
        f.write('gini_impurity: ' + str(Gini) + '\n')
        f.write('error: ' + str(error))
                
if __name__ == '__main__':
    Input = sys.argv[1]
    Output = sys.argv[2]
    main()
