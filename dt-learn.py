import arff
import numpy as np
import Node
import LeafNode
import RootNode
import sys

#Entropy Calculation
def entropy(s):
    value, counts = np.unique(s, return_counts = True)
    frequencies = counts.astype('float')/len(s)
    H = 0
    for p in frequencies:
        if(p != 0.0):
            H = H - p * np.log2(p)
    return H

#information gain for Nominal feature
def infoGain_Nominal(X,Y):
    infoGain = entropy(Y)
    value, counts = np.unique(X, return_counts=True)
    freq = counts.astype('float') / len(X)
    for p, v in zip(freq, value):
        conditionalIndex = [item for item in range(len(X)) if X[item] == v]
        conditionalSet = []
        for i in conditionalIndex:
                conditionalSet.append(Y[i])
        infoGain = infoGain - p * entropy(conditionalSet)
    return infoGain

# Information gain for numeric feature at specific real number split
def infoGain_Num(X,Y,C):
    freq = [0,0]
    smallSetIndex = [item for item in range(len(X)) if float(X[item]) <= C]
    largeSetIndex = [item for item in range(len(X)) if float(X[item]) > C]
    smallSet, largeSet = [], []
    freq[0] = float(len(smallSetIndex)) / len(X)
    freq[1] = float(len(largeSetIndex)) / len(X)
    for i in smallSetIndex:
        smallSet.append(Y[i])
    for i in largeSetIndex:
        largeSet.append(Y[i])
    infoGain = entropy(Y) - freq[0]*entropy(smallSet) - freq[1]*entropy(largeSet)
    return infoGain

#Determine candidate splits
def candidateSplits(D):
    X = np.delete(D, labelIndex, axis=1)
    C = {}
    for i in range(X.shape[1]):
        # For numeric features
        if(type(attributes[i][1]) is str):
            if (attributes[i][1].upper() == 'REAL' or attributes[i][1].upper() == 'NUMERIC'):
                C.update(candidateNumSplits(D, X[:,i], i))
        else:
            for x in np.unique(X[:,i]):
                C.setdefault(i,[])
                C[i].append(x)
    return C

# Numeric candidate splits
def candidateNumSplits(D, X, attr_index):
    C = {}
    val = [float(x) for x in np.unique(X)]
    val.sort()
    val = [str(x) for x in val]

    # Partitioned parts of D having same value for feature X
    S = []
    for v in val:
        vIndex = [item for item in range(len(X)) if X[item] == v]
        s = []
        for i in vIndex:
            s.append(D[i])
        S.append([s,v])

    contain = False

    for i in range(len(S) - 1):

        # Subset of data contains a value of v
        s, v = S[i][0], S[i][1]
        # Subset of data contains a value of v_next
        s_next, v_next = S[i+1][0], S[i+1][1]

        for s_index in range(len(s)):

            label = s[s_index][labelIndex]
            for s_next_index in range(len(s_next)):
                label_next = s_next[s_next_index][labelIndex]
                if( label != label_next):
                    contain = True
                    break

        if(contain == True):

            C.setdefault(attr_index, [])
            midVal = round((float(v) + float(v_next))/2.0, 5)
            C[attr_index].append(midVal)
    return C

#Find the best split
def bestSplit(D,C):
    X = np.delete(D, labelIndex, axis=1)
    Y = D[:, labelIndex]
    maxInfoGain = 0
    attr_Index = -1

    for i in C:
        # For numeric features
        if(type(attributes[i][1]) is str):
            if (attributes[i][1].upper() == 'REAL' or attributes[i][1].upper() == 'NUMERIC'):
                for j in range(len(C[i])):
                    infogain = infoGain_Num(X[:, i], Y, C[i][j])
                    if(infogain > maxInfoGain):
                        maxInfoGain = infogain
                        attr_Index = i
                        num_split = C[i][j]
                        S = [attr_Index, num_split]
                    elif(infogain == maxInfoGain):
                        if(i < attr_Index):
                            attr_Index = i
                            num_split = C[i][j]
                            S = [attr_Index, num_split]
        else:
            infogain = infoGain_Nominal(X[:, i], Y)
            if(infogain > maxInfoGain):
                maxInfoGain = infogain
                attr_Index = i
                S = [attr_Index]
            elif(infogain == maxInfoGain):
                if(i < attr_Index):
                    attr_Index = i
                    S = [attr_Index]
    return S

def makeSubTree(D, attr_index, operator_sign, split_value, stat_str, parent_label):
    # Read attribute features and labels
    X = np.delete(D, labelIndex, axis=1)
    Y = D[:, labelIndex]
    val, counts = np.unique(Y, return_counts=True)
    C = candidateSplits(D)
    stop = False
    # The training instances reach the node is the class
    if(len(val) == 1):
        stop = True
    # Fewer than m instances reach the node
    if(len(D) < m):
        stop = True
    # No feature has positive information gain
    posInfoGain = False
    for i in range(X.shape[1]):
        if(type(attributes[i][1]) is str):
            if (attributes[i][1] == 'REAL' or attributes[i][1] == 'NUMERIC') and (i in C.keys()):
                for j in range(len(C[i])):
                    infogain = infoGain_Num(X[:,i], Y, C[i][j])
                    if(infogain > 0):
                        posInfoGain = True
                        break
        else:
            infogain = infoGain_Nominal(X[:,i], Y)
            if(infogain > 0):
                posInfoGain = True
                break
    if(not posInfoGain):
        stop = True
    # No more remaining candidate splits at the node
    if(len(C) == 0):
        stop = True
    # Stop criteria met, generate leaf node
    if(stop):
        leafIndex = np.argwhere(counts == np.amax(counts))
        # The training instances reaching a leaf are equally represented,
        # the leaf predicts the most common class of instances reaching the parent node.
        if(len(leafIndex) == 1):
            label = val[np.argmax(counts)]
        else:
            label = parent_label
        node = LeafNode.LeafNode(D, label, attr_index, attributes[attr_index][0], operator_sign, split_value, stat_str)
    # Create internal node
    else:
        if(attr_index != None):
            node = Node.Node(attr_index, attributes[attr_index][0], operator_sign, split_value, stat_str)
        else:
            node = RootNode.RootNode()
        S = bestSplit(D,C)
        attr_index = S[0]
        if(attr_index >= 0):
            if(type(attributes[attr_index][1]) is str):
                if (attributes[attr_index][1] == 'REAL' or attributes[attr_index][1] == 'NUMERIC'):
                    split = S[1]
                    small_Set, large_Set = [],[]
                    for instance in D:
                        if(float(instance[attr_index]) <= split):
                            small_Set.append(instance)
                        else:
                            large_Set.append(instance)
                    small_Set = np.array(small_Set)
                    large_Set = np.array(large_Set)
                    node.addChild(makeSubTree(small_Set, attr_index, ' <= ', str(format(split, '0.6f')), stat(small_Set), determineLabel(D)))
                    node.addChild(makeSubTree(large_Set, attr_index, ' > ', str(format(split, '0.6f')), stat(large_Set), determineLabel(D)))
            else:
                nominal_Splits = np.unique(X[:,attr_index])
                for split in nominal_Splits:
                    subSet = []
                    for instance in D:
                        if (instance[attr_index] == split):
                            subSet.append(instance)
                    subSet = np.array(subSet)

                    childNode = makeSubTree(subSet, attr_index, ' = ', split, stat(subSet), determineLabel(D))
                    node.addChild(childNode)
    return node

def clf_dt(Node, instance):
    if(type(Node) is LeafNode.LeafNode):
        return Node.getClassLabel()
    else:
        for child in Node.getChildren():
            attr_Index = child.getSplitAttrIndex()
            operator_Sign = child.getSplitOpeSign()
            isBranch = False
            if(type(attributes[attr_Index][1]) is str):
                if (attributes[attr_Index][1] == 'REAL' or attributes[attr_Index][1] == 'NUMERIC'):
                    split_Value = float(child.getSplitValue())
                    instance_Value = float(instance[attr_Index])
                    if(operator_Sign == ' <= '):
                        if(instance_Value <= split_Value):
                            isBranch = True
                    elif(operator_Sign == ' > '):
                        if(instance_Value > split_Value):
                            isBranch = True
            else:
                split_Nomimal = str(child.getSplitValue())
                instance_Nominal = instance[attr_Index]
                if(split_Nomimal == instance_Nominal):
                    isBranch = True
            if isBranch:
                return clf_dt(child, instance)
    return

def stat(subSet):
    neg, pos = 0, 0
    value, counts = np.unique(subSet[:, labelIndex], return_counts=True)
    if(value[0] == 'negative'):
        neg = counts[0]
        if (len(counts) > 1):
            pos = counts[1]
    else:
        pos = counts[0]
        if (len(counts) > 1):
            neg = counts[1]
    return '[' + str(neg) + ' ' + str(pos) + ']'

def determineLabel(labelSet):
    neg, pos = 0, 0
    value, counts = np.unique(labelSet[:, labelIndex], return_counts=True)
    if(value[0] == 'negative'):
        neg = counts[0]
        if (len(counts) > 1):
            pos = counts[1]
    else:
        pos = counts[0]
        if (len(counts) > 1):
            neg = counts[1]
    if(pos > neg):
        return 'positive'
    else:
        return 'negative'

def printSubTree(Node, level):
    if(not (type(Node) is RootNode.RootNode)):
        print(Node.getSplit())
    if(type(Node) is LeafNode.LeafNode):
        return
    else:
        if(Node.hasChildren()):
            child_attr_index = Node.getChildren()[0].getSplitAttrIndex()
            if(not (type(attributes[child_attr_index][1]) is str)):
                sortedChildren = Node.sortedChildren(attributes[child_attr_index][1])
            else:
                sortedChildren = Node.getChildren()
            for child in sortedChildren:
                for i in range(level):
                    print ('|	', end='')

                printSubTree(child, level + 1)
    return


def printPrediction(testSet, root):
    test_Label = testSet[:, labelIndex]
    count, correct = 0, 0
    print('<Predictions for the Test Set Instances>')
    for instance, actual_label in zip(testSet, test_Label):
        predict_label = clf_dt(root, instance)
        count += 1
        if (actual_label == predict_label):
            correct += 1
        print(str(count) + ': Actual: ' + actual_label + ' Predicted: ' + predict_label)
    print('Number of correctly classified: ' + str(correct) + ' Total number of test instances: ' + str(count))

def accuracy(testSet, root):
    test_Label = testSet[:, labelIndex]
    count, correct = 0, 0
    for instance, actual_label in zip(testSet, test_Label):
        predict_label = clf_dt(root, instance)
        count += 1
        if (actual_label == predict_label):
            correct += 1
    return correct / count


# Training and test sets
# Sample split
train_file = sys.argv[1]
test_file = sys.argv[2]
m = int(sys.argv[3])

# Read training data set
# Generate attributes and training set
dataset = arff.load(open(train_file))
attributes = dataset['attributes']
relation = dataset['relation']
data = np.array(dataset['data'])

# Read test data set
dataset = arff.load(open(test_file))
data_test = np.array(dataset['data'])

for i in range(len(attributes)):
    if(attributes[i][0] == 'class'):
        labelIndex = i
    else:
        labelIndex = -1


 # Train Decision Tree and print the architecture
root = makeSubTree(data, None, None , None, None, None)
printSubTree(root,0)
printPrediction(data_test,root)



