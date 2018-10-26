# Node class used to illustrate the node of decision tree
class Node():
    #constructor with the attributes index, attribute name, 
    #operator sign(greater than, smaller than, equal), split value and stat string
    def __init__(self, attr_index, attr_name, operator_sign, split_value, stat):
        self.splitCriteria = attr_name + operator_sign + split_value + ' ' + stat
        self.attr_index = attr_index
        self.operator_sign = operator_sign
        self.split_value = split_value
        self.children = []

    # Children add method
    def addChild(self, childNode):
        self.children.append(childNode)

    # Boolean method to check whether the root node has children
    def hasChildren(self):
        if(len(self.children) == 0):
            return False
        return True

    # Return the children of the node
    def getChildren(self):
        return self.children

    # Split the certain node
    def getSplit(self):
        return self.splitCriteria

    # Return the value of split node
    def getSplitValue(self):
    	return self.split_value

    # Return the attribute index
    def getSplitAttrIndex(self):
	    return self.attr_index

    # Return the sign of the split node
    def getSplitOpeSign(self):
	    return self.operator_sign

    # Return sorted children
    def sortedChildren(self, orderList):
        numChildren = len(self.children)
        sortedList = []
        for i in range(numChildren):
            for childNode in self.children:
                if(childNode.getSplitValue() == orderList[i]):
                    sortedList.append(childNode)
        return sortedList
