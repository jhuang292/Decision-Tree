#RootNode Class used to illustrate the root of decision tree
class RootNode():

    # Constructor children list with the attributes index and name
    def __init__(self):
        self.children = []

    # Add children add method
    def addChild(self, childNode):
        self.children.append(childNode)

    # Boolean method to check whether the root node has children
    def hasChildren(self):
        if(len(self.children) == 0):
            return False
        return True

    # Return the list of children
    def getChildren(self):
        return self.children

    # Sort the list of children
    def sortedChildren(self, orderList):
        numChildren = len(self.children)
        sortedList = []
        for i in range(numChildren):
            for childNode in self.children:
                if(childNode.getSplitValue() == orderList[i]):
                    sortedList.append(childNode)
        return sortedList



