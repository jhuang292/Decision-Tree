# LeafNode class is used to illustrate the single leaf node
class LeafNode():
    def __init__(self, Data, label, attr_index, attr_name, operator_sign, split_value, stat):
        self.Data = Data
        self.label = label
        self.attr_index = attr_index
        self.operator_sign = operator_sign
        self.split_value = split_value
        self.splitCriteria = attr_name + operator_sign + split_value + ' ' + stat

    # Return the split node
    def getSplit(self):
        return self.splitCriteria + ': ' + self.label

    # Return the label of the leaf node
    def getClassLabel(self):
    	return self.label

    # Return the value of the split node
    def getSplitValue(self):
        return self.split_value

    # Return attribute index of the leaf node
    def getSplitAttrIndex(self):
        return self.attr_index

    # Return the sign of the split node
    def getSplitOpeSign(self):
        return self.operator_sign

