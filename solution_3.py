import math


class Tree(object):

    def __init__(self, feature=None, ys={}, subtrees={}):
        self.feature = feature
        self.ys = ys
        self.subtrees = subtrees

    @property
    def size(self):
        size = 1
        for subtree in self.subtrees.values():
            if type(subtree) == int:
                size += 1
            else:
                size += subtree.size
        return size

    @property
    def depth(self):
        max_depth = 0
        for subtree in self.subtrees.values():
            if type(subtree) == int:
                cur_depth = 1
            else:
                cur_depth = subtree.depth
            max_depth = max(cur_depth, max_depth)
        return max_depth + 1


def entropy(data):
    """Compute entropy of data.

    Args:
        data: A list of data points [(x_0, y_0), ..., (x_n, y_n)]

    Returns:
        entropy of data (float)
    """
    ### YOUR CODE HERE

    # p = 

    ys = {}

    for _,y in data:
        ys[y] = ys.get(y,0) + 1

    total = sum(ys.values())

    label_entropy = 0
    for _,y in ys.items():
      if y != 0:
        label_entropy += -(y)/total * math.log2(y/total)

    # I = -p*math.log2(p)-(1-p)*math.log2(1-p)

    return label_entropy

    ### END YOUR CODE


def gain(data, feature):
    """Compute the gain of data of splitting by feature.

    Args:
        data: A list of data points [(x_0, y_0), ..., (x_n, y_n)]
        feature: index of feature to split the data

    Returns:
        gain of splitting data by feature
    """
    ### YOUR CODE HERE
    labels_entropy = entropy(data)
    len_data = len(data)

    attributes = list(set([x[feature] for x,_ in data]))
    attribute_counts = dict() 
    labels = list(set([y for _,y in data])) 

    
    for (x,y) in data: 
        attribute_counts[x[feature]] = attribute_counts.get(x[feature],0) + 1
        attribute_counts[x[feature]+"_"+str(y)] = attribute_counts.get(x[feature]+"_"+str(y), 0) + 1

    feature_entropy = 0

    for attribute in attributes:

        attribute_total_count = attribute_counts[attribute] 

        p_attribute = attribute_total_count / len_data 
        counts = [] 

        for label in labels:
            if attribute_counts.get(attribute+"_"+str(label), 0) != 0:
                counts.append(attribute_counts[attribute+"_"+str(label)])

        total = sum(counts)

        branch_entropy = 0
        for count in counts:
          if count != 0:
            branch_entropy += -(count)/total * math.log2(count/total)

        feature_entropy += p_attribute * branch_entropy
    
    return labels_entropy - feature_entropy
    


    ### END YOUR CODE


def get_best_feature(data):
    """Find the best feature to split data.

    Args:
        data: A list of data points [(x_0, y_0), ..., (x_n, y_n)]

    Returns:
        index of feature to split data
    """
    ### YOUR CODE HERE
    gains = []

    n_features = len(data[0][0])    
    for feature_i in range(n_features):
      gains.append(gain(data, feature_i))

    max_gain = max(gains)
    best_feature_index = gains.index(max_gain)

    return best_feature_index

    ### END YOUR CODE


def build_tree(data):
    ys = {}
  
    for x, y in data:
        ys[y] = ys.get(y, 0) + 1
    if len(ys) == 1:
        return list(ys)[0]
    feature = get_best_feature(data)
    subtrees = {}
    ### YOUR CODE HERE

    attributes = list(set([x[feature] for x,_ in data]))
    # please split your data with feature and build sub-trees
    sub_data = {attribute:[] for attribute in attributes}
    # by calling build_tree recursively

    # sub_tree = build_tree(...)
    for x,y in data:
        sub_data[x[feature]].append((x,y))

    for attribute in attributes:
        subtrees[attribute] = build_tree(sub_data[attribute])

    
    ### END YOUR CODE
    return Tree(feature, ys, subtrees)


def test_entry(tree, entry):
    x, y = entry
    if type(tree) == int:
        return tree, y
    if x[tree.feature] not in tree.subtrees:
        return tree, max([(value, key) for key, value in tree.ys.items()])[1]
    return test_entry(tree.subtrees[x[tree.feature]], entry)


def test_data(tree, data):
    count = 0
    for d in data:
        y_hat, y = test_entry(tree, d)
        count += (y_hat == y)
    return round(count / float(len(data)), 4)


def prune_tree(tree, data):
    """Find the best feature to split data.

    Args:
        tree: a decision tree to prune
        data: A list of data points [(x_0, y_0), ..., (x_n, y_n)]

    Returns:
        a pruned tree
    """
    ### YOUR CODE HERE
    if type(tree) == int:
        return tree 
    
    attributes = tree.subtrees.keys()
    #iterate through each branch and reach leaf.
    for attribute in attributes:
        # filter data that reaches the branch.
        data_subset = [(x,y) for (x,y) in data if attribute == x[tree.feature]]

        if len(data_subset) != 0:
            tree.subtrees[attribute]= prune_tree(tree.subtrees[attribute], data_subset)
            
    new_tree = max([(value, key) for key, value in tree.ys.items()])[1]
    # compare accuracy/error and return best tree.
    old_accuracy = test_data(tree, data)
    pruned_accuracy = test_data(new_tree, data)

    # return the best tree.
    if (pruned_accuracy >= old_accuracy):
        return new_tree
    return tree
    ### END YOUR CODE
