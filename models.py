import numpy as np
import random
import copy
import math

def node_score_error(prob):
    '''
        TODO:
        Calculate the node score using the train error of the subdataset and return it.
        For a dataset with two classes, C(p) = min{p, 1-p}
    '''
    return min(prob, 1-prob)


def node_score_entropy(prob):
    '''
        TODO:
        Calculate the node score using the entropy of the subdataset and return it.
        For a dataset with 2 classes, C(p) = -p * log(p) - (1-p) * log(1-p)
        For the purposes of this calculation, assume 0*log0 = 0.
        HINT: remember to consider the range of values that p can take!
    '''
    if prob == 0:
        prob = 0.00001
    if prob == 1:
        prob = 0.99999

    return -prob * math.log(prob) - (1-prob) * math.log(1-prob)


def node_score_gini(prob):
    '''
        TODO:
        Calculate the node score using the gini index of the subdataset and return it.
        For dataset with 2 classes, C(p) = 2 * p * (1-p)
    '''
    return 2 * prob * (1-prob)



class Node:
    '''
    Helper to construct the tree structure.
    '''
    def __init__(self, left=None, right=None, depth=0, index_split_on=0, isleaf=False, label=1):
        self.left = left
        self.right = right
        self.depth = depth
        self.index_split_on = index_split_on
        self.isleaf = isleaf
        self.label = label
        self.info = {} # used for visualization


    def _set_info(self, gain, num_samples):
        '''
        Helper function to add to info attribute.
        You do not need to modify this. 
        '''

        self.info['gain'] = gain
        self.info['num_samples'] = num_samples


class DecisionTree:

    def __init__(self, data, validation_data=None, gain_function=node_score_entropy, max_depth=40):
        self.max_depth = max_depth
        self.root = Node()
        self.gain_function = gain_function

        indices = list(range(1, len(data[0])))

        self._split_recurs(self.root, data, indices)

        # Pruning
        if validation_data is not None:
            self._prune_recurs(self.root, validation_data)


    def predict(self, features):
        '''
        Helper function to predict the label given a row of features.
        You do not need to modify this.
        '''
        return self._predict_recurs(self.root, features)


    def accuracy(self, data):
        '''
        Helper function to calculate the accuracy on the given data.
        You do not need to modify this.
        '''
        return 1 - self.loss(data)


    def loss(self, data):
        '''
        Helper function to calculate the loss on the given data.
        You do not need to modify this.
        '''
        cnt = 0.0
        test_Y = [row[0] for row in data]
        for i in range(len(data)):
            prediction = self.predict(data[i])
            if (prediction != test_Y[i]):
                cnt += 1.0
        return cnt/len(data)


    def _predict_recurs(self, node, row):
        '''
        Helper function to predict the label given a row of features.
        Traverse the tree until leaves to get the label.
        You do not need to modify this.
        '''
        if node.isleaf or node.index_split_on == 0:
            return node.label
        split_index = node.index_split_on
        if not row[split_index]:
            return self._predict_recurs(node.left, row)
        else:
            return self._predict_recurs(node.right, row)


    def _prune_recurs(self, node, validation_data):
        '''
        TODO:
        Prune the tree bottom up recursively. Nothing needs to be returned.
        Do not prune if the node is a leaf.
        Do not prune if the node is non-leaf and has at least one non-leaf child.
        Prune if deleting the node could reduce loss on the validation data.
        NOTE:
        This might be slightly different from the pruning described in lecture.
        Here we won't consider pruning a node's parent if we don't prune the node 
        itself (i.e. we will only prune nodes that have two leaves as children.)
        HINT: Think about what variables need to be set when pruning a node!
        '''
        if node.isleaf:
            return
        if not node.left.isleaf:
            self._prune_recurs(node.left, validation_data)
        if not node.right.isleaf:
            self._prune_recurs(node.right, validation_data)
        
        if  node.left.isleaf and node.right.isleaf:
            curr_node_loss = self.loss(validation_data)
            node.isleaf = True

            changed_node_loss = self.loss(validation_data)

            if curr_node_loss < changed_node_loss:
                node.isleaf = False
            else:
                node.left = None
                node.right = None

    def _is_terminal(self, node, data, indices):
        '''
        TODO:
        Helper function to determine whether the node should stop splitting.
        Stop the recursion if:
            1. The dataset is empty.
            2. There are no more indices to split on.
            3. All the instances in this dataset belong to the same class
            4. The depth of the node reaches the maximum depth.
        Return:
            - A boolean, True indicating the current node should be a leaf and 
              False if the node is not a leaf.
            - A label, indicating the label of the leaf (or the label the node would 
              be if we were to terminate at that node). If there is no data left, you
              can return either label at random.
        '''
        
        if (len(data) == 0) or (len(indices) == 0) or (node.depth >= self.max_depth) or (len(set(data[:, 0].flatten())) == 1) :
            if node.isleaf:
                return True, node.label
            else:
                if data.size != 0:
                    labels = data[:, 0]
                    return True, np.argmax(np.bincount(labels))
                else:
                    return True, 1
        else:
            labels = data[:, 0]
            return False, np.argmax(np.bincount(labels))
            


    def _split_recurs(self, node, data, indices):
        '''
        TODO:
        Recursively split the node based on the rows and indices given.
        Nothing needs to be returned.

        First use _is_terminal() to check if the node needs to be split.
        If so, select the column that has the maximum infomation gain to split on.
        Store the label predicted for this node, the split column, and use _set_info()
        to keep track of the gain and the number of datapoints at the split.
        Then, split the data based on its value in the selected column.
        The data should be recursively passed to the children.
        '''
        bol, label = self._is_terminal(node, data, indices)
        node.label = label
        if bol == False:
            node.isleaf = False
            max_gain = float('-inf')
            max_gain_index = 1
            for index in indices:
                gain = self._calc_gain(data, index, self.gain_function)
                if gain > max_gain:
                    max_gain = gain
                    max_gain_index = index
            split_column = data[:, max_gain_index]
            node._set_info(max_gain, len(split_column))
            node.index_split_on = max_gain_index
            left_subset = []
            right_subset = []
            for r in range(len(split_column)):
                if split_column[r] == 0:
                    left_subset.append(data[r, :])
                if split_column[r] == 1:
                    right_subset.append(data[r, :])
            n_indices = indices.copy()
            n_indices.remove(max_gain_index)
            node.left = Node(depth=node.depth+1, isleaf=True, label=0)
            self._split_recurs(node.left, np.asarray(left_subset), n_indices)
            node.right = Node(depth=node.depth+1, isleaf=True, label=1)
            self._split_recurs(node.right, np.asarray(right_subset), n_indices)


    def _calc_gain(self, data, split_index, gain_function):
        '''
        TODO:
        Calculate the gain of the proposed splitting and return it.
        Gain = C(P[y=1]) - P[x_i=True] * C(P[y=1|x_i=True]) - P[x_i=False] * C(P[y=0|x_i=False])
        Here the C(p) is the gain_function. For example, if C(p) = min(p, 1-p), this would be
        considering training error gain. Other alternatives are entropy and gini functions.
        '''
        split_column = data[:, split_index]
        zero_count = 0
        one_count = 0
        left_subset = []
        right_subset = []
        for r in range(len(split_column)):
            if split_column[r] == 0:
                zero_count += 1
                left_subset.append(data[r, :])
            if split_column[r] == 1:
                one_count += 1
                right_subset.append(data[r, :])
        left_subset = np.asarray(left_subset).reshape(-1, data.shape[1])
        right_subset = np.asarray(right_subset).reshape(-1, data.shape[1])
        P_y1 = np.sum(data[:, 0])/data.shape[0]
        x_i_false = zero_count/len(split_column)
        x_i_true = one_count/len(split_column)
        if right_subset.shape[0] == 0:
            P_y1_true = 0
        else:
            P_y1_true = np.sum(right_subset[:, 0])/right_subset.shape[0]
        if left_subset.shape[0] == 0:
            P_y0_false = 0
        else:
            P_y0_false = 1 - np.sum(left_subset[:, 0])/left_subset.shape[0]
        gain = gain_function(P_y1) - x_i_true * gain_function(P_y1_true) - x_i_false * gain_function(P_y0_false)
        return gain
    

    def print_tree(self):
        '''
        Helper function for tree_visualization.
        Only effective with very shallow trees.
        You do not need to modify this.
        '''
        print('---START PRINT TREE---')
        def print_subtree(node, indent=''):
            if node is None:
                return str("None")
            if node.isleaf:
                return str(node.label)
            else:
                decision = 'split attribute = {:d}; gain = {:f}; number of samples = {:d}'.format(node.index_split_on, node.info['gain'], node.info['num_samples'])
            left = indent + '0 -> '+ print_subtree(node.left, indent + '\t\t')
            right = indent + '1 -> '+ print_subtree(node.right, indent + '\t\t')
            return (decision + '\n' + left + '\n' + right)

        print(print_subtree(self.root))
        print('----END PRINT TREE---')


    def loss_plot_vec(self, data):
        '''
        Helper function to visualize the loss when the tree expands.
        You do not need to modify this.
        '''
        self._loss_plot_recurs(self.root, data, 0)
        loss_vec = []
        q = [self.root]
        num_correct = 0
        while len(q) > 0:
            node = q.pop(0)
            num_correct = num_correct + node.info['curr_num_correct']
            loss_vec.append(num_correct)
            if node.left != None:
                q.append(node.left)
            if node.right != None:
                q.append(node.right)

        return 1 - np.array(loss_vec)/len(data)


    def _loss_plot_recurs(self, node, rows, prev_num_correct):
        '''
        Helper function to visualize the loss when the tree expands.
        You do not need to modify this.
        '''
        labels = [row[0] for row in rows]
        curr_num_correct = labels.count(node.label) - prev_num_correct
        node.info['curr_num_correct'] = curr_num_correct

        if not node.isleaf:
            left_data, right_data = [], []
            left_num_correct, right_num_correct = 0, 0
            for row in rows:
                if not row[node.index_split_on]:
                    left_data.append(row)
                else:
                    right_data.append(row)

            left_labels = [row[0] for row in left_data]
            left_num_correct = left_labels.count(node.label)
            right_labels = [row[0] for row in right_data]
            right_num_correct = right_labels.count(node.label)

            if node.left != None:
                self._loss_plot_recurs(node.left, left_data, left_num_correct)
            if node.right != None:
                self._loss_plot_recurs(node.right, right_data, right_num_correct)
