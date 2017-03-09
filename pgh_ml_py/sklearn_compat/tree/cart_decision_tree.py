"""CART on the Bank Note dataset"""
from __future__ import (absolute_import, division,
                        unicode_literals, print_function)
from collections import defaultdict
import math
# import json
# from random import seed  # randrange, sample

import numpy as np
import pandas as pd
# from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
# from sklearn.utils.multiclass import unique_labels

from pgh_ml_py.sklearn_compat.base import BaseClassifier


def display_tree(tree, classes, indent=''):
    """prints out decision tree, called recursively"""
    if not isinstance(tree, Node):
        # is leaf node
        print(str(tree))
    else:
        print("if feat[%d] <= %0.3f: (impurity: %.3f num_samples: %d %s)" %
              (tree.feat_index, tree.threshold, tree.impurity,
               tree.num_samples, tree.class_dist))
        # Print the branches
        print(indent + 'T->', end=" ")
        display_tree(tree.left, classes, indent + '  ')
        print(indent + 'F->', end=" ")
        display_tree(tree.right, classes, indent + '  ')


def gini_index(x, _unused):
    _, counts = np.unique(x, return_counts=True)
    proportion = counts / float(len(x))
    return np.sum(proportion * (1.0 - proportion))


def entropy(x):
    """calculates Shannon Entropy for items in 'x'

    :param x: array-line 1-d (one feature column in dataset)
    :return: a float representing Shannon Entropy
    """
    total = len(x)
    _, freqs = np.unique(x, return_counts=True)
    probs = freqs / total
    return -1 * probs.dot(np.log(probs))


def info_gain(x, y):
    """calculates the mutual information gain

    :param x: array-like 1-d (one feature column in dataset)
    :param y: array-like 1-d (target values)
    :return: a float representing information gain
    """
    total = len(y)
    vals, freqs = np.unique(x, return_counts=True)
    val_freqs = zip(vals, freqs)
    EA = 0.0
    for val, freq in val_freqs:
        EA += freq * entropy(y[x == val])
    return entropy(y) - EA / total


class Node(object):
    """represents a node in decision tree"""
    def __init__(self, feat_index, threshold, groups, impurity):
        self.feat_index = feat_index
        self.threshold = threshold
        self.groups = groups
        self.impurity = impurity
        samples = np.concatenate([group['y'] for group in groups])
        self.num_samples = len(samples)
        self.class_dist = np.bincount(samples)
        # defaultdict(int)
        # for group in groups:
        #     targets = group['y']
        #     self.samples += len(targets)
        #     for tgt in targets:
        #         self.dist[tgt] += 1
        self.left = None
        self.right = None

    def get_dist(self, targets):
        """returns an array of counts for each target value"""
        ret_val = []
        for target in targets:
            ret_val.append(self.dist.get(target, 0))
        return ret_val


class CartDecisionTreeClassifier(BaseClassifier):
    """CART algorithm based Decision Tree learner"""
    def __init__(self,
                 criterion="gini",
                 splitter="best",
                 max_depth=5,
                 min_samples_split=20):
        self.criterion = criterion
        self.splitter = splitter
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        # define these attributes to shut up the linter
        self.criterion_mthd = None
        self.tree_ = None
        self.classes_ = None
        self.X_ = None
        self.y_ = None


    def fit(self, X, y):
        """trains a decision tree"""
        self._pre_fit(X, y)

        if "entropy" == self.criterion:
            self.criterion_mthd = info_gain
        else:
            self.criterion_mthd = gini_index

        # _get_best_split() determines which feature and threshold is the best
        #  split point based on the current set of data
        # here we use it to determine what is the best root for the tree
        root = self._get_best_split(self.X_, self.y_)
        # flesh out the rest of the tree, '1' here is the current depth (root)
        # this function recursively calls itself until the tree is complete
        self._split(root, 1)
        # save our generated tree as an attribute named tree_
        self.tree_ = root
        # it is important that this method returns 'self'
        return self

    def predict(self, X):
        """outputs predicted value(s) for data X"""
        X = self._pre_predict(X)
        return [self._predict(self.tree_, row) for row in X]

    def _predict(self, node, row):
        """recursively follows branches of tree until it finds a terminal node
        and returns the terminal nodes value
        """
        if row[node.feat_index] <= node.threshold:
            if isinstance(node.left, Node):
                return self._predict(node.left, row)
            else:
                return node.left
        else:
            if isinstance(node.right, Node):
                return self._predict(node.right, row)
            else:
                return node.right

    # def _get_best_split(self, X, y):
    #     """
    #     returns the best split point for a dataset (what should be root of
    #     subtree)
    #     """
    #     # get distinct set of class values
    #     class_values = np.unique(y)
    #     # initialize some variables, these b_* variables represent the values
    #     # of the best candidate branch
    #     b_feat_index, b_threshold, b_impurity, b_groups = 999, 999, 999, None
    #     for feat_index in range(len(X[0])):
    #         for row in X:
    #             threshold = row[feat_index]
    #             # 'groups' represents the dataset split into (left, right) by
    #             # how each row's value of the current 'feat_index' compares to
    #             # 'theshold' of our current row
    #             candidate_feat_split = \
    #                 self._split_on_feature(feat_index, threshold, X, y)
    #             # the gini index is a value between 0.0 and 1.0.  A 'group'
    #             # with a lower gini index of predicting the class
    #             impurity = self._gini_index(candidate_feat_split, class_values)
    #             if impurity < b_impurity:
    #                 # this iteration is an improvement, store the values
    #                 b_feat_index, b_threshold, b_impurity, b_groups = \
    #                     feat_index, threshold, impurity, candidate_feat_split
    #     return Node(b_feat_index, b_threshold, b_groups, b_impurity)

    def _get_indices_except_current_row(self, y, row_idx):
        num_rows = len(y)
        left_idx = row_idx
        if left_idx == num_rows:
            left_idx -= 1
        left = np.arange(0, left_idx)
        right = np.arange(left_idx + 1, num_rows)
        indices = np.concatenate([left, right])
        return indices

    def _get_best_split(self, X, y):
        """
        returns the best split point for a dataset (what should be root of
        subtree)
        """
        # get distinct set of class values
        # class_values = np.unique(y)
        # initialize some variables, these b_* variables represent the values
        # of the best candidate branch
        b_feat_idx, b_impurity, b_threshold = 999, 999, 999
        num_feats = len(X[0])
        num_rows = len(X)
        for feat_idx in range(num_feats):
            for row_idx in range(num_rows):
                row = X[row_idx, :]
                threshold = row[feat_idx]
                # 'groups' represents the dataset split into (left, right) by
                # how each row's value of the current 'feat_index' compares to
                # 'theshold' of our current row
                indices = self._get_indices_except_current_row(y, row_idx)
                X_except_this_row = X[indices, :]
                y_except_this_row = y[indices]
                impurity = self.criterion_mthd(X_except_this_row,
                                               y_except_this_row)
                if impurity < b_impurity:
                    b_impurity = impurity
                    b_feat_idx = feat_idx
                    b_threshold = threshold

        groups = self._split_on_feature(b_feat_idx, b_threshold, X, y)
        return Node(b_feat_idx, b_threshold, groups, b_impurity)

    def _split_on_feature(self, feat_index, threshold, X, y):
        """
        return dataset split into two groups (left, right)
        based on how each of the datasets rows value for the feature identified
        by feat_idx compares to feat_value. if the current row's value
        is < feat_value place it in the left group, else right
        """
        feat_vals = X[:, feat_index]
        left_idx = np.where(feat_vals <= threshold)
        right_idx = np.where(feat_vals > threshold)
        # can I get away with the indices only rather than copying the data???
        left_X, left_y = X[left_idx], y[left_idx]
        right_X, right_y = X[right_idx], y[right_idx]
        return [{'X': left_X, 'y': left_y}, {'X': right_X, 'y': right_y}]
    #
    # def _entropy(self, X, feat_index):
    #     """
    #     Calculates the entropy of the given data set for the target attribute.
    #     """
    #     val_freq = defaultdict(float)
    #     data_entropy = 0.0
    #
    #     # Calculate the frequency of each of the values in the target attr
    #     for row in X:
    #         val_freq[row[feat_index]] += 1.0
    #         # if val_freq in record[target_attr]:
    #         #     val_freq[record[target_attr]] += 1.0
    #         # else:
    #         #     val_freq[record[target_attr]] = 1.0
    #
    #     # Calculate the entropy of the data for the target attribute
    #     for freq in val_freq.values():
    #         data_entropy += (-freq/len(X)) * math.log(freq/len(X), 2)
    #
    #     return data_entropy
    #
    # def _gain(self, X, feat_index, target_attr):
    #     """
    #     Calculates the information gain (reduction in entropy) that would
    #     result by splitting the data on the chosen attribute (attr).
    #     """
    #     val_freq = defaultdict(float)
    #     subset_entropy = 0.0
    #
    #     # Calculate the frequency of each of the values in the target attribute
    #     for row in X:
    #         val_freq[row[feat_index]] += 1.0
    #         # if val_freq in record[attr]:
    #         #     val_freq[record[attr]] += 1.0
    #         # else:
    #         #     val_freq[record[attr]] = 1.0
    #
    #     # Calculate the sum of the entropy for each subset of records weighted
    #     # by their probability of occuring in the training set.
    #     for val in val_freq:
    #         val_prob = val_freq[val] / sum(val_freq.values())
    #         data_subset = [row for row in X if row[feat_index] == val]
    #         subset_entropy += val_prob * self._entropy(data_subset, target_attr)
    #
    #     # Subtract the entropy of the chosen attribute from the entropy of the
    #     # whole data set with respect to the target attribute (and return it)
    #     return self._entropy(X, target_attr) - subset_entropy
    #
    # def gini(self, X, y, feat_index, threshold):
    #     gini_index = 0.0
    #     left = np.where(X)
    #     num_samples = len(y)
    #     for class_value in self.classes_:
    #         # print "curr class_value: %d" % class_value
    #         class_val_cnt = np.sum(y == class_value)
    #         # print "\tcurr class_val_cnt: %d" % class_val_cnt
    #         proportion = class_val_cnt / float(num_samples)
    #         # print "\tcurr proportion: %f" % proportion
    #         gini_index += (proportion * (1.0 - proportion))
    #         # print "\tcumulative gini_index: %f" % gini_index
    #     return gini_index
    #
    # def _gini_index(self, groups, class_values):
    #     """
    #     return the calculated Gini index for a split dataset
    #
    #     The name doesn't meaning anything - it was named after economist
    #     Corrando Gini, who first used it for comparing incomes in populations.
    #
    #     The index represents the equality of values in a set of data
    #     0.0 represents perfect equality whereas 1.0 represents totally
    #     inequality
    #     """
    #     gini = 0.0
    #     for class_value in class_values:
    #         for group in groups:
    #             group_targets = group['y']
    #             group_size = len(group_targets)
    #             # simple prevention of divide by zero
    #             if group_size == 0:
    #                 continue
    #             # count how many times class_value occurs in group's target
    #             # values
    #             grp_class_val_cnt = np.sum(group_targets == class_value)
    #             # proportion is how many times the current class_value is
    #             # present in the current group divided by the size of the
    #             # group
    #             proportion = grp_class_val_cnt / float(group_size)
    #             # add the normalized proportion value to the gini index
    #             gini += (proportion * (1.0 - proportion))
    #     return gini

    def _to_terminal(self, grp):
        """return a terminal node - simply the majority class in 'y'"""
        return np.argmax(np.bincount(grp['y']))

    def _split(self, node, depth):
        """Create child splits for a node or make terminal"""
        left_gr, right_gr = node.groups
        node.groups = None
        # check for a no split - both left and right will be same terminal node
        # and this node should be pruned later (how can I pre-prune?)
        if not len(left_gr['y']) or not len(right_gr['y']):
            term = None
            if len(left_gr['y']):
                term = self._to_terminal(left_gr)
            elif len(right_gr['y']):
                term = self._to_terminal(right_gr)
            else:
                raise Exception("ERROR: unable create terminal node. "
                                "both left and right splits are empty")
            node.left, node.right = term, term
            return
        # if we've hit max depth force both left and right to be terminal nodes
        if depth >= self.max_depth:
            node.left = self._to_terminal(left_gr)
            node.right = self._to_terminal(right_gr)
            return
        # process left child
        if len(left_gr['y']) <= self.min_samples_split:
            node.left = self._to_terminal(left_gr)
        else:
            node.left = self._get_best_split(left_gr['X'], left_gr['y'])
            self._split(node.left, depth+1)
        # process right child
        if len(right_gr['y']) <= self.min_samples_split:
            node.right = self._to_terminal(right_gr)
        else:
            node.right = self._get_best_split(right_gr['X'], right_gr['y'])
            self._split(node.right, depth+1)
