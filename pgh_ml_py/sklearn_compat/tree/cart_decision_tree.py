"""CART on the Bank Note dataset"""
from __future__ import (absolute_import, division,
                        unicode_literals, print_function)
# from collections import defaultdict
# import math
import sys

import numpy as np
# import pandas as pd

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


def group_gini_index(groups, class_values):
    """
    return the calculated Gini index for a split dataset

    The index represents the equality of values in a set of data
    0.0 represents perfect equality whereas 1.0 represents totally
    inequality
    """
    gini_index = 0.0
    for class_value in class_values:
        for group in groups:
            group_targets = group['y']
            group_size = len(group_targets)
            # simple prevention of divide by zero
            if group_size == 0:
                continue
            # count how many times class_value occurs in group's target
            # values
            grp_class_val_cnt = np.sum(group_targets == class_value)
            # proportion is how many times the current class_value is
            # present in the current group divided by the size of the
            # group
            proportion = grp_class_val_cnt / float(group_size)
            # add the normalized proportion value to the gini index
            gini_index += (proportion * (1.0 - proportion))
            # gini += (proportion ** 2)
    print ("grp_gini_index: %f" % gini_index)
    return gini_index

def gini(X, y, _unused):
    _, counts = np.unique(y, return_counts=True)
    proportion = counts / float(len(y))
    return np.sum(proportion * (1 - proportion))


def entropy(x):
    """calculates Shannon Entropy for items in 'x'

    :param x: array-line 1-d (one feature column in dataset)
    :return: a float representing Shannon Entropy
    """
    total = len(x)
    _, freqs = np.unique(x, return_counts=True)
    probs = freqs / float(total)
    return -1 * probs.dot(np.log(probs))


def info_gain(X, y, feat_index):
    """calculates the mutual information gain

    :param x: array-like 1-d (one feature column in dataset)
    :param y: array-like 1-d (target values)
    :return: a float representing information gain
    """
    total = len(y)
    if 0 == total:
        return 0.0
    x = X[:, feat_index]
    vals, freqs = np.unique(x, return_counts=True)
    val_freqs = zip(vals, freqs)
    # print("x: %s" % x)
    # print("y: %s" % y)
    # print("val_freqs: %s" % val_freqs)
    # print("len_x: %d" % len(x))
    # sys.exit(0)
    EA = 0.0
    for val, freq in val_freqs:
        # print("val: %s freq: %s" % (val, freq))
        idx = np.where(np.isclose(x, val))
        if idx:
            # print("idx: %s" % idx)
            h_idx = entropy(y[idx])
            EA += freq * h_idx
    h_y = entropy(y)
    gain = 0.0
    try:
        gain = h_y - EA / float(total)
    except ZeroDivisionError:
        print ("h_y: %s" % h_y)
        print ("EA: %s" % EA)
        print("total: %d" % total)

    return gain

#
# def _entropy(self, X, feat_index):
#     """
#     Calculates the entropy of the given dataset for the target attribute.
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
#     # Calculate the frequency of each value in the target attribute
#     for row in X:
#         val_freq[row[feat_index]] += 1.0
#         # if val_freq in record[attr]:
#         #     val_freq[record[attr]] += 1.0
#         # else:
#         #     val_freq[record[attr]] = 1.0
#
#     # Calculate the sum of the entropy for each subset of records
#     # weighted by their probability of occuring in the training set.
#     for val in val_freq:
#         val_prob = val_freq[val] / sum(val_freq.values())
#         data_subset = [row for row in X if row[feat_index] == val]
#         subset_entropy += val_prob * self._entropy(data_subset,
#                                                    target_attr)
#
#     # Subtract the entropy of the chosen attribute from the entropy of
#     # the whole data set with respect to the target attribute
#     return self._entropy(X, target_attr) - subset_entropy


class Split(object):
    """represents a split point in the dataset which may/may not become a node
    """
    def __init__(self, left_x, left_y, right_x, right_y):
        self.left_x = left_x
        self.left_y = left_y
        self.right_x = right_x
        self.right_y = right_y

    def __repr__(self):
        return "Split(left_x=%s, left_y=%s, right_x=%s, right_y=%s)" % \
            (self.left_x, self.left_y, self.right_x, self.right_y)


class Node(object):
    """represents a node in decision tree"""
    def __init__(self, feat_index, threshold, split, impurity):
        self.feat_index = feat_index
        self.threshold = threshold
        self.split = split
        self.impurity = impurity
        samples = np.concatenate([split.left_y, split.right_y])
        self.num_samples = len(samples)
        self.class_dist = np.bincount(samples)
        self.left = None
        self.right = None


class CartDecisionTreeClassifier(BaseClassifier):
    """CART algorithm based Decision Tree learner"""
    def __init__(self,
                 criterion="gini",
                 splitter="best",
                 max_depth=None,
                 min_samples_split=2):
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

        if "entropy" == self.criterion:  # noqa
            self.cost_func = info_gain
        else:
            self.cost_func = gini

        if self.max_depth is None:
            self.max_depth = float('inf')
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
    #
    # def _get_best_feature(self, X, y):
    #     best_impurity = float('inf')
    #     best_feat_idx = None
    #     for feat_idx in range(len(X[0])):
    #         # print("feat_idx: %d" % feat_idx)
    #         x = X[:, feat_idx]
    #         feat_impurity = self.criterion_mthd(x, y)
    #         # print("feat_impurity: %f" % feat_impurity)
    #         if feat_impurity < best_impurity:
    #             best_impurity = feat_impurity
    #             best_feat_idx = feat_idx
    #     return best_feat_idx
    #
    # def _get_best_threshold(self, X, y, feat_idx):
    #     best_threshold = float('inf')
    #     best_impurity = float('inf')
    #     num_rows = len(y)
    #     # print(num_rows)
    #     for row_idx in range(num_rows):
    #         # iterate through all rows, holding out the current row
    #         # (so we're not comparing the same thing each time)
    #         # and which held out row provides the most information
    #         current_row = X[row_idx, :]
    #         threshold = current_row[feat_idx]
    #         x = X[:, feat_idx]
    #         x_holdout = np.delete(x, row_idx)
    #         y_holdout = np.delete(y, row_idx)
    #         # this time we want the impurity of y, so reverse the params
    #         impurity = self.criterion_mthd(y_holdout, x_holdout)
    #         if impurity < best_impurity:
    #             best_impurity = impurity
    #             best_threshold = threshold
    #     return best_threshold, best_impurity
    #
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
    #             # how each row's value of the current 'feat_index' compares
    #             # to 'theshold' of our current row
    #             candidate_feat_split = \
    #                 self._split_on_feature(feat_index, threshold, X, y)
    #             # the gini index is a value between 0.0 and 1.0.  A 'group'
    #             # with a lower gini index of predicting the class
    #             impurity = self._gini_index(candidate_feat_split,
    #                                         class_values)
    #             if impurity < b_impurity:
    #                 # this iteration is an improvement, store the values
    #                 b_feat_index, b_threshold, b_impurity, b_groups = \
    #                     feat_index, threshold, impurity, candidate_feat_split
    #     return Node(b_feat_index, b_threshold, b_groups, b_impurity)

    # def _get_best_split(self, X, y):
    #     """
    #     returns the best split point for a dataset (what should be root of
    #     subtree)
    #     """
    #     # get distinct set of class values
    #     # class_values = np.unique(y)
    #     # initialize some variables, these b_* variables represent the values
    #     # of the best candidate branch
    #     b_feat_idx, b_impurity, b_threshold = 999, 999, 999
    #     num_feats = len(X[0])
    #     num_rows = len(X)
    #
    #     for feat_idx in range(num_feats):
    #         for row_idx in range(num_rows):
    #             row = X[row_idx, :]
    #             threshold = row[feat_idx]
    #             # 'groups' represents the dataset split into (left, right) by
    #             # how each row's value of the current 'feat_index' compares
    #             # to 'theshold' of our current row
    #             indices = self._get_indices_except_current_row(y, row_idx)
    #             X_except_this_row = X[indices, :]
    #             y_except_this_row = y[indices]
    #             impurity = self.criterion_mthd(X_except_this_row,
    #                                            y_except_this_row)
    #             if impurity < b_impurity:
    #                 b_impurity = impurity
    #                 b_feat_idx = feat_idx
    #                 b_threshold = threshold
    #
    #     groups = self._split_on_feature(b_feat_idx, b_threshold, X, y)
    #     return Node(b_feat_idx, b_threshold, groups, b_impurity)

    # def _get_best_split(self, X, y):
    #     """
    #     returns the best split point for a dataset (what should be root of
    #     subtree)
    #     """
    #     # get distinct set of class values
    #     # class_values = np.unique(y)
    #     # initialize some variables, these b_* variables represent the values
    #     # of the best candidate branch
    #     best_feat_idx, best_threshold, best_split = None, float('inf'), None
    #     # num_feats = len(X[0])
    #     # num_rows = len(y)
    #     #
    #     best_feat_idx = self._get_best_feature(X, y)
    #     best_threshold, best_impurity = self._get_best_threshold(X, y, best_feat_idx)
    #     groups = self._split_on_feature(best_feat_idx, best_threshold, X, y)
    #     return Node(best_feat_idx, best_threshold, groups, best_impurity)


    def _get_best_split(self, X, y):
        """
        returns the best split point for a dataset (what should be root of
        subtree)
        """
        best_feat_idx, best_split = None, None
        best_threshold, best_impurity = float('inf'), float('inf')
        num_feats = len(X[0])
        for feat_idx in range(num_feats):
            thresholds = np.unique(X[:, feat_idx])
            for threshold in thresholds:
                split = self._gen_split(X, y, feat_idx, threshold)
                # groups = ({'X': split.left_x, 'y': split.left_y},
                #           {'X': split.right_x, 'y': split.right_y})
                # group_gini_index(groups, self.classes_)
                cost_func = self.cost_func
                # print("split: %s" % split)
                # sys.exit(0)
                impurity = (cost_func(split.left_x, split.left_y, feat_idx) +
                            cost_func(split.right_x, split.right_y, feat_idx))
                # print("impurity: %f" % impurity)
                # print("""
                # group_impurity: %f
                # impurity: %f
                # """ % (group_impurity, impurity))

                if impurity < best_impurity:
                    best_impurity = impurity
                    best_threshold = threshold
                    best_feat_idx = feat_idx
                    best_split = split

        return Node(best_feat_idx, best_threshold, best_split, best_impurity)

    def _gen_split(self, X, y, feat_idx, threshold):
        """
        return dataset split into two groups (left, right)
        based on how each of the datasets rows value for the feature identified
        by feat_idx compares to feat_value. if the current row's value
        is < feat_value place it in the left group, else right
        """
        feat_vals = X[:, feat_idx]
        left_idx = np.where(feat_vals <= threshold)
        right_idx = np.where(feat_vals > threshold)
        return Split(X[left_idx], y[left_idx], X[right_idx], y[right_idx])

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
    #             # gini += (proportion ** 2)
    #
    #     return gini

    def _to_terminal(self, y):
        """return a terminal node - simply the majority class """
        return np.argmax(np.bincount(y))

    def _split(self, node, depth):
        """Create child splits for a node or make terminal"""
        ns = node.split
        l_X, l_y, r_X, r_y = ns.left_x, ns.left_y, ns.right_x, ns.right_y
        node.split = None
        # check for a no split - both left and right will be same terminal node
        # and this node should be pruned later (how can I pre-prune?)
        if not len(l_y) or not len(r_y):
            term = None
            if len(l_y):
                term = self._to_terminal(l_y)
            elif len(r_y):
                term = self._to_terminal(r_y)
            else:
                raise Exception("ERROR: unable create terminal node. "
                                "both left and right splits are empty")
            node.left, node.right = term, term
            return
        # if we've hit max depth force both left and right to be terminal nodes
        if depth >= self.max_depth:
            node.left = self._to_terminal(l_y)
            node.right = self._to_terminal(r_y)
            return
        # process left child
        if len(l_y) <= self.min_samples_split:
            node.left = self._to_terminal(l_y)
        else:
            node.left = self._get_best_split(l_X, l_y)
            self._split(node.left, depth+1)
        # process right child
        if len(r_y) <= self.min_samples_split:
            node.right = self._to_terminal(r_y)
        else:
            node.right = self._get_best_split(r_X, r_y)
            self._split(node.right, depth+1)
