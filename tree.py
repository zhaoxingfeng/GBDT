#!/usr/local/bin/python
# -*- coding: utf-8 -*-
"""
@Time: 2018/7/31 12:25
@Author: zhaoxingfeng
@Function：CART回归树，分裂准则为MSE
@Version: V1.0
"""
import pandas as pd
import numpy as np
from math import log, exp
import copy
import random


class Tree(object):
    def __init__(self):
        self.best_split_feature = None
        self.best_split_value = None
        self.tree_left = None
        self.tree_right = None
        self.leaf_node = None

    def calc_predict_value(self, dataset):
        if self.leaf_node:
            return self.leaf_node
        elif dataset[self.best_split_feature] <= self.best_split_value:
            return self.tree_left.calc_predict_value(dataset)
        else:
            return self.tree_right.calc_predict_value(dataset)

    def describe_tree(self):
        if not self.tree_left or not self.tree_right:
            return str(self.leaf_node)
        left_info = self.tree_left.describe_tree()
        right_info = self.tree_right.describe_tree()
        tree_structure = "{split_feature:" + str(self.best_split_feature) + \
                         ",split_value:" + str(self.best_split_value) + \
                         ",left_tree:" + left_info + \
                         ",right_tree:" + right_info + "}"
        return tree_structure

class BaseDecisionTree(object):
    def __init__(self, max_depth=2**31-1, min_samples_split=2, min_samples_leaf=1, subsample=1.0,
                 colsample_bytree=1.0, max_bin=100, random_state=None):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.random_state = random_state
        self.max_bin = max_bin
        self.tree = Tree()
        self.f_m = None

    def fit(self, dataset, targets):
        dataset_copy = copy.deepcopy(dataset)
        targets_copy = copy.deepcopy(targets)

        if self.random_state:
            random.seed(self.random_state)
        if self.subsample < 1.0:
            subset_index = random.sample(range(len(targets)), int(self.subsample*len(targets)))
            dataset_copy = dataset_copy.iloc[subset_index, :].reset_index(drop=True)
            targets_copy = targets_copy[subset_index].reset_index(drop=True)
        if self.colsample_bytree < 1.0:
            subcol_index = random.sample(dataset_copy.columns, int(self.colsample_bytree*len(dataset_copy.columns)))
            dataset_copy = dataset_copy[subcol_index]

        self.tree = self._fit(dataset_copy, targets_copy, depth=0)
        self.f_m = dataset.apply(lambda x: self.predict(x), axis=1)
        return self

    def _fit(self, dataset, targets, depth):
        if targets.unique().__len__() == 1 or dataset.__len__() <= self.min_samples_split:
            tree = Tree()
            tree.leaf_node = self.calc_leaf_value(targets)
            return tree

        if depth < self.max_depth:
            print(str(depth).center(20, '='))
            best_split_feature, best_split_value = self.choose_best_feature(dataset, targets)
            left_dataset, right_dataset, left_targets, right_targets = \
                self.split_dataset(dataset, targets, best_split_feature, best_split_value)

            tree = Tree()
            if left_dataset.__len__() <= self.min_samples_leaf or \
                    right_dataset.__len__() <= self.min_samples_leaf:
                tree.leaf_node = self.calc_leaf_value(targets)
                return tree
            else:
                tree.best_split_feature = best_split_feature
                tree.best_split_value = best_split_value
                tree.tree_left = self._fit(left_dataset, left_targets, depth + 1)
                tree.tree_right = self._fit(right_dataset, right_targets, depth + 1)
                return tree
        else:
            tree = Tree()
            tree.leaf_node = self.calc_leaf_value(targets)
            return tree

    def choose_best_feature(self, dataset, targets):
        best_mse = float('+inf')
        best_split_feature = None
        best_split_value = None

        for feature in dataset.columns:
            if dataset[feature].unique().__len__() <= 100:
                unique_values = dataset[feature].unique()
            else:
                unique_values = np.unique(np.percentile(dataset[feature], np.linspace(0, 100, self.max_bin)))
            for split_value in unique_values:
                left_targets = targets[dataset[feature] <= split_value]
                right_targets = targets[dataset[feature] > split_value]
                mse = self.calc_mse(left_targets) + self.calc_mse(right_targets)

                if mse < best_mse:
                    best_split_feature = feature
                    best_split_value = split_value
                    best_mse = mse
        return best_split_feature, best_split_value

    @staticmethod
    def calc_mse(targets):
        if len(targets) < 2:
            return 0
        mean = 1.0 * sum(targets) / len(targets)
        mse = sum([(y_i - mean) ** 2 for y_i in targets])
        return mse

    @staticmethod
    def split_dataset(dataset, targets, split_feature, split_value):
        left_dataset = dataset[dataset[split_feature] <= split_value]
        right_dataset = dataset[dataset[split_feature] > split_value]
        left_targets = targets[dataset[split_feature] <= split_value]
        right_targets = targets[dataset[split_feature] > split_value]
        return left_dataset, right_dataset, left_targets, right_targets

    @staticmethod
    def calc_leaf_value(targets):
        # 计算叶子节点值,Algorithm 5,line 5
        sum1 = sum(targets)
        sum2 = sum([abs(y_i) * (2 - abs(y_i)) for y_i in targets])
        return 1.0 * sum1 / sum2

    def predict(self, dataset):
        return self.tree.calc_predict_value(dataset)

    def print_tree(self):
        return self.tree.describe_tree()


if __name__ == '__main__':
    df = pd.read_csv(r"source/pima indians.csv")
    df['Class'] = df['Class'].map({1: 1, 0: -1})
    mean_targets = sum(df['Class']) * 1.0 / len(df['Class'])
    f0 = 1.0 / 2 * log((1 + mean_targets) * 1.0 / (1 - mean_targets))
    df['Class'] = df['Class'].apply(lambda x: 2.0 * x / (1 + exp(2.0 * x * f0)))

    decision_tree = BaseDecisionTree(max_depth=20,
                                     min_samples_split=20,
                                     min_samples_leaf=5,
                                     subsample=0.8,
                                     colsample_bytree=0.8,
                                     random_state=42,
                                     max_bin=100)

    decision_tree.fit(df.iloc[:, :-1], df['Class'])
    print(decision_tree.print_tree())
    print(decision_tree.predict(df.iloc[0, :-1]))
