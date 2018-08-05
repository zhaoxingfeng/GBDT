#!/usr/local/bin/python
# -*- coding: utf-8 -*-
"""
@Time: 2018/8/1 10:35
@Author: zhaoxingfeng
@Function：GBDT二分类，连续特征
@Version: V1.0
参考文献：
[1] Friedman JH. Greedy Function Approximation-A Gradient Boosting Machine[J].The Annals of Statistics,2001,29(5),1189-1232.
[2] kingsam_. GBDT原理与Sklearn源码分析-回归篇[DB/OL].https://blog.csdn.net/qq_22238533/article/details/79185969.
[3] liudragonfly. GBDT[DB/OL].https://github.com/liudragonfly/GBDT.
"""
import pandas as pd
import numpy as np
from math import exp, log
from tree import BaseDecisionTree
import warnings
warnings.filterwarnings('ignore')
pd.set_option('precision', 4)
pd.set_option('display.max_rows', 50)
pd.set_option('display.width', 1000)
pd.set_option('display.max_columns', 1000)
pd.set_option('expand_frame_repr', False)


class GradientBoostingClassifier(object):
    def __init__(self, n_estimators=100, max_depth=2**31-1, learning_rate=0.1, min_samples_split=2,
                 min_samples_leaf=1, subsample=1.0, colsample_bytree=1.0, max_bin=225, random_state=None):
        """Construct a gradient boosting model

        Parameters
        ----------
        n_estimators : int, optional (default=100)
            Number of boosted trees to fit.
        max_depth : int, optional (default=2**31-1)
            Maximum tree depth for base learners, -1 means no limit.
        learning_rate : float, optional (default=0.1)
            Boosting learning rate.
        min_samples_split : int, optional (default=2)
            The minimum number of samples required to split an internal node.
        min_samples_leaf : int, optional (default=1)
            The minimum number of samples required to be at a leaf node.
        subsample : float, optional (default=1.)
            Subsample ratio of the training instance.
        colsample_bytree : float, optional (default=1.)
            Subsample ratio of columns when constructing each tree.
        max_bin: int or None, optional (default=225))
            Max number of discrete bins for features.
        random_state : int or None, optional (default=None)
            Random number seed.
        """
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.max_bin = max_bin
        self.random_state = random_state
        self.f_0 = None
        self.trees = dict()

    def fit(self, dataset, targets):
        targets = targets.to_frame(name='label')
        targets['label'] = targets['label'].apply(lambda y: 1 if y == 1 else -1)
        if targets['label'].unique().__len__() != 2:
            raise ValueError("There must be two class for targets!")
        if len([x for x in dataset.columns if dataset[x].dtype in ['int32', 'float32', 'int64', 'float64']]) \
                != len(dataset.columns):
            raise ValueError("The features dtype must be int or float!")

        # 计算f_0,Algorithm 5,line 1
        mean = 1.0 * sum(targets['label']) / len(targets['label'])
        self.f_0 = 0.5 * log((1 + mean) / (1 - mean))
        targets['f_m'] = self.f_0

        for stage in range(self.n_estimators):
            print(str(stage).center(80, '='))
            # 计算梯度,Algorithm 5,line 3
            targets['y_tilde'] = targets.apply(lambda x: 2 * x['label'] / (1 + exp(2 * x['label'] * x['f_m'])), axis=1)
            # 建立单棵回归树,Algorithm 5,line 4
            tree = BaseDecisionTree(self.max_depth, self.min_samples_split, self.min_samples_leaf,
                                    self.subsample, self.colsample_bytree,
                                    self.max_bin, self.random_state)
            tree.fit(dataset, targets['y_tilde'])
            self.trees[stage] = tree
            # 更新f_m,Algorithm 5,line 6
            targets['f_m'] = targets['f_m'] + self.learning_rate * tree.f_m

    def predict_proba(self, dataset):
        res = []
        for index, row in dataset.iterrows():
            f_value = self.f_0
            for stage, tree in self.trees.items():
                f_value += self.learning_rate * tree.predict(row)
            p_0 = 1.0 / (1 + exp(2 * f_value))
            res.append([p_0, 1 - p_0])
        return np.array(res)

    def predict(self, dataset):
        res = []
        for p in self.predict_proba(dataset):
            label = 0 if p[0] >= p[1] else 1
            res.append(label)
        return np.array(res)


if __name__ == '__main__':
    df = pd.read_csv(r"source/pima indians.csv")
    gbdt = GradientBoostingClassifier(n_estimators=5,
                                      max_depth=6,
                                      learning_rate=0.3,
                                      min_samples_split=10,
                                      min_samples_leaf=3,
                                      subsample=0.8,
                                      max_bin=225,
                                      colsample_bytree=0.8,
                                      random_state=4)

    train_count = int(0.7 * len(df))
    gbdt.fit(df.ix[:train_count, :-1], df.ix[:train_count, 'Class'])
    prob = gbdt.predict_proba(df.ix[train_count:, :-1])[:, 1]

    from sklearn import metrics
    fpr, tpr, threshold = metrics.roc_curve(df.ix[train_count:, 'Class'], prob, pos_label=1)
    print(metrics.auc(fpr, tpr))
