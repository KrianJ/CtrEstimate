#!/usr/bin/env python
# -*- coding:utf-8 -*-
# author: KrianJ
# datetime:2022/5/6 20:21
# software: PyCharm-LR

import numpy as np


class LogisticRegression:
    def __init__(self, learning_rate=1e-2, n_iter=500, tolerance=1e-4):
        self.lr = learning_rate
        self.w = 0.
        self.b = 1.
        self.n_iter = n_iter
        self.tolerance = tolerance

        self.X = None
        self.y = None

    @staticmethod
    def sigmoid(x):
        """sigmoid函数"""
        return 1 / (1 + np.exp(-x))

    def gd_optimize(self, w, X, Y):
        """全量梯度下降"""
        n_samples = X.shape[0]
        # Gradient Descent
        p = self.sigmoid(np.dot(w, X.T))
        # 损失
        loss = -1 / n_samples * np.sum(Y * np.log(p) + (1 - Y) * np.log(1 - p))
        # 损失对w的偏导
        dw = -1 / n_samples * np.dot(X.T, (Y - p))
        db = -1 / n_samples * np.sum(p - Y)

        return loss, dw, db

    def sgd_optimize(self, w, x, y):
        pass

    def fit(self, dataset: np.ndarray, labels, weight_init='zero', method='gd'):
        """
        训练数据集
        :param dataset: 数据集X
        :param labels: 标签y
        :param weight_init: 权重初始化方式
        :param method: 梯度下降方式, ['gd', 'sgd']
        :return:
        """
        assert dataset.ndim == 2
        assert len(labels) == dataset.shape[0]

        n_samples, n_features = dataset.shape
        self.X = dataset
        self.y = labels
        # 初始化w
        self.w = np.random.normal(0, 0.01, n_features) if weight_init == 'random' else np.zeros(n_features)
        costs = []  # 记录损失

        # 1.求损失函数对w的梯度
        if method == 'gd':
            optim = self.gd_optimize
        elif method == 'sgd':
            optim = self.sgd_optimize

        for iter in range(self.n_iter):
            loss, dw, db = optim(self.w, self.X, self.y)
            # 更新权重w和偏置b
            self.w = self.w - self.lr * dw
            self.b = self.b - self.lr * db
            costs.append(loss)

            if iter % 100 == 0:
                print("iter: {}, loss: {}".format(iter, loss))
            if loss <= self.tolerance:
                print('loss({}) [le] tolerance({}), iter ended'.format(loss, self.tolerance))
                break
