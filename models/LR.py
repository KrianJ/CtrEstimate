#!/usr/bin/env python
# -*- coding:utf-8 -*-
# author: KrianJ
# datetime:2022/5/6 20:21
# software: PyCharm-LR

import numpy as np
from utils.base_func import sigmoid


class LogisticRegression:
    """
    Logistic Regression:
    LR一直是CTR预估的benchmark模型，具有简单、易于并行化实现、可解释性强等优点，
    但是LR模型中的特征是默认相互独立的，遇到具有交叉可能性的特征需进行大量的人工特征工程进行交叉(连续特征的离散化、特征交叉)，
    不能处理目标和特征之间的非线性关系。
    """
    def __init__(self, learning_rate=1e-2, n_iter=500, tolerance=1e-4):
        self.lr = learning_rate
        self.w = 0.
        self.b = 1.
        self.n_iter = n_iter
        self.tolerance = tolerance

        self.X = None
        self.y = None

    @staticmethod
    def gd_optimize(w, b, X, Y):
        """
        全量梯度下降
        :param w: 权重(m,)
        :param X: 数据集(n,m)
        :param Y: 标签(n,)
        :return:
        """
        n_samples = X.shape[0]
        # Gradient Descent
        p = sigmoid(np.dot(w, X.T) + b)
        # 损失
        loss = -1 / n_samples * np.sum(Y * np.log(p) + (1 - Y) * np.log(1 - p))
        # 损失对w的偏导
        dw = -1 / n_samples * np.dot(X.T, (Y - p))
        db = -1 / n_samples * np.sum(p - Y)

        return loss, dw, db

    def fit(self, dataset: np.ndarray, labels, weight_init='zero'):
        """
        训练数据集
        :param dataset: 数据集X
        :param labels: 标签y
        :param weight_init: 权重初始化方式
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

        for iter in range(self.n_iter):
            # 1.求损失函数对w,b的梯度
            loss, dw, db = self.gd_optimize(self.w, self.b, self.X, self.y)

            # 更新权重w和偏置b
            self.w = self.w - self.lr * dw
            self.b = self.b - self.lr * db
            costs.append(loss)

            if iter % 100 == 0:
                print("iter: {}, loss: {}".format(iter, loss))
            if loss <= self.tolerance:
                print('loss({}) [le] tolerance({}), iter ended'.format(loss, self.tolerance))
                break
