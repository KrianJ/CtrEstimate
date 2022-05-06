#!/usr/bin/env python
# -*- coding:utf-8 -*-
# author: KrianJ
# datetime:2022/5/6 22:12
# software: PyCharm-main

import numpy as np
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score

from models.LR import LinearRegression

X, y = load_iris(return_X_y=True)

lr = LinearRegression(learning_rate=1e-4)
w = lr.train(X, y)

pred = np.dot(w, np.insert(X, X.shape[1], 1, axis=1).T)
labels = [1 if i >= 0.5 else 0 for i in pred]
acc = accuracy_score(y, labels)
print(acc)

