#!/usr/bin/env python
# -*- coding:utf-8 -*-
# author: KrianJ
# datetime:2022/5/6 22:12
# software: PyCharm-main

import numpy as np
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression as sk_LR

from LR import LogisticRegression as LR

X, y = load_iris(return_X_y=True)
X, y = X[:100, :], y[:100]

# 自定义LR
lr = LR(learning_rate=1e-1, tolerance=1e-4, n_iter=500)
lr.fit(X, y)
w, b = lr.w, lr.b
pred = np.dot(w, X.T) + b

# sklearn的LR
sk_lr = sk_LR(l1_ratio=1e-1, tol=1e-4, max_iter=500, solver='sag')
sk_lr.fit(X=X, y=y)
sk_w = sk_lr.coef_
sk_b = sk_lr.intercept_
sk_pred = sk_lr.predict(X)

acc = accuracy_score(y, pred)
print(acc)

