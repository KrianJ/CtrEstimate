#!/usr/bin/env python
# -*- coding:utf-8 -*-
# author: KrianJ
# datetime:2022/5/8 12:05
# software: PyCharm-predict_fm_torch
import torch
from torch.optim import SGD
import torch.nn.functional as F
from sklearn.metrics import accuracy_score

from models.FM_torch import FM_Layer
from utils.load_data import load_criteo_data

if __name__ == '__main__':
    # 加载数据集
    (X_train, y_train), (X_test, y_test), _ = load_criteo_data('dataset/criteo_sample.csv')
    X_train = torch.tensor(X_train, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32)

    n, dim = X_train.shape
    # 初始化模型
    k = 8
    lr = 0.05
    epoch = 1000

    model = FM_Layer(dim=dim, k=k)      # 模型
    optimizer = SGD(model.parameters(), lr=lr)  # 优化器
    criterion = F.binary_cross_entropy       # 损失
    # 训练数据
    for i in range(epoch):
        model.train()
        logits = model(X_train)
        loss = criterion(logits, y_train)
        optimizer.zero_grad()
        loss.backward()         # 根据损失反向更新
        optimizer.step()
        if i % 100 == 0 and epoch:
            print("epoch: {}, loss: {}".format(i, loss))
    # 测试
    pred = model(X_test)
    pred = [1 if x > 0.5 else 0 for x in pred]
    print('Test acc: {}'.format(accuracy_score(y_test, pred)))

