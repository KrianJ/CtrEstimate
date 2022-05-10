#!/usr/bin/env python
# -*- coding:utf-8 -*-
# author: KrianJ
# datetime:2022/5/9 22:09
# software: PyCharm-predict_widedeep_torch

import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score

from utils.load_data import load_criteo_data
from models.WideDeep_torch import WideDeep

if __name__ == '__main__':
    # 加载数据
    (X_train, y_train), (X_test, y_test), (dense_fea, sparse_fea) = \
        load_criteo_data('dataset/criteo_sample.csv', sparse_encoding='both')
    X_train = torch.tensor(X_train, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32)

    # 模型参数
    sparse_one_hot_dim = sparse_fea['max_one_hot_dim']  # 每個类别特征的one-hot长度
    sparse_embed_dim = [8] * len(sparse_fea['feature'])  # 每個类别特征映射成embedding的长度
    hidden_units = [256, 128, 64]
    output_dim = 1
    lr = 0.01
    n_epoch = 100
    # 初始化模型
    model = WideDeep(dense_fea, sparse_fea['feature'], sparse_one_hot_dim, sparse_embed_dim,
                     hidden_units=hidden_units, output_dim=output_dim)
    optim = torch.optim.SGD(lr=lr, params=model.parameters())
    criterion = F.binary_cross_entropy

    # 训练模型
    for epoch in range(n_epoch):
        model.train()
        optim.zero_grad()
        logits = torch.reshape(model(X_train), (-1, ))
        loss = criterion(logits, y_train)
        loss.backward()
        optim.step()

        if epoch % 10 == 0 and epoch:
            print('epoch: {}, loss: {}'.format(epoch, loss))
    # 测试
    model.eval()
    pred = torch.reshape(model(X_test), (-1,))
    loss = criterion(pred, y_test)
    pred = [1 if x > 0.5 else 0 for x in pred]
    print('acc: {}, loss: {}'.format(accuracy_score(y_test, pred), loss))
