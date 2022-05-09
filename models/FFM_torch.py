#!/usr/bin/env python
# -*- coding:utf-8 -*-
# author: KrianJ
# datetime:2022/5/7 14:01
# software: PyCharm-FFM_torch

import torch
import torch.nn as nn
import torch.nn.functional as F


class FFM_Layer(nn.Module):
    def __init__(self, dense_features, sparse_features, sparse_feature_dim, k: int):
        super(FFM_Layer, self).__init__()
        self.k = k  # 单个隐向量维度
        self.dense_features = dense_features  # 稠密数值特征名称
        self.sparse_feature = sparse_features  # 稀疏离散特征名称
        self.sparse_feature_dim = sparse_feature_dim  # 稀疏类别特征的值数量(用于one hot编码)
        # 所有特征的数量(数值特征+类别特征)
        self.n_field = len(self.dense_features) + len(self.sparse_feature)
        # 经过编码后的数据集维度
        self.n_features = len(self.dense_features) + sum(self.sparse_feature_dim)

        self.w = nn.Linear(self.n_features, 1)
        self.v = nn.Parameter(torch.zeros((self.n_features, self.n_field, self.k)))

    def layer(self, x):
        dense_x = x[:, : len(self.dense_features)]  # 稠密数值特征
        sparse_x = x[:, len(self.dense_features):]  # 稀疏类别特征

        # 处理数据集
        inputs = dense_x.clone().detach()
        for i in range(sparse_x.shape[1]):  # 将label encoding之后的类别特征one-hot之后拼接到inputs
            inputs = torch.concat(
                [inputs, F.one_hot(sparse_x[:, i].to(torch.int64), self.sparse_feature_dim[i])],
                dim=1)
        # 线性部分
        linear_part = self.w(inputs)    # shape: (batch_size, 1)
        # 特征交叉部分
        cross_part = 0.
        field_f = torch.tensordot(inputs, self.v, dims=([1], [0]))  # 先计算V_ij * X_i, shape: (batch_size, n_field, k)
        # 计算每个域(field)的交叉参数
        for i in range(self.n_field):
            for j in range(i + 1, self.n_field):
                v_i, v_j = field_f[:, i], field_f[:, j]  # shape: [batch_size, k]
                cross_part += torch.sum(torch.multiply(v_i, v_j), dim=1, keepdim=True)  # shape: (batch_size, 1)
        output = linear_part + cross_part
        return torch.sigmoid(output)

    def forward(self, x):
        output = self.layer(x)
        return output
