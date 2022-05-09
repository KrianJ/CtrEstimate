#!/usr/bin/env python
# -*- coding:utf-8 -*-
# author: KrianJ
# datetime:2022/5/7 15:50
# software: PyCharm-FM_torch

import torch
import torch.nn as nn
import torch.nn.functional as F

"""torch1.10 实现的FM模型"""


class FM_Layer(nn.Module):
    """
    FM表达式:
    f(x) = w.T * x + sum(i=1, 1->m)sum(j=i+1, i+1->m)[w_ij * x_i * x_j]
    w_ij = <v_i, v_j>, v \belong R^k
    经因子分解可简化为
    f(x) = w.T * x + 1/2 * sum(f=1, 1->k)[sum(i=1, 1->m)v_if * x_i - sum(i=1, 1->m)(v_if)^2 * (x_i)^2]
    其中, m为样本维度，k为隐向量v的维度
    """

    def __init__(self, dim, k):
        super(FM_Layer, self).__init__()
        self.dim = dim  # 样本维度
        self.k = k  # 隐向量v维度

        self.w = nn.Linear(self.dim, 1)  # FM表达式第一项线性层权重w
        self.v = nn.Parameter(torch.randn(self.dim, self.k))  # 隐向量矩阵
        # 初始化v
        nn.init.normal_(self.v, 0, 0.1)

    def layer(self, x):
        # 线性部分
        linear_part = self.w(x)  # shape: (batch_size, 1)
        # FM表达式二次交叉项-第一项
        cross_part_1 = torch.pow(torch.mm(x, self.v), 2)  # shape: (batch_size, self.k)
        # FM二次交叉项-第二项
        cross_part_2 = torch.mm(torch.pow(x, 2), torch.pow(self.v, 2))  # shape: (batch_size, self.k)
        # FM二次交叉项结果
        cross_part = 1 / 2 * torch.sum(cross_part_1 - cross_part_2, dim=1, keepdim=True)  # shape: (batch_size, 1)
        output = linear_part + cross_part
        return torch.sigmoid(output)

    def forward(self, x):
        output = self.layer(x)
        return output
