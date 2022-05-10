#!/usr/bin/env python
# -*- coding:utf-8 -*-
# author: KrianJ
# datetime:2022/5/9 21:36
# software: PyCharm-WideDeep_torch
import torch
import torch.nn as nn
import torch.nn.functional as F


class DeepLayer(nn.Module):
    def __init__(self, input_dim, hidden_units: list, output_dim):
        """
        Deep model
        :param hidden_units:隐层节点数
        :param output_dim:输出维度
        :param activation:激活函数
        """
        super(DeepLayer, self).__init__()
        self.input_dim = input_dim
        self.hidden_units = hidden_units
        self.output_dim = output_dim
        self.init_layers()

    def init_layers(self):
        self.hidden_layers = nn.Sequential()
        units = [self.input_dim] + self.hidden_units
        for i in range(len(units) - 1):
            self.hidden_layers.add_module(name='hidden_{}'.format(i),
                                          module=nn.Linear(units[i], units[i + 1]))
            self.hidden_layers.add_module(name='activation_{}'.format(i), module=nn.ReLU())
        self.output_layer = nn.Linear(self.hidden_units[-1], self.output_dim)

    def forward(self, x):
        for layer in self.hidden_layers:
            x = layer(x)
        x = self.output_layer(x)
        return x


class WideDeep(nn.Module):
    def __init__(self, dense_features, sparse_features, sparse_one_hot_dim, sparse_embed_dim,
                 hidden_units, output_dim):
        """
        WideDeep模型 torch实现
        :param dense_features: 稠密数值特征名
        :param sparse_features: 稀疏类别特征名
        :param sparse_one_hot_dim: 每个类别特征one-hot后的维度
        :param sparse_embed_dim: 每个one-hot类别特征转成embedding的维度
        :param hidden_units: deep侧的隐层节点数
        :param output_dim: deep侧的输出维度
        """
        super(WideDeep, self).__init__()
        # Wide侧参数
        self.dense_features = dense_features
        self.sparse_features = sparse_features
        self.sparse_one_hot_dim = sparse_one_hot_dim
        # Deep侧参数
        self.sparse_embed_dim = sparse_embed_dim
        self.hidden_units = hidden_units
        self.output_dim = output_dim

        self.n_dense = len(self.dense_features)  # 数值特征数量
        self.n_sparse = len(self.sparse_features)  # 原始类别特征数量
        self.n_features = self.n_dense + sum(self.sparse_one_hot_dim)  # 数值特征+one-hot特征总维度(wide侧)

        # Wide部分 数值特征+稀疏one-hot特征
        self.wide = nn.Linear(self.n_features, 1)

        # embedding层：将one-hot向量映射到低维稠密向量,拼接成单个embedding作为deep侧输入
        self.deep_embeds = nn.ModuleDict({
            "embed_{}".format(fea):
                nn.Embedding(self.sparse_one_hot_dim[i], self.sparse_embed_dim[i])
            for i, fea in enumerate(self.sparse_features)
        })  # shape: one_hot_dim -> embed_dim
        # Deep: embedding -> output
        self.deep = DeepLayer(sum(self.sparse_embed_dim), hidden_units, output_dim)

    def forward(self, x):
        # 先将数据集中的数值特征，稀疏类别编码特征，稀疏one-hot特征分开
        dense_x = x[:, :self.n_dense]
        sparse_category_x = x[:, self.n_dense: (self.n_dense + self.n_sparse)].to(torch.int64)
        sparse_one_hot_x = x[:, (self.n_dense + self.n_sparse):]

        # wide侧输出: dense + sparse_one_hot
        wide_output = self.wide(torch.concat([dense_x, sparse_one_hot_x], dim=-1))

        # deep: sparse_category
        # 计算每个类别特征的embedding并拼接起来
        sparse_embed = []
        for i in range(sparse_category_x.shape[-1]):  # 遍历每个类别编码的特征，送到embedding层输出对应embedding向量
            layer = self.deep_embeds['embed_{}'.format(self.sparse_features[i])]
            sparse_embed.append(layer(sparse_category_x[:, i]))
        deep_input = torch.concat(sparse_embed, dim=-1)
        # Deep侧的输出
        deep_output = self.deep(deep_input)

        # wide_deep
        output = 0.5 * (wide_output + deep_output)
        output = torch.sigmoid(output)
        return output
