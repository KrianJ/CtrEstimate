#!/usr/bin/env python
# -*- coding:utf-8 -*-
# author: KrianJ
# datetime:2022/5/8 21:32
# software: PyCharm-WideDeep

import tensorflow as tf
from tensorflow.keras.layers import Embedding
from tensorflow.keras import Model

from models_tf2.base_layer import LR_Layer, DenseLayer


class WideDeep(Model):
    def __init__(self, dense_features, sparse_features, sparse_one_hot_dim, sparse_embed_dim,
                 hidden_units, output_dim, activation):
        """
        WideDeep模型
        :param dense_features: 稠密数值特征名
        :param sparse_features: 稀疏类别特征名
        :param sparse_one_hot_dim: 每个类别特征one_hot后的维度
        :param sparse_embed_dim: 每个one_hot类别特征转成embedding的维度
        :param hidden_units: deep侧的隐层节点数
        :param output_dim: deep侧的输出维度
        :param activation: deep侧激活函数
        """
        super(WideDeep, self).__init__()
        self.dense_features = dense_features  # 数值特征
        self.sparse_features = sparse_features  # 类别特征
        self.n_dense = len(self.dense_features)
        self.n_sparse = len(self.sparse_features)

        # Wide部分
        self.wide = LR_Layer()
        # Deep部分: sparse feature的embedding layer(高维one-hot -> 低维embedding)
        self.embedding_layers = {
            'embed_{}'.format(i): Embedding(input_dim=sparse_one_hot_dim[i],
                                            output_dim=sparse_embed_dim[i])
            for i, fea in enumerate(self.sparse_features)
        }
        self.deep = DenseLayer(hidden_units, output_dim, activation)

    def call(self, inputs, training=None, mask=None):
        """inputs = [数值特征向量  label_encoding类别特征向量  one_hot类别特征向量]"""
        dense_input = inputs[:, :self.n_dense]
        sparse_category_input = inputs[:, self.n_dense: (self.n_dense + self.n_sparse)]
        sparse_onehot_input = inputs[:, (self.n_dense + self.n_sparse):]

        # wide部分
        wide_input = tf.concat([dense_input, sparse_onehot_input], axis=-1)
        wide_output = self.wide(wide_input)
        # deep部分
        sparse_embeds = []  # 每个类别特征的embedding
        for i in range(sparse_category_input.shape[-1]):
            embed_layer = self.embedding_layers['embed_{}'.format(i)]
            sparse_embeds.append(embed_layer(sparse_category_input[:, i]))
        sparse_embeds = tf.concat(sparse_embeds, axis=-1)
        deep_output = self.deep(sparse_embeds)

        # wide+deep
        output = tf.nn.sigmoid(0.5 * (wide_output + deep_output))
        return output
