#!/usr/bin/env python
# -*- coding:utf-8 -*-
# author: KrianJ
# datetime:2022/5/8 21:32
# software: PyCharm-DeepFM_tf2

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Embedding
from models.FM_tf2 import FM_Layer
from models.WideDeep_tf2 import DeepLayer


class DeepFM(Model):
    """
    DeepFM实现
    与WideDeep不同, Deep部分和FM部分的输入共享embedding层的输出,
    而WideDeep中Wide侧输入dense + one_hot sparse, Deep侧输入为embedding sparse
    """
    def __init__(self, dense_features, sparse_features, sparse_one_hot_dim, sparse_embed_dim,
                 k, hidden_units, output_dim, activation, w_reg=1e-4, v_reg=1e-4):
        super(DeepFM, self).__init__()
        self.dense_features = dense_features
        self.sparse_features = sparse_features
        self.n_dense = len(self.dense_features)
        self.n_sparse = len(self.sparse_features)

        # embedding层
        self.embed_layers = {
            "embed_{}".format(fea): Embedding(sparse_one_hot_dim[i], sparse_embed_dim[i])
            for i, fea in enumerate(sparse_features)
        }
        self.fm = FM_Layer(k, w_reg, v_reg)
        self.deep = DeepLayer(hidden_units, output_dim, activation)

    def call(self, inputs, training=None, mask=None):
        """input为dense input + label_encoding sparse input"""
        dense_input = inputs[:, :self.n_dense]
        sparse_category_input = inputs[:, self.n_dense:]

        # 将one_hot sparse_input转换成embedding
        sparse_embedding = []
        for i in range(sparse_category_input.shape[1]):
            layer = self.embed_layers['embed_{}'.format(self.sparse_features[i])]
            sparse_embedding.append(layer(sparse_category_input[:, i]))
        sparse_embedding = tf.concat(sparse_embedding, axis=-1)
        x = tf.concat([dense_input, sparse_embedding])

        # FM侧
        fm_output = self.fm(x)
        # Deep侧
        deep_output = self.deep(x)
        output = tf.sigmoid(0.5 * (fm_output + deep_output))
        return output



