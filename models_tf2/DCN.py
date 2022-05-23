#!/usr/bin/env python
# -*- coding:utf-8 -*-
# author: KrianJ
# datetime:2022/5/11 20:39
# software: PyCharm-DCN

import tensorflow as tf
from tensorflow.keras.layers import Layer, Embedding, Dense
from tensorflow.keras import Model

from models_tf2.WideDeep import DenseLayer


class CrossLayer(Layer):
    def __init__(self, layer_num, w_reg=1e-4, b_reg=1 - 4):
        super(CrossLayer, self).__init__()
        self.layer_num = layer_num
        self.w_reg = w_reg
        self.b_reg = b_reg

    def build(self, input_shape):
        # 每层cross layer的权重
        self.cross_weight = [
            self.add_weight('w_{}'.format(i),
                            shape=(input_shape[-1], 1),
                            trainable=True,
                            initializer=tf.random_normal_initializer(),
                            regularizer=tf.keras.regularizers.l2(self.w_reg))
            for i in range(self.layer_num)
        ]
        # 每层cross layer的偏置
        self.cross_bias = [
            self.add_weight('b_{}'.format(i),
                            shape=(1,),
                            trainable=True,
                            initializer=tf.zeros_initializer(),
                            regularizer=tf.keras.regularizers.l2(self.b_reg))
            for i in range(self.layer_num)
        ]

    def call(self, inputs, **kwargs):
        x_0 = tf.expand_dims(inputs, axis=2)  # 扩展输入维度 shape: (batch_size, dim, 1)
        x_l = x_0
        # 前向计算 x_(l+1) = x_0 * x_l.T * w_l + b_l + x_l
        for i in range(self.layer_num):
            # 计算x_l.T * w_l标量,便于计算
            xl_w = tf.matmul(tf.transpose(x_l, [0, 2, 1]), self.cross_weight[i])    # shape: (batch_size, 1, 1)
            # 计算x_(l+1)
            x_l = tf.matmul(x_0, xl_w) + self.cross_bias[i] + x_l

        output = tf.squeeze(x_l, axis=2)
        return output


class DCN(Model):
    def __init__(self, dense_feature, sparse_feature, sparse_one_hot_dim, sparse_embed_dim,
                 n_cross_layer, hidden_units, output_dim, activation):
        super(DCN, self).__init__()
        self.dense_feature = dense_feature
        self.sparse_feature = sparse_feature
        self.sparse_one_hot_dim = sparse_one_hot_dim
        self.sparse_embed_dim = sparse_embed_dim

        # embedding输入层
        self.embed_layer = {
            'embed_{}'.format(fea): Embedding(input_dim=sparse_one_hot_dim[i],
                                              output_dim=sparse_embed_dim[i])
            for i, fea in enumerate(self.sparse_feature)
        }
        # Cross Layer
        self.cross_part = CrossLayer(n_cross_layer)
        # 全连接层
        self.dense_part = DenseLayer(hidden_units, output_dim, activation)
        # 输出层
        self.output_layer = Dense(1, activation)

    def call(self, inputs, training=None, mask=None):
        """inputs: dense_input + label_encoding input"""
        dense_input = inputs[:, :len(self.dense_feature)]
        sparse_input = inputs[:, len(self.dense_feature):]
        # 将sparse_input转换成embedding
        sparse_embed = [self.embed_layer['embed_{}'.format(fea)](sparse_input[:, i])
                        for i, fea in enumerate(self.sparse_feature)]
        sparse_embed = tf.concat(sparse_embed, axis=1)
        # 合并为输入
        x = tf.concat([dense_input, sparse_embed], axis=1)
        # Cross侧
        cross_output = self.cross_part(x)
        # Dense侧
        dnn_output = self.dense_part(x)

        output = tf.concat([cross_output, dnn_output], axis=1)
        output = tf.nn.sigmoid(self.output_layer(output))
        return output

