#!/usr/bin/env python
# -*- coding:utf-8 -*-
# author: KrianJ
# datetime:2022/5/15 10:31
# software: PyCharm-xDeepFM_tf2

import tensorflow as tf
from tensorflow.keras.layers import Layer, Embedding, Dense
from tensorflow.keras import Model

from models_tf2.WideDeep import LR_Layer, DenseLayer


class CIN(Layer):
    """Compressed Interaction Network: 压缩感知网络"""

    def __init__(self, cin_size):
        """n: field数量 k为输入的embedding维度"""
        super(CIN, self).__init__()
        self.cin_size = cin_size  # CIN每层的矩阵个数

    def build(self, input_shape):
        # input_shape: [None, n, k]
        self.field_num = [input_shape[1]] + self.cin_size

        # 用于压缩H_i三维矩阵得到H_(i+1)对应的field_num[i+1]个向量
        self.cin_W = [
            self.add_weight(
                name='w' + str(i),
                shape=(1, self.field_num[0] * self.field_num[i], self.field_num[i + 1]),
                initializer=tf.initializers.glorot_uniform(),
                regularizer=tf.keras.regularizers.l1_l2(1e-5),
                trainable=True)
            for i in range(len(self.field_num) - 1)]

    def call(self, inputs, **kwargs):
        # inputs: [None, n, k]
        k = inputs.shape[-1]
        res_list = [inputs]
        X_0 = tf.split(inputs, k, axis=-1)  # 将inputs分解成: k个(None, n, 1)
        for i, size in enumerate(self.field_num[1:]):
            X_i = tf.split(res_list[-1], k, axis=-1)  # 将上一个输出分解成: k * (None, field_num[i], 1)
            x = tf.matmul(X_0, X_i, transpose_b=True)  # k * (None, n, field_num[i])
            x = tf.reshape(x, shape=[
                k, -1, self.field_num[0] * self.field_num[i]])  # shape: (k, None, field_num[0]*field_num[i])
            x = tf.transpose(x, [1, 0, 2])  # shape: (None, k, field_num[0]*field_num[i])
            x = tf.nn.conv1d(input=x, filters=self.cin_W[i], stride=1, padding='VALID')  # 用卷积实现三维矩阵的压缩
            x = tf.transpose(x, [0, 2, 1])  # shape: (None, field_num[i+1], k)
            res_list.append(x)

        res_list = res_list[1:]  # 去掉X_0
        res = tf.concat(res_list, axis=1)  # (None, field_num[1]+...+field_num[n], k)
        output = tf.reduce_sum(res, axis=-1)  # (None, field_num[1]+...+field_num[n])
        return output


class xDeepFM(Model):
    def __init__(self, dense_features, sparse_features, sparse_one_hot_dim, sparse_embed_dim,
                 cin_size, hidden_units, output_dim, activation, dropout=0.):
        super(xDeepFM, self).__init__()
        self.n_dense = len(dense_features)
        self.n_sparse = len(sparse_features)
        self.embed_layers = [Embedding(sparse_one_hot_dim[i], sparse_embed_dim[i])
                             for i in range(self.n_dense)]
        self.linear = LR_Layer()
        self.dense_layer = DenseLayer(hidden_units=hidden_units, output_dim=output_dim, activation=activation,
                                      dropout=dropout)
        self.cin_layer = CIN(cin_size)
        self.out_layer = Dense(1, activation=None)

    def call(self, inputs, training=None, mask=None):
        dense_input, sparse_input = inputs[:, :self.n_dense], inputs[:, self.n_dense:]
        # 线性层输出
        linear_output = self.linear(inputs)

        # 对sparse_input构造embedding
        embed = [self.embed_layers[i][sparse_input[:, i]] for i in range(sparse_input.shape[-1])]

        # DNN侧输出
        dense_embed = tf.concat(embed, axis=-1)
        dense_embed = tf.concat([dense_input, dense_embed], axis=1)
        dense_out = self.dense_layer(dense_embed)

        # CIN侧输出
        cin_embed = tf.transpose(tf.convert_to_tensor(embed), [1, 0, 2])    # shape: (None, n, k)
        cin_output = self.cin_layer(cin_embed)

        output = self.out_layer(linear_output + dense_out, cin_output)
        return tf.nn.sigmoid(output)

