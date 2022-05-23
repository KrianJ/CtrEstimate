#!/usr/bin/env python
# -*- coding:utf-8 -*-
# author: KrianJ
# datetime:2022/5/15 13:59
# software: PyCharm-base_layer
"""一些通用的网络层"""
import tensorflow as tf
from tensorflow.keras.layers import Dense, Layer, Dropout
import tensorflow.keras.backend as K


class LR_Layer(Layer):
    """输入Dense feature + Sparse feature(one hot处理过的) + 特征组合"""

    def __init__(self):
        super(LR_Layer, self).__init__()
        self.w_reg = 1e-4

    def build(self, input_shape):
        self.w0 = self.add_weight(name='w0', shape=(1,),
                                  initializer=tf.zeros_initializer(),
                                  trainable=True)
        self.w = self.add_weight(name='w', shape=(input_shape[-1], 1),
                                 initializer=tf.random_normal_initializer(),
                                 trainable=True,
                                 regularizer=tf.keras.regularizers.l2(self.w_reg))

    def call(self, inputs, **kwargs):
        # 不做sigmoid, 与deep部分融合后再用sigmoid
        output = tf.matmul(inputs, self.w) + self.w0  # shape: (batch_size, 1)
        return output


class DenseLayer(Layer):

    def __init__(self, hidden_units, output_dim, activation, dropout=0.):
        super(DenseLayer, self).__init__()
        self.hidden_layers = [Dense(i, activation=activation) for i in hidden_units]
        self.output_layer = Dense(output_dim, activation=None)  # 不做sigmoid
        self.dropout = Dropout(dropout)

    def call(self, inputs, **kwargs):
        x = inputs
        for layer in self.hidden_layers:
            x = layer(x)
        x = self.dropout(x)
        output = self.output_layer(x)
        return output


class FM_Layer(tf.keras.layers.Layer):
    def __init__(self, k, w_reg, v_reg):
        super(FM_Layer, self).__init__()
        self.k = k  # 隐向量v的维度
        self.w_reg = w_reg  # 权重w的正则项系数
        self.v_reg = v_reg  # 隐向量v的正则项系数

    def build(self, input_shape):
        """相关变量权重"""
        # 偏置
        self.w0 = self.add_weight(name='w0', shape=(1,),
                                  initializer=tf.zeros_initializer(),
                                  trainable=True)
        # 线性部分权重 shape: (dim, 1)
        self.w = self.add_weight(name='w', shape=(input_shape[-1], 1),
                                 initializer=tf.random_normal_initializer(),
                                 trainable=True,
                                 regularizer=tf.keras.regularizers.l2(self.w_reg))
        # 隐向量 shape: (dim, k)
        self.v = self.add_weight(name='v', shape=(input_shape[-1], self.k),
                                 initializer=tf.random_normal_initializer(),
                                 trainable=True,
                                 regularizer=tf.keras.regularizers.l2(self.v_reg))

    def call(self, inputs, **kwargs):
        if K.ndim(inputs) != 2:
            raise ValueError("Unexpected inputs dimension {}, dim 2 required".format(K.ndim(inputs)))

        # FM表达式线性部分
        linear_part = tf.matmul(inputs, self.w) + self.w0  # shape: (batch_size, 1)
        # FM表达式二次交叉项-第一项
        cross_part_1 = tf.pow(tf.matmul(inputs, self.v), 2)  # shape: (batch_size, self.k)
        # FM二次交叉项-第二项
        cross_part_2 = tf.matmul(tf.pow(inputs, 2), tf.pow(self.v, 2))  # shape: (batch_size, self.k)
        # FM二次交叉项结果
        cross_part = 1 / 2 * tf.reduce_sum(cross_part_1 - cross_part_2, axis=-1,
                                           keepdims=True)  # shape: (batch_size, 1)
        # 输出结果
        output = linear_part + cross_part
        return tf.nn.sigmoid(output)
