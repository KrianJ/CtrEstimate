#!/usr/bin/env python
# -*- coding:utf-8 -*-
# author: KrianJ
# datetime:2022/5/7 14:01
# software: PyCharm-FM

import tensorflow as tf
import tensorflow.keras.backend as K


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


class FM(tf.keras.Model):
    def __init__(self, k, w_reg=1e-4, v_reg=1e-4):
        super(FM, self).__init__()
        self.fm = FM_Layer(k, w_reg, v_reg)

    def call(self, inputs, training=None, mask=None):
        output = self.fm(inputs)
        return output
