#!/usr/bin/env python
# -*- coding:utf-8 -*-
# author: KrianJ
# datetime:2022/5/8 21:32
# software: PyCharm-FFM_tf2
import tensorflow as tf

"""tensorflow2.0 实现的Field-aware模型"""


class FFM_layer(tf.keras.layers.Layer):
    def __init__(self, dense_features, sparse_features, sparse_feature_dim,
                 k: int, w_reg=1e-4, v_reg=1e-4):
        super(FFM_layer, self).__init__()
        self.dense_features = dense_features  # 稠密数值特征名称
        self.sparse_features = sparse_features  # 稀疏离散特征名称
        self.sparse_feature_dim = sparse_feature_dim    # 稀疏类别特征的值数量(用于one hot编码)

        # 所有特征的数量(数值特征+类别特征)
        self.n_field = len(self.dense_features) + len(self.sparse_features)
        # 经过编码后的数据集维度
        self.n_features = len(self.dense_features) + sum(self.sparse_feature_dim)

        self.k = k  # 隐向量v维度
        self.w_reg = w_reg  # 线性权重w正则化系数
        self.v_reg = v_reg  # 隐向量v正则化系数

    def build(self, input_shape):
        """初始化权重, 隐向量等参数"""
        # shape: (1, )
        self.w0 = self.add_weight(name='w0', shape=(1,),
                                  initializer=tf.zeros_initializer(),
                                  trainable=True)
        # shape: (m, 1)
        self.w = self.add_weight(name='w', shape=(self.n_features, 1),
                                 initializer=tf.random_normal_initializer(),
                                 trainable=True,
                                 regularizer=tf.keras.regularizers.l2(self.w_reg))
        # shape: (m, n_field, k),FFM将隐向量进一步细分,每个特征具有多个隐向量,即每个特征对应一个(n_field, k)的隐向量矩阵
        self.v = self.add_weight(name='v', shape=(self.n_features, self.n_field, self.k),
                                 initializer=tf.random_normal_initializer(),
                                 trainable=True,
                                 regularizer=tf.keras.regularizers.l2(self.v_reg))

    def call(self, inputs, **kwargs):
        # 将dense和sparse特征合并
        dense_x = inputs[:, :len(self.dense_features)]  # 数值特征需要放在inputs前面
        sparse_x = inputs[:, len(self.dense_features):]  # 后面为稀疏类别特征

        # 将dense_x与经过onehot encoding的类别特征拼接形成x
        x = tf.cast(dense_x, dtype=tf.float32)
        for i in range(sparse_x.shape[1]):
            x = tf.concat(
                [x, tf.one_hot(tf.cast(sparse_x[:, i], dtype=tf.int32),
                               depth=self.sparse_feature_dim[i])],
                axis=1)
        # 计算线性部分
        linear_part = tf.matmul(x, self.w) + self.w0  # shape: (batch_size, 1)
        # FFM交叉项第一部分
        cross_part = 0.
        field_f = tf.tensordot(x, self.v, axes=1)  # 先计算V_ij * X_i, shape: (batch_size, n_field, k)

        for i in range(self.n_field):  # 域Vi, Vj之间两两相乘, Vi, Vj为shape: (batch_size, k)的隐向量矩阵
            for j in range(i + 1, self.n_field):
                v_i, v_j = field_f[:, i], field_f[:, j]
                cross_part += tf.reduce_sum(tf.multiply(v_i, v_j), axis=1, keepdims=True)

        output = linear_part + cross_part
        return tf.nn.sigmoid(output)


class FFM(tf.keras.Model):
    def __init__(self, dense_features, sparse_features, sparse_feature_dim,
                 k: int, w_reg=1e-4, v_reg=1e-4):
        super(FFM, self).__init__()
        self.dense_features = dense_features  # 稠密数值特征
        self.sparse_features = sparse_features  # 稀疏离散特征
        self.sparse_feature_dim = sparse_feature_dim    # 稀疏离散特征one-hot编码长度
        self.ffm = FFM_layer(dense_features, sparse_features, sparse_feature_dim, k, w_reg, v_reg)

    def call(self, inputs, training=None, mask=None):
        """inputs = dense_data + sparse_category_data"""
        output = self.ffm(inputs)
        return output
