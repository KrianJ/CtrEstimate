#!/usr/bin/env python
# -*- coding:utf-8 -*-
# author: KrianJ
# datetime:2022/5/9 22:09
# software: PyCharm-predict_widedeep_tf2
import tensorflow as tf
from tensorflow.keras import optimizers, losses, metrics
from sklearn.metrics import accuracy_score

from models_tf2.WideDeep import WideDeep
from utils.load_data import load_criteo_data

if __name__ == '__main__':
    (X_train, y_train), (X_test, y_test), feature_info = \
        load_criteo_data('dataset/criteo_sample.csv', sparse_return='both')

    # 参数初始化
    lr = 0.05
    hidden_units = [256, 128, 64]
    output_dim = 1
    sparse_embed_dim = [8] * len(feature_info['sparse_feature'])
    activation = 'relu'
    n_epoch = 100
    # 模型初始化
    model = WideDeep(dense_features=feature_info['dense_feature'], sparse_features=feature_info['sparse_feature'],
                     sparse_one_hot_dim=feature_info['max_one_hot_dim'], sparse_embed_dim=sparse_embed_dim,
                     hidden_units=hidden_units, output_dim=output_dim, activation=activation)
    optim = optimizers.SGD(learning_rate=lr)
    # 训练模型
    for epoch in range(n_epoch):
        with tf.GradientTape() as tape:
            logits = model(X_train)
            loss = tf.reduce_mean(
                losses.binary_crossentropy(y_true=y_train, y_pred=logits)
            )
            grad = tape.gradient(loss, model.variables)  # 计算梯度
            optim.apply_gradients(grads_and_vars=(zip(grad, model.variables)))  # 将梯度更新到参数

            if epoch % 10 == 0 and epoch:
                print('epoch: {}, loss: {}'.format(epoch, loss))
    # 评估模型
    pred = model(X_test)
    loss = tf.reduce_mean(losses.binary_crossentropy(y_true=y_test, y_pred=pred))
    pred = [1 if val > 0.5 else 0 for val in pred]
    print('Acc: {}, loss: {}'.format(accuracy_score(y_test, pred), loss))
