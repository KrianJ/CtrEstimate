#!/usr/bin/env python
# -*- coding:utf-8 -*-
# author: KrianJ
# datetime:2022/5/11 21:37
# software: PyCharm-predict_DCN_tf2
import tensorflow as tf
from sklearn.metrics import accuracy_score

from models.DCN_tf2 import DCN
from utils.load_data import load_criteo_data

if __name__ == '__main__':
    (X_train, y_train), (X_test, y_test), feature_info = load_criteo_data('dataset/criteo_sample.csv',
                                                                          sparse_return='category')
    X_train = tf.cast(X_train, dtype=tf.float32)
    X_test = tf.cast(X_test, dtype=tf.float32)
    y_train = tf.cast(y_train, dtype=tf.float32)
    y_test = tf.cast(y_test, dtype=tf.float32)

    # 参数
    n_cross_layer = 6
    embedd_dim = [8] * len(feature_info['sparse_feature'])
    hidden_units = [256, 128, 64]
    output_dim = 1
    activation = 'relu'
    lr = 0.02
    n_epoch = 100
    # 模型初始化
    model = DCN(dense_feature=feature_info['dense_feature'], sparse_feature=feature_info['sparse_feature'],
                sparse_one_hot_dim=feature_info['max_one_hot_dim'],
                sparse_embed_dim=embedd_dim,
                n_cross_layer=n_cross_layer, hidden_units=hidden_units, output_dim=output_dim, activation=activation)
    optim = tf.optimizers.SGD(learning_rate=lr)
    # 模型训练
    for epoch in range(n_epoch):
        with tf.GradientTape() as tape:
            logits = tf.reshape(model(X_train), (-1,))
            loss = tf.losses.binary_crossentropy(y_pred=logits, y_true=y_train)
        grad = tape.gradient(loss, model.variables)
        optim.apply_gradients(grads_and_vars=zip(grad, model.variables))
        if epoch % 10 == 0 and epoch:
            print('epoch: {}, loos: {}'.format(epoch, loss))

    # 模型测试
    pred = model(X_test)
    pred = [1 if val > 0.5 else 0 for val in pred]
    print('Acc: {}'.format(accuracy_score(y_test, pred)))
