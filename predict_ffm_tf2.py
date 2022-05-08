#!/usr/bin/env python
# -*- coding:utf-8 -*-
# author: KrianJ
# datetime:2022/5/8 22:20
# software: PyCharm-predict_ffm_tf2
import tensorflow as tf
from tensorflow.keras import optimizers, losses, metrics
from sklearn.metrics import accuracy_score

from models.FFM_tf2 import FFM
from utils.load_data import load_criteo_data

if __name__ == '__main__':
    (X_train, y_train), (X_test, y_test), (dense_fea, sparse_fea) = load_criteo_data('dataset/criteo_sample.csv',
                                                                                     category_encoding='label_encoding')
    # 参数
    k = 8
    n_epoch = 300
    lr = 0.01
    # 初始化模型
    model = FFM(dense_features=dense_fea,
                sparse_features=sparse_fea['feature'], sparse_feature_dim=sparse_fea['max_depth'],
                k=k)
    optim = optimizers.SGD(learning_rate=lr)
    # 训练模型
    for epoch in range(n_epoch):
        with tf.GradientTape() as tape:
            logits = tf.reshape(model(X_train), (-1,))  # 前馈得到预测值
            loss = tf.reduce_mean(losses.binary_crossentropy(y_train, logits))  # 计算BCE平均损失
            grad = tape.gradient(loss, model.variables)  # 计算梯度
            optim.apply_gradients(grads_and_vars=(zip(grad, model.variables)))  # 将梯度更新到参数

            if epoch % 10 == 0 and epoch:
                print('epoch: {}, loss: {}'.format(epoch, loss))
    # 评估模型
    pred = model(X_test)
    pred = [1 if val > 0.5 else 0 for val in pred]
    print('Acc: {}'.format(accuracy_score(y_test, pred)))
