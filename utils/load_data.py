#!/usr/bin/env python
# -*- coding:utf-8 -*-
# author: KrianJ
# datetime:2022/5/8 22:16
# software: PyCharm-load_data
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split


def load_criteo_data(file, sparse_return=None, test_size=0.2):
    """
    加载criteo训练数据
    :param file: 路径
    :param sparse_return: 稀疏类别特征返回类型: one_hot(default), category, both
    :param test_size: 测试集比例
    :return:
    """
    if not sparse_return:
        sparse_return = 'one_hot'

    data = pd.read_csv(file)
    dense_fea = ['I{}'.format(i) for i in range(1, 14)]
    sparse_fea = ['C{}'.format(j) for j in range(1, 27)]

    # 处理缺失值
    data[dense_fea] = data[dense_fea].fillna(0)
    data[sparse_fea] = data[sparse_fea].fillna('-1')
    # 归一化
    data[dense_fea] = MinMaxScaler().fit_transform(data[dense_fea])

    dense_data = data[dense_fea]
    sparse_data = data[sparse_fea]

    # one_hot encoding
    sparse_one_hot_data = pd.get_dummies(sparse_data)
    # label encoding
    sparse_category_data = np.array([LabelEncoder().fit_transform(sparse_data[col]) for col in sparse_fea])
    sparse_category_data = pd.DataFrame(sparse_category_data.T, columns=sparse_fea)

    sparse_one_hot_dim = data[sparse_fea].nunique().values  # 每个类别特征的最大维度(对应的one hot维度)

    # 数据集划分
    if sparse_return == 'one_hot':
        X = pd.concat([dense_data, sparse_one_hot_data], axis=1).values
    elif sparse_return == 'category':
        X = pd.concat([dense_data, sparse_category_data], axis=1).values
    else:
        X = pd.concat([dense_data, sparse_category_data, sparse_one_hot_data], axis=1).values
    y = data['label'].values
    # 特征信息
    feature_info = {
        'dense_feature': dense_fea,
        'sparse_feature': sparse_fea,
        'max_one_hot_dim': sparse_one_hot_dim
    }

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)

    return (X_train, y_train), (X_test, y_test), feature_info
