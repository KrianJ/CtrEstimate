#!/usr/bin/env python
# -*- coding:utf-8 -*-
# author: KrianJ
# datetime:2022/5/8 22:16
# software: PyCharm-load_data
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split


def load_criteo_data(file, sparse_encoding="one_hot"):
    """
    加载criteo训练数据
    :param file: 路径
    :param sparse_encoding: 类别特征的编码方式, one_hot, label_encoding
    :return:
    """
    data = pd.read_csv(file)
    dense_fea = ['I{}'.format(i) for i in range(1, 14)]
    sparse_fea = ['C{}'.format(j) for j in range(1, 27)]

    # 处理缺失值
    data[dense_fea] = data[dense_fea].fillna(0)
    data[sparse_fea] = data[sparse_fea].fillna('-1')
    # 归一化
    data[dense_fea] = MinMaxScaler().fit_transform(data[dense_fea])

    # one-hot encoding
    if sparse_encoding == 'one_hot':
        data = pd.get_dummies(data)
        feature_info = [dense_fea, sparse_fea]
    # label encoding
    elif sparse_encoding == 'label_encoding':
        for col in sparse_fea:
            data[col] = LabelEncoder().fit_transform(data[col]).astype(int)
        # 稀疏类别特征的值数量
        sparse_fea_dim = data[sparse_fea].nunique().values
        feature_info = [dense_fea,  {"feature": sparse_fea, 'max_depth': sparse_fea_dim}]
    # both
    else:
        # 数值特征
        dense_df = data[dense_fea]
        # 类别编码
        category_encoding = np.array([LabelEncoder().fit_transform(data[col]).astype(int) for col in sparse_fea])
        category_df = pd.DataFrame(category_encoding.T, columns=sparse_fea, dtype=int)
        # one_hot编码
        one_hot_df = pd.get_dummies(data[sparse_fea])

        # 按照label, dense, sparse_category, sparse_one_hot重组
        data = pd.concat([data['label'], dense_df, category_df, one_hot_df], axis=1)
        # 稀疏类别特征的值数量
        sparse_fea_dim = data[sparse_fea].nunique().values
        feature_info = [dense_fea,  {"feature": sparse_fea, 'max_depth': sparse_fea_dim}]

    # 数据集划分
    X, y = data.drop('label', axis=1).values, data['label']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    return (X_train, y_train.values), (X_test, y_test.values), feature_info
