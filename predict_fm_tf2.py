import pandas as pd
import tensorflow as tf
from tensorflow.keras import optimizers, losses, metrics
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

from models.FM_tf2 import FM


def load_data(file):
    """处理数据"""
    data = pd.read_csv('dataset/train_fm.csv')
    dense_fea = ['I{}'.format(i) for i in range(1, 14)]
    sparse_fea = ['C{}'.format(j) for j in range(1, 27)]

    # 处理缺失值
    data[dense_fea] = data[dense_fea].fillna(0)
    data[sparse_fea] = data[sparse_fea].fillna('-1')
    # 归一化
    data[dense_fea] = MinMaxScaler().fit_transform(data[dense_fea])
    # one_hot
    data = pd.get_dummies(data)

    # 数据集划分
    X, y = data.drop('label', axis=1).values, data['label']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    return (X_train, y_train), (X_test, y_test)


if __name__ == '__main__':
    (X_train, y_train), (X_test, y_test) = load_data('dataset/train_fm.csv')
    k = 8
    w_reg = 1e-5
    v_reg = 1e-5
    n_epoch = 100

    model = FM(k=k, w_reg=w_reg, v_reg=v_reg)
    optim = optimizers.SGD(0.01)

    for epoch in range(n_epoch):
        with tf.GradientTape() as tape:
            y_pred = tf.reshape(model(X_train), (-1, ))     # 前馈得到预测值
            loss = tf.reduce_mean(losses.BinaryCrossentropy(y_train, y_pred))   # 计算交叉熵loss
            grad = tape.gradient(loss, model.variables)     # 计算梯度
            optim.apply_gradients(grads_and_vars=(zip(grad, model.variables)))  # 将梯度更新到对应参数

            if epoch % 10 == 0 and epoch:
                print("epoch: {}, loss: {}".format(epoch, loss))
    # 评估模型
    test_pred = model(X_test)
    pre = [1 if x > 0.5 else 0 for x in test_pred]
    print('acc: {}'.format(accuracy_score(y_test, test_pred)))
