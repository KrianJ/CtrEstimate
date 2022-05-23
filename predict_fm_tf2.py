import tensorflow as tf
from tensorflow.keras import optimizers, losses, metrics
from sklearn.metrics import accuracy_score

from models_tf2.FM import FM
from utils.load_data import load_criteo_data

if __name__ == '__main__':
    (X_train, y_train), (X_test, y_test), _ = load_criteo_data('dataset/criteo_sample.csv',
                                                               sparse_return='one_hot')
    k = 8
    w_reg = 1e-5
    v_reg = 1e-5
    n_epoch = 500

    model = FM(k=k, w_reg=w_reg, v_reg=v_reg)
    optim = optimizers.SGD(0.01)

    for epoch in range(n_epoch):
        with tf.GradientTape() as tape:
            y_pred = tf.reshape(model(X_train), (-1,))  # 前馈得到预测值
            loss = tf.reduce_mean(losses.binary_crossentropy(y_train, y_pred))  # 计算交叉熵loss
            grad = tape.gradient(loss, model.variables)  # 计算梯度
            optim.apply_gradients(grads_and_vars=(zip(grad, model.variables)))  # 将梯度更新到对应参数

            if epoch % 100 == 0 and epoch:
                print("epoch: {}, loss: {}".format(epoch, loss))
    # 评估模型
    pred = model(X_test)
    pred = [1 if x > 0.5 else 0 for x in pred]
    print('acc: {}'.format(accuracy_score(y_test, pred)))
