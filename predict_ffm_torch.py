import torch
from torch.optim import SGD
import torch.nn.functional as F
from sklearn.metrics import accuracy_score

from models.FFM_torch import FFM_Layer
from utils.load_data import load_criteo_data

if __name__ == '__main__':
    (X_train, y_train), (X_test, y_test), (dense_fea, sparse_fea) = load_criteo_data('dataset/criteo_sample.csv',
                                                                                     sparse_encoding='label_encoding')
    X_train = torch.tensor(X_train, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32)
    # 参数
    k = 8
    n_epoch = 10
    lr = 0.01
    # 初始化
    model = FFM_Layer(dense_features=dense_fea, sparse_features=sparse_fea['feature'],
                      sparse_feature_dim=sparse_fea['max_one_hot_dim'], k=k)
    optim = SGD(lr=lr, params=model.parameters())
    criterion = F.binary_cross_entropy
    # 训练模型
    for epoch in range(n_epoch):
        model.train()
        optim.zero_grad()
        logits = torch.reshape(model(X_train), (-1, ))
        loss = criterion(logits, y_train)
        # 更新权重
        loss.backward()
        optim.step()
        if epoch % 1 == 0 and epoch:
            print('epoch: {}, loss: {}'.format(epoch, loss))

    # 模型评估
    model.eval()
    pred = torch.reshape(model(X_test), (-1, ))
    loss = criterion(pred, y_test)
    pred = [1 if x > 0.5 else 0 for x in pred]
    print('acc: {}'.format(accuracy_score(y_test, pred)))
