import numpy as np

from models import MLP
from sklearn import datasets
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

if __name__ == '__main__':
    np.random.seed(123)

    '''
    1. データの準備
    '''
    N = 300
    x, t = datasets.make_moons(N, noise=0.2)
    t = t.reshape(N, 1)

    x_train, x_test, t_train, t_test = train_test_split(x, t, test_size=0.2)

    '''
    2. モデルの構築
    '''
    model = MLP(2, 3, 1)

    '''
    3. モデルの学習
    '''
    def compute_loss(t, y):
        return (-t * np.log(y) - (1 - t) * np.log(1 - y)).sum()
    
    def train_step(x, t):
        y = model(x)
        for i, layer in enumerate(model.layers[::-1]):
            if i == 0:
                delta = y - t
            else:
                delta = layer.backward(delta, W)

            dW, db = layer.compute_gradients(delta)

            alpha = 0.1
            layer.W = layer.W - alpha * dW
            layer.b = layer.b - alpha * db
            W = layer.W

        loss = compute_loss(t, y)
        return loss
    
    epochs = 100
    batch_size = 30
    n_batches = x_train.shape[0] // batch_size

    for epoch in range(epochs):
        train_loss = 0.0
        x_, t_ = shuffle(x_train, t_train)

        for n_batch in range(n_batches):
            start = n_batch * batch_size
            end = start + batch_size

            train_loss += train_step(x_[start:end], t_[start:end])

        if epoch % 10 == 0 or epoch == epoch - 1:
            message = f'epoch: {epoch+1: 10d}, loss: {train_loss: 10.3f}'
   
    '''
    4. モデルの評価
    '''
    preds = model(x_test) > 0.5
    acc = accuracy_score(t_test, preds)
    print(f'acc.: {acc:10.3f}')

    '''
    5. 可視化
    '''
    # plt.scatter(x=x[:,0], y=x[:,1], s=10, c=t)
    # plt.show()
    def plot_decision_boundary(model, x, t):
        # サンプルデータのプロット

        tk = t.reshape(N)
        plt.plot(x[:, 0][tk==0], x[:, 1][tk==0], 'bo')
        plt.plot(x[:, 0][tk==1], x[:, 1][tk==1], 'r^')
        # plt.scatter(x=x[:,0], y=x[:,1], s=10, c=t)
        plt.xlabel('x') # x 軸方向に x を表示
        plt.ylabel('y', rotation=0) # y 軸方向に y を表示
        
        # 描画範囲の設定
        margin = 0.1
        x1_min, x1_max = x[:, 0].min() - margin, x[:, 0].max() + margin
        x2_min, x2_max = x[:, 1].min() - margin, x[:, 1].max() + margin
        
        # 用意した間隔を使用してグリッドを作成
        _x, _y = np.meshgrid(np.arange(x1_min, x1_max, 0.01), np.arange(x2_min, x2_max, 0.01))
        
        # 多次元配列の結合
        xy = np.array([_x.ravel(), _y.ravel()]).T
        
        # 予測結果を算出し、分類境界線を図示
        y_pred = model(xy).reshape(_x.shape)
        custom_cmap = ListedColormap(['mediumblue', 'orangered'])
        plt.contourf(_x, _y, y_pred, cmap=custom_cmap, alpha=0.2)
        plt.show()
    plot_decision_boundary(model, x, t)