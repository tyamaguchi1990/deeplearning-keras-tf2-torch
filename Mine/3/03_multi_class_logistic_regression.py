import numpy as np
import matplotlib.pyplot as plt

from sklearn.utils import shuffle

class LogisticRegression(object):
    '''
    (多クラス)ロジスティック回帰
    '''
    def __init__(self, input_dim, output_dim):
        self.input_dim = input_dim
        self.W = np.random.normal(size=(input_dim, output_dim))
        self.b = np.zeros(output_dim)

    def __call__(self, x):
        return self.forward(x)
    
    def forward(self, x):
        return softmax(np.matmul(x, self.W) + self.b)
    
    def compute_gradients(self, x, t):
        y = self.forward(x)
        delta = y - t
        dW = np.matmul(x.T, delta)
        db = np.matmul(np.ones(x.shape[0]), delta)
        return dW, db
    
def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=1, keepdims=True)

if __name__ == '__main__':
    np.random.seed(123)

    '''
    1. データの準備
    '''
    M = 2     # Input Dimension
    K = 3     # Output Dimension
    n = 100   # Number of data for each class
    N = K * n # Number of all data

    x1 = np.random.randn(n, M) + np.array([ 0, 10])
    x2 = np.random.randn(n, M) + np.array([10, 10])
    x3 = np.random.randn(n, M) + np.array([10,  0])
    t1 = np.array([[1, 0, 0] for _ in range(n)])
    t2 = np.array([[0, 1, 0] for _ in range(n)])
    t3 = np.array([[0, 0, 1] for _ in range(n)])
    x = np.concatenate((x1, x2, x3), axis=0)
    t = np.concatenate((t1, t2, t3), axis=0)

    '''
    2. モデルの構築
    '''
    model = LogisticRegression(input_dim=M, output_dim=K)

    '''
    3. モデルの学習
    '''
    def compute_loss(t, y):
        return (-t * np.log(y)).sum(axis=1).mean() # データ数増加によるオーバーフロー回避のためにsum()ではなくmean()を使用する。

    def train_step(x, t):
        alpha = 0.1
        dW, db = model.compute_gradients(x, t)
        model.W = model.W - alpha * dW
        model.b = model.b - alpha * db
        loss = compute_loss(t, model(x))
        return loss
    
    epochs = 10
    batch_size = 50
    n_batches = x.shape[0] // batch_size

    for epoch in range(epochs):
        train_loss = 0.
        x_, t_= shuffle(x, t)

        for n_batch in range(n_batches):
            start = n_batch * batch_size
            end = start + batch_size

            train_loss += train_step(x_[start:end],
                                     t_[start:end])
            
        if epoch % 10 == 0 or epoch == epochs - 1:
            print('epoch: {}, loss: {:.3f}'.format(epoch+1, train_loss))

    '''
    4. モデルの評価
    ''' 
    x_, t_ = shuffle(x, t)
    preds = model(x_[0:5])
    classified = np.argmax(t_[0:5], axis=1) == np.argmax(preds[0:5], axis=1)
    print('Prediction matched:', classified)

    '''
    5. 分類結果の可視化
    ''' 
    from functools import partial
    z = np.arange(-5.0, 15.0, 0.01)
    def f(x1, k1, k2):
        w = model.W[:,k1] - model.W[:,k2]
        b = model.b[k1] - model.b[k2]
        w1 = w[0]
        w2 = w[1]
        x2 = - 1.0 / w2 * (w1 * x1 + b)
        return x2
    plt.plot(z, partial(f,k1=0,k2=1)(z))
    plt.plot(z, partial(f,k1=1,k2=2)(z))
    plt.plot(z, partial(f,k1=2,k2=0)(z))
    # --
    v1 = np.ones(n) * 1.0
    v2 = np.ones(n) * 2.0
    v3 = np.ones(n) * 3.0
    v = np.concatenate((v1, v2, v3), axis=0)
    plt.scatter(x=x[:,0], y=x[:,1], s=10, c=v)
    plt.show()