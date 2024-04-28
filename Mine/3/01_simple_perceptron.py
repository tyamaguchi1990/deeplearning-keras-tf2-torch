import numpy as np

class SimplePerceptron(object):
    '''
    単純パーセプトロン
    '''
    def __init__(self, input_dim):
        self.input_dim = input_dim
        self.w = np.random.normal(size=(input_dim,))
        self.b = 0.

    def forward(self, x):
        w = self.w
        y = step(np.matmul(w.T, x) + self.b)
        return y
    
    def compute_deltas(self, x, t):
        y = self.forward(x)
        delta = y - t
        dw = delta * x
        db = delta
        return dw, db

def step(x):
    return 1 * (x > 0)

if __name__ == '__main__':
    np.random.seed(123)

    '''
    1. データの準備
    '''
    d = 2  # Input dimension
    N = 20 # Number of all data

    mean = 5

    x1 = np.random.randn(N//2, d) + np.array([0, 0])
    x2 = np.random.randn(N//2, d) + np.array([mean, mean])

    t1 = np.zeros(N//2)
    t2 = np.ones(N//2)

    x = np.concatenate((x1, x2), axis=0) # Input data
    t = np.concatenate((t1, t2)) # Output data

    '''
    2. モデルの構築
    '''
    model = SimplePerceptron(input_dim=d)

    '''
    3. モデルの学習
    '''
    def compute_loss(dw, db):
        return all(dw == 0) * (db == 0)
    
    def train_step(x, t):
        dw, db = model.compute_deltas(x, t)

        # Debug
        # print("  x=", end='')
        # print(x, end=', ')
        # print("  dw=", end='')
        # print(dw, end=', ')
        # print("  db=", end='')
        # print(db)

        loss = compute_loss(dw, db)
        model.w = model.w - dw
        model.b = model.b - db
        return loss
    
    count = 0
    while True:
        count += 1
        print("Step: ", count)

        classified = True
        for i in range(N):
            loss = train_step(x[i], t[i])
            classified *= loss
        if classified:
            break

    '''
    4. モデルの評価
    '''
    print('w: ', model.w)
    print('b: ', model.b)

    print('(0, 0) =>', model.forward([0, 0]))  # => 0
    print('(5, 5) =>', model.forward([5, 5]))  # => 1

