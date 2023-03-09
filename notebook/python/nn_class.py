import numpy as np
class ReLU:
    def __call__(self, z):
        return np.maximum(z, 0)
    
    def gradient(self, z):
        return (z>0)
    
class SoftMax:
    def __call__(self, z):
        return np.exp(z)/sum(np.exp(z))
    
    def gradient(self, z):
        p = self.__call__(z)
        return p * (1-p)
    
def _xavier(l1, l2):
    return np.random.uniform(-1, 1, (l1, l2)) * np.sqrt(6/(l1 + l2))
    # return np.random.randn(l1, l2)

class DenseLayer:
    def __init__(self, hidden_layer, activation1, output_layer, activation2, lr = 0.01, n_iter = 500):
        self.hidden_layer = hidden_layer
        self.output_layer = output_layer
        self.af1 = activation1
        self.af2 = activation2
        self.lr = lr
        self.n_iter = n_iter

    def init_params(self, input_layer):
        
        self.W1 = _xavier(self.hidden_layer, input_layer)
        self.W2 = _xavier(self.output_layer,self.hidden_layer)
        self.b1 = _xavier(self.hidden_layer, 1)
        self.b2 = _xavier(self.output_layer, 1)
        print(f"{input_layer=} {self.hidden_layer=} {self.output_layer=} ")

    def forward(self, X):
        self.Z1 = self.W1.dot(X) + self.b1          # h1, M
        self.A1 = self.af1(self.Z1)                 # h1, M
        self.Z2 = self.W2.dot(self.A1) + self.b2    # o1, M
        self.A2 = self.af2(self.Z2)                 # o1, M
        return self.A2
    
    def onehot_enc(self, y):
        onehot = np.zeros((y.shape[0], y.max() +1 ))
        onehot[np.arange(y.size), y] = 1
        return onehot.T

    def backward_withUpdate(self, X, y):
        # Think in terms of chain rule from dx = dL/dx
        m = y.shape[1]
        dZ2 = self.yhat - y                                     # o1, M
        dW2 = 1 / m * (dZ2.dot(self.A1.T))                           # o1, h1
        db2 = 1 / m * np.sum(dZ2)                                  # o1, 1
        dZ1 = self.W2.T.dot(dZ2) * self.af1.gradient(self.Z1)       # h1, M
        dW1 = 1 / m * (dZ1.dot(X.T))                               # h1, input_layer
        db1 = 1 / m * np.sum(dZ1)                                   # h1, 1

        self.W1 -= self.lr * dW1
        self.W2 -= self.lr * dW2
        self.b1 -= self.lr * db1
        self.b2 -= self.lr * db2


    def fit(self, X, y):
        y = y.astype(int)
        self.init_params(X.shape[0])
        for i in range(1, self.n_iter + 1):
            self.yhat = self.forward(X)
            if i % 100 == 0:
                acc = np.sum(y == np.argmax(self.yhat, axis=0))/len(y)
                print(f"Iter {i} Accuracy : {acc}")
            # Running backward
            self.backward_withUpdate(X, self.onehot_enc(y))
        return self
    
    def predict(self, X):
        yhat = self.forward(X)
        pred  = np.argmax(yhat,0)


from sklearn.datasets import load_digits
digits = load_digits()
X,y = digits['data'], digits['target']
Xy = np.c_[X, y]
np.random.shuffle(Xy)
m = 1500
X,Y = Xy[:m, :64].T, Xy[:m, 64].T
Xtest,ytest = Xy[m:, :64], Xy[m:, 64]


nn = DenseLayer(20, ReLU(), 10, SoftMax(), lr = 0.01, n_iter = 500)
nn.fit(X, Y)