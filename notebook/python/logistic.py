import numpy as np
class LogisticReg:
    def __init__(self, lr = 0.01, n_iter = 1000, l2_reg = 0.01):
        self.lr = lr
        self.n_iter = n_iter
        self.l2_reg = l2_reg
    
    def h(self, X):
        z = X.dot(self.theta)
        return 1 / ( 1+ np.exp(-z) )
    
    def norm(self, X, fit):
        if fit:
            self.Xmean, self.Xstd = X.mean(axis=0), X.std(axis=0) + 0.001
        return (X - self.Xmean)/self.Xstd

    def add_bias(self, X):
        return np.c_[X, np.ones((X.shape[0],1))]
    
    def cost(self, y, yhat):
        return np.mean(-y * np.log(yhat) - (1-y) * np.log(1-yhat)) + self.l2_reg * self.theta.T.dot(self.theta).item()

    def fit(self, X ,y):
        # Normalize data
        X = self.norm(X, fit = True)
        # Add bias 
        X = self.add_bias(X)
        m, n = X.shape
        # random init weights
        self.theta = np.random.randn(n, 1)
        # Start grad descent
        for it in range(1, self.n_iter + 1 ):
            yhat = self.h(X)
            grad_l2 = self.l2_reg * self.theta
            grad_l2[-1] = 0
            gradtheta = 1/m * X.T.dot(yhat- y) + grad_l2
            self.theta -= self.lr * gradtheta
            if it % 100 == 0:
                print(f"Iterator : {it} Cost: {self.cost(y, yhat):.2f} ")
    
    def predict(self, X):
        X = self.add_bias(self.norm(X, fit=False))
        return np.round(self.h(X) >= 0.5)


import pandas as pd
df=  pd.read_csv('/Users/shravanshetty/Documents/GitHub/ml_algo_scratch/notebook/python/data/student_result_data.txt', header = None)
df.head()

nb = LogisticReg(lr= .01 , n_iter = 1000, l2_reg=0.01)
nb.fit(df.loc[:, [0,1]].values, df[[2]].values)
# 
df['predict'] = nb.predict(df.loc[:, [0,1]].values)
print(df.head())
print(nb.theta)
# [[0.00837496]
#  [0.00747461]
#  [0.34619282]]
print((df[2] == df['predict']).sum()/df.shape[0])