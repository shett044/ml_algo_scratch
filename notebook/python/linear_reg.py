import numpy as np
class LinearReg:
    def __init__(self, lr = 0.01, n_iter = 1000, l2_reg = 0.01):
        self.lr = lr
        self.n_iter = n_iter
        self.l2_reg = l2_reg
    
    def cost(self, y, yhat):
        return np.mean((y - yhat) ** 2) + self.l2_reg * self.theta.T.dot(self.theta).item()
    
    def norm(self, X, fit):
        if fit:
            self.Xmean, self.Xstd = X.mean(axis=0), X.std(axis=0) + 0.001
        return (X - self.Xmean)/self.Xstd

    def add_bias(self, X):
        return np.c_[X, np.ones((X.shape[0], 1))]

    def h(self, X):
        return X.dot(self.theta)

    def cost(self, y, yhat):
        return 0.5 * (np.mean(np.power(y - yhat, 2)) + self.l2_reg * self.theta.T.dot(self.theta).item())

    def fit(self, X, y):
        if y.ndim == 1:
            y = y.reshape(-1 ,1)
        # Norm
        X = self.norm(X, fit=True)
        # Add bias
        X  = self.add_bias(X)
        # define weights
        m, n = X.shape
        self.theta = np.random.randn(n, 1)
        # Learn from GD, by calculating the gradient
        for i in range(1, self.n_iter + 1):
            yhat = self.h(X)
            grad_theta = self.l2_reg * self.theta
            grad_theta[-1] = 0
            grad_theta+= X.T.dot(yhat - y) / m
            # Update the parameters
            self.theta -= self.lr * grad_theta
            if i % 100 ==0 :
                print(f"Iter : {i} Error : {self.cost(y, yhat): .2f}")
        
    def predict(self, X):
         # Norm
        X = self.norm(X, fit=False)
        # Add bias
        X  = self.add_bias(X)
        return self.h(X)
    
    def score(self, y, yhat, adj = False):
        SST = np.power(y - np.mean(y), 2).sum()
        SSR = np.power(y - yhat, 2).sum()
        ratio = SSR/SST
        rsq = 1 - ratio

        if adj:
            n = self.theta.shape[0]
            m = y.shape[0]
            rsq = 1 - (ratio * (m - 1)/(m - n-1))
        return rsq


import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.datasets import fetch_california_housing

# df=pd.read_csv('data/ex1data1.txt', header=None)
# X,y = df[[0]].values, df[[1]].values
raw_data = fetch_california_housing()
X,y = raw_data['data'] , raw_data['target']
print(X.shape)
# y = np.log1p(y)
l = LinearReg(l2_reg=0.01, lr = 0.1, n_iter=1000)
l.fit(X,y)
# l.fit_sgd(X,y,500)
df = pd.DataFrame(y, columns= ['target'])
df['predict'] = l.predict(X)
print(df.head())
# print(l.theta)
rsq = l.score(df['target'].values, df['predict'].values)
adj_rsq = l.score(df['target'].values, df['predict'].values, True)
rmse_pred, rmse_base = mean_squared_error(df['target'], df['predict'], squared=False), mean_squared_error(df['target'], np.ones(len(df)) * df['target'].mean(), squared=False)
print(f"RMSE: {rmse_pred:.2f} RMSE Base: {rmse_base:.2f} Norm RMSE %: {100 * rmse_pred/rmse_base:.2f}")
print(f"Rsq = {rsq}, AdjRsq = {adj_rsq} sklean = {r2_score(df['target'].values, df['predict'].values)}")
# l.plot_cost()
