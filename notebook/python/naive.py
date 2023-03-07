import numpy as np

class NaiveBayes:
    def __init__(self, tol = 0.001):
        self.tol = tol
    
    def fit(self, X, y):
        self.priors = []
        self.Xymean = []
        self.Xystd = []

        self.uniq_y = np.unique(y)
        for uy in self.uniq_y:
            mask  = (y == uy).squeeze()
            self.priors.append((mask.sum())/ len(mask))
            self.Xymean.append(X[mask].mean(axis = 0))
            self.Xystd.append(X[mask].std(axis = 0) + self.tol)
    
    def likelihood(self, X, uy):
        const = np.sqrt(2 * np.pi) * self.Xystd[uy]
        z = 0.5 *  ((X - self.Xymean[uy])/self.Xystd[uy])**2
        
        return np.product(1/const * np.exp(-z), axis=1)

    def predict(self, X):
        """
        Goal is :
        Run for each uy, and check the maximized prob based on bayes
        Pred is max_prob index
        """
        pred_y = []
        for uy in self.uniq_y:
            pred_y.append(self.likelihood(X, uy) * self.priors[uy])
        return np.argmax(np.stack(pred_y, axis=1), axis=1)
            
import pandas as pd
df =  pd.read_csv('/Users/shravanshetty/Documents/GitHub/ml_algo_scratch/notebook/python/data/student_result_data.txt', header = None)
df.head()

nb = NaiveBayes()
nb.fit(df.loc[:, [0,1]].values, df[[2]].values)

df['predict'] = nb.predict(df.loc[:, [0,1]].values)
print(df.head())
print((df[2] == df['predict']).sum()/df.shape[0])