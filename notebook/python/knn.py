import numpy as np

class KNN:
    def __init__(self, k):
        self.k = k
    def norm(self, X, fit):
        if fit:
            self.Xmean, self.Xstd = X.mean(), X.std() + 0.001
        
        return (X - self.Xmean )/self.Xstd
    def fit(self, X, y):
        self.X = self.norm(X, fit=True)
        self.y = y

    def vote(self, neighbors):
        y = self.y[neighbors]
        return np.argmax(np.bincount(y))

    
    def predict(self, Xtest):
        Xtest = self.norm(Xtest, fit=False)
        preds = []
        for obs in Xtest:
            dist = np.sum((self.X - obs)**2, axis=1)
            kneighbors =np.argsort(dist)[:self.k]
            preds.append(self.vote(kneighbors))
        return preds
    
        
import pandas as pd

k = KNN(5)
df=  pd.read_csv('/Users/shravanshetty/Documents/GitHub/ml_algo_scratch/notebook/python/data/student_result_data.txt', header = None)
df.head()

k.fit(df.loc[:, [0,1]].values, df[2].values)

df['predict'] = k.predict(df.loc[:, [0,1]].values)
print(df)
print((df[2] == df['predict']).sum()/df.shape[0])

