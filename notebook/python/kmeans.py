import numpy as np

class KMeans: 
    def __init__(self, k, n_reps = 10, n_iter = 100):
        self.k  = k
        self.n_reps = n_reps
        self.n_iter = n_iter
    
    def _assign_cluster(self, X, centroids):
        arr = []
        for i in range(self.k):
            dist = (X - centroids[i]) **2
            arr.append(dist.sum(axis=1))
        return np.argmin(np.stack(arr, axis=1), axis=1)

    def cluster_error(self, X, centroids, clusters):
        err = 0
        for i in range(self.k):
            mask = clusters == i
            dist = (X[mask] - centroids[i])**2
            err+= np.sum(dist)
        return err

    def _compute_centroid(self, X, clusters):
        centroids = []
        for i in range(self.k):
            mask = clusters == i
            centroids.append(X[mask].mean(axis=0))
        return centroids

    def fit(self, X):
        X = (X - X.mean(axis=0))/(X.std(axis=0) + 0.0001)
        rep_errs = []
        rep_centroids = []
        rep_clusters = []
        old_err = np.float("inf")
        for rep in range(1, self.n_reps + 1):
            centroids = X[np.random.choice(len(X), self.k)]
            print(f"Rep : {rep}")
            for it in range(1, self.n_iter + 1):
                clusters = self._assign_cluster(X, centroids)
                err = self.cluster_error(X, centroids, clusters)
                if np.isclose(old_err, err):
                    print("Early Stopping")
                    break
                centroids = self._compute_centroid(X, clusters)
                if it % 10 == 0:
                    print(f"Iter {it} Error : {err:.2f}")
                old_err = err
            rep_errs.append(err)
            rep_centroids.append(centroids)
            rep_clusters.append(clusters)
        self._best_rep = np.argmin(err)
        self._best_err = rep_errs[self._best_rep]
        self._clusters = rep_clusters[self._best_rep]
        return self._clusters


from sklearn.datasets import load_digits
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import pandas as pd

#Load Data
data = load_digits().data
pca = PCA(2)
  
#Transform the data
df = pca.fit_transform(data)
k = KMeans(10)

#Applying our function
label = k.fit(df)
print(k._best_err)
 
#Visualize the results
u_labels = np.unique(label)
for i in u_labels:
    plt.scatter(df[label == i , 0] , df[label == i , 1] , label = i)
plt.legend()

plt.show()