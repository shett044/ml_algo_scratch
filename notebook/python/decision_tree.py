import numpy as np
class DecisionNode:
    def __init__(self, featIdx=None, threshold=None, val=None, childs=None, gain=None):
        self.featIdx = featIdx
        self.gain = gain
        self.threshold = threshold
        self.val = val
        self.childs = childs


class DecisionTree():
    def  __init__(self, max_depth = 5, min_sample_split= 3):
        self.max_depth =max_depth
        self.min_sample_split  = min_sample_split
        self.root = None
    
    def diff_impurity(self, y, ygroups):
        def entropy(y):
            probs = np.bincount(y)/len(y)
            return -np.sum(p * np.log2(p) for p in probs if p > 0)
            # return 1 - np.sum([p**2 for p in probs])
            
        p_entropy = entropy(y)
        c_entropy = 0
        for yc in ygroups:
            c_entropy+= entropy(yc) * len(yc)/len(y)
        return p_entropy - c_entropy


    def leaf_calc(self, y):
        return np.argmax(np.bincount(y))

    def _bestSplit(self, X, y):
        self.bestGain = {'gain': -1}
        for fix in range(X.shape[1]):
            feat = X[:, fix]
            fval = np.unique(feat)
            for fv in fval:
                mask = feat <= fv
                ygroups = y[mask], y[~mask]
                if ygroups[0].shape[0] == 0 or ygroups[0].shape[0] == len(y):
                    # Split is not even
                    continue
            
                gain = self.diff_impurity(y, ygroups)
                if gain > self.bestGain['gain']:
                    self.bestGain['gain'] = gain
                    self.bestGain['featIdx'] = fix
                    self.bestGain['threshold'] = fv
                    self.bestGain['childs'] = [mask, ~mask]
        
        return self.bestGain
            
    

    def _buildTree(self, X, y, curr_depth):
        m, n = X.shape
        if curr_depth <= self.max_depth and self.min_sample_split<=m and len(np.unique(y)) > 1:
            split = self._bestSplit(X, y)
            split['childs'] = [self._buildTree(X[mask], y[mask], curr_depth + 1) for mask in split['childs']]
            return DecisionNode(**split)
        return DecisionNode(val = self.leaf_calc(y))

    def fit(self, X, y):
        self.root = self._buildTree(X, y, 0)
    
    def _predict(self, obs, node):
        if not node:
            return node
        if node.val is not None:
            return node.val
        child = node.childs[int(obs[node.featIdx] > node.threshold)]
        return self._predict(obs, child)


    def predict(self, X):
        pred = []
        for obs in X:
            pred.append(self._predict(obs, self.root))
        return pred


from sklearn.datasets import load_iris

iris = load_iris()

X = iris['data']
y = iris['target']


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.8, random_state=42)

model = DecisionTree()
model.fit(X_train, y_train)
preds = model.predict(X_test)

from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier

sk_model = DecisionTreeClassifier()
sk_model.fit(X_train, y_train)
sk_preds = sk_model.predict(X_test)


print("Acc Score: {} SkLearn Acc Score: {}".format(accuracy_score(y_test, preds), accuracy_score(y_test, sk_preds)))