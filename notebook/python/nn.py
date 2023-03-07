import numpy as np
from sklearn.datasets import load_digits
digits = load_digits()
X,y = digits['data'], digits['target']
Xy = np.c_[X, y]
np.random.shuffle(Xy)
m = 1500
X,Y = Xy[:m, :64], Xy[:m, 64]
Xtest,ytest = Xy[m:, :64], Xy[m:, 64]

# dataset = pd.DataFrame(X,columns={"X1","X2"})
# dataset["Y"] =Y


def init_parameters(inp_layer, h1_layer, output_layer):
    limit_1 =  np.sqrt(6 / (inp_layer + h1_layer))
    limit_2 =  np.sqrt(6 / (output_layer + h1_layer))
    W1 = np.random.uniform(-1, 1, (inp_layer, h1_layer)) * limit_1
    b1 = np.random.uniform(-1, 1, (h1_layer, 1)) * limit_1
    W2 = np.random.uniform(-1, 1, (h1_layer, output_layer)) * limit_2
    b2 = np.random.uniform(-1, 1, (output_layer, 1)) * limit_2
    return W1, b1, W2, b2


def ReLU(Z):
    return np.maximum(Z, 0)

def gradReLU(Z):
    return (Z>0).astype(int)

def softmax(Z):
    return np.exp(Z)/ sum(np.exp(Z))

def forward_prop(W1, b1, W2, b2, X):
    Z1 = X.dot(W1) + b1.T
    A1 = ReLU(Z1)
    Z2 = A1.dot(W2) + b2.T
    A2 = softmax(Z2)
    return Z1, A1, Z2, A2

def onehot_enc(Y):
    Y = Y.astype(int)
    onehot = np.zeros((Y.shape[0],Y.max() + 1 ))
    onehot[:, Y] = 1
    return onehot

def back_prop(Z1, W1, W2, A1, Z2, A2, y, X):
    m,n = X.shape
    Y = onehot_enc(y)
    dZ2 = A2 - Y
    dW2 = A1.T.dot(dZ2)/m
    db2 = np.mean(dZ2)
    dZ1 = dZ2.dot(W2.T) * gradReLU(A1)
    dW1 = X.T.dot(dZ1)/m
    db1 = np.mean(dZ1)
    return db1, dW1, db2, dW2

def update_param(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha):
    W1 = W1 - alpha * dW1
    b1 = b1 - alpha * db1    
    W2 = W2 - alpha * dW2  
    b2 = b2 - alpha * db2    
    return W1, b1, W2, b2

def get_predictions(A2):
    return np.argmax(A2, 1)

def get_accuracy(predictions, Y):
    print(predictions, Y)
    return np.sum(predictions == Y) / Y.size


def gradient_descent(X, y, input_layer, hidden_layer, output_layer, alpha, iter = 1000):
    W1, b1, W2, b2 = init_parameters(input_layer, hidden_layer, output_layer)
    for i in range(1, iter+1):
        Z1, A1, Z2, A2 = forward_prop(W1, b1, W2, b2, X)
        db1, dW1, db2, dW2 = back_prop(Z1, W1, W2, A1, Z2, A2, y, X)
        W1, b1, W2, b2 = update_param(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha)
        if i%100 == 0:
            print(f"Iteration : {i}")
            predictions = get_predictions(A2)
            print(get_accuracy(predictions, y))
            print("-----"*100)
    return W1, b1, W2, b2

W1, b1, W2, b2 = gradient_descent(X, Y.astype(int),64, 20, 10, 0.10, 500)





