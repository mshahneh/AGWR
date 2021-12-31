from sklearn.ensemble import RandomForestRegressor

from numpy import loadtxt
import xgboost
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler  
import numpy as np
import random


def random_forrest(_x, _coords, _y, process_count=-1):
    D_train = np.concatenate((_x, _coords), axis=1)
    rg = RandomForestRegressor(n_estimators=60, n_jobs=process_count)
    rg.fit(D_train, _y.reshape(-1, ))
    return rg


def neural_network(_x, _coords, _y, process_count=-1):
    D_train = np.concatenate((_x, _coords), axis=1)
    nn = NN(400)
    nn.train(D_train, _y.reshape(-1, ))
    # print("nn was trained", nn.W, nn.v)
    return nn


def xgb(_x, _coords, _y, process_count=-1):
    mxgb = xgboost.XGBRegressor()
    D_train = np.concatenate((_x, _coords), axis=1)
    mxgb.fit(D_train, _y.reshape(-1, ))
    return mxgb

def MLP(_x, _coords, _y, process_count=-1):
    layers = 700
    D_train = np.concatenate((_x, _coords), axis=1)
    clf = MLPRegressor(hidden_layer_sizes=(layers, layers, ))
    # print("hi", _y.reshape(-1, ).shape)
    clf.fit(D_train, _y.reshape(-1, ))
    # print("ye")
    return clf


class NN:
    def __init__(self, nodes):
        self.nodes = nodes
        self.W = None
        self.v = None

    def train(self, _x, _y):
        _x = np.c_[_x, np.ones(len(_x))]
        v, W = self.nural_classifier(self.nodes, _x, _y, 0.5)
        self.v = v
        self.W = W

    def nural_classifier(self, k, X, y, eta):
        d = X.shape[1]
        # W=np.random.randn(k,d)*0.01 # Input layer
        # v=np.random.randn(k)*0.01 # Output layer

        W = np.random.normal(0, 1 / k, (k, d))
        v = np.random.normal(0, 1 / 100, k)
        # eta=15  # learning rate
        N = X.shape[0]
        ITNUM = 30 * N
        for it in range(ITNUM):
            if it % (N//4) == 0:
                eta = 0.95 * eta
                # print(it)
                # print(v, W)
            i = random.randint(0, N - 1)
            grad_v, grad_W = self.get_grad(v, W, y[i], X[i, :])
            v -= eta * grad_v / N
            W -= eta * grad_W / N

        return v, W

    @staticmethod
    def relu(x):
        return np.maximum(x, 0)

    def predict(self, X):
        X = np.c_[X, np.ones(len(X))]
        prediction = np.zeros(X.shape[0])
        for i in range(X.shape[0]):
            prediction[i] = self.local_predict(self.v, self.W, X[i, :])
        return prediction

    def local_predict(self, v, W, x):
        return v.dot(self.relu(W.dot(x)))

    def get_grad(self, v, W, y, x):
        h = self.relu(W.dot(x))
        sigmap = W.dot(x) > 0 + 0.
        yh = self.local_predict(v, W, x)
        r = yh - y
        grad_v = r * h
        grad_W = r * np.outer(v * sigmap, x)
        return grad_v, grad_W