import numpy as np
from random import gauss
from random import seed
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
from Data.DataHandlers.storeData import store
import pickle
import os

seed(42)

def b0(loc1, loc2, l):
    return 3


def b1(loc1, loc2, l):
    return 1+((1/12)*(loc1+loc2))


def b2(loc1, loc2, l):
    temp1 = (l/4 - (loc1 / 2)) ** 2
    temp2 = (l/4 - (loc2 / 2)) ** 2
    return 1+((1/((l/2)**2)) * ((l/4)**2 - temp1) * ((l/4)**2 - temp2))


def b3(loc1, loc2, l):
    return 5+((1/4)*int(loc1/5))


def b4(loc1, loc2, l):
    temp1 = (l/4 - (loc1 / 2)) ** 2
    temp2 = (l/4 - (loc2 / 2)) ** 2
    return l - ((1 / 500) * ((l/4)**2 - temp1) * ((l/4)**2 - temp2))


def synthetic_data_generator(l, dim = 3):
    u = list(range(l))
    v = list(range(l))
    visual = np.zeros((l, l, dim))
    data = {'coords': [], 'X': [], 'y': []}
    coefficients = []

    for i in range(l):
        for j in range(l):
            y = 0
            x = []
            for k in range(dim):
                x.append(gauss(0, 1))
                visual[i, j, k] = globals()['b'+str(k)](u[i], v[j], l)
                y += x[k]*visual[i, j, k]
            y += gauss(0, 0.5)
            data['coords'].append((u[i], v[j]))
            data['y'].append(y)
            data['X'].append(x)
            coefficients.append(visual[i, j, :])

    y = np.asarray(data['y'])
    y = np.reshape(y, [-1, 1])
    x = np.asarray(data['X'])
    coords = data['coords']
    coords = np.asarray(coords)
    x = (x - np.mean(x, axis=0)) / np.std(x, axis=0)
    n = len(y)

    path = os.path.dirname(os.path.abspath(
        __file__ + str("/../../"))) + "/Data/syntheticData1/"

    for i in range(1, 6):
        store(x, y, coords, path, i)

    with open(path + '/_generated_data', 'wb') as file_handle:
        pickle.dump(data, file_handle)
    with open(path + '/coefficients.data', 'wb') as file_handle:
        pickle.dump(coefficients, file_handle)


if __name__ == "__main__":
    synthetic_data_generator(20)
