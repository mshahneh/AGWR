import sys
# sys.path.insert(0, '..')
from src.utils import local_dist
import os
import pickle
from src.GGWR_Model import GGWR_Model
import matplotlib.pyplot as plt
from mgwr.gwr import GWR, MGWR
from mgwr.sel_bw import Sel_BW
import src.utils as utils
import warnings
import time
import numpy as np
from src.hyperband import Hyperband
warnings.simplefilter(action='ignore', category=FutureWarning)

dataset = "kingHousePrices"
path = os.path.dirname(os.path.abspath(__file__ + str("/../../"))) + "/Data/" + dataset + "/"

with open(path + 'training_idx.data', 'rb') as filehandle:
    training_idx = pickle.load(filehandle)
with open(path + 'validation_idx.data', 'rb') as filehandle:
    validation_idx = pickle.load(filehandle)
with open(path + 'test_idx.data', 'rb') as filehandle:
    test_idx = pickle.load(filehandle)
with open(path + 'x.data', 'rb') as filehandle:
    x = pickle.load(filehandle)
with open(path + 'y.data', 'rb') as filehandle:
    y = pickle.load(filehandle)
with open(path + 'coords.data', 'rb') as filehandle:
    coords = pickle.load(filehandle)

X_training, X_validation, X_test = x[training_idx, :], x[validation_idx], x[test_idx, :]
y_training, y_validation, y_test = y[training_idx, :], y[validation_idx], y[test_idx, :]
coords_training, coords_validation, coords_test = coords[training_idx], coords[validation_idx], coords[test_idx]


kernelFuncs = ['triangular', 'uniform', 'quadratic', 'quartic', 'gaussian', 'bisquare', 'exponential']
point = 2
ggwr = GGWR_Model(X_training, X_validation, y_training, y_validation, coords_training, coords_validation, -1, 'gaussian', True, False, False)
ggwr.bandwidths = [43, 64, 43, 85, 48, 71, 51, 58]

print("for each point we show how a bandwidth distributes weights")
wi = ggwr._build_wi(point, ggwr.bandwidths, "training", list(range(ggwr.train_len)))
plt.suptitle('No cutoff')
for i in range(len(ggwr.bandwidths)):
    indices = np.where(ggwr.training_distances[point, :] < 30)[0]
    plt.scatter(ggwr.training_distances[point, indices], wi[indices, i], label='bandwidth='+str(ggwr.bandwidths[i]))
    plt.xlabel('distance')
    plt.ylabel('weight')
plt.legend()
plt.show()

print("for a point we show what will be the point where we cut below 10% of weight")

plt.suptitle('cutoff with bandwidth = 20')
cutoffs = [2, 3, 5, 10, -1]
for i in range(len(cutoffs)):
    ggwr.cutoff = cutoffs[i]
    wi = ggwr._build_wi(point, [20], "training", list(range(ggwr.train_len)))
    indices = np.where(ggwr.training_distances[point, :] < 20)[0]
    idx = np.argsort(ggwr.training_distances[point, indices])
    plt.plot(ggwr.training_distances[point, indices[idx]], wi[indices[idx]], label='cutoff='+str(cutoffs[i]))
    plt.xlabel('distance')
    plt.ylabel('weight')
plt.legend()
plt.savefig('cutoff.png', bbox_inches='tight', pad_inches=0, dpi=500)
plt.show()


plt.suptitle('cutoff with bandwidth = 5')
for i in range(len(cutoffs)):
    ggwr.cutoff = cutoffs[i]
    wi = ggwr._build_wi(point, [5], "training", list(range(ggwr.train_len)))
    idx = np.argsort(ggwr.training_distances[point, :])
    plt.plot(ggwr.training_distances[point, idx], wi[idx], label='cutoff='+str(cutoffs[i]))
    plt.xlabel('distance')
    plt.ylabel('weight')
plt.legend()
plt.show()

plt.suptitle('cutoff with bandwidth = 50')
for i in range(len(cutoffs)):
    ggwr.cutoff = cutoffs[i]
    wi = ggwr._build_wi(point, [50], "training", list(range(ggwr.train_len)))
    idx = np.argsort(ggwr.training_distances[point, :])
    plt.plot(ggwr.training_distances[point, idx], wi[idx], label='cutoff='+str(cutoffs[i]))
    plt.xlabel('distance')
    plt.ylabel('weight')
plt.legend()
plt.show()

# comparing the effect of cutoff on error value

# ggwr.bandwidths = [50 for _ in range(ggwr.numberOfFeatures)]
# ggwr.cutoff = -1
# warnings.simplefilter(action='ignore', category=FutureWarning)
# predictions = ggwr.predict(coords_test, X_test)
# print("the GGWR_Model prediction error is equal to: ", utils.R2(y_test, predictions))
#
# for i in range(len(cutoffs)):
#     warnings.simplefilter(action='ignore', category=FutureWarning)
#     ggwr.cutoff = cutoffs[i]
#     predictions = ggwr.predict(coords_test, X_test)
#     print("the GGWR_Model prediction with cutoff =", cutoffs[i], "error is equal to: ", utils.R2(y_test, predictions))


# comparing the effect of bandwidth
# a = [58, 58, 58, 58, 58, 58, 58, 58]  # 0.139
# b = [43, 64, 43, 85, 48, 71, 51, 52]  # 8.1
# print(ggwr.cutoff)
# ggwr.cutoff = -1
# B = ggwr.fit(a, "validation")
# valid_pred = utils.calculate_dependent(B, ggwr.X_validation)
# error = utils.R2(ggwr.y_validation, valid_pred)
# print(error)
# ggwr.bandwidths = a
# predictAll = ggwr.predict(coords, x)
# with open(path + '/Test1_predictAll.data', 'wb') as filehandle:
#     pickle.dump(predictAll, filehandle)
#
# B = ggwr.fit(b, "validation")
# valid_pred = utils.calculate_dependent(B, ggwr.X_validation)
# error = utils.R2(ggwr.y_validation, valid_pred)
# print(error)
# ggwr.bandwidths = b
# predictAll = ggwr.predict(coords, x)
# with open(path + '/Test2_predictAll.data', 'wb') as filehandle:
#     pickle.dump(predictAll, filehandle)

plt.suptitle('weights based on bandwidth')
ggwr.bandwidths = np.sort([20, 30, 130, 64, 35, 85, 48, 71, 51, 58])
for i in range(len(ggwr.bandwidths)):
    ggwr.cutoff = -1
    wi = ggwr._build_wi(point, [ggwr.bandwidths[i]], coords_test[point], list(range(len(coords_training))))
    dist = local_dist(coords_test[point], ggwr.coords_training[list(range(ggwr.train_len))], ggwr.spherical).reshape(-1)
    idx = np.argsort(dist)
    plt.plot(dist[idx], wi[idx], label='bandwidth='+str(ggwr.bandwidths[i]))
    plt.xlabel('distance')
    plt.ylabel('weight')
plt.legend()
plt.show()