import os
import sys
import pickle

from src.BackfittingModelImproved import ImprovedBackfitting
import src.utils as utils
import warnings
import time
from mgwr.gwr import GWR, MGWR
from mgwr.sel_bw import Sel_BW
from src.hyperband import Hyperband
import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt

from src.spaceSearch import SpaceSearch

dataset = "syntheticData1"
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
X_combined = np.concatenate((X_validation, X_training), axis=0)
coords_combined = np.concatenate((coords_validation, coords_training), axis=0)
y_combined = np.concatenate((y_validation, y_training), axis=0)

_x = X_combined
_y = y_combined
_coords = coords_combined

ggwr_model = ImprovedBackfitting(_x, _y, _coords)

fig, axs = plt.subplots(1, 3)
local_x = [[] for _ in range(ggwr_model.numberOfFeatures)]
local_y = [[] for _ in range(ggwr_model.numberOfFeatures)]
for feature in range(ggwr_model.numberOfFeatures):
    print(feature, end=" ")
    for bw in range(5, 300, 10):
        ggwr_model.bandwidths = [5 for _ in range(ggwr_model.numberOfFeatures)]
        ggwr_model.bandwidths[feature] = bw
        ggwr_model.fit(ggwr_model.bandwidths, iterations=10)
        predictions = ggwr_model.predict(coords_test, X_test)
        res = utils.R2(y_test, predictions)
        local_x[feature].append(bw)
        local_y[feature].append(res)
    axs[0].plot(local_x[feature], local_y[feature], label="feature "+str(feature))
axs[0].set_title('local')
axs[0].legend()

global_x = [[] for _ in range(ggwr_model.numberOfFeatures)]
global_y = [[] for _ in range(ggwr_model.numberOfFeatures)]
for feature in range(ggwr_model.numberOfFeatures):
    print(feature, end=" ")
    for bw in range(5, 300, 10):
        ggwr_model.bandwidths = [300 for _ in range(ggwr_model.numberOfFeatures)]
        ggwr_model.bandwidths[feature] = bw
        ggwr_model.fit(ggwr_model.bandwidths, iterations=10)
        predictions = ggwr_model.predict(coords_test, X_test)
        res = utils.R2(y_test, predictions)
        global_x[feature].append(bw)
        global_y[feature].append(res)
    axs[1].plot(global_x[feature], global_y[feature], label="feature "+str(feature))
axs[1].set_title('global')
axs[1].legend()

selector = Sel_BW(coords, y, x, kernel='gaussian', multi=True)
selector.search(criterion='CV', multi_bw_min=[2])
model = MGWR(coords, y, x, selector, sigma2_v1=True, kernel='gaussian', fixed=False, spherical=True)
correct_x = [[] for _ in range(ggwr_model.numberOfFeatures)]
correct_y = [[] for _ in range(ggwr_model.numberOfFeatures)]
for feature in range(ggwr_model.numberOfFeatures):
    print(feature, end=" ")
    for bw in range(5, 300, 10):
        ggwr_model.bandwidths = list(model.bws)
        ggwr_model.bandwidths[feature] = bw
        ggwr_model.fit(ggwr_model.bandwidths, iterations=10)
        predictions = ggwr_model.predict(coords_test, X_test)
        res = utils.R2(y_test, predictions)
        correct_x[feature].append(bw)
        correct_y[feature].append(res)
    axs[2].plot(correct_x[feature], correct_y[feature], label="feature "+str(feature))
axs[2].set_title('correct')
axs[2].legend()