import os
import pickle

from src.BackfittingModelImproved import ImprovedBackfitting
from src.Backfitting_Model import Backfitting
from src.GGWR import GGWR
from mgwr.gwr import GWR, MGWR
from mgwr.sel_bw import Sel_BW
import src.utils as utils
import warnings
import time
from src.hyperband import Hyperband
from sklearn.ensemble import RandomForestRegressor
import numpy as np
from src.hyperband import Hyperband
from src.spaceSearch import SpaceSearch
from src.GGWR_Model import GGWR_Model
warnings.filterwarnings('ignore')
import random

np.random.seed(42)
random.seed(42)

def random_forrest(_x, _coords, _y):
    D_train = np.concatenate((_x, _coords), axis=1)
    rg = RandomForestRegressor(n_estimators=60)
    rg.fit(D_train, _y.reshape(-1, ))
    return rg

def gwr_mine(_x, _coords, _y):
    bandwidth = Sel_BW(_coords, _y, _x, kernel='gaussian').search(criterion='CV')
    config = {"cutoff": -1}
    gwr_model = GGWR_Model(_x, _y, _coords, config)
    gwr_model.bandwidths = [bandwidth for _ in range(_x.shape[1]+1)]
    return gwr_model

def residual_gwr(_x, _coords, _y):
    bandwidth = Sel_BW(_coords, _y, _x, kernel='gaussian').search(criterion='CV')
    model = GWR(_coords, _y, _x, bw=bandwidth, fixed=False, kernel='gaussian', spherical=True)
    return model

def ggwr_model_train(_x, _coords, _y):
    n = len(_y)
    train_len = int(n*0.86)
    _x_train = _x[0:train_len, :]
    _x_validation = _x[train_len:, :]
    _coords_train = _coords[0:train_len, :]
    _coords_validation = _coords[train_len:, :]
    _y_train = _y[0:train_len, :]
    _y_validation = _y[train_len:, :]
    ggwr_model = GGWR_Model(_x, _y, _coords)
    bw = Sel_BW(_coords, _y, _x, kernel='gaussian').search(criterion='CV')
    SP = SpaceSearch(ggwr_model, 5)
    # bw = 15
    ggwr_model.bandwidths = [bw for i in range(_x_train.shape[1]+1)]
    sim_config = {"steps": 40, "updates": 30, "method": "gaussian_same_all"}
    bandwidths = SP.simulated_annealing([bw for i in range(_x_train.shape[1]+1)], sim_config)
    # print("step 1 done")
    sim_config = {"steps": 40, "updates": 30, "method": "gaussian_all"}
    bandwidths = SP.simulated_annealing(bandwidths, sim_config)
    # print("step 2 done")
    sim_config = {"steps": 120, "updates": 85, "method": "gaussian_one"}
    bandwidths = SP.hill_climbing(bandwidths, sim_config)
    ggwr_model.bandwidths = bandwidths
    return ggwr_model


def backfitting_model(_x, _coords, _y):
    ggwr_model = ImprovedBackfitting(_x, _y, _coords)

    SP = SpaceSearch(ggwr_model)
    start_time = time.time()
    bandwidths = SP.hyperband(81, 32, 3)
    ggwr_model.bandwidths = bandwidths
    print("time", time.time()-start_time)
    ggwr_model.fit(bandwidths)
    return ggwr_model

dataset = "artificialData"
path = os.path.dirname(os.path.abspath(__file__ + str("/../"))) + "/Data/" + dataset + "/"

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

print(X_training.shape, X_validation.shape, X_test.shape)
print(y_training.shape, y_validation.shape, y_test.shape)
print(coords_training.shape, coords_validation.shape, coords_test.shape)

X_combined = np.concatenate((X_validation, X_training), axis=0)
coords_combined = np.concatenate((coords_validation, coords_training), axis=0)
y_combined = np.concatenate((y_validation, y_training), axis=0)

backfitting = backfitting_model(X_combined, coords_combined, y_combined)
improved_ggwr_pred = backfitting.predict(coords_test, X_test)
print(utils.R2(y_test, improved_ggwr_pred))
