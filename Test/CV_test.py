import os
import pickle
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

def ggwr_model_train_cv(_x, _coords, _y):
    start = time.time()
    ggwr_model = GGWR_Model(_x, _y, _coords)
    bw = Sel_BW(_coords, _y, _x, kernel='gaussian').search(criterion='CV')
    SP = SpaceSearch(ggwr_model, 5)
    # bw = 15
    ggwr_model.bandwidths = [bw for _ in range(_x.shape[1]+1)]
    sim_config = {"steps": 20, "updates": 15, "method": "gaussian_same_all"}
    bandwidths = SP.simulated_annealing([bw for i in range(_x.shape[1]+1)], sim_config)
    sim_config = {"steps": 20, "updates": 15, "method": "gaussian_all"}
    bandwidths = SP.simulated_annealing(bandwidths, sim_config)
    sim_config = {"steps": 120, "updates": 85, "method": "gaussian_one"}
    bandwidths = SP.hill_climbing(bandwidths, sim_config)
    ggwr_model.bandwidths = bandwidths
    print("this section train time:", int((time.time() - start) * 10) / 10)
    return ggwr_model

def ggwr_model_train(_x, _coords, _y):
    start = time.time()
    ggwr_model = GGWR_Model(_x, _y, _coords)
    bw = Sel_BW(_coords, _y, _x, kernel='gaussian').search(criterion='CV')
    SP = SpaceSearch(ggwr_model, 1)
    # bw = 15
    ggwr_model.bandwidths = [bw for _ in range(_x.shape[1]+1)]
    sim_config = {"steps": 40, "updates": 30, "method": "gaussian_same_all"}
    bandwidths = SP.simulated_annealing([bw for i in range(_x.shape[1]+1)], sim_config)
    sim_config = {"steps": 40, "updates": 30, "method": "gaussian_all"}
    bandwidths = SP.simulated_annealing(bandwidths, sim_config)
    sim_config = {"steps": 120, "updates": 85, "method": "gaussian_one"}
    bandwidths = SP.hill_climbing(bandwidths, sim_config)
    ggwr_model.bandwidths = bandwidths
    print("this section train time:", int((time.time()-start)*10)/10)
    return ggwr_model

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

print(X_training.shape, X_validation.shape, X_test.shape)
print(y_training.shape, y_validation.shape, y_test.shape)
print(coords_training.shape, coords_validation.shape, coords_test.shape)

X_combined = np.concatenate((X_validation, X_training), axis=0)
coords_combined = np.concatenate((coords_validation, coords_training), axis=0)
y_combined = np.concatenate((y_validation, y_training), axis=0)

is_pipeline = False
print("pipelined:", is_pipeline)

ggwr = GGWR(ggwr_model_train, random_forrest,
            {"divide_method": "equalCount", "divide_sections": [2, 3], "pipelined": is_pipeline})
ggwr.train(X_combined, coords_combined, y_combined)
res = ggwr.predict(X_test, coords_test, y_test)
res = utils.R2(y_test.reshape(-1).tolist(), res)
print("ggwr without cv", res, "time", ggwr.train_time)


ggwr = GGWR(ggwr_model_train_cv, random_forrest,
            {"divide_method": "equalCount", "divide_sections": [2, 3], "pipelined": is_pipeline})
ggwr.train(X_combined, coords_combined, y_combined)
res = ggwr.predict(X_test, coords_test, y_test)
res = utils.R2(y_test.reshape(-1).tolist(), res)
print("ggwr with cv", res, "time", ggwr.train_time)


is_pipeline = True
print("pipelined:", is_pipeline)

ggwr = GGWR(ggwr_model_train, random_forrest,
            {"divide_method": "equalCount", "divide_sections": [2, 3], "pipelined": is_pipeline})
ggwr.train(X_combined, coords_combined, y_combined)
res = ggwr.predict(X_test, coords_test, y_test)
res = utils.R2(y_test.reshape(-1).tolist(), res)
print("ggwr without cv", res, "time", ggwr.train_time)


ggwr = GGWR(ggwr_model_train_cv, random_forrest,
            {"divide_method": "equalCount", "divide_sections": [2, 3], "pipelined": is_pipeline})
ggwr.train(X_combined, coords_combined, y_combined)
res = ggwr.predict(X_test, coords_test, y_test)
res = utils.R2(y_test.reshape(-1).tolist(), res)
print("ggwr with cv", res, "time", ggwr.train_time)
