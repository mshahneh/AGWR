from sklearn.ensemble import RandomForestRegressor

from src.BackfittingModelImproved import ImprovedBackfitting
from src.spaceSearch import SpaceSearch
from src.utils import local_dist, kernel_funcs, alt, calculate_dependent
from mgwr.gwr import GWR, MGWR
from mgwr.sel_bw import Sel_BW
import numpy as np
import pickle
import os
import time
import src.utils as utils
from src.DataDivider import Divider
from src.hyperband import Hyperband
from src.GGWR_Model import GGWR_Model
import warnings
warnings.filterwarnings('ignore')


def residual_gwr(_x, _coords, _y):
    bandwidth = Sel_BW(_coords, _y, _x, kernel='gaussian').search(criterion='CV')
    model = GWR(_coords, _y, _x, bw=bandwidth, fixed=False, kernel='gaussian', spherical=True)
    return model


def gwr_mine(_x, _coords, _y):
    bandwidth = Sel_BW(_coords, _y, _x, kernel='gaussian').search(criterion='CV')
    gwr_model = GGWR_Model(_x, None, _y, None, _coords, None, 3, 'gaussian',
                            True, False, True)
    return gwr_model


def backfitting_model(_x, _coords, _y):
    ggwr_model = ImprovedBackfitting(_x, _y, _coords)
    bw = Sel_BW(_coords, _y, _x, kernel='gaussian').search(criterion='CV')
    gwrModel = GWR(_coords, _y, _x, bw=bw, fixed=False,
                   kernel='gaussian', spherical=True)
    gwrModelFit = gwrModel.fit()
    ggwr_model.setB(gwrModelFit.params)
    # print("no_cv with", len(_y), "data", end=" ")
    # # bw = 15
    # ggwr_model.bandwidths = [bw for i in range(_x.shape[1]+1)]
    #
    # SP = SpaceSearch(ggwr_model, 1)
    # sim_config = {"steps": 30, "updates": 20, "method": "gaussian_same_all"}
    # bandwidths = SP.simulated_annealing(
    #     [bw for i in range(_x.shape[1]+1)], sim_config)
    # print("step 1 done", end=" ")
    # sim_config = {"steps": 30, "updates": 20, "method": "gaussian_all"}
    # bandwidths = SP.simulated_annealing(bandwidths, sim_config)
    # print("step 2 done", end=" ")
    # sim_config = {"steps": 120, "updates": 85, "method": "gaussian_one"}
    # bandwidths = SP.hill_climbing(bandwidths, sim_config)
    # ggwr_model.bandwidths = bandwidths
    ggwr_model.bandwidths = bandwidths = [100.,  60.,  10.,   4.,  24.,   7.]
    # ggwr_model.bandwidths = bandwidths = [26, 75, 10, 4, 6, 4]
    ggwr_model.fit(bandwidths)
    return ggwr_model

dataset = "artificialData"
# dataset = "artificialData"
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

# sm_learner = self.spatial_model(sm_x, sm_coords, sm_y)
# temp_pred = sm_learner.predict(sm_coords, sm_x)
# sm_preds[sm_indices] = temp_pred.predy.reshape((-1, 1))
gwr = residual_gwr(X_combined, coords_combined, y_combined)
temp_pred = gwr.predict(coords_test, X_test)
gwr_resy = temp_pred.predy.reshape((-1, 1))
print("gwr", utils.R2(y_test, gwr_resy))

ggwr = backfitting_model(X_combined, coords_combined, y_combined)
temp_pred = ggwr.predict(coords_test, X_test)
ggwr_resy = temp_pred.reshape((-1, 1))
print("\nggwr", utils.R2(y_test, ggwr_resy))