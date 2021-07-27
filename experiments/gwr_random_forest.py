from sklearn.ensemble import RandomForestRegressor
from src.utils import local_dist, kernel_funcs, alt, calculate_dependent
from mgwr.gwr import GWR, MGWR
from mgwr.sel_bw import Sel_BW
import numpy as np
import pickle
import time
import os
import src.utils as utils
from src.DataDivider import Divider
import warnings
warnings.filterwarnings('ignore')

def random_forrest(_x, _coords, _y):
    D_train = np.concatenate((_x, _coords), axis=1)
    rg = RandomForestRegressor(n_estimators=60)
    rg.fit(D_train, _y.reshape(-1, ))
    return rg


def residual_gwr(_x, _coords, _y):
    bandwidth = Sel_BW(_coords, _y, _x, kernel='gaussian').search(criterion='CV')
    model = GWR(_coords, _y, _x, bw=bandwidth, fixed=False, kernel='gaussian', spherical=True)
    return model

# dataset = "kingHousePrices"
dataset = "artificialData"
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

# x = x[:, 1:]


# X_training, X_validation, X_test = x[training_idx, 1:], x[validation_idx, 1:], x[test_idx, 1:]
# y_training, y_validation, y_test = y[training_idx, :], y[validation_idx], y[test_idx, :]
# coords_training, coords_validation, coords_test = coords[training_idx], coords[validation_idx], coords[test_idx]

training_idx = np.concatenate((training_idx, validation_idx))
X_training, X_test = x[training_idx, 1:], x[test_idx, 1:]
y_training, y_test = y[training_idx, :], y[test_idx, :]
coords_training, coords_test = coords[training_idx], coords[test_idx]
print("done reading")

numberOfLearners = 6
learners = [{} for _ in range(numberOfLearners)]
n = len(y_training)
batchSize = int(n/numberOfLearners)
sections = [2, 3]
divider = Divider()
indices = divider.grid_divide(coords_training, sections[0], sections[1], "equalCount")

rounds_count = 10
rf_round = [0 for _ in range(rounds_count)]
combined = [0 for _ in range(rounds_count)]
combine_times = [0 for _ in range(rounds_count)]
rf_rimes = [0 for _ in range(rounds_count)]

start = time.time()
gwr_preds = np.zeros(len(y_training)).reshape((-1, 1))
for i in range(numberOfLearners):
    print(i, end=" ")
    gwr_indices = indices[i]
    gwr_x = X_training[gwr_indices]
    gwr_coords = coords_training[gwr_indices]
    gwr_y = y_training[gwr_indices]
    gwr_learner = residual_gwr(gwr_x, gwr_coords, gwr_y)
    learners[i] = {"gwr": gwr_learner}
    gwr_learner = residual_gwr(gwr_x, gwr_coords, gwr_y)
    temp_pred = gwr_learner.predict(gwr_coords, gwr_x)
    gwr_preds[gwr_indices] = temp_pred.predy.reshape((-1, 1))

weights = divider.predict_weight(coords_test, False)
gwr_results = np.zeros((len(y_test), sections[0]*sections[1]))
for i in range(numberOfLearners):
    gwr_results[:, i] = learners[i]["gwr"].predict(coords_test, X_test).predy.reshape(-1)
gwr_results = gwr_results*weights
gwr_results = np.sum(gwr_results, axis=1).reshape((-1, 1))
gwr_time = (time.time() - start)/2

for rounds in range(rounds_count):
    start = time.time()
    rf_learner = random_forrest(X_training, coords_training, y_training-gwr_preds)
    combine_times[rounds] = time.time()-start + gwr_time
    D_test = np.concatenate((X_test, coords_test), axis=1)
    rf_results = rf_learner.predict(D_test).reshape((-1, 1))
    print(rounds, "divided gwr only", utils.R2(y_test.reshape(-1).tolist(), gwr_results), end=" ")
    resy = gwr_results + rf_results
    print("rf and gwr", utils.R2(y_test, resy))

    D_train = np.concatenate((X_training, coords_training), axis = 1)
    D_test = np.concatenate((X_test, coords_test), axis = 1)
    rg = RandomForestRegressor(n_estimators=60)
    start = time.time()
    rg.fit(D_train, y_training.reshape(-1, ))
    rf_rimes[rounds] = time.time()-start
    print("rf only", utils.R2(y_test.reshape(-1).tolist(), rg.predict(D_test)))

    rf_round[rounds] = utils.R2(y_test.reshape(-1).tolist(), rg.predict(D_test))
    combined[rounds] = utils.R2(y_test, resy)

print("mean of 10 rounds rf:", np.mean(rf_round), "with time:", np.mean(rf_rimes))
print("mean of 10 combined:", np.mean(combined), "with time:", np.mean(combine_times))
start = time.time()
bandwidth = Sel_BW(coords_training, y_training, X_training, kernel='gaussian').search(criterion='CV')
model = GWR(coords_training, y_training, X_training, bw=bandwidth, fixed=False, kernel='gaussian', spherical=True)
gwr_times = time.time() - start
results = model.predict(coords_test, X_test)
best_error = utils.R2(y_test, results.predy)
print("gwr only:", best_error, "with time", gwr_times)