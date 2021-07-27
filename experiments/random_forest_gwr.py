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

for rounds in range(rounds_count):
    start = time.time()
    for i in range(numberOfLearners):
        print(i, end=" ")
        rf_indices = []
        for j in range(len(indices)):
            if j != i:
                rf_indices.extend(indices[j])
        gwr_indices = indices[i]

        rf_x = X_training[rf_indices]
        rf_y = y_training[rf_indices]
        rf_coords = coords_training[rf_indices]
        rf_learner = random_forrest(rf_x, rf_coords, rf_y)

        gwr_x = X_training[gwr_indices]
        gwr_coords = coords_training[gwr_indices]
        D_test = np.concatenate((gwr_x, gwr_coords), axis=1)
        RF_pred = rf_learner.predict(D_test).reshape((-1, 1))
        gwr_y = y_training[gwr_indices] - RF_pred
        gwr_learner = residual_gwr(gwr_x, gwr_coords, gwr_y)
        learners[i] = {"rf":rf_learner, "gwr":gwr_learner}

    combine_times[rounds] = time.time()-start

    resys = [1 for _ in range(numberOfLearners)]

    # results = np.zeros((len(y_test), sections[0] * sections[1]))
    # for i in range(sections[0] * sections[1]):
    #     results[:, i] = models[i].predict(coords_test, X_test).predy.reshape(-1)

    weights = divider.predict_weight(coords_test, False)
    # results = results * weights

    for i in range(numberOfLearners):
        D_test = np.concatenate((X_test, coords_test), axis=1)
        gwr_results = learners[i]["gwr"].predict(coords_test, X_test)
        rf_results = learners[i]["rf"].predict(D_test).reshape((-1, 1))
        print(i, "rf only", utils.R2(y_test.reshape(-1).tolist(), rf_results), end=" ")
        resy = gwr_results.predy.reshape((-1, 1))*weights[:,i].reshape((-1, 1)) + rf_results
        resys[i] = resy
        print("rf and gwr", utils.R2(y_test, resy))

    temp = np.mean(resys, axis=0)
    print("mean", utils.R2(y_test, temp))
    D_train = np.concatenate((X_training, coords_training), axis = 1)
    D_test = np.concatenate((X_test, coords_test), axis = 1)
    rg = RandomForestRegressor(n_estimators=60)
    start = time.time()
    rg.fit(D_train, y_training.reshape(-1, ))
    rf_rimes[rounds] = time.time()-start
    print("rf only", utils.R2(y_test.reshape(-1).tolist(), rg.predict(D_test)))

    rf_round[rounds] = utils.R2(y_test.reshape(-1).tolist(), rg.predict(D_test))
    combined[rounds] = utils.R2(y_test, temp)

print("mean of 10 rounds rf:", np.mean(rf_round), "with time:", np.mean(rf_rimes))
print("mean of 10 combined:", np.mean(combined), "with time:", np.mean(combine_times))
start = time.time()
bandwidth = Sel_BW(coords_training, y_training, X_training, kernel='gaussian').search(criterion='CV')
model = GWR(coords_training, y_training, X_training, bw=bandwidth, fixed=False, kernel='gaussian', spherical=True)
gwr_times = time.time() - start
results = model.predict(coords_test, X_test)
best_error = utils.R2(y_test, results.predy)
print("gwr only:", best_error, "with time", gwr_times)