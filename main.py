import os
import pickle
from src.GGWR_Model import GGWR_Model
from mgwr.gwr import GWR, MGWR
from mgwr.sel_bw import Sel_BW
import src.utils as utils
import warnings
import time
from src.hyperband import Hyperband
from src.spaceSearch import simulated_annealing
import numpy as np

dataset = "kingHousePrices"
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

# X_combined = np.concatenate((X_validation, X_test), axis = 0)
# coords_combined = np.concatenate((coords_validation, coords_test), axis = 0)
# y_combined = np.concatenate((y_validation, y_test), axis = 0)

start = time.time()
ggwr = GGWR_Model(X_training, X_validation, y_training, y_validation, coords_training, coords_validation, 3, 'gaussian', True, False)
# bw = Sel_BW(coords_training, y_training, X_training[:, 1:], kernel='gaussian').search(criterion='CV')
bw = 15
# ggwr.bandwidths = [bw for i in range(X_training.shape[1])]
ggwr.bandwidths = [14, 15, 15, 14, 16, 17, 14, 17, 14]
# ggwr.bandwidths = [9, 17, 20, 17, 43, 19]


config = {"steps": 40, "updates": 25, "method": "gaussian_all"}
# bandwidths = simulated_annealing(ggwr, [bw for i in range(X_training.shape[1])], config)
end = time.time()
print("time elapsed", end-start)
warnings.simplefilter(action='ignore', category=FutureWarning)
predictions = ggwr.predict(coords_test, X_test)
print("the GGWR_Model prediction error is equal to: ", utils.R2(y_test, predictions))

from sklearn import metrics
print("the GGWR_Model rmse is", np.sqrt(metrics.mean_squared_error(y_test, predictions)))

# predictAll = ggwr.predict(coords, x)
# with open(path + '/predictAll.data', 'wb') as filehandle:
#     pickle.dump(predictAll, filehandle)

# start_time = time.time()
# mgwr_x_training = X_training[:, 1:]
# selector = Sel_BW(coords_training, y_training, mgwr_x_training, kernel='gaussian', multi=True)
# selector.search(criterion='CV', multi_bw_min=[2])
# #
# model2 = MGWR(coords_training, y_training, mgwr_x_training, selector, fixed=False, kernel='gaussian',  sigma2_v1=True)
# results2 = model2.fit()
# print("--- MGWR runtime %s seconds ---" % (time.time() - start_time))
# print("the MGWR error is equal to: ", sum((y_training - results2.predy)**2))



# hyperband = Hyperband(ggwr)
# best_config, best_error = hyperband.compute(243, 100, ggwr.train_len, 3)
# ggwr.bandwidths = list(best_config.values())
# print("done")
# start = time.time()
#
# B = ggwr.fit(ggwr.bandwidths, "validation")
# valid_pred = utils.calculate_dependent(B, X_validation)
# error = utils.R2(y_validation, valid_pred)
# n = len(y)
# i = 0
# jump = 4
# dir = 1
# updated = 0
# shake = 0
# while updated < len(ggwr.bandwidths):
#     newBandwidth = list(ggwr.bandwidths)
#     if (newBandwidth[i] + dir * jump >= n or newBandwidth[i] + dir * jump < 3):
#         updated += 1
#         i = (i + 1) % len(ggwr.bandwidths)
#         continue
#     newBandwidth[i] += dir * jump
#
#     B = ggwr.fit(newBandwidth, "validation")
#     valid_pred = utils.calculate_dependent(B, X_validation)
#     tempError = utils.R2(y_validation, valid_pred)
#
#     print(ggwr.bandwidths, newBandwidth, tempError, i, error, jump, dir)
#     if tempError < error:
#         error = tempError
#         ggwr.bandwidths = newBandwidth
#         jump = jump + 1
#         updated = 0
#         shake = 0
#     else:
#         if (shake == 0 and jump > 3):
#             jump = max(round(jump / 2), 1)
#         elif (shake < 3):
#             shake = shake + 1
#             jump = 4 + shake * 2
#         elif (dir == 1):
#             dir = -1
#             jump = 4
#             shake = 0
#         else:
#             dir = 1
#             i = (i + 1) % len(ggwr.bandwidths)
#             updated = updated + 1
#
# end = time.time()
# print("time elapsed", end-start)
#
# B = ggwr.fit(ggwr.bandwidths, "training")
# train_pred = utils.calculate_dependent(B, X_training)
# print("GGWR_Model train error is", utils.R2(y_training, train_pred))
#
# warnings.simplefilter(action='ignore', category=FutureWarning)
# predictions = ggwr.predict(coords_test, X_test)
# print("the GGWR_Model prediction error is equal to: ", utils.R2(y_test, predictions))