import numpy as np
import pickle
from mgwr.gwr import GWR, MGWR
from mgwr.sel_bw import Sel_BW
import time
import os
import src.utils as utils
from src.DataDivider import Divider
from sklearn import metrics

dataset = "kingHousePrices"  # pysalBerlin #pysalBaltimore
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

# warnings.filterwarnings("ignore")
x = x[:, 1:]
X_training, X_validation, X_test = x[training_idx, :], x[validation_idx], x[test_idx, :]
y_training, y_validation, y_test = y[training_idx, :], y[validation_idx], y[test_idx, :]
coords_training, coords_validation, coords_test = coords[training_idx], coords[validation_idx], coords[test_idx]

# start_time = time.time()
# bandwidth = Sel_BW(coords_training, y_training, X_training, kernel='gaussian').search(criterion='CV')
# model = GWR(coords_training, y_training, X_training, bw=bandwidth, fixed=False, kernel='gaussian', spherical=True)
# results = model.predict(coords_test, X_test)
# print("naive gwr predict R2 error:" , utils.R2(y_test, results.predy), "time:", time.time()-start_time)

# sections = 6
# start_time = time.time()
# indices = random_divide(coords_training, sections)
# models = []
# for i in range(sections):
#     bandwidth = Sel_BW(coords_training[indices[i]], y_training[indices[i]], X_training[indices[i]], kernel='gaussian').search(criterion='CV')
#     models.append(GWR(coords_training[indices[i]], y_training[indices[i]], X_training[indices[i]], bw=bandwidth, fixed=False, kernel='gaussian', spherical=True))
#
# results = np.zeros((len(y_test), sections))
# for i in range(sections):
#     results[:, i] = models[i].predict(coords_test, X_test).predy.reshape(-1)
#
# print("random with 6 gwr predict R2 error:", utils.R2(y_test, np.mean(results, axis=1)), "time", time.time()-start_time)
#
#
# def is_inside(coords, boundaries):
#     if coords[0] < boundaries[0][0] or coords[0] >= boundaries[1][0]:
#         return False
#     if coords[1] < boundaries[0][1] or coords[1] >= boundaries[1][1]:
#         return False
#     return True
#
#
# sections = [2, 3]
# start_time = time.time()
# indices = grid_divide(coords_training, sections[0], sections[1], "equalCount")
#
#
# models = []
# for i in range(sections[0]):
#     for j in range(sections[1]):
#         ind = indices[i][j]["indices"]
#         bandwidth = Sel_BW(coords_training[ind], y_training[ind], X_training[ind], kernel='gaussian').search(criterion='CV')
#         models.append(GWR(coords_training[ind], y_training[ind], X_training[ind], bw=bandwidth, fixed=False, kernel='gaussian', spherical=True))
#
# results = np.ones(len(y_test))
# tags = [[] for _ in range(sections[0]*sections[1])]
# for i in range(len(y_test)):
#     for j in range(sections[0]):
#         for k in range(sections[1]):
#             if is_inside(coords_test[i], indices[j][k]["boundaries"]):
#                 tags[j*3+k].append(i)
#
# for i in range(sections[0]*sections[1]):
#     results[tags[i]] = models[i].predict(coords_test[tags[i], :], X_test[tags[i], :]).predy.reshape(-1)
#
# print("equal size grid 2x3 gwr predict R2 error:", utils.R2(y_test, results), "time", time.time()-start_time)
#
# start_time = time.time()
# models = []
# for i in range(sections[0]):
#     for j in range(sections[1]):
#         ind = indices[i][j]["indices"]
#         bandwidth = Sel_BW(coords_training[ind], y_training[ind], X_training[ind], kernel='gaussian').search(criterion='CV')
#         models.append(GWR(coords_training[ind], y_training[ind], X_training[ind], bw=bandwidth, fixed=False, kernel='gaussian', spherical=True))
# results2 = np.zeros(len(y_test))
# weights = np.zeros((len(y_test), sections[0]*sections[1]))
# centers = []
# for j in range(sections[0]):
#     for k in range(sections[1]):
#         centers.append(indices[j][k]["center"])
#
# for i in range(len(y_test)):
#     weights[i, :] = weighted_mean(centers, coords_test[i], spherical=True)
#
# for j in range(sections[0]*sections[1]):
#     temp = models[j].predict(coords_test, X_test).predy.reshape(-1)
#     results2[:] += weights[:, j]*temp
# print("equal size grid 2x3 gwr predict with weighted mean R2 error:", utils.R2(y_test, results2), "time", time.time()-start_time)



print("and now the new system")
sections = [2, 2]
divider = Divider()
# indices = divider.grid_divide(coords_training, sections[0], sections[1], "equalCount")
indices = divider.kmeans_divide(coords_training, sections[0]*sections[1])
models = []
for i in range(sections[0]*sections[1]):
    bandwidth = Sel_BW(coords_training[indices[i]], y_training[indices[i]], X_training[indices[i]], kernel='gaussian').search(criterion='CV')
    models.append(GWR(coords_training[indices[i]], y_training[indices[i]], X_training[indices[i]], bw=bandwidth, fixed=False, kernel='gaussian', spherical=True))
results = np.zeros((len(y_test), sections[0]*sections[1]))
for i in range(sections[0]*sections[1]):
    results[:, i] = models[i].predict(coords_test, X_test).predy.reshape(-1)

weights = divider.predict_weight(coords_test, False)
results = results*weights

print("random with 6 gwr predict R2 error:", utils.R2(y_test, np.sum(results, axis=1)))
