import sys
import warnings
import libpysal as ps
import numpy as np
import pickle
from mgwr.gwr import GWR, MGWR
from mgwr.sel_bw import Sel_BW
import geopandas as gp
# from scgwr import scgwr
import pandas as pd
import os
import time
import src.utils as utils
from sklearn import metrics
import multiprocessing
from mgwr.utils import compare_surfaces, truncate_colormap
from sklearn.linear_model import LinearRegression
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel


def gwr_test(dataset, path, config={}):
    default_config = {"process_count": 1}
    for key in config.keys():
        default_config[key] = config[key]
    config = default_config

    res = {}
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
    X_training, X_validation, X_test = x[training_idx, :], x[validation_idx], x[test_idx, :]
    y_training, y_validation, y_test = y[training_idx, :], y[validation_idx], y[test_idx, :]
    coords_training, coords_validation, coords_test = coords[training_idx], coords[validation_idx], coords[test_idx]

    # krnl = Kernel(coords_training, coords_test, fixed=False, function='gaussian', eps=1.0000001, spherical=False)


    # warnings.simplefilter(action='ignore', category=FutureWarning)
    # modelprez = GWR(coords_training, y_training, X_training, bw=10, fixed=False, kernel='gaussian')
    # wi = GWR.build_wi(modelprez, 0, 10).reshape(-1, 1)
    # from spglm.iwls import iwls, _compute_betas_gwr
    # try:
    #     betas, inv_xtx_xt = _compute_betas_gwr(y_training, X_training, wi)
    # except:
    #     print("here")

    pool_of_process = None
    if config["process_count"] > 1:
        pool_of_process = multiprocessing.Pool(config["process_count"])
    start_time = time.time()
    bandwidth = Sel_BW(coords_training, y_training, X_training, kernel='gaussian').search(criterion='CV', pool=pool_of_process)
    model = GWR(coords_training, y_training, X_training, bw=bandwidth, fixed=False, kernel='gaussian', spherical=True)
    # results = model.predict(coords_test, X_test)
    # print("the GWR prediction error is equal to: ", utils.R2(y_test, results.predy))
    # from sklearn import metrics
    # print("the gwr rmse is", np.sqrt(metrics.mean_squared_error(y_test, results.predy)))

    results = model.predict(coords_validation, X_validation)
    best_error = utils.R2(y_validation, results.predy)
    best_res = bandwidth
    # for i in range(10, 70, 2):
    #     model = GWR(coords_training, y_training, X_training, bw=i, fixed=False, kernel='gaussian', spherical=True)
    #     results = model.predict(coords_validation, X_validation)
    #     error = utils.R2(y_validation, results.predy)
    #     if error < best_error:
    #         best_error = error
    #         best_res = i

    # print("bandwidth equals", best_res)
    res["bandwidths"] = best_res
    res["time"] = time.time()-start_time
    model = GWR(coords_training, y_training, X_training, bw=best_res, fixed=False, kernel='gaussian', spherical=True)
    results = model.predict(coords_training, X_training)
    res["train_R2"] = utils.R2(y_training, results.predy)
    res["train_rmse"] = np.sqrt(metrics.mean_squared_error(y_training, results.predy))

    model = GWR(coords_training, y_training, X_training, bw=best_res, fixed=False, kernel='gaussian', spherical=True)
    results = model.predict(coords_validation, X_validation)
    res["validation_R2"] = utils.R2(y_validation, results.predy)
    res["validation_rmse"] = np.sqrt(metrics.mean_squared_error(y_validation, results.predy))

    model = GWR(coords_training, y_training, X_training, bw=best_res, fixed=False, kernel='gaussian', spherical=True)
    results = model.predict(coords_test, X_test)
    res["test_R2"] = utils.R2(y_test, results.predy)
    res["test_rmse"] = metrics.mean_squared_error(y_test, results.predy)
    if pool_of_process is not None:
        pool_of_process.close()
    # print("the GWR prediction error is equal to: ", utils.R2(y_test, results.predy))
    # print("the gwr rmse is", np.sqrt(metrics.mean_squared_error(y_test, results.predy)))

    print(utils.RMSE(y_test.reshape(1, -1), results.predy.reshape(1, -1)))

    return res


def main():
    # dataset = "pysalGeorgia" #pysalBerlin #pysalBaltimore
    dataset = "kingHousePrices"
    path = os.path.dirname(os.path.abspath(__file__ + str("/../../"))) + "/Data/" + dataset + "/"
    result = gwr_test(dataset, path, {"process_count": 1})

    print("the GWR train time is: ", result["time"])
    print("the GWR bandwidths: ", result["bandwidths"])
    print("the GWR train r2 error is: ", result["train_R2"])
    print("the GWR train rmse is: ", result["train_rmse"])
    print("the GWR validation r2 error is: ", result["validation_R2"])
    print("the GWR validation rmse is: ", result["validation_rmse"])
    print("the GWR prediction error is: ", result["test_R2"])
    print("the GWR rmse is: ", result["test_rmse"])


if __name__ == "__main__":
    main()
