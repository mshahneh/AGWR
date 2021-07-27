import sys
# sys.path.insert(0,'..')
import warnings
import libpysal as ps
import numpy as np
import pickle
import os
from mgwr.gwr import GWR, MGWR
from mgwr.sel_bw import Sel_BW
from sklearn import metrics
import geopandas as gp
# from scgwr import scgwr
import pandas as pd
import time
import src.utils as utils
from src.BackfittingModelImproved import ImprovedBackfitting
import multiprocessing
from src.GGWR_Model import GGWR_Model
from mgwr.utils import compare_surfaces, truncate_colormap
from sklearn.linear_model import LinearRegression
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel


def mgwr_test(dataset, path, config={}):
    default_config = {"process_count": 2}
    for key in config.keys():
        default_config[key] = config[key]
    config = default_config

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
    X_combined = np.concatenate((X_validation, X_training), axis=0)
    coords_combined = np.concatenate((coords_validation, coords_training), axis=0)
    y_combined = np.concatenate((y_validation, y_training), axis=0)

    x = X_combined
    y = y_combined
    coords = coords_combined

    # krnl = Kernel(coords_training, coords_test, fixed=False, function='gaussian', eps=1.0000001, spherical=False)

    # print(X_training.shape, X_validation.shape, X_test.shape)
    # print(y_training.shape, y_validation.shape, y_test.shape)
    # print(X_training[1, :])

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
    selector = Sel_BW(coords, y, x, kernel='gaussian', multi=True)
    selector.search(criterion='CV', multi_bw_min=[2], pool=pool_of_process)
    model = MGWR(coords, y, x, selector, sigma2_v1=True, kernel='gaussian', fixed=False, spherical=True)

    res = {}
    res["time"] = time.time()-start_time
    # print("train time = ", time.time()-start_time)

    bandwidths = model.bws
    res["bandwidths"] = bandwidths
    # print(bandwidths)
    temp = model.fit(pool=pool_of_process)
    if pool_of_process is not None:
        pool_of_process.close()

    predictions = temp.predy
    res["train_R2"] = utils.R2(y, predictions)
    # print("the MGWR train r2 error is equal to: ", utils.R2(y_training, predictions))
    res["train_rmse"] = np.sqrt(
        metrics.mean_squared_error(y, predictions))

    mgwr_pred_model = ImprovedBackfitting(x, y, coords, {"enable_cache": False})
    mgwr_pred_model.setB(temp.params)
    mgwr_pred_model.bandwidths = model.bws
    predictions = mgwr_pred_model.predict(coords_test, X_test)

    res["test_R2"] = utils.R2(y_test, predictions)
    # print("the MGWR prediction error is equal to: ", utils.R2(y_test, predictions))
    res["test_rmse"] = np.sqrt(metrics.mean_squared_error(y_test, predictions))
    # print("the MGWR rmse is", np.sqrt(metrics.mean_squared_error(y_test, predictions)))

    return res


def main():
    dataset = "pysalBerlin"
    path = os.path.dirname(os.path.abspath(
        __file__ + str("/../../"))) + "/Data/" + dataset + "/"
    warnings.simplefilter(action='ignore', category=FutureWarning)
    result = mgwr_test(dataset, path,  {"process_count": 4})
    print("the MGWR train time is: ", result["time"])
    print("the MGWR bandwidths: ", result["bandwidths"])
    print("the MGWR train r2 error is: ", result["train_R2"])
    print("the MGWR train rmse is: ", result["train_rmse"])
    print("the MGWR prediction error is: ", result["test_R2"])
    print("the MGWR rmse is: ", result["test_rmse"])


if __name__ == "__main__":
    main()
