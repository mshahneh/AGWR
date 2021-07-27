import random
import os
import pickle
from src.BackfittingModelImproved import ImprovedBackfitting
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
import multiprocessing

warnings.filterwarnings('ignore')

np.random.seed(42)
random.seed(42)


def random_forrest(_x, _coords, _y, process_count=-1):
    D_train = np.concatenate((_x, _coords), axis=1)
    rg = RandomForestRegressor(n_estimators=60, n_jobs=process_count)
    rg.fit(D_train, _y.reshape(-1, ))
    return rg


def residual_gwr(_x, _coords, _y, process_count=-1):
    if process_count > 1:
        pool_of_process = multiprocessing.Pool(process_count)
    else:
        pool_of_process = None
    bandwidth = Sel_BW(
        _coords, _y, _x, kernel='gaussian').search(criterion='CV', pool=pool_of_process)
    model = GWR(_coords, _y, _x, bw=bandwidth, fixed=False,
                kernel='gaussian', spherical=True)
    return model


def backfitting_model(_x, _coords, _y, learned_bandwidths=[], process_count=-1):
    ggwr_model = ImprovedBackfitting(_x, _y, _coords)
    if _x.shape[0] > 500:
        SP = SpaceSearch(ggwr_model, 1)
    else:
        SP = SpaceSearch(ggwr_model, 5)
    if len(learned_bandwidths) < 1 or len(learned_bandwidths) > ggwr_model.numberOfFeatures:
        bandwidths = ggwr_model.bandwidths
        # sim_config = {"steps": 30, "updates": 20, "method": "gaussian_same_all"}
        # bandwidths = SP.simulated_annealing(bandwidths, sim_config)
        # if len(bandwidths) <= 6:
        #     bandwidths = SP.thorough_search(bandwidths, {})
        # else:
        #     bandwidths = SP.bayesian_optimization(bandwidths, {"random_count": 50, "iter_count": 50})
        if _x.shape[0] > 500:
            bandwidths, _ = SP.successive_halving(256, 64, -1, 4)
            bandwidths = list(bandwidths.values())
        sim_config = {"steps": 10}
        bandwidths = SP.SPSA(bandwidths, sim_config)
        bandwidths = SP.bayesian_optimization(bandwidths, {"is_local": True, "locality_range": 20,
                                                           "random_count": 25, "iter_count": 40})
        print("done global")
        sim_config = {"steps": 80, "updates": 50, "method": "gaussian_one"}
        bandwidths = SP.hill_climbing(bandwidths, sim_config)
        ggwr_model.bandwidths = bandwidths
    else:
        bandwidths = learned_bandwidths
        bandwidths = SP.bayesian_optimization(bandwidths, {"is_local": True, "locality_range": 30,
                                                           "random_count": 35, "iter_count": 50})
        sim_config = {"steps": 55, "updates": 35, "method": "gaussian_one"}
        bandwidths = SP.hill_climbing(bandwidths, sim_config)
        ggwr_model.bandwidths = bandwidths
    ggwr_model.fit(ggwr_model.bandwidths, iterations=10)
    return ggwr_model


def MGWR_model(_x, _coords, _y, learned_bandwidths=[], process_count=-1):
    if process_count > 1:
        pool_of_process = multiprocessing.Pool(process_count)
    else:
        pool_of_process = None
    selector = Sel_BW(_coords, _y, _x, kernel='gaussian', multi=True)
    selector.search(criterion='CV', multi_bw_min=[2], pool=pool_of_process)
    model = MGWR(_coords, _y, _x, selector, sigma2_v1=True, kernel='gaussian', fixed=False, spherical=True)

    ggwr_model = ImprovedBackfitting(_x, _y, _coords)
    ggwr_model.bandwidths = model.bws
    ggwr_model.fit(ggwr_model.bandwidths, iterations=10)
    return ggwr_model


def modular_test(dataset, path, config={}):
    res = {}
    default_config = {"sections": [4, 5], "is_pipeline": True, "spatial_model": "GGWR",
                      "divide_method": "equalCount", "process_count": 17, "overlap":0}
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

    X_training, X_validation, X_test = x[training_idx, :], x[validation_idx], x[test_idx, :]
    y_training, y_validation, y_test = y[training_idx, :], y[validation_idx], y[test_idx, :]
    coords_training, coords_validation, coords_test = coords[training_idx], coords[validation_idx], coords[test_idx]

    # print(X_training.shape, X_validation.shape, X_test.shape)
    # print(y_training.shape, y_validation.shape, y_test.shape)
    # print(coords_training.shape, coords_validation.shape, coords_test.shape)

    X_combined = np.concatenate((X_validation, X_training), axis=0)
    coords_combined = np.concatenate((coords_validation, coords_training), axis=0)
    y_combined = np.concatenate((y_validation, y_training), axis=0)

    print("pipelined:", config["is_pipeline"])
    print("sections:", config["sections"])

    if config["spatial_model"] == "GWR":
        ggwr = GGWR(residual_gwr, random_forrest,
                    {"process_count": config["process_count"], "divide_method": config["divide_method"], "overlap":config["overlap"],
                     "divide_sections": config["sections"],
                     "pipelined": config["is_pipeline"]})
    elif config["spatial_model"] == "GGWR":
        ggwr = GGWR(backfitting_model, random_forrest,
                    {"process_count": config["process_count"], "divide_method": config["divide_method"], "overlap":config["overlap"],
                     "divide_sections": config["sections"],
                     "pipelined": config["is_pipeline"]})
    else:
        ggwr = GGWR(MGWR_model, random_forrest,
                    {"process_count": config["process_count"], "divide_method": config["divide_method"], "overlap":config["overlap"],
                     "divide_sections": config["sections"],
                     "pipelined": config["is_pipeline"]})
    ggwr.train(X_combined, coords_combined, y_combined)
    pred = ggwr.predict(X_test, coords_test, y_test)
    test_r2 = utils.R2(y_test.reshape(-1).tolist(), pred)

    res["time"] = ggwr.train_time
    res["test_R2"] = test_r2
    res["test_rmse"] = utils.RMSE(y_test.reshape(-1).tolist(), pred)

    return res


def main():
    # dataset = "pysalGeorgia" #pysalBerlin #pysalBaltimore artificialData
    dataset = "kingHousePrices"
    # dataset = "pysalTokyo"
    path = os.path.dirname(os.path.abspath(
        __file__ + str("/../../"))) + "/Data/" + dataset + "/"
    result = modular_test(dataset, path)

    print("the modular train time is: ", result["time"])
    # print("the GWR bandwidths: ", result["bandwidths"])
    print("the modular prediction error is: ", result["test_R2"])
    print("the modular rmse is: ", result["test_rmse"])


if __name__ == "__main__":
    main()
