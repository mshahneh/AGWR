import os
import sys
import pickle

from src.BackfittingModelImproved import ImprovedBackfitting
from src.GGWR_Model import GGWR_Model
from mgwr.gwr import GWR, MGWR
from mgwr.sel_bw import Sel_BW
import src.utils as utils
import warnings
import time
from src.hyperband import Hyperband
import numpy as np
from sklearn import metrics

from src.spaceSearch import SpaceSearch


def  ggwr_test(dataset, path, config={"space_search_method": "SPSA"}, budget = 1):
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

    _x = X_combined
    _y = y_combined
    _coords = coords_combined

    start = time.time()

    ggwr_model = ImprovedBackfitting(_x, _y, _coords)
    bandwidths = ggwr_model.bandwidths
    SP = SpaceSearch(ggwr_model, 1)

    if config["space_search_method"] == "simulated_annealing":
        sim_config = {"steps": budget*180, "updates": budget*125, "method": "gaussian_all"}
        bandwidths = SP.simulated_annealing(bandwidths, sim_config)
        ggwr_model.bandwidths = bandwidths

    elif config["space_search_method"] == "advanced_combined":
        bandwidths = SP.bayesian_optimization(bandwidths, {"random_count": budget*20, "iter_count": budget*30})
        bandwidths = SP.bayesian_optimization(bandwidths, {"is_local": True, "locality_range": 20,
                                                           "random_count": budget*20, "iter_count": budget*30})
        sim_config = {"steps": budget*35, "updates": budget*15, "method": "gaussian_one"}
        bandwidths = SP.hill_climbing(bandwidths, sim_config)
        ggwr_model.bandwidths = bandwidths

    elif config["space_search_method"] == "bayesian_optimization":
        bandwidths = SP.bayesian_optimization(bandwidths, {"random_count": budget*60, "iter_count": budget*66})
        ggwr_model.bandwidths = bandwidths

    
    elif config["space_search_method"] == "successive_halving":
        max_budget = len(_y)
        bandwidths, _ = SP.successive_halving(budget*648, 27, max_budget, 3)
        bandwidths = list(bandwidths.values())
        ggwr_model.bandwidths = bandwidths

    elif config["space_search_method"] == "bayesian_hill":
        bandwidths = SP.bayesian_optimization(bandwidths, {"random_count": budget*45, "iter_count": budget*55})
        sim_config = {"steps": budget*40, "updates": budget*24, "method": "gaussian_one"}
        bandwidths = SP.hill_climbing(bandwidths, sim_config)
        ggwr_model.bandwidths = bandwidths

    elif config["space_search_method"] == "hyperband":
        ggwr_model.bandwidths = SP.hyperband(budget*243, 64, 3)

    elif config["space_search_method"] == "hill_climbing":
        sim_config = {"steps": budget*185, "updates": budget*110, "method": "gaussian_one"}
        bandwidths = SP.hill_climbing(bandwidths, sim_config)
        ggwr_model.bandwidths = bandwidths

    elif config["space_search_method"] == "combined":
        sim_config = {"steps": budget*9}
        bandwidths = SP.SPSA(bandwidths, sim_config)
        print('done SPSA')
        bandwidths = SP.bayesian_optimization(bandwidths, {"is_local": True, "locality_range": 10,"random_count": budget*15, "iter_count": budget*20})
        print('done basian')
        sim_config = {"steps": budget*55, "updates": budget*40, "method": "gaussian_one"}
        bandwidths = SP.hill_climbing(bandwidths, sim_config)
        ggwr_model.bandwidths = bandwidths

    elif config["space_search_method"] == "golden_search":
        sim_config = {"steps": budget*220}
        ggwr_model.bandwidths = SP.golden_section(bandwidths, sim_config)

    elif config["space_search_method"] == "fibonacci_all":
        sim_config = {"steps": budget*270, "updates": budget*210, "method": "fibonacci_all"}
        bandwidths = SP.simulated_annealing(bandwidths, sim_config)
        ggwr_model.bandwidths = bandwidths

    elif config["space_search_method"] == "fibonacci_same_all":
        sim_config = {"steps": 7, "updates": 4, "method": "fibonacci_same_all"}
        bandwidths = SP.simulated_annealing(bandwidths, sim_config)
        ggwr_model.bandwidths = bandwidths
        
    elif config["space_search_method"] == "fibonacci_one":
        sim_config = {"steps": budget*175, "updates": budget*105, "method": "fibonacci_one"}
        bandwidths = SP.hill_climbing(bandwidths, sim_config)
        ggwr_model.bandwidths = bandwidths

    elif config["space_search_method"] == "SPSA":
        sim_config = {"steps": budget*60}
        bandwidths = SP.SPSA(bandwidths, sim_config)
        ggwr_model.bandwidths = bandwidths
    elif config["space_search_method"] == "SPSAgs":
        sim_config = {"steps": budget*20}
        bandwidths = SP.SPSA(bandwidths, sim_config)
        sim_config = {"steps": budget*100}
        ggwr_model.bandwidths = SP.golden_section(bandwidths, sim_config)
        ggwr_model.bandwidths = bandwidths
    elif config["space_search_method"] == "FDSA":
        sim_config = {"steps": budget*25}
        bandwidths = SP.FDSA(bandwidths, sim_config)
        bandwidths = SP.bayesian_optimization(bandwidths, {"is_local": True, "locality_range": 10,
                                                           "random_count": budget*15, "iter_count": budget*25})
        sim_config = {"steps": budget*40, "updates": budget*25, "method": "gaussian_one"}
        bandwidths = SP.hill_climbing(bandwidths, sim_config)
        ggwr_model.bandwidths = bandwidths
    else:
        print("Bad space search method :<")
        return None
    ggwr_model.fit(ggwr_model.bandwidths, iterations=10)
    end = time.time()
    res = {"time": end-start}
    # print("time elapsed", end-start)
    warnings.simplefilter(action='ignore', category=FutureWarning)
    res["bandwidths"] = ggwr_model.bandwidths

    predictions = ggwr_model.predict(coords_test, X_test)
    res["test_R2"] = utils.R2(y_test, predictions)
    res["test_rmse"] = np.sqrt(metrics.mean_squared_error(y_test, predictions))

    return res


def main():
    dataset = "syntheticData1"
    path = os.path.dirname(os.path.abspath(__file__ + str("/../../"))) + "/Data/" + dataset + "/"
    result = ggwr_test(dataset, path, config={"space_search_method": "SPSA"})
    print("the GGWR_Model train time is: ", result["time"])
    print("the GGWR_Model bandwidths: ", result["bandwidths"])
    # print("the GGWR_Model train r2 error is: ", result["train_R2"])
    # print("the GGWR_Model train rmse is: ", result["train_rmse"])
    # print("the GGWR_Model validation r2 error is: ", result["validation_R2"])
    # print("the GGWR_Model validation rmse is: ", result["validation_rmse"])
    print("the GGWR_Model test r2 error is: ", result["test_R2"])
    print("the GGWR_Model test rmse is: ", result["test_rmse"])


if __name__ == "__main__":
    main()
