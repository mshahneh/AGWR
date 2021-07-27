from src.BackfittingModelImproved import ImprovedBackfitting
import warnings
import pickle
import src.utils as utils
import time
from prettytable import PrettyTable
from src.spaceSearch import SpaceSearch
import random
import numpy as np
np.random.seed(42)
random.seed(42)
import os

# data_sets = ["pysalGeorgia", "kingHousePrices", "artificialData"]
data_sets = ["syntheticData1", "syntheticData2"]
warnings.simplefilter(action='ignore', category=FutureWarning)
res = {}
results_location = os.path.dirname(os.path.abspath(__file__)) + "/Results/Cache_experiments.data"
for dataset in data_sets:
    res[dataset] = {}
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
    x = X_combined
    y = y_combined
    coords = coords_combined

    print("\n\n", dataset)
    cutoff = [-1, 1, 3, 5, 7]
    for j in range(len(cutoff)):
        print(cutoff[j])
        res[dataset][str(cutoff[j])] = {}
        start = time.time()
        config = {"cutoff": cutoff[j]}
        ggwr_model = ImprovedBackfitting(x, y, coords)
        SP = SpaceSearch(ggwr_model)

        start_time = time.time()
        bandwidths = ggwr_model.bandwidths
        bandwidths = SP.thorough_search(bandwidths, {})
        sim_config = {"steps": 20, "updates": 12, "method": "gaussian_all"}
        bandwidths = SP.simulated_annealing(bandwidths, sim_config)
        sim_config = {"steps": 80, "updates": 50, "method": "gaussian_one"}
        bandwidths = SP.hill_climbing(bandwidths, sim_config)
        ggwr_model.bandwidths = bandwidths
        ggwr_model.fit(bandwidths)
        B = ggwr_model.trainB

        end = time.time()
        res[dataset][str(cutoff[j])]["time"] = end-start

        predictions = ggwr_model.predict(coords_test, X_test)
        res[dataset][str(cutoff[j])]["r2"] = utils.R2(y_test, predictions)

    with open(results_location, 'wb') as filehandle:
        pickle.dump(res, filehandle)

    t = PrettyTable(["cutoff", "time", "mean R2 error"])
    for i in cutoff:
        t.add_row([i, res[dataset][str(i)]["time"], res[dataset][str(i)]["r2"]])

    print(dataset)
    print(t)
