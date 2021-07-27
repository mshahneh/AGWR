import pickle

from src.Backfitting_Model import Backfitting
from src.BackfittingModelImproved import ImprovedBackfitting
from src.DataHandlers.syntheticData import synthetic_data_generator
from src.GGWR_Model import GGWR_Model
from mgwr.gwr import GWR, MGWR
from mgwr.sel_bw import Sel_BW
from src.Cache import Cache
import src.utils as utils
import warnings
import time
import os
import random
import numpy as np
np.random.seed(42)
random.seed(42)

from src.experiments.display_results import display_results
from src.hyperband import Hyperband
# from src.spaceSearch import simulated_annealing, hill_climbing
import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm

from src.spaceSearch import SpaceSearch
from src.utils import local_dist, kernel_funcs

def backfitting_model(_x, _coords, _y):
    ggwr_model = ImprovedBackfitting(_x, _y, _coords)

    SP = SpaceSearch(ggwr_model)
    start_time = time.time()
    bandwidths = ggwr_model.bandwidths
    sim_config = {"steps": 15}
    bandwidths = SP.SPSA(bandwidths, sim_config)
    sim_config = {"steps": 75, "updates": 45, "method": "gaussian_one"}
    bandwidths = SP.hill_climbing(bandwidths, sim_config)
    ggwr_model.bandwidths = bandwidths

    # ggwr_model.bandwidths = bandwidths = [878., 862.,  11.,   3. ,  9. ,  3.]
    # ggwr_model.bandwidths = bandwidths = [44., 52.,  7.,  3., 20.,  6.]
    # ggwr_model.fit(bandwidths)
    return ggwr_model

def golden_search_model(_x, _coords, _y):
    ggwr_model = ImprovedBackfitting(_x, _y, _coords)

    SP = SpaceSearch(ggwr_model)
    start_time = time.time()
    config = {"steps": 300}
    bandwidths = SP.golden_section(ggwr_model.bandwidths, config)
    ggwr_model.bandwidths = bandwidths

    # ggwr_model.bandwidths = bandwidths = [878., 862.,  11.,   3. ,  9. ,  3.]
    # ggwr_model.bandwidths = bandwidths = [44., 52.,  7.,  3., 20.,  6.]
    # ggwr_model.fit(bandwidths)
    return ggwr_model


categories = ["train_r2", "train_time", "coef_rmse", "test_r2"]
algorithms = ["GWR", "MGWR", "GGWR", "GGWR_bw", "golden_ggwr"]
results = {}
for algorithm in algorithms:
    results[algorithm] = {}
    for categorie in categories:
        results[algorithm][categorie] = []


for it in range(10):
    print("\n iteration:", it)
    synthetic_data_generator(27, 5)
    dataset = "syntheticData1"
    path = os.path.dirname(os.path.abspath(__file__ + str("/../../"))) + "/Data/" + dataset + "/"
    warnings.simplefilter(action='ignore', category=FutureWarning)
    with open(path + 'x.data', 'rb') as filehandle:
        x = pickle.load(filehandle)
    with open(path + 'y.data', 'rb') as filehandle:
        y = pickle.load(filehandle)
    with open(path + 'coords.data', 'rb') as filehandle:
        coords = pickle.load(filehandle)
    with open(path + 'training_idx.data', 'rb') as filehandle:
        training_idx = pickle.load(filehandle)
    with open(path + 'validation_idx.data', 'rb') as filehandle:
        validation_idx = pickle.load(filehandle)
    with open(path + 'test_idx.data', 'rb') as filehandle:
        test_idx = pickle.load(filehandle)
    with open(path + 'coefficients.data', 'rb') as filehandle:
        coefficients = pickle.load(filehandle)


    X_training, X_validation, X_test = x[training_idx, :], x[validation_idx], x[test_idx, :]
    y_training, y_validation, y_test = y[training_idx, :], y[validation_idx], y[test_idx, :]
    coords_training, coords_validation, coords_test = coords[training_idx], coords[validation_idx], coords[test_idx]
    X_combined = np.concatenate((X_validation, X_training), axis=0)
    print(np.max(X_combined, axis=0))

    coords_combined = np.concatenate((coords_validation, coords_training), axis=0)
    y_combined = np.concatenate((y_validation, y_training), axis=0)
    coefficients = np.asarray(coefficients)
    coefficients_b_validation = coefficients[validation_idx, :]
    coefficients_b_training = coefficients[training_idx, :]
    coefficients_b = np.concatenate((coefficients_b_validation, coefficients_b_training), axis=0)

    x = X_combined
    y = y_combined
    coords = coords_combined


    #GWR
    start_time = time.time()
    GWRbandwidth = Sel_BW(coords, y, x, kernel='gaussian').search(criterion='CV')
    results["GWR"]["train_time"].append(time.time()-start_time)
    gwrModel = GWR(coords, y, x, bw=GWRbandwidth, fixed=False, kernel='gaussian', spherical=True)
    gwrModelFit = gwrModel.fit()

    gwr_pred = gwrModel.predict(coords_test, X_test).predy
    gwr_b = gwrModelFit.params
    gwr_b = gwr_b[:, 1:]
    results["GWR"]["train_r2"].append(utils.R2(y, gwrModelFit.predy))
    results["GWR"]["coef_rmse"].append(np.sum(np.sum(np.multiply(gwr_b - coefficients_b, gwr_b - coefficients_b))))
    results["GWR"]["test_r2"].append(utils.R2(y_test, gwr_pred))

    print("gwr time:", results["GWR"]["train_time"][-1])
    print("gwr train r2:", results["GWR"]["train_r2"][-1])
    print("gwr coefficients rmse:", results["GWR"]["coef_rmse"][-1])
    print("gwr test r2:", results["GWR"]["test_r2"][-1])


    #improved
    start_time = time.time()
    improvedBackfitting = backfitting_model(x, coords, y)
    results["GGWR"]["train_time"].append(time.time()-start_time)
    print("agwr bandwidth", improvedBackfitting.bandwidths)
    improvedBackfitting.fit(improvedBackfitting.bandwidths, iterations=10)
    B = improvedBackfitting.trainB
    GGWR_train_pred = np.sum(B*improvedBackfitting.X_training, axis=1)
    improved_ggwr_pred = improvedBackfitting.predict(coords_test, X_test)
    B = B[:, 1:]

    results["GGWR"]["train_r2"].append(utils.R2(y, GGWR_train_pred))
    results["GGWR"]["coef_rmse"].append(np.sum(np.sum(np.multiply(B-coefficients_b, B-coefficients_b))))
    results["GGWR"]["test_r2"].append(utils.R2(y_test, improved_ggwr_pred))

    print("aGWR time:", results["GGWR"]["train_time"][-1])
    print("aGWR train r2:", results["GGWR"]["train_r2"][-1])
    print("aGWR coefficients rmse:", results["GGWR"]["coef_rmse"][-1])
    print("aGWR test r2:", results["GGWR"]["test_r2"][-1])


    #MGWR
    start_time = time.time()
    selector = Sel_BW(coords, y, x, kernel='gaussian', multi=True)
    selector.search(criterion='CV', multi_bw_min=[2])
    results["MGWR"]["train_time"].append(time.time()-start_time)
    model = MGWR(coords, y, x, selector, sigma2_v1=True, kernel='gaussian', fixed=False,
                 spherical=False)
    print("MGWR bandwidths", model.bws)
    temp = model.fit()
    mgwr_pred_model = ImprovedBackfitting(x, y, coords)
    mgwr_pred_model.setB(temp.params)
    mgwr_pred_model.bandwidths = model.bws
    mgwr_pred = mgwr_pred_model.predict(coords_test, X_test)

    mgwr_b = temp.params
    mgwr_b = mgwr_b[:, 1:]

    results["MGWR"]["train_r2"].append(utils.R2(y, temp.predy))
    results["MGWR"]["coef_rmse"].append(np.sum(np.sum(np.multiply(mgwr_b-coefficients_b, mgwr_b-coefficients_b))))
    results["MGWR"]["test_r2"].append(utils.R2(y_test, mgwr_pred))

    print("MGWR time:", results["MGWR"]["train_time"][-1])
    print("MGWR train r2:", results["MGWR"]["train_r2"][-1])
    print("MGWR coefficients rmse:", results["MGWR"]["coef_rmse"][-1])
    print("MGWR test r2:", results["MGWR"]["test_r2"][-1])

    #ggwr with mgwr bandwidth
    start_time = time.time()
    GGWR_bw = ImprovedBackfitting(x, y, coords)
    GGWR_bw.bandwidths = model.bws
    SP = SpaceSearch(GGWR_bw)
    sim_config = {"steps": 10}
    bandwidths = SP.FDSA(GGWR_bw.bandwidths, sim_config)
    sim_config = {"steps": 30, "updates": 20, "method": "gaussian_one"}
    bandwidths = SP.hill_climbing(bandwidths, sim_config)
    GGWR_bw.bandwidths = bandwidths
    results["GGWR_bw"]["train_time"].append(time.time()-start_time)
    print("GGWR_bw bandwidth", GGWR_bw.bandwidths)
    GGWR_bw.fit(GGWR_bw.bandwidths, iterations=10)
    B = GGWR_bw.trainB
    GGWR_bw_train_pred = np.sum(B*GGWR_bw.X_training, axis=1)
    GGWR_bw_pred = GGWR_bw.predict(coords_test, X_test)
    B = B[:, 1:]

    results["GGWR_bw"]["train_r2"].append(utils.R2(y, GGWR_bw_train_pred))
    results["GGWR_bw"]["coef_rmse"].append(np.sum(np.sum(np.multiply(B-coefficients_b, B-coefficients_b))))
    results["GGWR_bw"]["test_r2"].append(utils.R2(y_test, GGWR_bw_pred))

    print("GGWR_bw time:", results["MGWR"]["train_time"][-1] + results["GGWR_bw"]["train_time"][-1])
    print("GGWR_bw train r2:", results["GGWR_bw"]["train_r2"][-1])
    print("GGWR_bw coefficients rmse:", results["GGWR_bw"]["coef_rmse"][-1])
    print("GGWR_bw test r2:", results["GGWR_bw"]["test_r2"][-1])

    # #golden search
    # start_time = time.time()
    # golden_model = golden_search_model(x, coords, y)
    # results["golden_ggwr"]["train_time"].append(time.time()-start_time)
    # print("golden_ggwr bandwidth", golden_model.bandwidths)
    # golden_model.fit(golden_model.bandwidths, iterations=10)
    # B = golden_model.trainB
    # golden_ggwr_train_pred = np.sum(B*golden_model.X_training, axis=1)
    # improved_golden_ggwr_pred = golden_model.predict(coords_test, X_test)
    # B = B[:, 1:]

    # results["golden_ggwr"]["train_r2"].append(utils.R2(y, golden_ggwr_train_pred))
    # results["golden_ggwr"]["coef_rmse"].append(np.sum(np.sum(np.multiply(B-coefficients_b, B-coefficients_b))))
    # results["golden_ggwr"]["test_r2"].append(utils.R2(y_test, improved_golden_ggwr_pred))

    # print("golden_ggwr time:", results["golden_ggwr"]["train_time"][-1])
    # print("golden_ggwr train r2:", results["golden_ggwr"]["train_r2"][-1])
    # print("golden_ggwr coefficients rmse:", results["golden_ggwr"]["coef_rmse"][-1])
    # print("golden_ggwr test r2:", results["golden_ggwr"]["test_r2"][-1])
    


experiments_results = {}
for categorie in categories:
    experiments_results[categorie] = {}
    for algorithm in algorithms:
        experiments_results[categorie][algorithm] = np.mean(results[algorithm][categorie])
        print("mean of", categorie, "in", algorithm, "is:", experiments_results[categorie][algorithm])

results_location = os.path.dirname(os.path.abspath(__file__)) + "/"
with open(results_location + "Results/comparison_vs_mgwr.data", 'wb') as filehandle:
    pickle.dump(experiments_results, filehandle)
