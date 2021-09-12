# importing the libraries
import numpy as np
import pickle
import os
import os
import pickle
import utils as utils
import numpy as np
import time
import math
import multiprocessing
from helpers.Visualizer.visualizer import visualizer
from helpers.SampleModules.SpatialModules import gwr_module, mgwr_module, smgwr_module
from helpers.SampleModules.MLModules import random_forrest, neural_network, xgb
from ModularFramework.ModularFramework import ModularFramework


# read input
def read_input(dataset, validation=False):
    """
    reads the data
    :param dataset: dataset name to read
    :param validation: if true, returns the validation set as well
    :return: returns X, coords, and y for train, (validation), and test
    """
    path = os.path.dirname(os.path.abspath(__file__)) + "/Data/" + dataset + "/"

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

    x = x.astype(float)
    y = y.astype(float)
    coords = coords.astype(float)


    X_training, X_validation, X_test = x[training_idx, :], x[validation_idx], x[test_idx, :]
    y_training, y_validation, y_test = y[training_idx, :], y[validation_idx], y[test_idx, :]
    coords_training, coords_validation, coords_test = coords[training_idx], coords[validation_idx], coords[test_idx]

    if not validation:
        X_combined = np.concatenate((X_validation, X_training), axis=0)
        coords_combined = np.concatenate((coords_validation, coords_training), axis=0)
        y_combined = np.concatenate((y_validation, y_training), axis=0)

        return X_combined, coords_combined, y_combined, X_test, coords_test, y_test
    else:
        return X_training, coords_training, y_training, X_validation, coords_validation, y_validation, X_test, coords_test, y_test


# create modules and their setting
def module_selection(spatial, ml):
    """
    selects the spatial and ml module
    :param spatial: spatial module name (GWR, MGWR, SMGWR)
    :param ml: ml module name (Random Forest)
    :return: returns the selected modules
    """
    if spatial == "GWR":
        spatial_module = gwr_module
    elif spatial == "MGWR":
        spatial_module = mgwr_module
    elif spatial == "SMGWR":
        spatial_module = smgwr_module
    else:
        spatial_module = None

    if ml == "RF":
        ml_module = random_forrest
    elif ml == "NN":
        ml_module = neural_network
    elif ml == "XGB":
        ml_module = xgb
    else:
        ml_module = None

    return spatial_module, ml_module


def solve(spatial_module_name, ML_module_name, dataset):
    
    # reading the dataset values
    X_training, coords_training, y_training, X_test, coords_test, y_test = read_input(dataset, False)

    # selecting spatial and ml modules
    spatial_module, ML_module = module_selection(spatial_module_name, ML_module_name)

    if spatial_module == None:
        start_time = time.time()
        ml = ML_module(X_training, coords_training, y_training)
        D_test = np.concatenate((X_test, coords_test), axis=1)
        end_time = time.time()

        pred = ml.predict(D_test)
        test_r2 = utils.R2(y_test.reshape(-1).tolist(), pred)

    elif ML_module == None:
        start_time = time.time()
        sp_m = spatial_module(X_training, coords_training, y_training)
        end_time = time.time()
        prediction = sp_m.predict(coords_test, X_test)

        try:
            pred = prediction.predy
        except:
            pred = prediction
        
        test_r2 = utils.R2(y_test.reshape(-1).tolist(), pred)

    else:
        start_time = time.time()

        # creating the A-GWR setting
        temp = len(X_training)
        temp /= 900
        processes = min(multiprocessing.cpu_count(), math.ceil(temp))
        sec1 = max(1, math.floor(temp ** (0.5)))
        sec2 = max(1, math.ceil(temp ** (0.5)))

        print(sec1, sec2)

        A_GWR_config = {"process_count": processes, "divide_method": "equalCount",
                        "divide_sections": [sec1, sec2], "pipelined": False}  # a-gwr configurations
        A_GWR = ModularFramework(spatial_module, ML_module, A_GWR_config)

        # train model
        A_GWR.train(X_training, coords_training, y_training)

        end_time = time.time()

        # predict and print result
        pred = A_GWR.predict(X_test, coords_test, y_test)
        test_r2 = utils.R2(y_test.reshape(-1).tolist(), pred)

    return end_time-start_time, test_r2
        

def main():
    # the dataset name
    spatial_module_names = ["SMGWR"]
    ML_module_names = [None]
    # data_sets = ["kingHousePrices", "syntheticData2", "syntheticData1"]
    # data_sets = ["pysalTokyo", "pysalGeorgia", "pysalClearwater", "pysalBerlin", "syntheticData1", "syntheticData2"]
    data_sets = ["pysalBerlin"]
    rep_count = 1

    try:
        results = pickle.load( open( "all_models.p", "rb" ) )
    except:
        results = {}
    for spatial_module_name in spatial_module_names:
        results[spatial_module_name] = {}
        for ML_module_name in ML_module_names:
            results[spatial_module_name][ML_module_name] = {}
            for dataset in data_sets:
                if spatial_module_name == None and ML_module_name == None:
                    continue
                results[spatial_module_name][ML_module_name][dataset] = {"time":[], "error":[]}
                print(spatial_module_name, ML_module_name, dataset, ":\n")
                for rep in range(rep_count):
                    print("rep", rep, end=": ")
                    try:
                        time, test_error = solve(spatial_module_name, ML_module_name, dataset)
                        results[spatial_module_name][ML_module_name][dataset]["time"].append(time)
                        results[spatial_module_name][ML_module_name][dataset]["error"].append(test_error)
                        pickle.dump(results, open( "all_models.p", "wb" ))
                        # break
                    except Exception as inst:
                        print("**error**", inst)
                    print("done", end="  ")
            visualizer(results, "table")
    print(results)


if __name__ == "__main__":
    main()
