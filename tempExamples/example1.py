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
from helpers.SampleModules.MLModules import random_forrest, neural_network, xgb, MLP
from ModularFramework.ModularFramework import ModularFramework


# read input
def read_input(dataset, validation=False):
    """
    reads the data
    :param dataset: dataset name to read
    :param validation: if true, returns the validation set as well
    :return: returns X, coords, and y for train, (validation), and test
    """
    path = os.path.dirname(os.path.abspath(__file__+"/../")) + "/Data/" + dataset + "/"

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
def module_selection(spatial, mls):
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

    ml_module = []
    for ml in mls:
        if ml == "RF":
            ml_module.append(random_forrest)
        elif ml == "NN":
            ml_module.append(neural_network)
        elif ml == "MLP":
            ml_module.append(MLP)
        elif ml == "XGB":
            ml_module.append(xgb)
        else:
            ml_module.append(None)

    return spatial_module, ml_module


def solve(spatial_module_name, ML_module_names, dataset):
    
    # reading the dataset values
    X_training, coords_training, y_training, X_test, coords_test, y_test = read_input(dataset, False)
    test_r2 = []

    if spatial_module_name == None:
        # selecting spatial and ml modules
        spatial_module, ML_module = module_selection(spatial_module_name, ML_module_names)
        
        for i in range(len(ML_module)):
            if ML_module[i] is None:
                test_r2.append(None)
                continue
            start_time = time.time()
            ml = ML_module[i](X_training, coords_training, y_training)
            D_test = np.concatenate((X_test, coords_test), axis=1)
            end_time = time.time()

            pred = ml.predict(D_test)
            test_r2.append(utils.R2(y_test.reshape(-1).tolist(), pred))

    else:
        spatial_module, ML_module = module_selection(spatial_module_name, ML_module_names)
        print(spatial_module_name, ML_module_names)
        start_time = time.time()

        # creating the A-GWR setting
        temp = len(X_training)
        temp /= 900
        processes = min(multiprocessing.cpu_count(), math.ceil(temp))
        sec1 = max(1, math.floor(temp ** (0.5)))
        sec2 = max(1, math.ceil(temp ** (0.5)))

        print(sec1, sec2)

        A_GWR_config = {"process_count": processes, "divide_method": "equalCount",
                        "divide_sections": [sec1, sec2], "pipelined": True}  # a-gwr configurations
        A_GWR = ModularFramework(spatial_module, ML_module, A_GWR_config)

        # train model
        A_GWR.train(X_training, coords_training, y_training)

        end_time = time.time()

        # predict and print result
        pred = A_GWR.predict(X_test, coords_test, y_test)

        for j in range(len(ML_module)):
            test_r2.append(utils.R2(y_test.reshape(-1).tolist(), pred[j]))

    return test_r2
        

def main():
    # the dataset name
    # spatial_module_names = [None, "GWR", "SMGWR"]
    spatial_module_names = ["GWR"]
    ML_module_names = [None]
    # data_sets = ["pysalGeorgia"]
    data_sets = ["pysalTokyo", "pysalGeorgia", "pysalClearwater", "pysalBerlin", "syntheticData1", "syntheticData2", "kingHousePrices"]
    rep_count = 5

    results = {}
    
    for spatial_module_name in spatial_module_names:
        if spatial_module_name not in results.keys():
            results[spatial_module_name] = {}
        for ML_module_name in ML_module_names:
            if ML_module_name not in results[spatial_module_name].keys():
                results[spatial_module_name][ML_module_name] = {}
            for dataset in data_sets:
                results[spatial_module_name][ML_module_name][dataset] = {"time":[], "error":[]}
    for spatial_module_name in spatial_module_names:
        for dataset in data_sets:
            print(dataset, ":")
            for rep in range(rep_count):
                print("rep", rep, end=": ")
                # try:
                test_error = solve(spatial_module_name, ML_module_names, dataset)
                for i in range(len(ML_module_names)):
                    ML_module_name = ML_module_names[i]
                    # results[spatial_module_name][ML_module_name][dataset]["time"].append(time[i])
                    results[spatial_module_name][ML_module_name][dataset]["error"].append(test_error[i])
                    print(test_error[i])
                    # break
                # except Exception as inst:
                #     print("**error**", inst)
                print("done", end="  ")
        visualizer(results, "table")
    print(results)


if __name__ == "__main__":
    main()
