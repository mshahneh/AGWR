# from src.Test import gwrTest, mgwrTest, ggwrTest, xgboostTest, randomForrestTest, modular_test
from src.Test import gwrTest, randomForrestTest, modular_test, mgwrTest, xgboostTest
import warnings
import math
import random
import numpy as np
import pickle
import os
from src.experiments.display_results import display_results
from src.DataHandlers.syntheticData import synthetic_data_generator

np.random.seed(42)
random.seed(42)


def main():
    synthetic_data_generator(80, 5)
    data_sets = ["syntheticData1"]
    warnings.simplefilter(action='ignore', category=FutureWarning)
    results_location = os.path.dirname(os.path.abspath(__file__)) + "/"
    with open(results_location + "Results/divide_size.data", 'rb') as filehandle:
        experiments_results = pickle.load(filehandle)
    # experiments_results = {}
    for dataset in data_sets:
        path = os.path.dirname(os.path.abspath(
            __file__ + str("/../../"))) + "/Data/" + dataset + "/"
        print(path)
        with open(path + 'training_idx.data', 'rb') as filehandle:
            training_idx = pickle.load(filehandle)
        with open(path + 'validation_idx.data', 'rb') as filehandle:
            validation_idx = pickle.load(filehandle)
        with open(path + 'test_idx.data', 'rb') as filehandle:
            test_idx = pickle.load(filehandle)
        with open(path + 'x.data', 'rb') as filehandle:
            x = pickle.load(filehandle)
        X_training, X_validation, X_test = x[training_idx,
                                           :], x[validation_idx], x[test_idx, :]
        print("\n\n", dataset)
        print("train", X_training.shape, "validation",
              X_validation.shape, "test", X_test.shape, "\n")


        for sec1 in range(1, 6):
            for sec2 in range(1, 6):
                # if (sec1 == 1 and sec2 == 1):
                #     continue
                result = modular_test.modular_test(dataset, path, {"sections": [sec1, sec2], "is_pipeline": True, "spatial_model": "GWR"})
                experiments_results = display_results(dataset, result, "pipelined modular " + str(sec1)+"*"+str(sec2),
                                                    experiments_results, results_location + "Results/divide_size.data")

                result = modular_test.modular_test(dataset, path, {"sections": [sec1, sec2], "is_pipeline": False, "spatial_model": "GWR"})
                experiments_results = display_results(dataset, result, "ens modular " + str(sec1)+"*"+str(sec2),
                                                    experiments_results, results_location + "Results/divide_size.data")


# if __name__ == "__main__":
main()
