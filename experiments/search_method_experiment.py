from src.Test import ggwrTest
import warnings
import pickle
import os
from src.experiments.display_results import display_results
from src.DataHandlers.syntheticData import synthetic_data_generator
import random
import numpy as np
np.random.seed(42)
random.seed(42)


def main():
    experiments_results = {}
    synthetic_data_generator(36, 5)
    data_sets = ["syntheticData1"]
    results_location = os.path.dirname(os.path.abspath(__file__)) + "/"
    warnings.simplefilter(action='ignore', category=FutureWarning)
    for dataset in data_sets:
        path = os.path.dirname(os.path.abspath(__file__ + str("/../../"))) + "/Data/" + dataset + "/"
        with open(path + 'training_idx.data', 'rb') as filehandle:
            training_idx = pickle.load(filehandle)
        with open(path + 'validation_idx.data', 'rb') as filehandle:
            validation_idx = pickle.load(filehandle)
        with open(path + 'test_idx.data', 'rb') as filehandle:
            test_idx = pickle.load(filehandle)
        with open(path + 'x.data', 'rb') as filehandle:
            x = pickle.load(filehandle)
        X_training, X_validation, X_test = x[training_idx, :], x[validation_idx], x[test_idx, :]
        print("\n\n", dataset)

        methods = ["bayesian_optimization", "bayesian_hill", "advanced_combined", "golden_search",
                   "hyperband", "combined", "hill_climbing", "fibonacci_one"]

        repCount = 5
        for method in methods:
            mean_time = 0
            mean_rmse = 0
            mean_R2 = 0
            for i in range(repCount):
                result = ggwrTest.ggwr_test(dataset, path, {"space_search_method": method})
                mean_time += result["time"]
                mean_rmse += result["test_rmse"]
                mean_R2 += result["test_R2"]
                print(method, result["time"], result["test_rmse"], result["test_R2"])

            result["time"] = mean_time/repCount
            result["test_rmse"] = mean_rmse/repCount
            result["test_R2"] = mean_R2/repCount

            experiments_results = display_results(dataset, result, method, experiments_results,
                                                  results_location + "Results/search_method_experiment.data")


    with open(results_location + 'Results/search_method_experiment.data', 'wb') as filehandle:
        pickle.dump(experiments_results, filehandle)



#if __name__ == "__main__":
main()
