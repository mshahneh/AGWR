# from src.Test import gwrTest, mgwrTest, ggwrTest, randomForrestTest, modular_test
from src.Test import gwrTest, randomForrestTest, modular_test, mgwrTest
import warnings
import math
import random
import numpy as np
import pickle
import os
from src.experiments.display_results import display_results
import multiprocessing
np.random.seed(42)
random.seed(42)


def main():
    # data_sets = ["pysalGeorgia", "artificialData", "kingHousePrices", "pysalBaltimore"]
    # data_sets = ["artificialData", "kingHousePrices", "NYCAirBnb", "pysalGeorgia",
    #              "pysalBerlin"]
    data_sets = ['pysalGeorgia', 'pysalTokyo', 'pysalClearwater', 'pysalBerlin', "syntheticData1",
                 "syntheticData2", "kingHousePrices", "NYCAirBnb"]
    warnings.simplefilter(action='ignore', category=FutureWarning)
    results_location = os.path.dirname(os.path.abspath(__file__)) + "/"
    # experiments_results = {}
    with open(results_location + "Results/parallel_run_all.data", 'rb') as filehandle:
        experiments_results = pickle.load(filehandle)

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
        repCount = 5

        temp = len(training_idx) + len(validation_idx)
        temp /= 1000
        processes = min(multiprocessing.cpu_count(), math.ceil(temp))
        process_config = {"process_count": processes}

        sec1 = max(1, math.floor(temp ** (0.5)))
        sec2 = max(1, math.ceil(temp ** (0.5)))
        print("sec:", sec1, sec2, processes)


        ## Modular MGWR
        time = 0
        test_r2 = 0
        for i in range(repCount):
            print("i:", i)
            tempResult = modular_test.modular_test(dataset, path,
                                           {"sections": [sec1, sec2], "is_pipeline": True, "spatial_model": "MGWR", "process_count": processes})
            time += tempResult["time"]
            test_r2 += tempResult["test_R2"]
        result = tempResult
        result["time"] = time/repCount
        result["test_R2"] = test_r2/repCount
        experiments_results = display_results(dataset, result, "pipelined modular MGWR",
                                            experiments_results, results_location + "Results/parallel_run_all.data")


        # ## Modular GGWR
        # time = 0
        # test_r2 = 0
        # temperr = []
        # for i in range(repCount):
        #     tempResult = modular_test.modular_test(dataset, path,
        #                                    {"sections": [sec1, sec2], "is_pipeline": True, "spatial_model": "GGWR",
        #                                     "process_count": processes, "overlap": 0.1})
        #     time += tempResult["time"]
        #     test_r2 += tempResult["test_R2"]
        #     temperr.append(tempResult["test_R2"])
        # result = tempResult
        # result["time"] = time/repCount
        # result["test_R2"] = test_r2/repCount
        # result["temp_err"] = temperr
        # experiments_results = display_results(
        #     dataset, result, "pipelined modular GGWR equal count with overlap", experiments_results,
        #     results_location + "Results/parallel_run_all.data")


        ## Modular GWR
        time = 0
        test_r2 = 0
        for i in range(repCount):
            tempResult = modular_test.modular_test(dataset, path,
                                               {"sections": [sec1, sec2], "is_pipeline": True, "spatial_model": "GWR",
                                                "process_count": processes, "overlap": 0.1})
            time += tempResult["time"]
            test_r2 += tempResult["test_R2"]
        result = tempResult
        result["time"] = time / repCount
        result["test_R2"] = test_r2 / repCount
        experiments_results = display_results(dataset, result, "pipelined modular GWR with overlap",
                                              experiments_results, results_location + "Results/parallel_run_all.data")


        # ## GWR + Random Forest
        # time = 0
        # test_r2 = 0
        # for i in range(repCount):
        #     tempResult = modular_test.modular_test(dataset, path,
        #                                    {"sections": [1, 1], "is_pipeline": True, "spatial_model": "GWR"})
        #     time += tempResult["time"]
        #     test_r2 += tempResult["test_R2"]
        # result = tempResult
        # result["time"] = time / repCount
        # result["test_R2"] = test_r2 / repCount

        # experiments_results = display_results(dataset, result, "GWR and RF",
        #                                       experiments_results, results_location + "Results/parallel_run_all.data")


        # ## Modular GWR with kmeans
        # time = 0
        # test_r2 = 0
        # for i in range(repCount):
        #     tempResult = modular_test.modular_test(dataset, path,
        #                                    {"divide_method": "kmeans", "sections": [sec1, sec2], "is_pipeline": True,
        #                                     "spatial_model": "GWR"})
        #     time += tempResult["time"]
        #     test_r2 += tempResult["test_R2"]
        # result = tempResult
        # result["time"] = time / repCount
        # result["test_R2"] = test_r2 / repCount
        # experiments_results = display_results(dataset, result, "pipelined GWR kmeans",
        #                                       experiments_results, results_location + "Results/parallel_run_all.data")

        # if (sec1 == 1 and sec2 == 1):
        #     sec2 = 2
        # result = modular_test.modular_test(
        #     dataset, path, {"sections": [sec1, sec2], "is_pipeline": False})
        # experiments_results = display_results(
        #     dataset, result, "ens modular", experiments_results, results_location + "Results/run_all.data")


        ##MGWR
        # try:
        #     result = mgwrTest.mgwr_test(dataset, path, process_config)
        #     experiments_results = display_results(dataset, result, "MGWR", experiments_results,
        #                                           results_location + "Results/parallel_run_all.data")
        # except:
        #     print("error in MGWR!")

        # result = xgboostTest.xgboost_test(dataset, path)
        # experiments_results = display_results(
        #     dataset, result, "xgboost", experiments_results, results_location + "Results/run_all.data")


        ## Random Forest
        # time = 0
        # test_r2 = 0
        # for i in range(repCount):
        #     print(i, end=" ")
        #     tempResult = randomForrestTest.random_forrest_test(dataset, path, process_config)
        #     time += tempResult["time"]
        #     test_r2 += tempResult["test_R2"]
        # print()
        # result = tempResult
        # result["time"] = time / repCount
        # result["test_R2"] = test_r2 / repCount
        # experiments_results = display_results(
        #     dataset, result, "Random Forest", experiments_results, results_location + "Results/parallel_run_all.data")



        ## GWR
        # try:
        #     result = gwrTest.gwr_test(dataset, path, process_config)
        #     experiments_results = display_results(
        #         dataset, result, "gwr", experiments_results, results_location + "Results/parallel_run_all.data")
        # except:
        #     print("error in GWR")


if __name__ == "__main__":
    main()
