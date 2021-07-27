from src.Test import ggwrTest
import warnings
import pickle
import os
from src.experiments.display_results import display_results
from src.DataHandlers.syntheticData import synthetic_data_generator
from src.BackfittingModelImproved import ImprovedBackfitting
from mgwr.gwr import GWR, MGWR
from mgwr.sel_bw import Sel_BW
from prettytable import PrettyTable
import src.utils as utils
import matplotlib.pyplot as plt
import random
import numpy as np
np.random.seed(42)
random.seed(42)

results_location = os.path.dirname(os.path.abspath(__file__)) + "/"
# methods = ["bayesian_optimization", "bayesian_hill", "advanced_combined", "golden_search",
#                    "hyperband", "combined", "hill_climbing", "fibonacci_one", "SPSA", 'combined', 'successive_halving', 'simulated_annealing']
methods = ["bayesian_optimization",  "hyperband", "hill_climbing", "fibonacci_one", "SPSA", 'successive_halving', 'simulated_annealing']
method_names = ["Bayesian Optimization", "Hyperband", "Hill Climbing", "Fibonacci Hill Climbing", "SPSA", 'Successive Halving', 'Simulated Annealing']
data_sets = ["syntheticData1"]

def main():
    experiments_results = {}
    with open(results_location + "Results/search_method_experiment_with_charts.data", 'rb') as filehandle:
        experiments_results = pickle.load(filehandle)
    # synthetic_data_generator(27, 5)

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



        sizes = list(np.round(np.arange(0.1, 1.01, 0.1), 2))
        # sizes = list(np.round(np.arange(1, 1.01, 0.1), 2))

        repCount = 5
        for method in methods:
            history = {"sizes":sizes}
            history["results"] = []
            for size in sizes:
                mean_time = 0
                mean_rmse = 0
                mean_R2 = 0
                print(method, size)
                for i in range(repCount):
                    result = ggwrTest.ggwr_test(dataset, path, {"space_search_method": method}, size)
                    mean_time += result["time"]
                    mean_rmse += result["test_rmse"]
                    mean_R2 += result["test_R2"]
                    print(method, result["time"], result["test_rmse"], result["test_R2"])

                result["time"] = mean_time/repCount
                result["test_rmse"] = mean_rmse/repCount
                result["test_R2"] = mean_R2/repCount
                result["sizes"] = sizes

                history["results"].append(result)
                print(result)

            experiments_results = display_results(dataset, history, method, experiments_results,
                                                  results_location + "Results/search_method_experiment_with_charts.data")


    with open(results_location + 'Results/search_method_experiment_with_charts.data', 'wb') as filehandle:
        pickle.dump(experiments_results, filehandle)


def draw():
    #plot configuration
    linestyle_tuple = [
     ('loosely dotted',        (0, (1, 10))),
     ('dotted',                (0, (1, 1))),
     ('densely dotted',        (0, (1, 1))),

     ('loosely dashed',        (0, (5, 10))),
     ('dashed',                (0, (5, 5))),
     ('densely dashed',        (0, (5, 1))),

     ('dashdotted',            (0, (3, 5, 1, 5))),
     ('densely dashdotted',    (0, (3, 1, 1, 1))),

     ('dashdotdotted',         (0, (3, 5, 1, 5, 1, 5))),
     ('densely dashdotdotted', (0, (3, 1, 1, 1, 1, 1)))]

    markers = ['o', 'v', '^', 's', 'h', 'P', 'd', 'X', '8', '*', '1']
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))


    # Reading Data from input
    path = os.path.dirname(os.path.abspath(__file__ + str("/../../"))) + "/Data/" + data_sets[0] + "/"
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

    GWRbandwidth = Sel_BW(_coords, _y, _x, kernel='gaussian').search(criterion='CV')
    gwrModel = GWR(_coords, _y, _x, bw=GWRbandwidth, fixed=False, kernel='gaussian', spherical=True)
    gwrModelFit = gwrModel.fit()
    gwr_pred = gwrModel.predict(coords_test, X_test).predy
    gwr_pred = utils.R2(y_test, gwr_pred)




    #MGWR
    selector = Sel_BW(_coords, _y, _x, kernel='gaussian', multi=True)
    selector.search(criterion='CV', multi_bw_min=[2])
    model = MGWR(_coords, _y, _x, selector, sigma2_v1=True, kernel='gaussian', fixed=False,
                 spherical=False)
    temp = model.fit()
    mgwr_pred_model = ImprovedBackfitting(_x, _y, _coords)
    mgwr_pred_model.setB(temp.params)
    mgwr_pred_model.bandwidths = model.bws
    mgwr_pred = mgwr_pred_model.predict(coords_test, X_test)
    mgwr_pred = utils.R2(y_test, mgwr_pred)
    # mgwr_pred = 0.0016



    results_location = os.path.dirname(os.path.abspath(__file__)) + "/"
    with open(results_location + "Results/search_method_experiment_with_charts.data", 'rb') as filehandle:
        experiments_results = pickle.load(filehandle)
    
    # print(experiments_results[data_sets[0]][methods[0]]["results"])
    t = PrettyTable(["method", "R_2"])
    temp = []
    for method in methods:
        temp.append(round(experiments_results[data_sets[0]][method]["results"][-1]["test_R2"], 6))
    
    pltx = list(range(0,101,10))
    plty = [mgwr_pred for _ in range(len(pltx))]
    plt.plot(pltx, plty, label="MGWR baseline", marker = markers[-2])
    
    indices = np.argsort(temp)
    for i in range(len(methods)):
        method = methods[indices[i]]
        row = [method]
        # row.append(round(experiments_results[data_sets[0]][method]["results"][-1]["time"], 6))
        row.append(round(experiments_results[data_sets[0]][method]["results"][-1]["test_R2"], 6))
        t.add_row(row)
        # pltx = list(experiments_results[data_sets[0]][method]["sizes"])
        # pltx.insert(0, 0)
        plty = [gwr_pred]
        print(method, pltx, plty, len( experiments_results[data_sets[0]][method]["sizes"]))
        for item in experiments_results[data_sets[0]][method]["results"]:
            plty.append(item['test_R2'])
        plt.plot(pltx, plty, label=method_names[indices[i]], marker=markers[i])
    


    plty = [gwr_pred for _ in range(len(plty))]
    plt.plot(pltx, plty, label="GWR", marker = markers[-1])

    row = ["GWR"]
    # row.append('-')
    row.append(round(gwr_pred, 6))
    t.add_row(row)

    row = ["MGWR"]
    # row.append('-')
    row.append(round(mgwr_pred, 6))
    t.add_row(row)

    print(t)
    plt.xlabel("Time (s)", fontsize=18)
    plt.ylabel("Error", fontsize=16)
    plt.legend(loc='upper right', bbox_to_anchor=(1, 0.9))
    plt.savefig(results_location + 'search_methods.png', bbox_inches='tight', pad_inches=0, dpi=500)
    plt.show()


    print('MGWR bandwidths:', model.bws, 'best:', experiments_results[data_sets[0]][methods[indices[0]]]["results"][-1]["bandwidths"])
    
    

#if __name__ == "__main__":
draw()
