import os
import pickle
import numpy as np
from prettytable import PrettyTable

def latex(dict):
    a = 1

def table(dict):
    spatial_methods = list(dict.keys())
    ml_methods = list(dict[spatial_methods[0]].keys())

    datasets = set()
    for spatial_method in spatial_methods:
        for ml_method in ml_methods:
            for dataset in dict[spatial_method][ml_method].keys():
                datasets.add(dataset)
    # datasets = list(dict[spatial_methods[0]][ml_methods[0]].keys())
    temp = ['method']
    temp.extend(datasets)
    table_times = PrettyTable(temp)
    table_errors = PrettyTable(temp)
    for spatial_method in spatial_methods:
        for ml_method in ml_methods:
            if spatial_method == None and ml_method == None:
                continue
            try:
                train_times = [str(spatial_method) + " + " + str(ml_method)]
                r2_errors = [str(spatial_method) + " + " + str(ml_method)]
                for dataset in datasets:
                    try:
                        times = dict[spatial_method][ml_method][dataset]['time']
                        errors = dict[spatial_method][ml_method][dataset]['error']
                    except:
                        times=[]
                        errors=[]
                    if len(errors) == 0:
                        mean_time = '-'
                        mean_errors = '-'
                    else:
                        mean_time = round(np.mean(times), 2)
                        mean_errors = round(np.mean(errors), 4)

                    train_times.append(mean_time)
                    r2_errors.append(str(mean_errors) + " + " + str(round(np.std(errors), 4)))
                table_times.add_row(train_times)
                table_errors.add_row(r2_errors)
            except:
                a = "do nothing"
    
    print(table_errors)

def html(dict):
    a = 1


def visualizer(dict, type="table"):
    if type == "latex":
        latex(dict)
    elif type == "table":
        table(dict)
    elif type == "html":
        html(dict)

def main():
    path = path = os.path.dirname(os.path.abspath(__file__ + "/../../")) + "/fast_all_models.p"
    dict = pickle.load( open( path, "rb" ) )
    type = "table"

    visualizer(dict, type)


if __name__ == "__main__":
    main()
