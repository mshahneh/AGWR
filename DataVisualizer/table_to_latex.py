import pickle
import numbers
import os
from prettytable import PrettyTable

path = path = os.path.dirname(os.path.abspath(
    __file__ + str("/.."))) + "/experiments/Results/parallel_run_all.data"

with open(path, 'rb') as filehandle:
    experiment_result = pickle.load(filehandle)

# keys = ["time", "bandwidths", "test_R2", "test_rmse"]

keys = ["test_R2"]

# def convert_to_table(result):
#     temp = ['Name']
#     temp.extend(keys)
#     t = PrettyTable(temp)
#     del temp
#     for method_name in result.keys():
#         row = [method_name]
#         for key in keys:
#             if key in result[method_name].keys():
#                 if isinstance(result[method_name][key], numbers.Number):
#                     row.append(round(result[method_name][key], 6))
#                 else:
#                     row.append(result[method_name][key])
#                 # print("the", method_name, key, "is: ", result[key])
#             else:
#                 row.append("-")
#         t.add_row(row)

#     return t


# for dataset in experiment_result.keys():
#     print("Dataset:", dataset)
#     t = convert_to_table(experiment_result[dataset])
#     print(t)


method_names = []
for dataset in experiment_result.keys():
    results = experiment_result[dataset]
    for method_name in results.keys():
        if method_name not in method_names:
            method_names.append(method_name)
# method_names = list(method_names)
# method_names = sorted(method_names)

datasets = []
for dataset in experiment_result.keys():
    datasets.append(dataset)
# datasets = ["syntheticData1", "syntheticData2", "kingHousePrices", "NYCAirBnb"]
# datasets = ["pysalClearwater", "pysalGeorgia", "pysalTokyo", "pysalBerlin"]
for dataset in datasets:
    # print("&\\multicolumn{2}{c|}{", dataset, "}", end=" ")
    print("&", dataset, end=" ")
print("\\\\")
print("\\hline")
for dataset in datasets:
    print(" & error", end=" ")
print("\\\\")
print("\\hline")
for method_name in method_names:
    print(method_name, end=" ")

    for dataset in datasets:
        for key in keys:
            try:
                print("&", round(experiment_result[dataset]
                            [method_name][key], 4), end=" ")
            except:
                print("&", end=" ")

    print("\\\\")
