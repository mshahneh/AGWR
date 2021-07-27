import pickle
import numbers
from prettytable import PrettyTable

# keys = ["time", "bandwidths", "train_R2", "train_rmse",
#         "validation_R2", "validation_rmse", "test_R2", "test_rmse"]
keys = ["time", "bandwidths", "test_R2", "test_rmse"]


def convert_to_table(result):
    temp = ['Name']
    temp.extend(keys)
    t = PrettyTable(temp)
    del temp
    for method_name in result.keys():
        row = [method_name]
        for key in keys:
            if key in result[method_name].keys():
                if isinstance(result[method_name][key], numbers.Number):
                    row.append(round(result[method_name][key], 6))
                else:
                    row.append(result[method_name][key])
                # print("the", method_name, key, "is: ", result[key])
            else:
                row.append("-")
        t.add_row(row)

    return t


def display_results(dataset, result, method_name, experiments_results, experiment_path):
    # t = convert_to_table({method_name:result})
    # print(t)
    # print("-------------------")
    if dataset not in experiments_results.keys():
        experiments_results[dataset] = {}
    experiments_results[dataset][method_name] = result

    # print(experiments_results)
    with open(experiment_path, 'wb') as filehandle:
        pickle.dump(experiments_results, filehandle)

    t = convert_to_table(experiments_results[dataset])
    print("-------------------")
    print(t)
    return experiments_results


if __name__ == "__main__":
    path = "Results/run_all.data"
    with open(path, 'rb') as filehandle:
        experiments_results = pickle.load(filehandle)

    for dataset in experiments_results.keys():
        print("Dataset:", dataset)
        t = convert_to_table(experiments_results[dataset])
        print(t)
