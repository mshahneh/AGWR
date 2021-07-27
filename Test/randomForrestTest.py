import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score
import statsmodels.formula.api as sm
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Lasso
from sklearn.model_selection import GridSearchCV
import matplotlib
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import sys
# sys.path.insert(0,'..')

import numpy as np
import pickle
import time
import src.utils as utils


def random_forrest_test(dataset, path, config={}):
    default_config = {"process_count": 3}
    for key in config.keys():
        default_config[key] = config[key]
    config = default_config

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

    X_training, X_validation, X_test = x[training_idx,
                                         :], x[validation_idx], x[test_idx, :]
    y_training, y_validation, y_test = y[training_idx,
                                         :], y[validation_idx], y[test_idx, :]
    coords_training, coords_validation, coords_test = coords[
        training_idx], coords[validation_idx], coords[test_idx]

    D_train = np.concatenate((X_training, coords_training), axis=1)
    D_validation = np.concatenate((X_validation, coords_validation), axis=1)
    D_test = np.concatenate((X_test, coords_test), axis=1)
    y_training = y_training.reshape(-1)
    y_test = y_test.reshape(-1)

    start_time = time.time()

    rf = RandomForestRegressor(n_jobs=config["process_count"])
    n_estimators = [int(x) for x in np.linspace(start=20, stop=500, num=10)]
    max_features = ['auto', 'sqrt']
    max_depth = [int(x) for x in np.linspace(10, 110, num=11)]
    max_depth.append(None)
    min_samples_split = [2, 5, 10]
    min_samples_leaf = [1, 2, 4]
    bootstrap = [True, False]
    random_grid = {'n_estimators': n_estimators,
                   'max_features': max_features,
                   'max_depth': max_depth,
                   'min_samples_split': min_samples_split,
                   'min_samples_leaf': min_samples_leaf,
                   'bootstrap': bootstrap}

    rf_random = RandomizedSearchCV(estimator=rf, param_distributions=random_grid,
                                   n_iter=50, cv=3, verbose=0,
                                   random_state=42, n_jobs=config["process_count"])
    rf_random.fit(D_train, y_training)
    best_config = rf_random.best_params_
    rg = RandomForestRegressor(
        n_estimators=best_config["n_estimators"],
        min_samples_split=best_config["min_samples_split"],
        min_samples_leaf=best_config["min_samples_leaf"],
        max_features=best_config["max_features"],
        max_depth=best_config["max_depth"],
        n_jobs=config["process_count"]
    )
    rg.fit(D_train, y_training.reshape(-1, ))
    res = {}
    res["time"] = time.time() - start_time
    res["train_rmse"] = np.sqrt(
        mean_squared_error(y_training, rg.predict(D_train)))
    res["train_R2"] = utils.R2(
        y_training.reshape(-1).tolist(), rg.predict(D_train))

    res["validation_rmse"] = np.sqrt(mean_squared_error(
        y_validation, rg.predict(D_validation)))
    res["validation_R2"] = utils.R2(
        y_validation.reshape(-1).tolist(), rg.predict(D_validation))

    res["test_rmse"] = np.sqrt(mean_squared_error(y_test, rg.predict(D_test)))
    res["test_R2"] = utils.R2(y_test.reshape(-1).tolist(), rg.predict(D_test))

    # print("random forrest train RMSE is:", np.sqrt(mean_squared_error(y_training, rg.predict(D_train))))
    # print("random forrest train R2 is:", utils.R2(y_training.reshape(-1).tolist(), rg.predict(D_train)))
    #
    # print("random forrest validation RMSE is:", np.sqrt(mean_squared_error(y_validation, rg.predict(D_validation))))
    # print("random forrest validation R2 is:", utils.R2(y_validation.reshape(-1).tolist(), rg.predict(D_validation)))
    #
    # print("random forrest test RMSE is:", np.sqrt(mean_squared_error(y_test, rg.predict(D_test))))
    # print("random forrest test R2 is:", utils.R2(y_test.reshape(-1).tolist(), rg.predict(D_test)))

    # D_all = np.concatenate((x, coords), axis = 1)
    # predictAll = rg.predict(D_all)
    # with open(path + '/predictAll_ranforr.data', 'wb') as filehandle:
    #     pickle.dump(predictAll, filehandle)

    return res


def main():
    dataset = "kingHousePrices"
    # dataset = "artificialData"
    path = os.path.dirname(os.path.abspath(
        __file__ + str("/../../"))) + "/Data/" + dataset + "/"
    result = random_forrest_test(dataset, path)
    print("the random forrest train time is: ", result["time"])
    print("the random forrest train r2 error is: ", result["train_R2"])
    print("the random forrest train rmse is: ", result["train_rmse"])
    print("the random forrest validation r2 error is: ",
          result["validation_R2"])
    print("the random forrest validation rmse is: ", result["validation_rmse"])
    print("the random forrest prediction error is: ", result["test_R2"])
    print("the random forrest rmse is: ", result["test_rmse"])


if __name__ == "__main__":
    main()
