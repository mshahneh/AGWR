import sys
# sys.path.insert(0,'..')
import xgboost as xgb
import numpy as np
import pickle
import time
import os
import src.utils as utils
from xgboost import XGBRegressor, plot_importance
from sklearn.metrics import mean_squared_error
from sklearn.metrics import explained_variance_score, median_absolute_error, r2_score


def xgboost_test(dataset, path, config={}):
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

    D_train = np.concatenate((np.concatenate((X_training, X_validation), axis=0), np.concatenate(
        (coords_training, coords_validation), axis=0)), axis=1)
    L_train = np.concatenate((y_training, y_validation), axis=0)
    D_test = np.concatenate((X_test, coords_test), axis=1)

    y_mean = np.mean(L_train)
    start_time = time.time()
    xgb_params = {'eta':  0.05,
                  'max_depth':  8,
                  'subsample': 0.80,
                  'objective':  'reg:linear',
                  'eval_metric': 'rmse',
                  'base_score':  y_mean,
                  'nthread': 1}

    # xgb = XGBRegressor(n_estimators=100, learning_rate=0.08, gamma=0, subsample=0.75, colsample_bytree=1, max_depth=7)

    model = XGBRegressor()
    model.fit(X_training, y_training)
    res = {}
    res["time"] = time.time() - start_time
    preds = model.predict(X_training)
    res["train_rmse"] = utils.RMSE(y_training.reshape(-1).tolist(), preds)
    res["train_R2"] = utils.R2(y_training.reshape(-1).tolist(), preds)
    # print("the XGBoost train error is equal to: ", utils.RMSE(y_training.reshape(-1).tolist(), preds))
    # print("the XGBoost train r2 error is equal to: ", utils.R2(y_training.reshape(-1).tolist(), preds))

    preds = model.predict(X_test)
    res["test_rmse"] = utils.RMSE(y_test.reshape(-1).tolist(), preds)
    res["test_R2"] = utils.R2(y_test.reshape(-1).tolist(), preds)
    # print("the XGBoost error is equal to: ", utils.RMSE(y_test.reshape(-1).tolist(), preds))
    # print("the XGBoost r2 error is equal to: ", utils.R2(y_test.reshape(-1).tolist(), preds))

    return res


def main():
    dataset = "kingHousePrices"
    path = os.path.dirname(os.path.abspath(
        __file__ + str("/../../"))) + "/Data/" + dataset + "/"
    result = xgboost_test(dataset, path)
    print("the XGboost train time is: ", result["time"])
    print("the XGboost train r2 error is: ", result["train_R2"])
    print("the XGboost train rmse is: ", result["train_rmse"])
    print("the XGboost prediction error is: ", result["test_R2"])
    print("the XGboost rmse is: ", result["test_rmse"])


if __name__ == "__main__":
    main()
