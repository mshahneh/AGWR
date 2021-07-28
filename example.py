# importing the libraries
import geopandas as gpd
import json
from bokeh.io import show
from bokeh.models import (CDSView, ColorBar, ColumnDataSource,
                          CustomJS, CustomJSFilter,
                          GeoJSONDataSource, HoverTool,
                          LinearColorMapper, Slider)
from bokeh.layouts import column, row, widgetbox
from bokeh.palettes import brewer
from bokeh.plotting import figure
from bokeh.palettes import Spectral4
import pandas as pd
import numpy as np
import pickle
import os
import random
import os
import pickle
from SMGWR.SMGWRModel import SMGWRModel
from ModularFramework.ModularFramework import ModularFramework
from mgwr.gwr import GWR, MGWR
from mgwr.sel_bw import Sel_BW
import utils as utils
import warnings
import time
from SMGWR.Hyperband import Hyperband
from sklearn.ensemble import RandomForestRegressor
import numpy as np
from SMGWR.SpaceSearch import SpaceSearch
import multiprocessing
from helpers.SampleModules.SpatialModules import gwr_module, mgwr_module, smgwr_module
from helpers.SampleModules.MLModules import random_forrest
from ModularFramework.ModularFramework import ModularFramework


# read input
def read_input(dataset, validation=False):
    """
    reads the data
    :param dataset: dataset name to read
    :param validation: if true, returns the validation set as well
    :return: returns X, coords, and y for train, (validation), and test
    """
    path = os.path.dirname(os.path.abspath(__file__)) + "/Data/" + dataset + "/"

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

    if not validation:
        X_combined = np.concatenate((X_validation, X_training), axis=0)
        coords_combined = np.concatenate((coords_validation, coords_training), axis=0)
        y_combined = np.concatenate((y_validation, y_training), axis=0)

        return X_combined, coords_combined, y_combined, X_test, coords_test, y_test
    else:
        return X_training, coords_training, y_training, X_validation, coords_validation, y_validation, X_test, coords_test, y_test


# create modules and their setting
def module_selection(spatial, ml):
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

    if ml == "RF":
        ml_module = random_forrest
    else:
        ml_module = None

    return spatial_module, ml_module


def main():
    # the dataset name
    dataset = "syntheticData1"  # "kingHousePrices"

    # reading the dataset values
    X_training, coords_training, y_training, X_test, coords_test, y_test = read_input(dataset, False)

    # selecting spatial and ml modules
    spatial_module, ML_module = module_selection("GWR", "RF")

    # creating the A-GWR setting
    A_GWR_config = {"process_count": 4, "divide_method": "equalCount",
                    "divide_sections": [1, 2], "pipelined": True}  # a-gwr configurations
    A_GWR = ModularFramework(spatial_module, ML_module, A_GWR_config)

    # train model
    A_GWR.train(X_training, coords_training, y_training)

    # predict and print result
    pred = A_GWR.predict(X_test, coords_test, y_test)
    test_r2 = utils.R2(y_test.reshape(-1).tolist(), pred)
    print(test_r2)


if __name__ == "__main__":
    main()
