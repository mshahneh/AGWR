import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib
import matplotlib.pyplot as plt
# import seaborn as sns
plt.style.use('fivethirtyeight')
matplotlib.rcParams['font.family'] = "Arial"
#
# from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
# import plotly as py
# import plotly.graph_objs as go
# from plotly.subplots import make_subplots

# init_notebook_mode(connected=True)

import collections
import itertools

import scipy.stats as stats
from scipy.stats import norm
from scipy.special import boxcox1p

import statsmodels
import statsmodels.api as sm
#print(statsmodels.__version__)

from sklearn.preprocessing import scale, StandardScaler, RobustScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, KFold, StratifiedKFold, RandomizedSearchCV
from sklearn.linear_model import Ridge, RidgeCV, Lasso, LassoCV, LinearRegression, ElasticNet,  HuberRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.kernel_ridge import KernelRidge
from sklearn.utils import resample

#Model interpretation modules
# import eli5
# import lime
# import lime.lime_tabular
# import shap
# shap.initjs()
from sklearn.ensemble import RandomForestRegressor
from src.utils import local_dist, kernel_funcs, alt, calculate_dependent
from mgwr.gwr import GWR, MGWR
from mgwr.sel_bw import Sel_BW
import numpy as np
import pickle
import time
import src.utils as utils
from src.DataDivider import Divider
import warnings
warnings.filterwarnings('ignore')

dataset = "artificialData"
path = "../../Data/" + dataset + "/"
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

training_idx = np.concatenate((training_idx, validation_idx))
X_training, X_test = x[training_idx, 1:], x[test_idx, 1:]
y_training, y_test = y[training_idx, :], y[test_idx, :]
coords_training, coords_test = coords[training_idx], coords[test_idx]

D_train = np.concatenate((X_training, coords_training), axis=1)
D_test = np.concatenate((X_test, coords_test), axis=1)

y_training = y_training.reshape(-1)
y_test = y_test.reshape(-1)

n_folds = 5

scaler = RobustScaler()
X_train = scaler.fit_transform(D_train)
X_test = scaler.fit_transform(D_test)


start_time = time.time()
rf = RandomForestRegressor()
n_estimators = [int(x) for x in np.linspace(start = 20, stop=2000,num=10)]
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

rf_random = RandomizedSearchCV(estimator=rf, param_distributions = random_grid, n_iter=10, cv = 3, verbose=2, random_state=42, n_jobs=-1)
rf_random.fit(D_train, y_training)
best_config = rf_random.best_params_
rg = RandomForestRegressor(
    n_estimators=best_config["n_estimators"],
    min_samples_split=best_config["min_samples_split"],
    min_samples_leaf=best_config["min_samples_leaf"],
    max_features=best_config["max_features"],
    max_depth=best_config["max_depth"]
)
rg.fit(D_train, D_test)
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

print(rfr_best_results)