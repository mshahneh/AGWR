from sklearn.ensemble import RandomForestRegressor
import numpy as np


def random_forrest(_x, _coords, _y, process_count=-1):
    D_train = np.concatenate((_x, _coords), axis=1)
    rg = RandomForestRegressor(n_estimators=60, n_jobs=process_count)
    rg.fit(D_train, _y.reshape(-1, ))
    return rg