import pickle
import pandas as pd
import numpy as np
from Data.DataHandlers.storeData import store
import os
from libpysal.examples import explain, get_path, load_example
import libpysal as ps
import pysal as pysal

psDatas = ['pysalGeorgia', 'pysalBerlin', 'pysalTokyo', 'pysalClearwater']


for dataset in psDatas:
    if dataset == "pysalClearwater":
        data = os.path.dirname(os.path.abspath(
            __file__ + str("/../../"))) + "/Data/pysalClearwater/landslides.csv"
        data = pd.read_csv(data)
        coords = list(zip(data['X'], data['Y']))
        coords = [list(item) for item in coords]
        coords = np.asanyarray(coords).astype(float)
        y = np.array(data['Landslid'].values.astype(float)).reshape((-1, 1))
        x = data[["Elev", "Slope", "SinAspct", "CosAspct", "AbsSouth", "DistStrm"]].values.astype(float)
    if dataset == "pysalTokyo":
        tokyo = ps.io.open(get_path('Tokyomortality.csv'))
        data = pd.DataFrame(tokyo.data, columns=tokyo.header)
        coords = list(zip(data['X_CENTROID'], data['Y_CENTROID']))
        coords = [list(item) for item in coords]
        coords = np.asanyarray(coords).astype(float)
        y = np.array(data['db2564'].values.astype(float)).reshape((-1, 1))
        x = data[["eb2564", "OCC_TEC", "OWNH", "POP65", "UNEMP"]].values.astype(float)
        print(x[0])
    if dataset == 'pysalGeorgia':
        data = ps.io.open(ps.examples.get_path('GData_utm.csv'))
        coords = list(zip(data.by_col('X'), data.by_col('Y')))
        coords = [list(item) for item in coords]
        coords = np.asanyarray(coords)
        y = np.array(data.by_col('PctBach')).reshape((-1, 1))
        x = data.by_col_array(["PctBlack", "PctFB", "PctEld", "TotPop90"])
       # x = data.by_col_array(
       #     ["PctRural", "PctEld", "PctFB", "PctPov", "PctBlack"])

    if dataset == "pysalBerlin":
        data = os.path.dirname(os.path.abspath(
            __file__ + str("/../../"))) + "/Data/pysalBerlin/prenzlauer.csv"
        data = pd.read_csv(data)
        y = np.log(data['price'].values.reshape((-1, 1)))
        x = data[['review_scores_rating',
                  'bedrooms',
                  'bathrooms',
                  'beds',
                  'accommodates']].values
        u = data['X']
        v = data['Y']
        coords = list(zip(u, v))
        coords = np.asanyarray(coords)

    path = os.path.dirname(os.path.abspath(
        __file__ + str("/../../"))) + "/Data/" + dataset +"/"
    for i in range(1, 6):
        store(x, y, coords, path, i)
