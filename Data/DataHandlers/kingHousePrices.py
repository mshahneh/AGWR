import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
import pandas as pd
import numpy as np
import random
import pickle
import os
from Data.DataHandlers.storeData import store

path = os.path.dirname(os.path.abspath(__file__ + str("/../../"))) + "/Data/kingHousePrices/"
data = pd.read_csv(path + "kc_house_data.csv")
print(data.shape, data.keys())

data = data[data["yr_renovated"] == 0]
data = data[data["waterfront"] == 0]
data = data[data["view"] < 1]
# data = data.loc[~data['Province_State'].isin(['Alaska', 'Hawaii'])]
print(data.shape)


categories = ['price', 'bedrooms', 'bathrooms', 'sqft_living',
              'sqft_lot', 'floors', 'condition', 'grade', 'yr_built',
              'lat', 'long']
data = data[categories]
nan_indexes = []
for category in categories:
    nan_indexes.extend(np.where(np.isnan(data[category]))[0])
nan_indexes = np.asarray(nan_indexes)
nan_indexes = np.unique(nan_indexes)
data = data.drop(data.index[nan_indexes.tolist()])
# indexes = random.sample(range(len(data)), k=2000)
# data = data.iloc[indexes]

print('shape', data.shape)


u = data['lat']
v = data['long']
coords = list(zip(u, v))
coords = np.asanyarray(coords)

y = data['price'].values.reshape((-1, 1))
data.to_csv(path + "/data.csv")
data = data.drop(columns=['price', 'lat', 'long'])

x = data.values
for i in range(1, 6):
    store(x, y, coords, path, i)
