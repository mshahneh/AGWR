import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
import pandas as pd
import numpy as np
import random
import pickle
import os
from Data.DataHandlers.storeData import store

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import warnings
warnings.filterwarnings('ignore')

path = os.path.dirname(os.path.abspath(__file__ + str("/../../"))) + "/Data/NYCAirBnb/"
airbnb = pd.read_csv(path + "/AB_NYC_2019.csv")


print(airbnb.shape, airbnb.keys())
airbnb.drop(['name', 'host_id', 'id', 'neighbourhood_group', 'neighbourhood', 'host_name', 'last_review'], axis=1, inplace=True)
categories = airbnb.keys()

print("number of rows with atleast one missing value:", sum(airbnb.apply(lambda x: sum(x.isnull().values), axis = 1)>0))
airbnb.isnull().sum()
airbnb.dropna(how='any', inplace=True)
print('shape', airbnb.shape)

airbnb = pd.concat([airbnb, pd.get_dummies(airbnb['room_type'], prefix='')],axis=1)
airbnb.drop(['room_type'], axis=1, inplace=True)
airbnb = airbnb[np.log1p(airbnb['price']) < 8]
airbnb = airbnb[np.log1p(airbnb['price']) > 3]
airbnb['price'] = np.log1p(airbnb['price'])
print('shape', airbnb.shape)

u = airbnb['latitude']
v = airbnb['longitude']
coords = list(zip(u, v))
coords = np.asanyarray(coords)

y = airbnb['price'].values.reshape((-1, 1))
airbnb.to_csv(path + "/data.csv")
data = airbnb.drop(columns=['price', 'latitude', 'longitude'])

x = data.values
for i in range(1, 6):
    store(x, y, coords, path, i)
