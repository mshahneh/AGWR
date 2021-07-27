import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
import pandas as pd
import numpy as np
import random
import pickle
import os
from storeData import store

path = "./Data/FORMGMT"
data = pd.read_csv(path + "/FORMGMT.csv")

print(data.keys())

u = data['Longi']
v = data['Lat']
coords = list(zip(u, v))
coords = np.asanyarray(coords)

y = data['Y'].values.reshape((-1, 1))

data.to_csv(path + "/data.csv")

data = data.drop(columns=['Y', 'Longi', 'Lat'])
print(data.shape)

x = data.values

store(x, y, coords, path)
