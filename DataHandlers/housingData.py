import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
import pandas as pd
import numpy as np
import pickle
import os

data = pd.read_csv("../../Data/housePrices/housing.csv")

print(data.shape, data.keys(), data['ocean_proximity'].unique())

y = data['median_house_value'].values.reshape((-1, 1))
categories = ['housing_median_age', 'total_rooms', 'total_bedrooms',
          'population', 'households', 'median_income']

nan_indexes = []
for category in categories:
    nan_indexes.extend(np.where(np.isnan(data[category]))[0])

nan_indexes = np.asarray(nan_indexes)
nan_indexes = np.unique(nan_indexes)

x = data[['housing_median_age', 'total_rooms', 'total_bedrooms',
          'population', 'households', 'median_income']].values

oceanValues = ['NEAR BAY', '<1H OCEAN', 'INLAND', 'NEAR OCEAN', 'ISLAND']
oceanColumns = np.zeros((len(y), len(oceanValues)))

for index, row in data.iterrows():
    for j in range(len(oceanValues)):
        if row['ocean_proximity'] == oceanValues[j]:
            oceanColumns[index, j] = 1


# x = np.concatenate((x, oceanColumns), axis=1)

print(y.shape, x.shape)
print(x[0:5, :])

u = data['longitude']
v = data['latitude']
coords = list(zip(u, v))
coords = np.asanyarray(coords)

x = np.delete(x, nan_indexes, 0)
y = np.delete(y, nan_indexes, 0)
coords = np.delete(coords, nan_indexes, 0)
print(y.shape, x.shape)
print(y[0:10], max(y), min(y))

x = x[0:3000]
y = y[0:3000]
coords = coords[0:3000]

x = (x - np.mean(x, axis=0)) / np.std(x, axis=0)

n = len(y)
x = np.insert(x, 0, 1, axis=1)

indices = np.random.permutation(n)
testLen = int(0.2 * n)
validationLen = int(0.16 * n)
trainLen = n - testLen - validationLen
training_idx, validation_idx, test_idx = indices[:trainLen], indices[trainLen:trainLen + validationLen],\
                                         indices[trainLen + validationLen:]
path = "../../Data/housePrices"

try:
    os.mkdir(path)
except OSError:
    print("Creation of the directory %s failed" % path)
else:
    print("Successfully created the directory %s " % path)

with open(path + '/training_idx.data', 'wb') as filehandle:
    pickle.dump(training_idx, filehandle)
with open(path + '/validation_idx.data', 'wb') as filehandle:
    pickle.dump(validation_idx, filehandle)
with open(path + '/test_idx.data', 'wb') as filehandle:
    pickle.dump(test_idx, filehandle)
with open(path + '/x.data', 'wb') as filehandle:
    pickle.dump(x, filehandle)
with open(path + '/y.data', 'wb') as filehandle:
    pickle.dump(y, filehandle)
with open(path + '/coords.data', 'wb') as filehandle:
    pickle.dump(coords, filehandle)
