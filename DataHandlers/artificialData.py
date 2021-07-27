import numpy as np
from random import gauss
from random import seed
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
from src.DataHandlers.storeData import store
import pickle
import os

l = 32
u = list(range(l))
v = list(range(l))
seed(1)


def b0(loc1, loc2):
    return 3


def b1(loc1, loc2):
    return 1+((1/12)*(loc1+loc2))


def b2(loc1, loc2):
    temp1 = (l/4 - (loc1 / 2)) ** 2
    temp2 = (l/4 - (loc2 / 2)) ** 2
    return 1+((1/((l/2)**2)) * ((l/4)**2 - temp1) * ((l/4)**2 - temp2))


def b3(loc1, loc2):
    return 5+((1/4)*int(loc1/5))


def b4(loc1, loc2):
    temp1 = (l/4 - (loc1 / 2)) ** 2
    temp2 = (l/4 - (loc2 / 2)) ** 2
    return l - ((1 / 500) * ((l/4)**2 - temp1) * ((l/4)**2 - temp2))


visual = np.zeros((l, l, 5))
data = {'coords': [], 'X': [], 'y': []}
coefficients = []

for i in range(l):
    for j in range(l):
        y = 0
        x = []
        for k in range(5):
            x.append(gauss(0, 1))
            visual[i, j, k] = globals()['b'+str(k)](u[i], v[j])
            y += x[k]*visual[i, j, k]
        y += gauss(0, 0.5)
        data['coords'].append((u[i], v[j]))
        data['y'].append(y)
        data['X'].append(x)
        coefficients.append(visual[i, j, :])

y = np.asarray(data['y'])
y = np.reshape(y, [-1, 1])
x = np.asarray(data['X'])
coords = data['coords']
coords = np.asarray(coords)
x = (x - np.mean(x, axis=0)) / np.std(x, axis=0)
n = len(y)

print(len(coefficients), coefficients[0])

fig, axs = plt.subplots(1, 5, figsize=(10, 3))
for i in range(5):
    axs[i].imshow(visual[:, :, i], interpolation='nearest', cmap=cm.Blues)

plt.show()

path = os.path.dirname(os.path.abspath(
    __file__ + str("/../../"))) + "/Data/artificialData/"
store(x, y, coords, path)

# try:
#     os.mkdir(path)
# except OSError:
#     print("Creation of the directory %s failed" % path)
# else:
#     print("Successfully created the directory %s " % path)

# with open(path + '/training_idx.data', 'wb') as filehandle:
#     pickle.dump(training_idx, filehandle)
# with open(path + '/validation_idx.data', 'wb') as filehandle:
#     pickle.dump(validation_idx, filehandle)
# with open(path + '/test_idx.data', 'wb') as filehandle:
#     pickle.dump(test_idx, filehandle)
# with open(path + '/x.data', 'wb') as filehandle:
#     pickle.dump(x, filehandle)
# with open(path + '/y.data', 'wb') as filehandle:
#     pickle.dump(y, filehandle)
# with open(path + '/coords.data', 'wb') as filehandle:
#     pickle.dump(coords, filehandle)
with open(path + '/_generated_data', 'wb') as file_handle:
    pickle.dump(data, file_handle)
with open(path + '/coefficients.data', 'wb') as file_handle:
    pickle.dump(coefficients, file_handle)
