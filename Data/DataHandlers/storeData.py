from sklearn.model_selection import train_test_split
import numpy as np
import pickle
import os
import sys


def store(x, y, coords, path, random_state=None):
    x = (x - np.mean(x, axis=0)) / np.std(x, axis=0)

    n = len(y)
    # x = np.insert(x, 0, 1, axis=1)
    indices = list(range(n))
    training_idx, test_idx = train_test_split(indices, test_size=0.2, random_state=random_state)
    training_idx, validation_idx = train_test_split(training_idx, test_size=0.1, random_state=random_state)

    if random_state is not None:
        seedpath = path + "seed" + str(random_state)
    else:
        seedpath = path

    # print(testLen)
    try:
        os.mkdir(path)
    except OSError:
        print("Creation of the directory %s failed" % path)
    else:
        print("Successfully created the directory %s " % path)
    
    try:
        os.mkdir(seedpath)
    except OSError:
        print("Creation of the directory %s failed" % path)
    else:
        print("Successfully created the directory %s " % path)

    with open(seedpath + '/training_idx.data', 'wb') as filehandle:
        pickle.dump(training_idx, filehandle)
    with open(seedpath + '/validation_idx.data', 'wb') as filehandle:
        pickle.dump(validation_idx, filehandle)
    with open(seedpath + '/test_idx.data', 'wb') as filehandle:
        pickle.dump(test_idx, filehandle)
    with open(path + '/x.data', 'wb') as filehandle:
        pickle.dump(x, filehandle)
    with open(path + '/y.data', 'wb') as filehandle:
        pickle.dump(y, filehandle)
    with open(path + '/coords.data', 'wb') as filehandle:
        pickle.dump(coords, filehandle)
