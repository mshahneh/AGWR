import numpy as np
import pickle
import os
import sys


def store(x, y, coords, path):
    x = (x - np.mean(x, axis=0)) / np.std(x, axis=0)

    n = len(y)
    # x = np.insert(x, 0, 1, axis=1)

    indices = np.random.permutation(n)
    testLen = int(0.20 * n)
    validationLen = int(0.10 * n)
    trainLen = n - testLen - validationLen
    training_idx, validation_idx, test_idx = indices[:trainLen], indices[trainLen:trainLen + validationLen], \
                                             indices[trainLen + validationLen:]

    # print(testLen)
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
