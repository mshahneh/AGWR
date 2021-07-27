import sys
from scipy import linalg
import numpy as np
import time
import random
from src.Cache import Cache
from src.utils import local_dist, kernel_funcs, alt, calculate_dependent


class GGWR_Model:
    """
        The class for the GGWR_Model
    """
    bandwidths = None

    def  __init__(self, X_training,  y_training, coords_training, config={}):
        """
        Initialize class
        """

        self.config = {"cutoff": 3, "kernel_function": 'gaussian', "eps": 1.0000001,
                       "spherical": True, "fixed": False, "enable_cache": True, "validation": "normal"}
        for key in config.keys():
            self.config[key] = config[key]

        X_training = np.insert(X_training, 0, 1, axis=1)

        self.X_training = X_training
        self.y_training = y_training
        self.coords_training = coords_training
        self.train_len = self.X_training.shape[0]
        self.numberOfFeatures = self.X_training.shape[1]

        # self.spherical = spherical
        # self.fixed = fixed
        # self.cutoff = cutoff
        # self.kernel_function = kernel_function
        # self.eps = 1.0000001

        # if X_validation is not None:
        #     X_validation = np.insert(X_validation, 0, 1, axis=1)
        #     self.X_validation = X_validation
        #     self.y_validation = y_validation
        #     self.coords_validation = coords_validation
        #     self.valid_len = self.X_validation.shape[0]
        #     self.validation_distances = np.zeros((self.valid_len, self.train_len))
        #     for i in range(len(self.coords_validation)):
        #         self.validation_distances[i, :] = local_dist(self.coords_validation[i], self.coords_training,
        #                                                      self.spherical).reshape(-1)

        self.trainB = None

        if self.train_len < 10000:
            self.distances = np.zeros((self.train_len, self.train_len))
            for i in range(len(self.coords_training)):
                self.distances[i, :] = local_dist(self.coords_training[i], self.coords_training, self.config["spherical"]).reshape(-1)
        # self.enable_cache = enable_cache
        # if enable_cache:
        #     self.cache = Cache(self.train_len)
        if self.config["enable_cache"]:
            self.cache = Cache(self.train_len, int(0.2*self.train_len)) ###FIX IT

    def _select_training(self, wi, indices):
        """
        selecting all the non zero weights
        :param wi: weights related to i-th observation
        :return: selected training, selected dependent, selected weights
        """
        nonzero = list(np.where(~np.all(wi == 0, axis=1))[0])
        indices = indices[nonzero]
        return self.X_training[indices, :], self.y_training[indices, :], wi[nonzero, :]

    def _build_wi(self, index, bandwidths, data, indices):
        weights = np.zeros((len(bandwidths), len(indices)))
        for k in range(len(bandwidths)):
            if self.config["fixed"]:
                bw = float(bandwidths[k])
            else:
                pilot = min(int(bandwidths[k]), len(indices) - 1)
                if type(data) == str:
                    if data == "training_data":
                        bw = np.partition(self.distances[index][indices], pilot - 1)[pilot - 1] * self.config["eps"]
                else:
                    dist = local_dist(data, self.coords_training[indices], self.config["spherical"]).reshape(-1)
                    bw = np.partition(dist, pilot - 1)[pilot - 1] * self.config["eps"]

            flag = False
            if self.config["enable_cache"]:
                flag, values = self.cache.isHit(bandwidths[k], index)
            if flag:

                weights[k][:] = values
            else:
                bw = max(bw, self.config["eps"])
                for i in range(len(indices)):
                    if type(data) == str and data == "training_data":
                        weights[k][i] = kernel_funcs(self.distances[index][i] / bw, self.config["kernel_function"])
                    else:
                        weights[k][i] = kernel_funcs(dist[i] / bw, self.config["kernel_function"])

                if self.config["cutoff"] > 0:
                    pilot = int(bandwidths[k] * self.config["cutoff"])
                    if pilot < len(indices):
                        idx = np.argpartition(weights[k, :], -pilot)
                        weights[k][idx[:-pilot]] = 0

                if self.config["enable_cache"]:
                    self.cache.update(bandwidths[k], index, weights[k][:])

        weights = np.transpose(weights)
        return weights

    def _local_fit(self, i, bw, data, indices):
        # if self.trainB is None:
        #     importance = np.ones((1, self.numberOfFeatures))
        # else:
        #     importance = self.trainB[i]*self.X_training[i]

        importance = np.ones((1, self.numberOfFeatures))
        wi = self._build_wi(i, bw, data, indices)
        x_in, y_in, w_in = self._select_training(wi, indices)

        w_in = np.sqrt(w_in)
        A = w_in * x_in
        At = A.T
        b = alt(w_in, importance, 'mean') * y_in
        temp2 = np.matmul(At, b)
        return linalg.solve(np.matmul(At, A), temp2).T

    def fit(self, bw, validation_indices=[], training_indices=[]):
        """
        :return: the fitted model for the bw
        """
        if len(validation_indices) == 0:
            validation_indices = list(range(self.train_len))
        if len(training_indices) == 0:
            training_indices = list(range(self.train_len))
        validation_indices = np.asarray(validation_indices)
        training_indices = np.asarray(training_indices)
        B = np.zeros((len(validation_indices), self.numberOfFeatures))
        for i in range(len(validation_indices)):
            B[i, :] = self._local_fit(validation_indices[i], bw, "training_data", training_indices)
        return B

    def setB(self, B):
        self.trainB = B

    def setB(self, B):
        self.trainB = B

    def setB_ind(self, B, indices):
        self.trainB[indices, :] = B

    def predict(self, test_coords, test_x):
        """
        :return: prediction of the values based on the model and bandwidths.
        """
        test_x = np.insert(test_x, 0, 1, axis=1)
        # print(test_x.shape, len(self.bandwidths))
        temp_cache_status = self.config["enable_cache"]
        self.config["enable_cache"] = False
        indices = np.asarray(list(range(self.train_len)))
        coefficients = np.zeros((len(test_x), len(self.bandwidths)))
        for i in range(len(test_coords)):
            coefficients[i, :] = self._local_fit(i, self.bandwidths, test_coords[i], indices)

        predictions = calculate_dependent(coefficients, test_x)
        self.config["enable_cache"] = temp_cache_status
        return predictions
