import sys
from scipy import linalg
import numpy as np
import src.utils as utils
import time
import random
from src.Cache import Cache
from src.utils import local_dist, kernel_funcs, alt, calculate_dependent


class Backfitting:
    """
        The class for the GGWR_Model
    """
    bandwidths = None

    def __init__(self, X_training,  y_training, coords_training, config={}):
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

        self.distances = np.zeros((self.train_len, self.train_len))
        for i in range(len(self.coords_training)):
            self.distances[i, :] = local_dist(
                self.coords_training[i], self.coords_training, self.config["spherical"]).reshape(-1)

        self.trainB = np.zeros(X_training.shape)

        if self.config["enable_cache"]:
            self.cache = Cache(self.train_len, self.train_len)  # FIX IT

    def _select_training(self, wi, indices):
        """
        selecting all the non zero weights
        :param wi: weights related to i-th observation
        :return: selected training, selected dependent, selected weights
        """
        return self.X_training[indices, :], self.y_training[indices, :], wi[indices]

    def _build_wi(self, index, bandwidth, data, indices):
        weights = np.zeros(len(indices))
        if self.config["fixed"]:
            bw = float(bandwidth)
        else:
            pilot = min(int(bandwidth), len(indices) - 1)
            if type(data) == str:
                if data == "training_data":
                    bw = np.partition(
                        self.distances[index][indices], pilot - 1)[pilot - 1] * self.config["eps"]
            else:
                dist = local_dist(
                    data, self.coords_training[indices], self.config["spherical"]).reshape(-1)
                bw = np.partition(
                    dist, pilot - 1)[pilot - 1] * self.config["eps"]

        flag = False
        if self.config["enable_cache"] and type(data) == str:
            flag, values = self.cache.isHit(bandwidth, index)
        if flag:
            weights = values
        else:
            bw = max(bw, 1)
            for i in range(len(indices)):
                if type(data) == str and data == "training_data":
                    weights[i] = kernel_funcs(
                        self.distances[index][i] / bw, self.config["kernel_function"])
                else:
                    weights[i] = kernel_funcs(
                        dist[i] / bw, self.config["kernel_function"])

            if self.config["cutoff"] > 0:
                pilot = int(bandwidth * self.config["cutoff"])
                if pilot < len(indices):
                    idx = np.argpartition(weights, -pilot)
                    weights[idx[:-pilot]] = 0

            if self.config["enable_cache"]:
                self.cache.update(bandwidth, index, weights)

        return weights

    def _local_fit(self, i, j, bw, data, indices):
        # if self.trainB is None:
        #     importance = np.ones((1, self.numberOfFeatures))
        # else:
        #     importance = self.trainB[i]*self.X_training[i]

        wi = self._build_wi(i, bw[j], data, indices)
        wi = wi.reshape((-1, 1))
        x_in, y_in, w_in = self._select_training(wi, indices)
        temp = (x_in*self.trainB)
        temp = np.delete(temp, j, 1)
        pred = np.sum(temp, axis=1).reshape((-1, 1))
        # print(temp.shape, y_in.shape, pred.shape)
        # if j > 0:
        #     print("y: ", np.mean(y_in), end=" ")
        y_in = y_in - pred
        # if j > 0:
        #     print(np.mean(y_in), np.mean(pred))
        # print(np.mean(self.y_training))
        x = x_in[:, j].reshape((-1, 1))
        # if j > 0:
        #     print(i, "x: ", self.X_training[1:3, j], np.mean(x), np.mean(self.trainB[:, j]), np.mean(self.trainB[:, j]*x_in[:, j]))
        # print(x, self.X_training)
        # print(x.shape, y_in.shape)
        y = wi*y_in
        x = wi*x
        x_mean = np.mean(x)
        y_mean = np.mean(y)
        x_res = x - x_mean
        y_res = y - y_mean
        x_res2 = np.sum(x_res**2)
        # print(self.trainB[i, j], np.sum(x_res*y_res), np.sum(x_res**2))

        if x_res2 == 0:
            return y_mean
        else:
            # a = (np.sum(x_res * y_res) / x_res2 )
            # print("res:", a*x[i], y[i], x[i]*self.trainB[i, j])
            # print(np.sum(x_res * y_res), x_res2, "\n")
            return np.sum(x_res * y_res) / x_res2

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

        convergence = 0
        while convergence < 5:
            convergence += 1
            print(convergence, np.mean(self.trainB, axis=0))
            pred = np.sum(self.trainB * self.X_training, axis=1)
            print("ggwr mean of", utils.R2(self.y_training, pred))
            for j in range(self.numberOfFeatures):
                for i in range(len(validation_indices)):
                    self.trainB[validation_indices[i], j] = self._local_fit(
                        validation_indices[i], j, bw, "training_data", training_indices)

            new_pred = utils.R2(self.y_training[validation_indices],
                                utils.calculate_dependent(self.trainB[validation_indices, :],
                                                          self.X_training[validation_indices, :]))
            print(new_pred)
        return self.trainB

    def setB(self, B):
        self.trainB = B

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

        convergence = 0
        while convergence < 5:
            convergence += 1
            # print(convergence, np.mean(self.trainB, axis=0))
            # pred = np.sum(self.trainB * self.X_training, axis=1)
            # print("ggwr mean of", utils.R2(self.y_training, pred))
            for j in range(self.numberOfFeatures):
                for i in range(len(test_coords)):
                    coefficients[i, j] = self._local_fit(
                        i, j, self.bandwidths, test_coords[i], indices)

        predictions = calculate_dependent(coefficients, test_x)
        self.config["enable_cache"] = temp_cache_status
        return predictions
