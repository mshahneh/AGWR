import sys
from scipy import linalg
import numpy as np
import utils as utils
import time
import random
from mgwr.gwr import GWR, MGWR
from mgwr.sel_bw import Sel_BW
from numpy.linalg import inv
from SMGWR.Cache import Cache
from utils import local_dist, kernel_funcs, alt, calculate_dependent


class SMGWRModel:
    """
        The class for the S-MGWR
    """

    def __init__(self, X_training, y_training, coords_training, config={}):
        """
        :param X_training: training features
        :param y_training:  training values
        :param coords_training:  training coordinates
        :param config: S-MGWR setting
            - cutoff indicates the cutoff value. If -1, there would be no cutoff
            - kernel function indicates the weight function applied to points around regression point
            - spherical indicates whether the coordinates are longitude and latitude or not
            - fixed indicates whether the bandwdith is adaptive or fixed
            - enable_cache indicates enable_cache to use cache or not
        """

        self.config = {"cutoff": -1, "kernel_function": 'gaussian', "eps": 1.0000001,
                       "spherical": True, "fixed": False, "enable_cache": True, "validation": "normal"}
        for key in config.keys():
            self.config[key] = config[key]

        bw = Sel_BW(coords_training, y_training, X_training, kernel='gaussian').search(criterion='CV')
        gwrModel = GWR(coords_training, y_training, X_training, bw=bw, fixed=False,
                       kernel='gaussian', spherical=True)
        gwrModelFit = gwrModel.fit()

        X_training = np.insert(X_training, 0, 1, axis=1)

        self.X_training = X_training
        self.y_training = y_training
        self.coords_training = coords_training
        self.train_len = self.X_training.shape[0]
        self.numberOfFeatures = self.X_training.shape[1]

        self.trainB = gwrModelFit.params
        pred = np.sum(self.trainB * X_training, axis=1).reshape((-1, 1))
        self.y_residuals = self.y_training - pred
        self.defaultB = np.copy(self.trainB)
        self.default_y_residuals = np.copy(self.y_residuals)
        self.bandwidths = [bw for _ in range(X_training.shape[1])]

        self.distances = np.zeros((self.train_len, self.train_len))
        for i in range(len(self.coords_training)):
            self.distances[i, :] = local_dist(
                self.coords_training[i], self.coords_training, self.config["spherical"]).reshape(-1)

        if self.config["enable_cache"]:
            self.cache = Cache(self.train_len, self.train_len)

    def _select_training(self, index, wi, indices):
        """
        selecting all the non zero weights
        :param index: index of the item, remove for cross validation
        :param wi: weights related to i-th observation
        :return: selected training, selected dependent, selected weights
        """
        if self.config["cutoff"] != -1:
            nonzero = list(np.where(~np.all(wi == 0, axis=1))[0])
            intersect = np.intersect1d(nonzero, indices)
            return intersect
        else:
            return indices

    def _build_wi(self, index, bandwidth, data, indices):
        """
        builds the weight matrix for regression point
        :param index: index of the regression point
        :param bandwidth: bandwidth value for the weight kernel
        :param data: rest of the values
        :param indices: indices to consider (the weight matrix is only build for these indices)
        :return: returns the weight matrix built around regression point for points indicated in indices
        """
        flag = False
        if self.config["enable_cache"] and type(data) == str:
            flag, values = self.cache.isHit(bandwidth, index)
        if flag:
            weights = values
        else:
            weights = np.zeros(self.train_len)
            if self.config["fixed"]:
                bw = float(bandwidth)
            else:
                pilot = min(int(bandwidth), len(indices) - 1)
                if type(data) == str and data == "training_data":
                    bw = np.partition(self.distances[index][indices], pilot - 1)[pilot - 1] * self.config["eps"]
                else:
                    dist = local_dist(data, self.coords_training[indices], self.config["spherical"]).reshape(-1)
                    bw = np.partition(dist, pilot - 1)[pilot - 1] * self.config["eps"]
            bw = max(bw, 1)

            if type(data) == str and data == "training_data":
                weights[indices] = kernel_funcs(self.distances[index][indices] / bw, self.config["kernel_function"])
            else:
                weights[indices] = kernel_funcs(dist[indices] / bw, self.config["kernel_function"])

            if self.config["cutoff"] > 0:
                pilot = int(bandwidth * self.config["cutoff"])
                if pilot < self.train_len:
                    idx = np.argpartition(weights, -pilot)
                    weights[idx[:-pilot]] = 0

            if self.config["enable_cache"] and type(data) == str and data == "training_data":
                self.cache.update(bandwidth, index, weights)

        return weights

    def _local_fit(self, i, j, bw, data, indices):
        """
        fits a weighted regression around the regression point for specific feature
        :param i: regression point index
        :param j: feature index
        :param bw: bandwidth
        :param data: rest of the points
        :param indices: indices to consider
        :return: returns the coefficient of the fitted model
        """
        wi = self._build_wi(i, bw[j], data, indices)
        wi = wi.reshape((-1, 1))
        indices = self._select_training(i, wi, indices)
        x_in = self.X_training[indices, :]
        y_in = self.y_training[indices, :]
        w_in = wi[indices, :]
        position = -1
        x = x_in[:, j].reshape((-1, 1))

        if self.y_residuals is None:
            temp = (x_in * self.trainB)
            temp = np.delete(temp, j, 1)
            pred = np.sum(temp, axis=1).reshape((-1, 1))
            y_in = y_in - pred
        else:
            if position == -1:
                y_in = self.y_residuals[indices] + ((x_in[:, j] * self.trainB[indices, j]).reshape((-1, 1)))
            else:
                new_ind = np.delete(indices, position)
                y_in = self.y_residuals[new_ind] + ((x_in[:, j] * self.trainB[indices, j]).reshape((-1, 1)))
        temp = np.transpose(x*w_in)
        left = np.dot(temp, x)
        right = np.dot(temp, y_in)
        if left == 0:
            return 0
        return inv(left)*right

    def fit(self, bw, indices=[], iterations=3):
        """
        fit a model for a specific bandwdith
        :param bw: bandwdith
        :param indices: indices to consider
        :param iterations: number of iterations the adjustment of coefficients is repeated
        :return: the error for the fitted model of the specific bandwidth
        """
        if len(indices) == 0:
            indices = list(range(self.train_len))
        indices = np.asarray(indices)

        self.trainB[indices] = np.copy(self.defaultB[indices])
        self.y_residuals[indices] = np.copy(self.default_y_residuals[indices])

        convergence_it = 0
        old_pred = 10
        new_pred = utils.R2(self.y_training[indices],
                            utils.calculate_dependent(self.trainB[indices, :], self.X_training[indices, :]))

        while convergence_it < iterations and (new_pred / old_pred) < 0.999:
            convergence_it += 1
            old_pred = new_pred
            for j in range(self.numberOfFeatures):
                for i in range(len(indices)):
                    self.trainB[indices[i], j] = self._local_fit(indices[i], j, bw, "training_data", indices)
                    if self.y_residuals is not None:
                        self.y_residuals[indices[i]] = self.y_training[indices[i]] - np.sum(
                            self.trainB[indices[i], :] * self.X_training[indices[i], :])

            new_pred = utils.R2(self.y_training[indices], self.y_training[indices] + self.y_residuals[indices])

    def setB(self, B):
        self.trainB = B
        pred = np.sum(self.trainB * self.X_training, axis=1).reshape((-1, 1))
        self.y_residuals = self.y_training - pred

        self.defaultB = np.copy(self.trainB)
        self.default_y_residuals = np.copy(self.y_residuals)

    def setB_ind(self, B, indices):
        self.trainB[indices, :] = B
        pred = np.sum(
            self.trainB[indices, :] * self.X_training[indices, :], axis=1).reshape((-1, 1))
        self.y_residuals[indices] = self.y_training[indices] - pred

        self.defaultB[indices, :] = np.copy(self.trainB[indices, :])
        self.default_y_residuals = np.copy(self.y_residuals[indices, :])

    def validate(self, state, validation_indices, train_indices):
        """
        predicts the values for the state based on the validation method and indices
        :param state: the state to be evaluated
        :return: returns the prediction values for the validation set
        """
        coefficients = np.zeros((len(validation_indices), len(self.bandwidths)))

        for j in range(self.numberOfFeatures):
            for i in range(len(validation_indices)):
                coefficients[i, j] = self._local_fit(
                    validation_indices[i], j, state, "training_data", train_indices)

        predictions = calculate_dependent(coefficients, self.X_training[validation_indices, :])
        return predictions

    def predict(self, test_coords, test_x):
        """
        :return: prediction of the values based on the model and bandwidths.
        """
        test_x = np.insert(test_x, 0, 1, axis=1)
        if self.config["enable_cache"]:
            self.cache.clear()
        indices = np.asarray(list(range(self.train_len)))
        coefficients = np.zeros((len(test_x), len(self.bandwidths)))
        test_indices = np.arange(len(test_coords))
        for j in range(self.numberOfFeatures):
            for i in range(len(test_coords)):
                coefficients[i, j] = self._local_fit(
                    i, j, self.bandwidths, test_coords[i], indices)

        predictions = calculate_dependent(coefficients, test_x)
        if self.config["enable_cache"]:
            self.cache.clear()
        return predictions
