import random
import numpy as np
import math
from utils import kernel_funcs, local_dist
from sklearn.cluster import KMeans

eps=1e-6


def equal_width(coords, count, axis, overlap=0):
    """
    divides the coords to count many grid on the defined axis (each section has the same width)
    :param coords: coordinates of the points ti divide
    :param count: number of sections to divide the points
    :param axis: if 0 divides the x part which means column wise if 1 divides y which means row wise
    :param overlap: overlap is the ratio of overlap 0 percent means no overlap and 1 means compelete overlap
    :return: returns the sections. Each section has:
        -indices of the points inside it
        -boundries of the section
        -center of the section
    """
    result = [{"indices": [], "boundaries": [], "center": []} for _ in range(count)]

    n = coords.shape[0]
    if count == 0 and overlap > 0:
        overlap = 0
    Min = np.min(coords[:, axis])
    Max = np.max(coords[:, axis])+eps
    chunk = (Max - Min) / count
    for i in range(count):
        section_start = Min + i * chunk
        section_end = Min + (i + 1) * chunk
        result[i]["boundaries"] = [section_start, section_end]
        result[i]["center"] = (section_end + section_start) / 2

        if i == 0:
            section_end = min(Max, section_end+(chunk*overlap))
        elif i == count - 1:
            section_start = max(Min, section_start-(chunk*overlap))
        else:
            section_start = max(Min, section_start-(chunk*overlap/2))
            section_end = min(Max, section_end+(chunk*overlap/2))
        result[i]["indices"] = [x for x in range(n) if section_start <= coords[x][axis] < section_end]
        if i == 0:
            result[i]["boundaries"][0] = -math.inf
        if i == count - 1:
            result[i]["boundaries"][1] = math.inf

    return result


def equal_count(coords, count, axis, overlap=0):
    """ same as equal width but each section has equal number of points in it """
    result = [{"indices": [], "boundaries": [], "center": []} for _ in range(count)]
    if count == 0 and overlap > 0:
        overlap = 0

    n = coords.shape[0]
    sorted_indices = np.argsort(coords[:, axis])
    chunk = int(n / count)
    boundary_start = boundary_end = 0
    for i in range(count):
        section_start = i * chunk
        section_end = (i + 1) * chunk
        if count == 1:
            boundary_start = -math.inf
            boundary_end = math.inf
        elif i == 0:
            boundary_start = -math.inf
            boundary_end = (coords[sorted_indices[section_end]][axis] + coords[sorted_indices[section_end - 1]][axis]) / 2
        elif i == count - 1:
            boundary_start = (coords[sorted_indices[section_start]][axis] + coords[sorted_indices[section_start - 1]][axis]) / 2
            boundary_end = math.inf
        else:
            boundary_start = (coords[sorted_indices[section_start]][axis] + coords[sorted_indices[section_start - 1]][axis]) / 2
            boundary_end = (coords[sorted_indices[section_end]][axis] + coords[sorted_indices[section_end - 1]][axis]) / 2
        
        if i == 0:
            section_end = int(min(n, section_end+(chunk*overlap)))
        elif i == count - 1:
            section_start = int(max(0, section_start-(chunk*overlap)))
        else:
            section_start = int(max(0, section_start-(chunk*overlap/2)))
            section_end = int(min(n, section_end+(chunk*overlap/2)))
        result[i]["indices"] = sorted_indices[section_start:section_end]
        result[i]["center"] = (coords[sorted_indices[section_start]][axis] + coords[sorted_indices[section_end - 1]][axis]) / 2
        result[i]["boundaries"] = [boundary_start, boundary_end]
    return result


class Divider:
    """
    The divider class, divides data into sections based on the indicated method
    """
    def __init__(self, sections, method, overlap=0):
        self.section_indices = []
        self.method = method
        self.sections = sections
        self.overlap = overlap

    def divide(self, coords):
        if self.method == "no_divide":
            return self._no_divide(coords)
        elif self.method == "random_divide":
            return self._random_divide(coords, self.sections[0])
        elif self.method == "equalCount":
            return self._grid_divide(coords, self.sections[0], self.sections[1], "equalCount")
        elif self.method == "equalWidth":
            return self._grid_divide(coords, self.sections[0], self.sections[1], "equalWidth")
        elif self.method == "kmeans":
            return self._kmeans_divide(coords, self.sections[0])
        else:
            print("Error, wrong method: ", self.method)
            return None

    def _no_divide(self, coords):
        self.method = "no_divide"
        n = len(coords)
        self.section_indices = list(range(n))
        return self.section_indices

    def _random_divide(self, coords, count):
        """ takes the coordinates and count, break the coordinates to count many random sets.
        :returns count arrays of indices chosen at random """
        self.section_indices = []
        self.method = None
        self.settings = dict()

        n = len(coords)
        indices = list(range(n))
        random.shuffle(indices)
        batchSize = int(n / count)
        results = [[] for _ in range(count)]
        for i in range(0, count):
            results[i - 1] = indices[i*batchSize:(i+1)*batchSize]

        self.method = "random_divide"
        self.section_indices = results
        return self.section_indices

    def _grid_divide(self, coords, row, column, method="equalCount"):
        """
        divides data into grid
        :param coords: coordinates of the data points
        :param row: number of rows in the final grid
        :param column: number of columns in the final grid
        :param method: equalCount or equalWidth
        :return: returns the array of indices in each section
        """
        self.section_indices = []
        self.method = None
        self.settings = dict()

        coords = np.asarray(coords)
        if method == "equalWidth":
            functionName = "equal_width"
        elif method == "equalCount":
            functionName = "equal_count"
        else:
            print("bad method")
            return None
        results = [[{"indices": [], "boundaries": [], "center":[]} for _ in range(column)] for _ in range(row)]
        row_divides = eval(functionName + "(coords, row, 1, self.overlap)")
        for i in range(row):
            column_divides = eval(functionName + "(coords[row_divides[i]['indices']], column, 0, self.overlap)")
            for j in range(column):
                row_bound = row_divides[i]["boundaries"]
                col_bound = column_divides[j]["boundaries"]
                results[i][j]["boundaries"] = [(col_bound[0], row_bound[0]), (col_bound[1], row_bound[1])]
                results[i][j]["center"] = [column_divides[j]["center"], row_divides[i]["center"]]
                results[i][j]["indices"] = [row_divides[i]["indices"][x] for x in column_divides[j]["indices"]]

        self.settings["rows"] = row
        self.settings["columns"] = column
        self.settings["centers"] = []
        self.settings["boundaries"] = []
        for i in range(row):
            for j in range(column):
                self.section_indices.append(results[i][j]["indices"])
                self.settings["centers"].append(results[i][j]["center"])
                self.settings["boundaries"].append(results[i][j]["boundaries"])

        self.method = method
        return self.section_indices

    def _kmeans_divide(self, coords, clusters):
        """
        performs k-means clustering
        :param coords: coordinates of the points
        :param clusters: number of clusters
        :return: returns the array of indices in each section
        """
        self.section_indices = []
        self.method = None
        self.settings = dict()

        kmeans = KMeans(n_clusters=clusters, random_state=0).fit(coords)
        for i in range(clusters):
            self.section_indices.append(np.where(kmeans.labels_ == i)[0])

        self.method = "kmeans"
        self.settings = kmeans
        return self.section_indices

    def predict_weight(self, points, weighted=False, spherical=True):
        """
        predicts the how much each point belongs to a section
        :param points: test points
        :param weighted: if true, the weights are based on the distance to center of each section.
            -Otherwise, the result is a one-hot vector
        :return: returns the matrix of weights (number of points by number of sections)
        """
        result = np.zeros((len(points), len(self.section_indices)))
        for i in range(len(points)):
            if self.method == "random_divide":
                result[i, :] = 1/len(self.section_indices)
            elif self.method == "kmeans":
                if weighted:
                    result[i, :] = self.weighted_mean(self.settings.cluster_centers_, points[i], spherical=spherical)
                else:
                    j = self.settings.predict(np.asarray(points[i]).reshape(1, -1))[0]
                    result[i][j] = 1
            else:
                if weighted:
                    result[i, :] = self.weighted_mean(self.settings["centers"], points[i], spherical=spherical)
                else:
                    for j in range(len(self.section_indices)):
                        if self.is_inside(points[i], self.settings["boundaries"][j]):
                            result[i][j] = 1
                            break
        return result

    def is_inside(self, coords, boundaries):
        """
        checks if the point is inside boundary or not
        :param coords: coordinate of the point
        :param boundaries: boundaries of the area
        :return: true if the point is inside, false otherwise
        """
        if coords[0] < boundaries[0][0] or coords[0] >= boundaries[1][0]:
            return False
        if coords[1] < boundaries[0][1] or coords[1] >= boundaries[1][1]:
            return False
        return True

    def weighted_mean(self, centers, point, method="gaussian", spherical=False):
        """ :returns: the weights of each point in section based on its distance to the center """
        centers = np.asarray(centers)
        weights = []
        distances = local_dist(point, centers, spherical)
        bw = min(distances)
        for i in range(len(centers)):
            weights.append(kernel_funcs(distances[i]/bw, method))

        weights = weights/sum(weights)
        return weights

# def (point, centers, coords, setting):
