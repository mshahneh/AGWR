import numpy as np
from scipy import linalg
import math


def R2(y, predictions):
    y = np.asarray(y).reshape(-1)
    predictions = np.asarray(predictions).reshape(-1)
    mean_y = np.mean(y)
    return np.sum((y-predictions)**2)/np.sum((y - mean_y)**2)


def MAPE(y, predictions):
    size = len(y)
    return (np.sum(np.abs(y - predictions) / y))/size


def MSE(y, predictions):
    y = np.asarray(y).reshape(-1, 1)
    predictions = np.asarray(predictions).reshape(-1, 1)
    size = len(y)
    return (np.sum((y - predictions)**2))/size


def RMSE(y, predictions):
    return np.sqrt(MSE(y, predictions))


def local_dist(coords_i, coords, spherical):
    """
    Compute Haversine (spherical=True) or Euclidean (spherical=False) distance for a local kernel.
    """
    if spherical:
        dLat = np.radians(coords[:, 1] - coords_i[1])
        dLon = np.radians(coords[:, 0] - coords_i[0])
        lat1 = np.radians(coords[:, 1])
        lat2 = np.radians(coords_i[1])
        a = np.sin(
            dLat / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dLon / 2)**2
        c = 2 * np.arcsin(np.sqrt(a))
        R = 6371.0
        return R * c
    else:
        return np.sqrt(np.sum((coords_i - coords)**2, axis=1))


def kernel_funcs(zs, function):
    zs = np.abs(zs)
    # functions follow Anselin and Rey (2010) table 5.4
    if function == 'triangular':
        return 1 - zs
    elif function == 'uniform':
        return np.ones(zs.shape) * 0.5
    elif function == 'quadratic':
        return (3. / 4) * (1 - zs**2)
    elif function == 'quartic':
        return (15. / 16) * (1 - zs**2)**2
    elif function == 'gaussian':
        return np.exp(-0.5 * (zs)**2)
    elif function == 'bisquare':
        return (1 - (zs)**2)**2
    elif function == 'exponential':
        return np.exp(-zs)
    else:
        print('Unsupported kernel function', function)


def alt(wi, importance, type='mean'):
    if type == 'mean':
        return np.sum(wi * importance, axis=1).reshape((-1, 1)) / np.sum(importance)

    if type == "sum":
        return np.sum(wi * importance, axis=1).reshape((-1, 1))


def calculate_dependent(beta, data):
    return np.sum(beta * data, axis=1).reshape((-1, 1))


def get_distance_from_lat_lon_in_km(lat1, lon1, lat2, lon2):
    R = 6371  # Radius of the earth in km
    dLat = deg2rad(lat2-lat1)  # deg2rad below
    dLon = deg2rad(lon2-lon1)
    a = math.sin(dLat/2) * math.sin(dLat/2) + math.cos(deg2rad(lat1)) * math.cos(deg2rad(lat2)) * math.sin(dLon/2) * math.sin(dLon/2)

    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    d = R * c  # Distance in km
    return d


def deg2rad(deg):
    return deg * (math.pi/180)