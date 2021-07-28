import numpy as np

class Cache:
    """
    A LRU cache module keeps cache_size different sets of n*m values
    """
    def __init__(self, n, m, cache_size=50):
        """
            :param cache_size: number of different matrices to cache (number of bandwidths to keep in memory)
            :param n: number of training points
            :param m: number of values stored for each point
        """
        self.cacheSize = cache_size
        self.n = n
        self.memory = np.zeros((cache_size, n, m))
        self.bandwidthMap = [-1 for i in range(cache_size)]
        self.timestamp = [-1 for i in range(cache_size)]
        self.isValid = np.zeros((cache_size, n))

    def isHit(self, bandwidth, index):
        """
        :param bandwidth: desired bandwidth value
        :param index: index of the regression point
        :return: true if the bandwidth value exists in cache, false otherwise
        """
        loc = -1
        for i in range(self.cacheSize):
            if self.bandwidthMap[i] == bandwidth:
                loc = i
        if loc != -1 and self.isValid[loc, index] == 1:
            return True, self.memory[loc, index]
        else:
            return False, None

    def update(self, bandwidth, index, data):
        """
        updates the cache with data based on the bandwidth and index of the regression point
        :param bandwidth: desired bandwidth value
        :param index: index of the regression point
        :param data: vector to replace with the old value for index and bandwidth
        :return: void function
        """
        Min = 0
        Max = -1
        loc = -1
        for i in range(self.cacheSize):
            if self.timestamp[i] < self.timestamp[Min]:
                Min = i
            Max = max(self.timestamp[i], Max)
            if self.bandwidthMap[i] == bandwidth:
                loc = i

        if loc != -1:
            self.memory[loc, index] = data
            self.isValid[loc, index] = 1
            for i in range(self.cacheSize):
                if self.timestamp[i] > self.timestamp[loc]:
                    self.timestamp[i] -= 1
            self.timestamp[loc] = Max
        else:
            self.bandwidthMap[Min] = bandwidth
            for i in range(self.cacheSize):
                self.timestamp[i] -= 1
            self.timestamp[Min] = Max
            self.memory[Min, index] = data
            for i in range(self.n):
                self.isValid[Min, i] = 0
            self.isValid[Min, index] = 1

    def clear(self):
        """
        clears the cache
        :return: void function
        """
        self.bandwidthMap = [-1 for i in range(self.cacheSize)]

