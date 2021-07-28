import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH
import numpy as np
import math
import utils

INF = 10000000


def get_configspace(bandwidths, n):
    """
    creates the bandwidth search space
    for our experiments, we limited the number of neighbor points between 5 and 100.
        - It can be changed to use n for the upper or lower limits
    :param bandwidths: number of bandwidths
    :param n: number of examples
    :return: returns the search space
    """
    cs = CS.ConfigurationSpace()
    bandwidth_spaces = []
    for i in range(bandwidths):
        bandwidth_spaces.append(
            CSH.UniformIntegerHyperparameter('bandwidth' + str(i), lower=5, upper=100, default_value=30, log=False))

    cs.add_hyperparameters(bandwidth_spaces)
    return cs


class Hyperband:
    """
    Hyperband algorithm for bandwidth search for S-MGWR
    """

    def __init__(self, smgwr):
        """
        :param smgwr: an instance of smgwr model
        """
        self.smgwr = smgwr
        self.cs = get_configspace(smgwr.numberOfFeatures, smgwr.train_len)

    def compute(self, start_configs_count, min_budget, max_budget, eta):
        """
        :param start_configs_count: number of initial configurations
        :param min_budget: minimum budget
        :param max_budget: maximum budget
        :param eta: the eta variable
        :return: returns the best set of bandwidths and its corresponding error
        """
        smax = math.ceil(math.log(start_configs_count, eta))
        best_error = INF
        for i in range(smax + 1):
            n = math.ceil((smax * (eta ** i)) / max(i, 1))
            config, error = self.successive_halving(n, min(max_budget, min_budget * (eta ** (smax - i))), max_budget,
                                                    eta)
            if error < best_error:
                best_error = error
                best_config = config
            # print(n, best_error, best_config.values())
        return best_config, best_error

    def successive_halving(self, start_configs_count, min_budget, max_budget, eta):
        """
        performs the successive halving algorithm (it is also one of the inner steps of the Hyperband)
        :return: returns the best configuration found by successive halving
        """
        n = start_configs_count
        configs = [(self.cs.sample_configuration().get_dictionary(), 0) for _ in range(n)]
        budget = min_budget
        while n > 0 and budget <= max_budget:
            for i in range(len(configs)):
                configs[i] = (configs[i][0], self.evaluate(configs[i][0], budget))
            configs.sort(key=lambda tup: tup[1])
            if n == 1 or budget == max_budget:
                break
            n = math.ceil(n / eta)
            configs = configs[0:n]
            budget = min(budget * eta, max_budget)
        return configs[0]

    def evaluate(self, config, budget):
        """
        evaluates the error for a configuration over a part of training data
        :param config: the bandwidths set to be evaluated
        :param budget: indicates the indexes to run the evaluation on
        :return: returns the error of evaluation
        """
        budget = int(budget)
        bandwth = list(config.values())
        B = self.smgwr.fit(bandwth, "validation", budget)
        valid_pred = utils.calculate_dependent(B, self.smgwr.X_validation)
        error = utils.R2(self.smgwr.y_validation, valid_pred)
        return error
