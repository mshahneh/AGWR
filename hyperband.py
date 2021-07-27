import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH
import numpy as np
import math
from src import utils
INF = 10000000


def get_configspace(bandwidths, n):
    cs = CS.ConfigurationSpace()
    bandwidth_spaces = []
    for i in range(bandwidths):
        bandwidth_spaces.append(
            CSH.UniformIntegerHyperparameter('bandwidth' + str(i), lower=5, upper=100, default_value=30, log=False))

    cs.add_hyperparameters(bandwidth_spaces)
    return cs


class Hyperband:

    def __init__(self, ggwr):
        self.ggwr = ggwr
        self.cs = get_configspace(ggwr.numberOfFeatures, ggwr.train_len)

    def compute(self, start_configs_count, min_budget, max_budget, eta):
        smax = math.ceil(math.log(start_configs_count, eta))
        best_error = INF
        for i in range(smax+1):
            n = math.ceil((smax*(eta**i))/max(i, 1))
            config, error = self.successive_halving(n, min(max_budget, min_budget*(eta**(smax-i))), max_budget, eta)
            if error < best_error:
                best_error = error
                best_config = config
            # print(n, best_error, best_config.values())
        return best_config, best_error

    def successive_halving(self, start_configs_count, min_budget, max_budget, eta):
        n = start_configs_count
        configs = [(self.cs.sample_configuration().get_dictionary(), 0) for _ in range(n)]
        budget = min_budget
        while n > 0 and budget <= max_budget:
            for i in range(len(configs)):
                configs[i] = (configs[i][0], self.evaluate(configs[i][0], budget))
            configs.sort(key=lambda tup: tup[1])
            if n == 1 or budget == max_budget:
                break
            n = math.ceil(n/eta)
            configs = configs[0:n]
            budget = min(budget*eta, max_budget)
        return configs[0]

    def evaluate(self, config, budget):
        budget = int(budget)
        bandwth = list(config.values())
        B = self.ggwr.fit(bandwth, "validation", budget)
        valid_pred = utils.calculate_dependent(B, self.ggwr.X_validation)
        error = utils.R2(self.ggwr.y_validation, valid_pred)
        return error
