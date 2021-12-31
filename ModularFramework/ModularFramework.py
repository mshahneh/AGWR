from SMGWR.SMGWRModel import SMGWRModel
from ModularFramework.DataDivider import Divider
from mgwr.gwr import GWR, GWRResults
from multiprocessing import Manager
from functools import partial
import utils as utils
import multiprocessing
import numpy as np
import time

'''
config:
    string divide_method: [no_divide, random, kmeans, ]
    array divide_sections: 2, [3, 4]
    bool pipelined
'''


class ModularFramework:
    """
    implementation of the modular framework
    """
    def __init__(self, spatial_model=None, fast_model=None, config={}):
        default_config = {"overlap": 0}
        for key in config.keys():
            default_config[key] = config[key]
        
        config = default_config
        
        if "divide_sections" not in config:
            config["divide_sections"] = [1]
            config["divide_method"] = "no_divide"
        elif "divide_method" not in config:
            config["divide_method"] = "random"

        if isinstance(config["divide_sections"], int):
            config["divide_sections"] = [config["divide_sections"]]
        
        self.config = config

        self.spatial_model = spatial_model

        if isinstance(fast_model, list):
            self.fast_model = fast_model
            self.return_list_predict = True
        else:
            self.return_list_predict = False
            self.fast_model = [fast_model]
        self.divider = None
        self.spatial_learners = None
        self.ml_learners = None
        if config["process_count"] is not None:
            self.process_count = config["process_count"]
        else:
            self.process_count = 1
        self.train_time = 0

    def pipeline(self, i, x_train, coords_train, y_train, learned_bandwidths):
        """
        performs the pipeline model
        if the learned_bandwidths is passed to the function, it will be used during the training to reduce the training time
        :param i: index of the section
        :param x_train: training features
        :param coords_train: training coordinates
        :param y_train: training dependant
        :param learned_bandwidths: bandwidths learned by a previous section
        :return: returns the model learned for the section
        """
        start = time.time()
        sm_indices = self.divider.section_indices[i]
        sm_x = x_train[sm_indices]
        sm_coords = coords_train[sm_indices]
        sm_y = y_train[sm_indices]
        if len(learned_bandwidths) > 1:
            sm_learner = self.spatial_model(sm_x, sm_coords, sm_y, learned_bandwidths)
        else:
            sm_learner = self.spatial_model(sm_x, sm_coords, sm_y)

        self.spatial_learners[i] = sm_learner
        if isinstance(sm_learner, GWR): #a copy is required due to a problem in mgwr/gwr library implementation
            sm_learner = self.spatial_model(sm_x, sm_coords, sm_y)

        temp_pred = sm_learner.predict(sm_coords, sm_x)
        prediction_array = self._get_predict_array(temp_pred)
        if len(learned_bandwidths) < 1 and isinstance(sm_learner, SMGWRModel):
            for _ in sm_learner.bandwidths:
                learned_bandwidths.append(_)

        return prediction_array.reshape((-1, 1)), self.spatial_learners[i]

    def train(self, x_train, coords_train, y_train):
        """
        based on the instance configuration, performs pipeline or ensemble
        :return: returns the trained model
        """
        train_time = time.time()
        self.divider = Divider(self.config["divide_sections"], self.config["divide_method"], self.config["overlap"])
        self.divider.divide(coords_train)
        numberOfLearners = len(self.divider.section_indices)
        self.spatial_learners = [None] * numberOfLearners
        if "pipelined" not in self.config.keys() or not self.config["pipelined"]:  # performs ensemble
            if (len(self.fast_model) > 1):
                raise "multi ml model not supported for the ensumble architecture"
            self.ml_learners = [None] * numberOfLearners
            learned_bandwidths = []
            ggwr_flag = False
            for i in range(numberOfLearners):
                start = time.time()
                fm_indices = []
                if numberOfLearners != 1:
                    for j in range(numberOfLearners):
                        if j != i:
                            fm_indices.extend(self.divider.section_indices[j])
                else:
                    fm_indices.extend(self.divider.section_indices[0])
                sm_indices = self.divider.section_indices[i]

                fm_x = x_train[fm_indices]
                fm_y = y_train[fm_indices]
                fm_coords = coords_train[fm_indices]
                fm_learner = self.fast_model[0](fm_x, fm_coords, fm_y)

                sm_x = x_train[sm_indices]
                sm_coords = coords_train[sm_indices]
                D_test = np.concatenate((sm_x, sm_coords), axis=1)
                fm_pred = fm_learner.predict(D_test).reshape((-1, 1))

                sm_y = y_train[sm_indices] - fm_pred
                if ggwr_flag:
                    sm_learner = self.spatial_model(sm_x, sm_coords, sm_y, learned_bandwidths)
                else:
                    sm_learner = self.spatial_model(sm_x, sm_coords, sm_y)
                if isinstance(sm_learner, SMGWRModel):
                    ggwr_flag = True
                    learned_bandwidths = sm_learner.bandwidths
                self.spatial_learners[i] = sm_learner
                self.ml_learners[i] = fm_learner

        else: #performs pipeline
            self.ml_learners = [None] * len(self.fast_model)
            learned_bandwidths = []
            ggwr_flag = False
            sm_preds = np.zeros(len(y_train)).reshape((-1, 1))
            listOfLearners = list(range(numberOfLearners))

            numOfThreads = min(self.process_count, numberOfLearners)  # number of processes to create
            manager = Manager()  # using manager to communicate between processes
            learned_bandwidths = manager.list([])
            with multiprocessing.Pool(numOfThreads) as p:
                prediction_array = p.map(partial(self.pipeline, x_train=x_train, coords_train=coords_train,
                                                 y_train=y_train, learned_bandwidths=learned_bandwidths), listOfLearners)
            p.close()
            for i in range(numberOfLearners):
                self.spatial_learners[i] = prediction_array[i][1]
                sm_indices = self.divider.section_indices[i]
                sm_preds[sm_indices] = prediction_array[i][0]
            
            for j in range(len(self.fast_model)):
                if self.fast_model[j] is not None:
                    fm_learner = self.fast_model[j](x_train, coords_train, y_train - sm_preds)
                else:
                    fm_learner = None
                self.ml_learners[j] = fm_learner

        self.train_time = time.time() - train_time

    def predict(self, x_test, coords_test, setting=None):
        """
        predicts the error for the test data
        :param x_test: features of test data
        :param coords_test: coordinates of the test data
        :param setting: setting of the prediction
        :return:
        """
        numberOfLearners = len(self.divider.section_indices)
        resys = np.zeros((len(x_test), numberOfLearners))
        weights = self.divider.predict_weight(coords_test, False)

        if "pipelined" not in self.config.keys() or not self.config["pipelined"]:
            for i in range(numberOfLearners):
                indices = np.where(weights[:, i] > 0)[0]
                selected_coords = coords_test[indices, :]
                selected_X = x_test[indices, :]

                D_test = np.concatenate((selected_X, selected_coords), axis=1)
                sm_results = self.spatial_learners[i].predict(selected_coords, selected_X)
                fm_results = self.ml_learners[i].predict(D_test).reshape((-1, 1))
                prediction_array = self._get_predict_array(sm_results)
                resys[indices, i] = (prediction_array.reshape((-1, 1)) + fm_results).reshape(-1)
            temp = np.sum(resys*weights, axis=1)
            return temp
        else:
            sm_results = np.zeros((x_test.shape[0], numberOfLearners))
            for i in range(numberOfLearners):
                indices = np.where(weights[:, i] > 0)[0]
                selected_coords = coords_test[indices, :]
                selected_X = x_test[indices, :]
                prediction_array = self._get_predict_array(
                    self.spatial_learners[i].predict(selected_coords, selected_X))
                sm_results[indices, i] = prediction_array.reshape(-1)
                # sm_results[:, i] = self.learners[i]["sm"].predict(coords_test, x_test).reshape(-1)
            sm_results = sm_results * weights
            sm_results = np.sum(sm_results, axis=1).reshape((-1, 1))

            D_test = np.concatenate((x_test, coords_test), axis=1)
            res_array = []
            for j in range(len(self.fast_model)):
                if self.ml_learners[j] is not None:
                    fm_results = self.ml_learners[j].predict(D_test).reshape((-1, 1))
                    resy = sm_results + fm_results
                else:
                    resy = sm_results
                res_array.append(resy)
            if not self.return_list_predict:
                return res_array[0]
            else:
                return res_array

    def _get_predict_array(self, prediction):
        """
        if the model is GWR, the prediction values are actually inside prediction.predy
        """
        try:
            return prediction.predy
        except:
            return prediction
