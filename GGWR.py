from src.BackfittingModelImproved import ImprovedBackfitting
from src.DataDivider import Divider
from mgwr.gwr import GWR, GWRResults
from multiprocessing import Manager
from functools import partial
import src.utils as utils
import multiprocessing
import numpy as np
import time

'''
config:
    string divide_method: [no_divide, random, kmeans, ]
    array divide_sections: 2, [3, 4]
    bool pipelined
'''

class GGWR:
    def __init__(self, spatial_model=None, fast_model=None, config={}):
    
        default_config = {"overlap":0}
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
        self.fast_model = fast_model
        self.divider = None
        self.learners = None
        if config["process_count"] is not None:
            self.process_count = config["process_count"]
        else:
            self.process_count = 1
        self.train_time = 0

    def pipeline(self, i, x_train, coords_train, y_train, learned_bandwidths):
        start = time.time()
        print("in pipeline, iteration, learned bw:", i, learned_bandwidths, end=" ")
        sm_indices = self.divider.section_indices[i]
        sm_x = x_train[sm_indices]
        sm_coords = coords_train[sm_indices]
        sm_y = y_train[sm_indices]
        if len(learned_bandwidths) > 1:
            sm_learner = self.spatial_model(sm_x, sm_coords, sm_y, learned_bandwidths)
        else:
            sm_learner = self.spatial_model(sm_x, sm_coords, sm_y)

        self.learners[i] = {"sm": sm_learner}
        if isinstance(sm_learner, GWR):
            sm_learner = self.spatial_model(sm_x, sm_coords, sm_y)
        else:
            print(sm_learner.bandwidths, end=" ")

        temp_pred = sm_learner.predict(sm_coords, sm_x)
        prediction_array = self._get_predict_array(temp_pred)

        if len(learned_bandwidths) < 1 and isinstance(sm_learner, ImprovedBackfitting):
            for _ in sm_learner.bandwidths:
                learned_bandwidths.append(_)

        print("section", i, " train took", time.time() - start)
        return prediction_array.reshape((-1, 1)), self.learners[i]

    def train(self, x_train, coords_train, y_train):
        train_time = time.time()
        self.divider = Divider(self.config["divide_sections"], self.config["divide_method"], self.config["overlap"])
        self.divider.divide(coords_train)
        numberOfLearners = len(self.divider.section_indices)
        # print("numberOfLearners", numberOfLearners, self.divider.section_indices[0])
        self.learners = [{} for _ in range(numberOfLearners)]
        if "pipelined" not in self.config.keys() or not self.config["pipelined"]:
            learned_bandwidths = []
            ggwr_flag = False
            for i in range(numberOfLearners):
                start = time.time()
                print(i, end=" ")
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
                fm_learner = self.fast_model(fm_x, fm_coords, fm_y)

                sm_x = x_train[sm_indices]
                sm_coords = coords_train[sm_indices]
                D_test = np.concatenate((sm_x, sm_coords), axis=1)
                fm_pred = fm_learner.predict(D_test).reshape((-1, 1))
                sm_y = y_train[sm_indices] - fm_pred
                if ggwr_flag:
                    sm_learner = self.spatial_model(sm_x, sm_coords, sm_y, learned_bandwidths)
                else:
                    sm_learner = self.spatial_model(sm_x, sm_coords, sm_y)
                if isinstance(sm_learner, ImprovedBackfitting):
                    ggwr_flag = True
                    learned_bandwidths = sm_learner.bandwidths
                self.learners[i] = {"fm": fm_learner, "sm": sm_learner}
                print("this section train took", time.time() - start)

        else:
            learned_bandwidths = []
            ggwr_flag = False
            sm_preds = np.zeros(len(y_train)).reshape((-1, 1))
            listOfLearners = list(range(numberOfLearners))

            numOfThreads = min(self.process_count, numberOfLearners)
            print("numOfThreads:", numOfThreads, "list of learners", listOfLearners)
            # prediction_array = []
            # for i in range(numberOfLearners):
            #     prediction_array.append(self.pipeline(i = i, x_train=x_train, coords_train=coords_train, y_train=y_train))
            manager = Manager()
            learned_bandwidths = manager.list([])
            with multiprocessing.Pool(numOfThreads) as p:
                prediction_array = p.map(partial(self.pipeline, x_train=x_train, coords_train=coords_train,
                                                 y_train=y_train, learned_bandwidths=learned_bandwidths), listOfLearners)
            p.close()
            # for i in range(numberOfLearners):
            #     start = time.time()
            #     print(i, end=" ")
            #     sm_indices = self.divider.section_indices[i]
            #     sm_x = x_train[sm_indices]
            #     sm_coords = coords_train[sm_indices]
            #     sm_y = y_train[sm_indices]
            #
            #     if ggwr_flag:
            #         sm_learner = self.spatial_model(sm_x, sm_coords, sm_y, learned_bandwidths)
            #     else:
            #         sm_learner = self.spatial_model(sm_x, sm_coords, sm_y)
            #
            #     if isinstance(sm_learner, ImprovedBackfitting):
            #         ggwr_flag = True
            #         learned_bandwidths = sm_learner.bandwidths
            #
            #     self.learners[i] = {"sm": sm_learner}
            #     if isinstance(sm_learner, GWR):
            #         sm_learner = self.spatial_model(sm_x, sm_coords, sm_y)
            #     else:
            #         print(sm_learner.bandwidths, end=" ")
            #
            #     temp_pred = sm_learner.predict(sm_coords, sm_x)
            #     prediction_array = self._get_predict_array(temp_pred)
            #     sm_preds[sm_indices] = prediction_array.reshape((-1, 1))
            #     print("this section train took", time.time() - start)

            # print(prediction_array)
            for i in range(numberOfLearners):
                self.learners[i] = prediction_array[i][1]
                sm_indices = self.divider.section_indices[i]
                sm_preds[sm_indices] = prediction_array[i][0]
            fm_learner = self.fast_model(x_train, coords_train, y_train - sm_preds)
            self.learners[0]["fm"] = fm_learner

        self.train_time = time.time() - train_time

    def predict(self, x_test, coords_test, y_test, setting=None):
        numberOfLearners = len(self.divider.section_indices)
        resys = np.zeros((len(x_test), numberOfLearners))
        weights = self.divider.predict_weight(coords_test, False)
        # results = results * weights
        if "pipelined" not in self.config.keys() or not self.config["pipelined"]:
            for i in range(numberOfLearners):
                indices = np.where(weights[:, i] > 0)[0]
                selected_coords = coords_test[indices, :]
                selected_X = x_test[indices, :]

                D_test = np.concatenate((selected_X, selected_coords), axis=1)
                sm_results = self.learners[i]["sm"].predict(selected_coords, selected_X)
                fm_results = self.learners[i]["fm"].predict(D_test).reshape((-1, 1))
                print(i, "fm only", utils.R2(y_test[indices].reshape(-1).tolist(), fm_results), end=" ")
                prediction_array = self._get_predict_array(sm_results)
                # resy = sm_results.reshape((-1, 1)) * weights[:, i].reshape((-1, 1)) + fm_results
                resys[indices, i] = (prediction_array.reshape((-1, 1)) + fm_results).reshape(-1)
                print("fm and sm", utils.R2(y_test[indices], resys[indices, i]))
            temp = np.sum(resys*weights, axis=1)
            return temp
        else:
            sm_results = np.zeros((len(y_test), numberOfLearners))
            for i in range(numberOfLearners):
                indices = np.where(weights[:, i] > 0)[0]
                selected_coords = coords_test[indices, :]
                selected_X = x_test[indices, :]
                prediction_array = self._get_predict_array(
                    self.learners[i]["sm"].predict(selected_coords, selected_X))
                sm_results[indices, i] = prediction_array.reshape(-1)
                # sm_results[:, i] = self.learners[i]["sm"].predict(coords_test, x_test).reshape(-1)
            sm_results = sm_results * weights
            sm_results = np.sum(sm_results, axis=1).reshape((-1, 1))

            D_test = np.concatenate((x_test, coords_test), axis=1)
            fm_results = self.learners[0]["fm"].predict(
                D_test).reshape((-1, 1))
            resy = sm_results + fm_results
            return resy

    def _get_predict_array(self, prediction):
        try:
            return prediction.predy
        except:
            return prediction
