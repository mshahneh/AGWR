import multiprocessing
from mgwr.gwr import GWR, MGWR
from mgwr.sel_bw import Sel_BW
from SMGWR.SMGWRModel import SMGWRModel
from SMGWR.SpaceSearch import SpaceSearch


def gwr_module(_x, _coords, _y, process_count=-1):
    if process_count > 1:
        pool_of_process = multiprocessing.Pool(process_count)
    else:
        pool_of_process = None
    bandwidth = Sel_BW(
        _coords, _y, _x, kernel='gaussian').search(criterion='CV', pool=pool_of_process)
    model = GWR(_coords, _y, _x, bw=bandwidth, fixed=False,
                kernel='gaussian', spherical=True)
    return model


def mgwr_module(_x, _coords, _y, learned_bandwidths=[], process_count=-1):
    if process_count > 1:
        pool_of_process = multiprocessing.Pool(process_count)
    else:
        pool_of_process = None
    selector = Sel_BW(_coords, _y, _x, kernel='gaussian', multi=True)
    selector.search(criterion='CV', multi_bw_min=[2], pool=pool_of_process)
    model = MGWR(_coords, _y, _x, selector, sigma2_v1=True, kernel='gaussian', fixed=False, spherical=True)

    ggwr_model = SMGWRModel(_x, _y, _coords)
    ggwr_model.bandwidths = model.bws
    ggwr_model.fit(ggwr_model.bandwidths, iterations=10)
    return ggwr_model


def smgwr_module(_x, _coords, _y, learned_bandwidths=[], process_count=-1):
    smgwr_model = SMGWRModel(_x, _y, _coords)
    if _x.shape[0] > 500:
        SP = SpaceSearch(smgwr_model, 1)
    else:
        SP = SpaceSearch(smgwr_model, 5)
    if len(learned_bandwidths) < 1 or len(learned_bandwidths) > smgwr_model.numberOfFeatures:
        bandwidths = smgwr_model.bandwidths
        # sim_config = {"steps": 30, "updates": 20, "method": "gaussian_same_all"}
        # bandwidths = SP.simulated_annealing(bandwidths, sim_config)
        # else:
        #     bandwidths = SP.bayesian_optimization(bandwidths, {"random_count": 50, "iter_count": 50})
        if _x.shape[0] > 500:
            bandwidths, _ = SP.successive_halving(256, 64, -1, 4)
            bandwidths = list(bandwidths.values())
            sim_config = {"steps": 10}
            bandwidths = SP.SPSA(bandwidths, sim_config)
        # print(bandwidths)
            try:
                bandwidths = SP.bayesian_optimization(bandwidths, {"is_local": True, "locality_range": 20, "random_count": 25, "iter_count": 40})
            except Exception as inst:
                print("**error**", inst)
        else:
            bandwidths, _ = SP.successive_halving(64, 64, -1, 4)
            bandwidths = list(bandwidths.values())
            if len(bandwidths) <= 6:
                bandwidths = SP.thorough_search(bandwidths, {})
            sim_config = {"steps": 10}
            bandwidths = SP.SPSA(bandwidths, sim_config)
            print('through done')

        print("done global")
        sim_config = {"steps": 80, "updates": 50, "method": "gaussian_one"}
        bandwidths = SP.hill_climbing(bandwidths, sim_config)
        smgwr_model.bandwidths = bandwidths
    else:
        bandwidths = learned_bandwidths
        bandwidths = SP.bayesian_optimization(bandwidths, {"is_local": True, "locality_range": 30,
                                                           "random_count": 35, "iter_count": 50})
        sim_config = {"steps": 55, "updates": 35, "method": "gaussian_one"}
        bandwidths = SP.hill_climbing(bandwidths, sim_config)
        smgwr_model.bandwidths = bandwidths
    smgwr_model.fit(smgwr_model.bandwidths, iterations=10)
    return smgwr_model


