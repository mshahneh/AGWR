#FastMGWR MPI Script
#Author: Ziqi Li
#Email: liziqi1992@gmail.com

import math
import numpy as np
from mpi4py import MPI
from scipy.spatial.distance import cdist,pdist
import argparse
from copy import deepcopy
from FastGWR import FastGWR


class FastMGWR(FastGWR):
    """
    FastMGWR class.
    
    Parameters
    ----------
    comm        : MPI communicators initialized with mpi4py.
    parser      : The parser object contains model arguments.
    
    Attributes
    ----------
    comm        : MPI communicators initialized with mpi4py.
    parser      : The parser object contains model arguments.
    y           : array
                  n*1, dependent variable

    X           : array
                  n*k, independent variables (include constant, if any)
    coords      : array
                  n*2, collection of n sets of (x,y) coordinates used for
                  calibration locations
    n           : int
                  number of observations
    k           : int
                  number of independent variables
    minbw       : float
                  lower-bound bandwidth in the search range
    maxbw       : float
                  upper-bound bandwidth in the search range

    """
    def __init__(self, comm, parser):
        """
        Initialize class
        """
    
        FastGWR.__init__(self, comm, parser)
        
        #Standardizaing data
        if self.constant:
            stds = np.std(self.X, axis=0)
            stds[0] = 1
            self.X = (self.X - np.mean(self.X,axis=0))/stds
            self.X[:,0] = 1
        else:
            self.X = (self.X - np.mean(self.X,axis=0))/np.std(self.X, axis=0)
        self.y = (self.y - np.mean(self.y,axis=0))/np.std(self.y, axis=0)
            
        
    def backfitting(self):
        """
        Backfitting MGWR model and obtain parameter estimates
        and covariate-specific bandwidths.
        see Fotheringham et al. 2017. Annals of AAG.
        """
        
        if self.comm.rank ==0:
            print("MGWR Backfitting...",flush=True)
            print("Data are standardized",flush=True)
        
        #Initalization
        betas,bw = self.fit(init_mgwr=True,mgwr=True)

        self.bw_init = bw
        
        if self.comm.rank ==0:
            print("Initialization Done...",flush=True)
        XB = betas*self.X
        err = self.y.reshape(-1) - np.sum(XB,axis=1)
        bws = [None]*self.k
        bw_stable_counter = 0
        bws_history = []

        for mgwr_iters in range(1,201):
            newXB = np.empty(XB.shape, dtype=np.float64)
            newbetas = np.empty(XB.shape, dtype=np.float64)
            
            for j in range(self.k):
                temp_y = (XB[:,j] + err).reshape(-1,1)
                temp_X = self.X[:,j].reshape(-1,1)

                if bw_stable_counter >= 5:
                    #If in backfitting, all bws not changing in bws_same_times (default 5) iterations
                    bw_j = bws[j]
                    betas = self.mpi_gwr_fit(temp_y,temp_X,bw_j,final=True,mgwr=True)
                else:
                    betas,bw_j = self.fit(y=temp_y,X=temp_X,init_mgwr=False,mgwr=True)
                XB_j = (betas*temp_X).reshape(-1)
                err = temp_y.reshape(-1) - XB_j
                newXB[:,j] = XB_j
                newbetas[:,j] = betas.reshape(-1)
                bws[j] = bw_j
            
            if (mgwr_iters > 1) and np.all(bws_history[-1] == bws):
                bw_stable_counter += 1
            else:
                bw_stable_counter = 0
        
            bws_history.append(deepcopy(bws))
            num = np.sum((newXB - XB)**2) / self.n
            den = np.sum(np.sum(newXB, axis=1)**2)
            score = (num / den)**0.5
            XB = newXB
            
            if self.comm.rank ==0:
                print("Iter:",mgwr_iters,"SOC:","{:.2e}".format(score),flush=True)
                print("bws:",bws,flush=True)
            
            if score < 1e-5:
                break
        
        self.bws_history = np.array(bws_history)
        self.RSS = np.sum(err**2)
        self.TSS = np.sum((self.y - np.mean(self.y))**2)
        self.R2 = 1 - self.RSS/self.TSS
        self.err = err
        self.params = newbetas
        
        if self.comm.rank == 0 and self.estonly:
            header="index,residual,"
            varNames = np.genfromtxt(self.fname, dtype=str, delimiter=',',names=True, max_rows=1).dtype.names[3:]
            if self.constant:
                varNames = ['intercept'] + list(varNames)
            for x in varNames:
                header += ("b_"+x+',')
            
            self.output_diag(None,None,self.R2)
            index = np.arange(self.n).reshape(-1,1)
            output = np.hstack([index,self.err.reshape(-1,1),self.params])
            self.save_results(output,header)
        
        
    def _chunk_compute_R(self, chunk_id=0):
        """
        Compute MGWR inference by chunks to reduce memory footprint.
        See Li and Fotheringham, 2020. IJGIS and Yu et al., 2019. GA.
        """
        
        n = self.n
        k = self.k
        n_chunks = self.n_chunks
        chunk_size = int(np.ceil(float(n / n_chunks)))
        ENP_j = np.zeros(k)
        CCT = np.zeros((n, k))

        chunk_index = np.arange(n)[chunk_id * chunk_size:(chunk_id + 1) *
                                   chunk_size]
                                   
        init_pR = np.zeros((n, len(chunk_index)))
        init_pR[chunk_index, :] = np.eye(len(chunk_index))
        pR = np.zeros((n, len(chunk_index),k))  #partial R: n by chunk_size by k

        for i in range(n):
            wi = self.build_wi(i, self.bw_init).reshape(-1, 1)
            xT = (self.X * wi).T
            P = np.linalg.solve(xT.dot(self.X), xT).dot(init_pR).T
            pR[i, :, :] = P * self.X[i]

        err = init_pR - np.sum(pR, axis=2)  #n by chunk_size

        for iter_i in range(self.bws_history.shape[0]):
            for j in range(k):
                pRj_old = pR[:, :, j] + err
                Xj = self.X[:, j]
                n_chunks_Aj = n_chunks
                chunk_size_Aj = int(np.ceil(float(n / n_chunks_Aj)))
                for chunk_Aj in range(n_chunks_Aj):
                    chunk_index_Aj = np.arange(n)[chunk_Aj * chunk_size_Aj:(
                        chunk_Aj + 1) * chunk_size_Aj]
                    pAj = np.empty((len(chunk_index_Aj), n))
                    for i in range(len(chunk_index_Aj)):
                        index = chunk_index_Aj[i]
                        wi = self.build_wi(index, self.bws_history[iter_i, j]).reshape(-1)
                        xw = Xj * wi
                        pAj[i, :] = Xj[index] / np.sum(xw * Xj) * xw
                    
                    pR[chunk_index_Aj, :, j] = pAj.dot(pRj_old)
                err = pRj_old - pR[:, :, j]

        for j in range(k):
            CCT[:, j] += ((pR[:, :, j] / self.X[:, j].reshape(-1, 1))**2).sum(axis=1)
        for i in range(len(chunk_index)):
            ENP_j += pR[chunk_index[i], i, :]

        return ENP_j, CCT
        
    
    def mgwr_fit(self,n_chunks=2):
        """
        Fit MGWR model and output results
        """
        if self.estonly:
            return
        if self.comm.rank ==0:
            print("Computing Inference with",n_chunks,"Chunk(s)",flush=True)
        self.n_chunks = self.comm.size * n_chunks
        self.chunks = np.arange(self.comm.rank*n_chunks, (self.comm.rank+1)*n_chunks)
        
        ENP_list = []
        CCT_list = []
        for r in self.chunks:
            ENP_j_r, CCT_r = self._chunk_compute_R(r)
            ENP_list.append(ENP_j_r)
            CCT_list.append(CCT_r)
        
        ENP_list = np.array(self.comm.gather(ENP_list, root=0))
        CCT_list = np.array(self.comm.gather(CCT_list, root=0))
        
        if self.comm.rank == 0:
            ENP_j = np.sum(np.vstack(ENP_list), axis=0)
            CCT = np.sum(np.vstack(CCT_list), axis=0)
            
            header="index,residual,"
            varNames = np.genfromtxt(self.fname, dtype=str, delimiter=',',names=True, max_rows=1).dtype.names[3:]
            if self.constant:
                varNames = ['intercept'] + list(varNames)
            for x in varNames:
                header += ("b_"+x+',')
            for x in varNames:
                header += ("se_"+x+',')
            
            
            trS = np.sum(ENP_j)
            sigma2_v1 = self.RSS/(self.n-trS)
            aicc = self.compute_aicc(self.RSS, trS)
            self.output_diag(aicc,ENP_j,self.R2)
            
            bse = np.sqrt(CCT*sigma2_v1)
            index = np.arange(self.n).reshape(-1,1)
            output = np.hstack([index,self.err.reshape(-1,1),self.params,bse])
            
            self.save_results(output,header)
            
        return
