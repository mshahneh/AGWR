#FastGWR Class
#Author: Ziqi Li
#Email: liziqi1992@gmail.com

from mpi4py import MPI
import math
import numpy as np
from scipy.spatial.distance import cdist,pdist
import argparse


class FastGWR:
    """
    FastGWR class.
    
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
    bw          : float
                  user-specified bandwidth

    
    """
    def __init__(self, comm, parser):
        """
        Class Initialization
        """
        self.comm = comm
        self.parser = parser
        self.X = None
        self.y = None
        self.coords = None
        self.n = None
        self.k = None
        self.iter = None
        self.minbw = None
        self.maxbw = None
        self.bw = None
        
        self.parse_gwr_args()
        
        
        if self.comm.rank ==0:
            self.read_file()
            self.k = self.X.shape[1]
            self.iter = np.arange(self.n)
            
        #broadcast data to other nodes
        self.X = comm.bcast(self.X,root=0)
        self.y = comm.bcast(self.y,root=0)
        self.coords = comm.bcast(self.coords,root=0)
        self.iter = comm.bcast(self.iter,root=0)
        self.n = comm.bcast(self.n,root=0)
        self.k = comm.bcast(self.k,root=0)
        
        #split arrays to chunks for parralel processing
        m = int(math.ceil(float(len(self.iter)) / self.comm.size))
        self.x_chunk = self.iter[self.comm.rank*m:(self.comm.rank+1)*m]
        
    
    def parse_gwr_args(self):
        """
        Parsing arguments from the command line
        """
        parser_arg = self.parser.parse_args()
        self.fname = parser_arg.data
        self.fout  = parser_arg.out
        self.fixed = parser_arg.fixed
        self.constant = parser_arg.constant
        self.estonly = parser_arg.estonly

        if parser_arg.bw:
            if self.fixed:
                self.bw = float(parser_arg.bw)
            else:
                self.bw = int(parser_arg.bw)
        
        if parser_arg.minbw:
            if self.fixed:
                self.minbw = float(parser_arg.minbw)
            else:
                self.minbw = int(parser_arg.minbw)
        
        if self.comm.rank == 0:
            print("-"*60,flush=True)
            print("Starting FastGWR with",self.comm.size,"Processors",flush=True)
            if self.fixed:
                print("Spatial Kernel: Fixed Gaussian",flush=True)
            else:
                print("Spatial Kernel: Adaptive Bisquare",flush=True)
                
            print("Data Input Path:",self.fname,flush=True)
            print("Output Result Path:",self.fout,flush=True)
            print("Intercept:",self.constant,flush=True)


    def read_file(self):
        """
        Read file from the path
        """
        input = np.genfromtxt(self.fname, dtype=float, delimiter=',',skip_header=True)
        #Converting things into matrices
        self.y = input[:,2].reshape(-1,1)
        self.n = input.shape[0]
        if self.constant:
            self.X = np.hstack([np.ones((self.n,1)),input[:,3:]])
        else:
            self.X = input[:,3:]
        self.coords = input[:,:2]
        
        
    def set_search_range(self,init_mgwr=False,mgwr=False):
        """
        Define the search range in golden section
        """
        if self.fixed:
            max_dist = np.max(np.array([np.max(cdist([self.coords[i]],self.coords)) for i in range(self.n)]))
            self.maxbw = max_dist*2
            
            if self.minbw is None:
                min_dist = np.min(np.array([np.min(np.delete( cdist(self.coords[[i]],self.coords),i)) for i in range(self.n)]))
                
                self.minbw = min_dist/2
            
        else:
            self.maxbw = self.n

            if self.minbw is None:
                self.minbw = 40 + 2 * self.k
            if not init_mgwr and mgwr:
                self.minbw = 40 + 2
                
                
    def build_wi(self, i, bw):
        """
        Build the local weight matrix for location i
        """
    
        dist = cdist([self.coords[i]], self.coords).reshape(-1)
        #dist = np.sqrt(np.sum((self.coords[i] - self.coords)**2, axis=1)).reshape(-1)
        
        #fixed gaussian
        if self.fixed:
            wi = np.exp(-0.5*(dist/bw)**2).reshape(-1,1)
        #adaptive bisquare
        else:
            maxd = np.partition(dist, int(bw)-1)[int(bw)-1]*1.0000001
            zs = dist/maxd
            zs[zs>=1] = 1
            wi = ((1-(zs)**2)**2).reshape(-1,1)
        
        return wi
        

    def local_fit(self, i, y, X, bw, final=False, mgwr=False):
        """
        Fit the local regression model at location i
        """
    
        wi = self.build_wi(i, bw)
        #Last fitting, return more stats
        if final:
            xT = (X * wi).T
            xtx_inv_xt = np.dot(np.linalg.inv(np.dot(xT, X)), xT)
            betas = np.dot(xtx_inv_xt, y).reshape(-1)
            if mgwr:
                return betas
            
            ri = np.dot(X[i],xtx_inv_xt)
            predy = np.dot(X[i],betas)
            err = y[i][0] - predy
            CCT = np.diag(np.dot(xtx_inv_xt,xtx_inv_xt.T))
            
            return np.concatenate(([i,err,ri[i]],betas,CCT))
            
        #During bandwidth selection, return selected stats
        else:
            X_new = X*np.sqrt(wi)
            Y_new = y*np.sqrt(wi)
            temp = np.dot(np.linalg.inv(np.dot(X_new.T,X_new)),X_new.T)
            hat = np.dot(X_new[i],temp[:,i])
            yhat = np.sum(np.dot(X_new,temp[:,i]).reshape(-1,1)*Y_new)
            err = Y_new[i][0]-yhat
            return err*err,hat

        

    def golden_section(self, a, c, function):
        """
        Golden-section search bandwidth optimization
        """
        delta = 0.38197
        b = a + delta * np.abs(c-a)
        d = c - delta * np.abs(c-a)
        opt_bw = None
        score = None
        diff = 1.0e9
        iters  = 0
        dict = {}
        while np.abs(diff) > 1.0e-6 and iters < 200:
            iters += 1
            if not self.fixed:
                b = np.round(b)
                d = np.round(d)
    
            if b in dict:
                score_b = dict[b]
            else:
                score_b = function(b)
                dict[b] = score_b
    
            if d in dict:
                score_d = dict[d]
            else:
                score_d = function(d)
                dict[d] = score_d
    
            if self.comm.rank == 0:
                if score_b <= score_d:
                    opt_score = score_b
                    opt_bw = b
                    c = d
                    d = b
                    b = a + delta * np.abs(c-a)
                else:
                    opt_score = score_d
                    opt_bw = d
                    a = b
                    b = d
                    d = c - delta * np.abs(c-a)
    
                diff = score_b - score_d
                score = opt_score
    
            b = self.comm.bcast(b,root=0)
            d = self.comm.bcast(d,root=0)
            opt_bw = self.comm.bcast(opt_bw,root=0)
            diff = self.comm.bcast(diff,root=0)
            score = self.comm.bcast(score,root=0)
            
        return opt_bw
        
    
    def mpi_gwr_fit(self, y, X, bw, final=False, mgwr=False):
        """
        Fit the GWR model given a bandwidth
        """
        k = X.shape[1]
        #Need Parameter estimates
        if final:
            if mgwr:
                sub_Betas = np.empty((self.x_chunk.shape[0],k), dtype=np.float64)
            else:
                sub_Betas = np.empty((self.x_chunk.shape[0],2*k+3), dtype=np.float64)
            
            
            pos = 0
            for i in self.x_chunk:
                sub_Betas[pos] = self.local_fit(i, y, X, bw, final=True,mgwr=mgwr)
                pos+=1
    
            '''
            offset = rank*sub_Betas.nbytes
    
            fh.Write_at_all(offset, sub_Betas.reshape(-1))
            fh.Close()
            '''
        
            Betas_list = self.comm.gather(sub_Betas, root=0)
            
            if mgwr:
                Betas_list = self.comm.bcast(Betas_list, root=0)
                data = np.vstack(Betas_list)
                
                return data
    
            if self.comm.rank ==0:
                data = np.vstack(Betas_list)
    
                print("Fitting GWR Using Bandwidth:",bw,flush=True)

                RSS = np.sum(data[:,1]**2)
                TSS = np.sum((y - np.mean(y))**2)
                R2 = 1- RSS/TSS
                trS = np.sum(data[:,2])
                #trSTS = np.sum(data[:,4])
                #sigma2_v1v2 = RSS/(n-2*trS+trSTS)
                sigma2_v1 = RSS/(self.n-trS)
                aicc = self.compute_aicc(RSS, trS)
                data[:,-k:] = np.sqrt(data[:,-k:]*sigma2_v1)
                
                
                header="index,residual,influ,"
                varNames = np.genfromtxt(self.fname, dtype=str, delimiter=',',names=True, max_rows=1).dtype.names[3:]
                if self.constant:
                    varNames = ['intercept'] + list(varNames)
                for x in varNames:
                    header += ("b_"+x+',')
                for x in varNames:
                    header += ("se_"+x+',')
                

                #print and save results
                self.output_diag(aicc,trS,R2)
                self.save_results(data,header)
            
            return
        
        #Not final run
        sub_RSS = 0
        sub_trS = 0
        for i in self.x_chunk:
            err2,hat = self.local_fit(i,y,X,bw,final=False)
            sub_RSS += err2
            sub_trS += hat
    
        RSS_list = self.comm.gather(sub_RSS, root=0)
        trS_list = self.comm.gather(sub_trS, root=0)

        if self.comm.rank == 0:
            RSS = sum(RSS_list)
            trS = sum(trS_list)
            aicc = self.compute_aicc(RSS, trS)
            if not mgwr:
                print("BW, AICc",bw, aicc,flush=True)
            return aicc
            
        return
        
   
    def fit(self, y=None, X=None, init_mgwr=False, mgwr=False):
        """
        Fit the model if given bandwidth then call self.mpi_gwr_fit()
        If not, then using golden-section search for finding the optimal bandwidth
        which minimizes AICc.
        """
        if y is None:
            y = self.y
            X = self.X
        if self.bw:
            self.mpi_gwr_fit(y,X,self.bw,final=True)
            return
        
        if self.comm.rank ==0:
            self.set_search_range(init_mgwr=init_mgwr,mgwr=mgwr)
            if not mgwr:
                print("Optimal Bandwidth Searching...",flush=True)
                print("Range:",self.minbw,self.maxbw,flush=True)
        self.minbw = self.comm.bcast(self.minbw,root=0)
        self.maxbw = self.comm.bcast(self.maxbw,root=0)


        gwr_func = lambda bw: self.mpi_gwr_fit(y,X,bw,mgwr=mgwr)
        opt_bw = self.golden_section(self.minbw, self.maxbw, gwr_func)
        if self.fixed:
            opt_bw = round(opt_bw,2)
        
        data = self.mpi_gwr_fit(y,X,opt_bw,final=True,mgwr=mgwr)
        return data,opt_bw
   
        
    def compute_aicc(self, RSS, trS):
        """
        Compute AICc
        """
        aicc = self.n*np.log(RSS/self.n) + self.n*np.log(2*np.pi) + self.n*(self.n+trS)/(self.n-trS-2.0)
        return aicc
    
    def output_diag(self,aicc,trS,R2):
        """
        Output diagnostics to screen
        """
        if self.comm.rank == 0:
            print("Diagnostic Information:")
            print("AICc:",aicc)
            print("ENP:",trS)
            print("R2:",R2)
            
    def save_results(self,data,header):
        """
        Save results to .csv file
        """
        if self.comm.rank == 0:
            np.savetxt(self.fout, data, delimiter=',',header=header[:-1],comments='')
        
        
