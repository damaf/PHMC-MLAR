from sys import exit
import numpy as np
from regime_switching_VARM import RSVARM

############################################################################
## @class Gaussian
#  Regime switching vectoc auto-regressive model with gaussian innovation
#
class Gaussian_X(RSVARM):

    ## @fn 
    #  @brief 
    #  
    #  @param dimension Data dimension
    #  @param order Autoregressive process order
    #  @param nb_regime Number of regimes
    #  @param data Vector of length S, where S is the numbers of observed 
    #  sequences, data[s] is s^th sequence, is a matrix T_s x dimension.
    #  @param initial_values Matrix orderx1, initial values of X.
    #  @param init_method Initialization method to be considered. Can take two
    #   values :
    #       * "datadriven" for datadriven initialization. See function 
    #          datadriven_init for further details.
    #       * "rand1" 
    #       * "rand2"
    #       
    #   
    def __init__(self, dimension, order, nb_regime, data, initial_values, \
                 init_method):

        #---assertion
        assert (dimension > 0)
        assert (order >= 0)
        assert (nb_regime > 0)
        assert (initial_values[0].shape[1] == data[0].shape[1])
        
        if (dimension != data[0].shape[1]):
            print()
            print("ERROR: class Gaussian_X: dimension must be equal to the number of column within data!\n")
            exit(1)
            
        #---model dimension and data
        self.innovation = "gaussian"
        self.dimension = dimension
        self.order = order
        self.nb_regime = nb_regime
        self.data = data
        self.initial_values = initial_values
        
        #---initial law parameters
        # The initial_values[.][order-i] for i = 1, ..., order, follows 
        # a multi-variate normal distribution of mean self.psi["means"][i] and 
        # covariance matrix self.psi["covar"])
        self.psi = {}
        self.psi["means"] = []
        for _ in range(self.order):         
            self.psi["means"].append(np.zeros(dtype=np.float64, \
                                                        shape=self.dimension))                     
        self.psi["covar"] = np.zeros(dtype=np.float64, \
                                       shape=(self.dimension,self.dimension))
        
        #---VAR parameters initialization
        self.intercept = []
        self.sigma = []
        self.coefficients = []
                        
        if(init_method == "datadriven"):
            self.datadriven_init()
        elif(init_method == "rand1" or init_method == "rand2"):
            self.random_init(init_method)
        else:
            print("ERROR: in class Gaussian_X, unkown initialization method!\n")
            exit(1)
        
        return 
    
        
    ## @fn datadriven_init
    #  @brief VAR Covariance matrices are set at time series's empirical 
    #   covariance matrix; coefficients are set at zeros; and intercepts are 
    #   randomly chosen within interval [m-2s, m+2s] where m is time series' 
    #   empirical mean and s is their standard deviation. 
    #           
    def datadriven_init(self):
        
        #---compute the unconditional mean and covariance of data
        unc_mean = np.zeros(dtype=np.float64, shape=self.dimension)
        unc_covar = np.zeros(dtype=np.float64, \
                                     shape=(self.dimension, self.dimension))
        S = len(self.data) 
        for i in range(S):
            unc_mean = unc_mean + np.mean(self.data[i], axis=0)
            unc_covar = unc_covar + np.cov(self.data[i], rowvar=False)
            
        unc_mean = unc_mean / S
        unc_covar = unc_covar / S
        
        #---parameters initialization begins
        for k in range(self.nb_regime):
            
            #-intercepts are randomly chosen within 2-standard deviations
            #arround the unconditional mean
            two_std = 2*np.sqrt(np.diag(unc_covar))
            self.intercept.append( \
                       np.random.uniform(unc_mean-two_std, unc_mean+two_std)) 
                                      
            #-covariance matrices are set at the unconditional covariance matrix
            self.sigma.append( np.array(unc_covar) )
                                    
            #--autoregressive coefficients are set at zero in order that 
            #the conditional means are equal to the conditional mean
            self.coefficients.append([])
            for i in range(self.order):
                self.coefficients[k].append( np.zeros(dtype=np.float64, \
                                 shape=(self.dimension, self.dimension)) )
                   
        return 

     
    ## @fn random_one_init 
    #  @brief VAR coefficients are randomly chosen within [-1,1] and 
    #   covariance matrices are set at data empirical covariance matrix.
    #   Intercepts initialization depends of init_method:
    #   * if init_method == "rand1" then intercepts are set at zeros
    #   * if init_method == "rand2", b_k = unc_mean - sum_i=1..order C_i^k * X_{t-i} 
    #   
    #  NB: ADDED 2022/05/25
    #
    def random_init(self, init_method):
        
        #---compute the unconditional mean and covariance of data
        unc_mean = np.zeros(dtype=np.float64, shape=self.dimension)
        unc_covar = np.zeros(dtype=np.float64, \
                                     shape=(self.dimension, self.dimension))
        S = len(self.data) 
        for i in range(S):
            unc_mean = unc_mean + np.mean(self.data[i], axis=0)
            unc_covar = unc_covar + np.cov(self.data[i], rowvar=False)
        unc_mean = unc_mean / S
        unc_covar = unc_covar / S
        two_std = 2*np.sqrt(np.diag(unc_covar))
        
        #---parameters initialization begins
        for k in range(self.nb_regime):
                                    
            #----autoregressive coefficients are randomly chosen within [-1,1]
            self.coefficients.append([])
            for i in range(self.order):
                self.coefficients[k].append( np.random.uniform(-1.0, 1.0, \
                                             (self.dimension, self.dimension)))
                
            #----covariance matrices are set to random diagonal matrices
            self.sigma.append( np.array(unc_covar) )
            
            #----intercepts are initialization
            if(init_method == "rand1"):
                self.intercept.append(np.zeros(dtype=np.float64, \
                                       shape=self.dimension))
            else:
                # choose state k conditional mean within 2-standard 
                # deviations arround the unconditional mean
                mean_k = np.random.uniform(unc_mean-two_std, unc_mean+two_std)
                # choose the time series used to initialize intercepts
                chosen_s = np.random.choice([s for s in range(S)])
                T_s = self.data[chosen_s].shape[0]
                
                tmp_intecept_k = []
                
                for t in range(self.order, T_s):
                    tmp = np.zeros(dtype=np.float64, shape=self.dimension)
                    for i in range(1, self.order+1):
                        tmp = tmp + np.matmul(self.coefficients[k][i-1], \
                                          self.data[chosen_s][t-i, :])
                    tmp_intecept_k.append(mean_k - tmp)
                
                tmp_intecept_k = np.array(tmp_intecept_k)
                self.intercept.append(np.mean(tmp_intecept_k, axis=0))                
                        
        return 
                       
    
    ## @fn
    #  @brief
    #
    #  @param intercept
    #  @param coefficients
    #  @param sigma
    #
    def set_parameters(self, intercept, coefficients, sigma):
        
        #NB: object intercept, coefficients and sigma are not copied       
        self.intercept = intercept
        self.coefficients = coefficients  
        self.sigma = sigma
        
        return 


def compute_means_k(ar_coefficients, ar_intercepts, previous_vals, K, order, dim):
        
    assert((previous_vals.shape[0] == order) and (previous_vals.shape[1] == dim))
    
    #output
    means = np.zeros(shape=(K,dim), dtype=np.float64)
    
    for i in range(K):        
        for j in range(1, order+1):
            means[i,:] = means[i,:] + \
                 np.matmul(ar_coefficients[i][j-1], previous_vals[order-j, :])
            
        means[i,:] = means[i,:] + ar_intercepts[i]
        
    return means

