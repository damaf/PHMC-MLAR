#NB: numba is a parallelization library. It does support neither scipy library
#nor class members also named class functions.
#import numba as nb

import numpy as np
import math
from scipy.stats import multivariate_normal
from scipy.linalg import inv, pinv
import concurrent.futures
#####################################################################################
##  @class VARM
#   Regime switching vector auto-regressive model
#
class RSVARM():

    # Class attributes
               
    def __init__(self):
        ## @brief Distribution of residuals, can be any distribution 
        #  parameterized by its mean and variance, e.g. gaussian, gamma, 
        #  log-normal ...
        #  
        self.innovation = ""
        
        ## @brief Order of the autoregressive process X.
        #  
        self.order = 0
        
        ## @brief Dimension of process X.
        #  
        self.dimension = 0
        
        ## @brief The number of possible execution regimes of X.
        #  
        self.nb_regime = 0
        
        ## @brief Auto--regressive coefficients: \n
        #  A list of list of dimension x dimension matrices.
        #  self.coefficients[k] is a list of order matrices corresponding to the 
        #  auto-regressive coefficients associated to regime k. 
        #  Where matrix self.coefficients[k][i] is the autoregressive 
        #  coefficients related to the i^{th} lagged values:
        #  X_{t,j} | S_t=k := \sum_{i=1}^p self.coefficients[k][i][j,:]  X_{t-i} 
        #  for j=1, ..., dimension, t=1, ..., T.
        #  Note that the j^{th} line of self.coefficients[k][i] is associated
        #  with the j^{th} dimension of X.
        #
        self.coefficients = {}
        
        ## @brief Standard deviation:  \n
        #  List of nb_regime dimension x dimension matrices.
        #  self.sigma[k] is the covariance matrix of vector X within regime k.
        # 
        self.sigma = {}
        
        ## @brief Intercept parameter : \n
        #  List of nb_regime arrays of length dimension.
        #  self.intercept[k] is the intercept associated to regime k.
        # 
        self.intercept = {}
        
        ## @brief List of length S, where S is the number of observed
        #  time series. data[s], the s^th time series, is a T_s x dimension matrix 
        #  where T_s denotes its size starting at timestep t = order + 1 included
        #
        self.data = {}
        
        ## @brief List of length S, where S is the number of observed
        #  time series. self.initial_values[s] is a order x dimension matrix of 
        #  initial values associated with the s^th time series.
        #
        self.initial_values = {}
        
        ## @brief parameters of initial_values law. A multi-variate normal 
        #   distribution of dimension self.order is considered.
        #   Dictionnary having two entries:
        #    * entry "means" A list of 1-D arrays of length self.dimension.
        #      The i^{th} array corresponds to the mean of the distribution of
        #      X_{1-i} for i=1, ..., p
        #    * entry "covar" self.dimension x self.dimension 
        #      covariance matrix
        #
        self.psi = {}
 
        return

       
    ## @fn
    #
    def __str__(self):
        return "innovation " + self.innovation   
 
    
    ## @fn estimate_psi_MLE
    #  @brief compute the MLE estimator of the initial law which is a 
    #   multi-variate normal distribution specific to each single initial value.
    #   That is initial_values[.][order-i] for i = 1, ..., order follows a 
    #   Normal(self.psi["means"][i], matrix self.psi["covar"]).
    #   Note that the same covariance matrix is shared over all initial values.
    #
    def estimate_psi_MLE(self):
        #nb sequences
        S = len(self.initial_values) 
                
        #------vector of means estimations
        for i in range(1, self.order+1):
            for s in range(0, S):
                
                #self.psi["means"][i-1] is initialized with zeros
                self.psi["means"][i-1] = self.psi["means"][i-1] + \
                                    + self.initial_values[s][(self.order-i),:]
            
            self.psi["means"][i-1] = self.psi["means"][i-1] / S
            
        #------variance-covariance matrix estimation     
        for s in range(0, S):
            for i in range(1, self.order+1):
                    
                tmp_vec = self.initial_values[s][(self.order-i),:] - \
                                                        self.psi["means"][i-1]
                tmp_vec = tmp_vec.reshape((self.dimension,1))
                
                #self.psi["covar"] is initialized with zeros
                self.psi["covar"] = self.psi["covar"] + \
                                    np.matmul(tmp_vec, np.transpose(tmp_vec))        
        if(self.order != 0):
            self.psi["covar"] = self.psi["covar"] / (S*self.order)
        
        return
     
       
    ## @fn init_val_ll
    #  @brief compute log-likelihood of the sequence of initial values
    #
    def init_val_ll(self):
        #nb sequences
        S = len(self.initial_values) 
        
        #if few sequences have been observed
        if(S < 10):
            return 0.0
        else:
            ll = 0.0
            for s in range(0, S):
                for i in range(1, self.order+1):
                    ll = ll + np.log(multivariate_normal.pdf(\
                        x=self.initial_values[s][(self.order-i),:] , \
                        mean=self.psi["means"][i-1], cov=self.psi["covar"], \
                        allow_singular=True))
                
            return ll
       
    
    ## @fn total_likelihood_s
    #  @brief
    #
    #  @param s
    #  
    #  @return LL likelihood matrix of dimension T_s x nb_regime where: \n
    #  * LL[t,z] = g(x_t | x_{t-order}^{t-1}, Z_t=z ; \theta^{(X,z)}) \n
    #  * \theta^{(X,z)} is the set of parameter related to z^th regime.
    #
    def total_likelihood_s(self, s):
        
        #assertion
        assert (s >= 0 and s < len(self.data))
        
        #final result initialization
        T_s = self.data[s].shape[0]
        LL = np.zeros(dtype=np.float64, shape=(T_s, self.nb_regime))
                
        for k in range(self.nb_regime):           
            LL[:, k] = likelihood_k(self.coefficients[k], self.intercept[k], \
                                    self.sigma[k], self.data, \
                                    self.initial_values, s, self.order, \
                                    self.innovation)
               
        return LL
                    
              
    ## @fn update_parameters
    #
    #  @brief Computes the step M-X of EM algorithm.
    #  For gaussian innovations, the parameters' reestimation formulas are used.
    #  Otherwise innovations are not supported
    #
    #  @param list_Gamma A list of matrix where list_Gamma[s] is a T_sxM 
    #  matrix of time dependent marginal a posteriori probabilities relative 
    #  to s^th observed sequence, s in 1, ..., S, with S the number of 
    #  observed sequences.
    #
    #  
    def update_parameters(self, list_Gamma):
        
        if(self.innovation != "gaussian"):
            print()
            print("Error: file regime_switching_ARM.py: given distribution is not supported!")
            exit(1)
        
        #-------------Update regime parameters in parallel  
        #### BEGIN multi-process parallel execution 
        futures = []        
        
        with concurrent.futures.ProcessPoolExecutor(max_workers=self.nb_regime) as executor:               
            #lauch tasks
            for k in range(self.nb_regime):
                futures.append( executor.submit(call_update_parameters_regime_k, \
                                               self, list_Gamma, k) )      
                
            #collect the results as tasks complete
            for f in concurrent.futures.as_completed(futures):
                #update regime k parameters
                ((coefficients_k, intercept_k), sigma_k, k) = f.result()  
                self.coefficients[k] = coefficients_k
                self.intercept[k] = intercept_k
                self.sigma[k] = sigma_k       
        #### End multi-process parallel execution    
                              
                        
        return (self.intercept, self.coefficients, self.sigma)
    
        
    ## @fn update_parameters_regime_k
    #  @brief Compute the reestimation formulas of regime k parameters.
    #
    #  @param list_Gamma
    #  @param k
    #
    #  @return The reestimation of regime k parameters and the index of regime
    #  is used in main process to properly collect the results of child processes.
    #
    def update_parameters_regime_k(self, list_Gamma, k):
        
        #----Update autoregressive coefficients and intercept
        mu_k = self.compute_mu_k(list_Gamma, k)
        
        #----Update covariance matrix
        sigma_k = self.compute_sigma_k(list_Gamma, k, mu_k)
               
        return (mu_k, sigma_k, k)
    
    
    ## @fn
    #  @brief Compute the reestimation of autoregressive coefficients and 
    #   intercept within regime k.
    #
    #  @param list_Gamma
    #  @param k
    #
    #  @return mu_k = (coefficients_k, intercept_k) with
    #   * coefficients_k List of order dimension x dimension matrices
    #   * intercept_k Array of length dimension
    #
    def compute_mu_k(self, list_Gamma, k):
        
        #model dimensions
        dim = self.dimension
        order = self.order
        
        #----compute data matrix W_k:
        #dimension x (1+dimension*order) matrix
        W_k = compute_data_matrix_W_k(self.data, self.initial_values, \
                                      list_Gamma, k, dim, order)
        
        #----compute data matrix U_k: 
        #(1+dimension*order) x (1+dimension*order) matrix
        U_k = compute_data_matrix_U_k(self.data, self.initial_values, \
                                      list_Gamma, k, dim, order)
        
        #----inverse U_k in O((1+dimension*order)^3) operations at most
        try: 
            #try true inversion 
            U_k_inverse = inv(a=U_k, overwrite_a=False, check_finite=False)
        except np.linalg.LinAlgError:
            #if U_k is singular then pseudo-inverse is performed
            U_k_inverse = pinv(a=U_k, return_rank=False, check_finite=False)
            print("Warning: regime {}: U_k is singular, {}".format(k, \
                  "pseudo inverse has been performed!"))  
            
        #----compute mu_k = (b_k, C_1^k, ..., C_p^k) with b_k in R^{dimx1},
        #C_i^k in R^{dimxdim}
        #So m_k is a dimension x (1+dimension*order) matrix
        mu_k = np.matmul(W_k, U_k_inverse)
        
        #assertion
        assert( (mu_k.shape[0] == dim) and  (mu_k.shape[1] == 1+dim*order) )
        
        #----split mu_k = ()
        intercept_k = mu_k[:,0]
        coefficients_k = []
        for i in range(1, order+1):
            b_ind = dim*(i-1) + 1
            e_ind = dim*i + 1
            coefficients_k.append( mu_k[:,b_ind:e_ind] )
            
            #assertions
            assert(coefficients_k[i-1].shape[0] == coefficients_k[i-1].shape[1])
            assert(coefficients_k[i-1].shape[1] == dim)
        
        
        return (coefficients_k, intercept_k)
    
    
    ## @fn
    #  @brief Compute the reestimation of of covariance matrix within regime k
    #   using the newly estimation of mu_k
    #
    #  @param list_Gamma
    #  @param k
    #  @param mu_k = (coefficients_k, intercept_k) with
    #   * coefficients_k List of order dimension x dimension matrices
    #   * intercept_k Array of length dimension
    #
    #  @return A dimension x dimension matrix  
    #
    def compute_sigma_k(self, list_Gamma, k, mu_k):
        
        #mu_k decomposition
        coefficients_k = mu_k[0]
        intercept_k = mu_k[1]
        
        #initialization
        S = len(self.data)        
        regime_k_freq = 0.
        sigma_k = np.zeros(dtype=np.float64, shape=(self.dimension,\
                                                    self.dimension))
        
        for s in range(S):           
            #---update regime k frequency
            regime_k_freq = regime_k_freq + np.sum(list_Gamma[s][:,k])
            
            #---update sigma_k
            #T_s x dimension matrix: conditional mean of each observation of
            #the s^{th} time series
            cond_means = conditional_mean_of_timeSeries(coefficients_k, \
                                                        intercept_k, \
                                                        self.data[s], \
                                                        self.initial_values[s])            
            T_s = self.data[s].shape[0]
            for t in range(T_s):
                tmp_vec = self.data[s][t, :] - cond_means[t, :]
                tmp_vec = tmp_vec.reshape( (self.dimension, 1) )
                sigma_k = sigma_k + ( np.matmul(tmp_vec, np.transpose(tmp_vec)) * \
                                 list_Gamma[s][t,k] )
        
        sigma_k = sigma_k / regime_k_freq
                       
        return sigma_k
    
    
    
############################### BEGIN UTILS FUNCTIONS 
        
## @fn call_update_parameters_regime_k
#  @brief Call function update_parameters_regime_k on the given rsvarm object
#
#  @param rsvarm_object
#  @param list_Gamma
#  @param k
#
def call_update_parameters_regime_k(rsvarm_object, list_Gamma, k):
    return rsvarm_object.update_parameters_regime_k(list_Gamma, k) 
   

## @fn compute_data_matrix_U_k
#  @brief 
#
#  @param data
#  @param list_Gamma
#  @param k
#
#  @return A (1+dimension*order) x (1+dimension*order) matrix
#
def compute_data_matrix_U_k(data, initial_values, list_Gamma, k, dim, order):
    
    #number of sequences
    S = len(data)    
    #output
    U_k = np.zeros(dtype=np.float64, shape=(1+dim*order, 1+dim*order))
    
    #sum over matrices U_k(s,t)
    for s in range(S):
        
        T_s = data[s].shape[0]
        #sequence s and its first values are stacked vertically
        total_data_s = np.vstack((initial_values[s], data[s]))
        
        for t in range(order, T_s+order):
            
            #---build the first line and the first column of U_k(s,t)
            #then add then to U_k
            d_0 = list_Gamma[s][t-order, k]  #real number
            U_k[0,0] = U_k[0,0] + d_0
            for l in range(1, order+1):              
                
                x_t_l = total_data_s[t-l, :] 
                d_l = x_t_l * list_Gamma[s][t-order, k]  #vector of len dim
                
                b_ind = dim*(l-1) + 1
                e_ind = dim*l + 1
                
                U_k[b_ind:e_ind, 0] = U_k[b_ind:e_ind, 0] + d_l         
                U_k[0, b_ind:e_ind] = U_k[0, b_ind:e_ind] + d_l
                
            #---buil remaining part of U_k(s,t) by block of dimxdim matrices
            #then add then to U_k
            for i in range(1, order+1):
                
                b_ind_line = dim*(i-1) + 1
                e_ind_line = dim*i + 1
                    
                for j in range(1, order+1):
                    
                    x_t_i = total_data_s[t-i, :].reshape( (dim, 1) )
                    x_t_j_trans = total_data_s[t-j, :].reshape( (1, dim) )                  
                    #dim x dim matrix
                    d_i_j = np.matmul(x_t_i, x_t_j_trans) * \
                                                    list_Gamma[s][t-order, k]
                                                                        
                    b_ind_column = dim*(j-1) + 1
                    e_ind_column = dim*j + 1
                    
                    U_k[b_ind_line:e_ind_line, b_ind_column:e_ind_column] = \
                        U_k[b_ind_line:e_ind_line, b_ind_column:e_ind_column] \
                        + d_i_j
                                     
    return U_k

        
## @fn compute_data_matrix_W_k
#  @brief 
#
#  @param data
#  @param list_Gamma
#  @param k
#
#  @return A dimension x (1+dimension*order) matrix
#
def compute_data_matrix_W_k(data, initial_values, list_Gamma, k, dim, order):
    
    #number of sequences
    S = len(data)    
    #output
    W_k = np.zeros(dtype=np.float64, shape=(dim, 1+dim*order))
    
    #sum over matrices W_k(s,t)
    for s in range(S):
    
        T_s = data[s].shape[0]
        #sequence s and its first values are stacked vertically
        total_data_s = np.vstack((initial_values[s], data[s]))
        
        for t in range(order, T_s+order):
            
            x_t = total_data_s[t, :].reshape( (dim, 1) )
                  
            #---build the first column of W_k(s,t) then add it to W_k          
            e_0 = x_t[:, 0] * list_Gamma[s][t-order, k]    #real number
            W_k[:, 0] = W_k[:, 0] + e_0
                
            #---build other columns, by block of dim columns then add them to W_k
            for i in range(1, order+1):
                
                x_t_i_trans = total_data_s[t-i, :].reshape( (1, dim) )
                #vector of len dim
                e_i = np.matmul(x_t, x_t_i_trans) * list_Gamma[s][t-order, k]
                
                b_ind = dim*(i-1) + 1
                e_ind = dim*i + 1  
                
                W_k[:, b_ind:e_ind] = W_k[:, b_ind:e_ind] + e_i
                
    return W_k

    
## @fn pdf
#  @brief Conditional probability density function of a vector of d conditionally 
#  independent random variables: 
#  P(X_t=x_t|X_{t-1}^{t-p}) = \prod_{j=1}^d P_j(X_{t,j}=x_{t,j} | X_{t-1}^{t-p}) \n
#  Note that the mean of each marginal conditional density P_j depends of the 
#  past values of all variables. Thus both inter-variable correlations and 
#  intra-variable correlations are modelled by theses means following a vector 
#  autoregressive model
#  
#  @param distribution Either "gaussien" or "gamma"
#  @param x 1D-Array whose values are within the support of the distribution 
#   defined by innovation attribute
#  @param mean 1D-Array of means
#  @param covariance Covariance matrix 
#
#  @return If distribution is supported return a dimension length error
#
def pdf(distribution, x, mean, covariance):
        
    if (distribution != "gaussian"):   
        print("*************************************************************************")
        print("Error: file regime_switching_ARM.py: given distribution is not supported!") 
        exit(1)   
    else:
        #assertions
        assert(x.shape[0] == mean.shape[0] == covariance.shape[0] == covariance.shape[1])
        assert(np.sum(np.isnan(x)) == 0)
        
        den_x = multivariate_normal.pdf(x=x, mean=mean, cov=covariance, \
                                        allow_singular=True)  
        
                
        return den_x               
            

## @fn conditional_mean
#  @brief Compute the conditional mean of X_t within regime k
#
#  @param coefficients_k List of order dimension x dimension matrices
#  @param intercept_k Array of length dimension
#  @param past_values order x dimension matrix of past values.
#   They are stacked from X_{t-order} to  X_{t-1}.
#
#  @return An array of length dimension
#
def conditional_mean(coefficients_k, intercept_k, past_values):
    
    #observations' dimension
    dim = past_values.shape[1]
    #autoregressive order
    order = len(coefficients_k)    
    #output
    cond_mean = np.zeros(dtype=np.float64, shape=dim)
            
    #---compute the autoregressive part of conditional means
    for i in range(1, order+1):
        cond_mean = cond_mean + np.matmul(coefficients_k[i-1], \
                                          past_values[order-i, :])
        
    #---add intercepts
    cond_mean = cond_mean + intercept_k 
    
    return cond_mean


## @fn conditional_mean_of_timeSeries
#  @brief Compute the conditional mean of each observation of the given 
#   time series within regime k.
#
#  @param coefficients_k List of order dimension x dimension matrices
#  @param intercept_k Array of length dimension
#  @param time_series T_s x dimension matrix of time series observations
#  @âram first_values order x dimension matrix of initial values
#
#  @return A matrix T_s x dimension
#
def conditional_mean_of_timeSeries(coefficients_k, intercept_k, time_series, \
                                   first_values):
    #time series length
    T_s = time_series.shape[0]
    #observations' dimension
    dim = time_series.shape[1] 
    #autoregressive order
    order = len(coefficients_k)   
    #output
    cond_means = np.zeros(dtype=np.float64, shape=(T_s, dim))
    
    #assertion
    assert(order == first_values.shape[0])
    
    #vertical stacking of initial_values and time_series
    total_data = np.vstack((first_values, time_series))
       
    for t in range(order, T_s+order):              
        #conditional mean
        cond_means[t-order, :] = conditional_mean(coefficients_k, intercept_k, \
                                                  total_data[(t-order):t, :])

    return cond_means

   
## @fn likelihood_k
#
#  @param coefficients_k List of order dimension x dimension matrices
#  @param intercept_k Array of length dimension
#  @parm sigma_k Matix of size dimension x dimension
#
#  @param data List of length S, where S is the numbers of observation
#  sequences, data[s] is the s^th sequence.
#  @param initial_values List of length S, where S is the numbers of observed
#  sequences, initial_values[s] is a matrix orderx1 of initial values associated
#  with s^th sequence.
#
#  @param s 
#  @param order
#  @param distribution
#
#  @return Likelihood of data[s] within k^th regime, which is a vector of
#  length T_s.
# 
def likelihood_k(coefficients_k, intercept_k, sigma_k, data, initial_values, \
                 s, order, distribution):
  
    #assertion
    assert (s >= 0 and s < len(data))
        
    #output
    T_s = data[s].shape[0]
    LL = np.ones(dtype=np.float64, shape=T_s) * (-1.0)
    
    #---compute the conditional mean of each observation X_t in data[s]
    cond_means = conditional_mean_of_timeSeries(coefficients_k, intercept_k,\
                                                data[s], initial_values[s])       
            
    #---likelihood computing
    for t in range(T_s):           
        LL[t] = pdf(distribution, data[s][t, :], cond_means[t, :], sigma_k)
        
        #assertions
        assert(not math.isnan(LL[t]))
        assert(LL[t] >= 0.)
        #continuous PDF can be greater than 1 as an integral within an 
        #interval. Only mass function are bordered within [0, 1].
        
        
    
    return LL
   
############################### END UTILS FUNCTIONS 
