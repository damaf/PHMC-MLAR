import numpy as np
import math
from scipy.stats import multivariate_normal, multinomial

from regime_switching_VARM import likelihood_k




############################################################################
## @package Util functions
# 

#/////////////////////////////////////////////////////////////////////////////
# INFERENCE ASSESSMENT
#/////////////////////////////////////////////////////////////////////////////

## @fun segmention_uncertainty
#  Entropy of class probabilities is a quantification of MAP classification.
#  Entropy_t = - sum p[t,k] ln(p[t,k])
#
#  @return For each observed sequence, returns the uncertainty of MAP clustering.
#   Return a list of S array
#
def segmentation_uncertainty(list_Gamma):
    S = len(list_Gamma)
    H = [ segmentation_uncertainty_S(list_Gamma[s]) for s in range(S) ]     
    return H

def segmentation_uncertainty_S(Gamma):
    
    (T, M) = np.shape(Gamma)
    H = np.zeros(shape=T, dtype=np.float64)
    
    for t in range(T):
        aux = 0.0
        for m in range(M):
            if(Gamma[t, m] != 0):
                aux = aux + Gamma[t, m] * np.log(Gamma[t, m])                
        H[t] = -aux
        
    return H
      

"""
## @fn
#  @brief For all metrics, the best value is 1 and the worst value is 0
#
#  @param event_seq 1D array of events represented by their ID
#  @param event_categories
#  @param predictions 1D array
#
def event_predic_accurracy(event_seq, predictions):
    
    #accuracy or precision
    accurr = np.mean(event_seq == predictions)
    
    #confusion matrix
    conf_mat = confusion_matrix(event_seq, predictions, normalize='true')
    
    #recall = tp /(tp + fn)
    # Intuitively it is the ability of the classifier to find all the positive samples
    recall = recall_score(event_seq, predictions, average='macro')
    
    #F1-score = 2 * (precion * recall) / (precison + recall)
    f1Score = f1_score(event_seq, predictions, average='macro')
            
    return (accurr, conf_mat, recall, f1Score)


## @fn
#  @brief Mean Average Rank precision score
#
def mar_precision_score(event_seq, probabilities):
    
    nb_occ = event_seq.shape[0]
    nb_evnt_types = probabilities.shape[1]    
    indicator_vect_format = np.array( [ [0 if event_seq[n] != j  else 1 for \
                        j in range(nb_evnt_types)] for n in  range(nb_occ) ] )
        
    score = label_ranking_average_precision_score(indicator_vect_format, \
                                                  probabilities)
    
    return score

    

## TODO: event_sequences contains events_to_be_ignored. Maybe filter out them 
#  upstream
#
def prediction_assessment(hawkes_learner_Exp, hawkes_realizations, \
                           event_sequences):
    
    nb_seq = len(event_sequences)
    #outputs
    list_acc = []
    list_conf_mat = []
    list_recall = []
    list_f1_score = []
    list_mar = []
    
    for n in range(nb_seq):
        (probabilities, predictions) =\
                event_predic_knowing_occ_time(hawkes_learner_Exp, \
                                              hawkes_realizations[n], \
                                              event_sequences[n][:,0]/60)
        # accuracy metrics
        tmp = event_predic_accurracy(event_sequences[n][:,1], predictions)
        list_acc.append( tmp[0] )
        list_conf_mat.append( tmp[1] )
        list_recall.append( tmp[2] )
        list_f1_score.append( tmp[3] )
        
        # MAR
        list_mar.append( mar_precision_score(event_sequences[n][:,1], \
                                             probabilities) )
        
    return (list_acc, list_conf_mat, list_recall, list_f1_score, list_mar)
"""



#/////////////////////////////////////////////////////////////////////////////
# Sample likelihood computing, used in inference algorithm
#/////////////////////////////////////////////////////////////////////////////
## @fn
#
#  @param coefficients
#  @param intercept 
#  @param sigma 
#  @param innovation
#  @param Obs_seq Observation sequence, Txdimention matrix
#
#  @return likelihood of Obs_seq, a matrix (T-p)xM, with T the sequence 
#  length, M the number of regimes and p the autoregressive order.
#  LL[t,z] = g(x_t | x_{0}^{t-1}, Z_t, \theta_{z_t})
#  theta_z is the set of parameter related to the z^th regime
#
def compute_LL(coefficients, intercept, sigma, innovation, Obs_seq):
    
    #model dimensions
    order = len(coefficients[0])
    M = len(intercept)
    
    #effective number of observations
    T = Obs_seq.shape[0] - order
    
    #observed sequence length must be greater than VAR order
    assert(T > 0) 
    
    #output
    LL = np.zeros(shape=(T, M), dtype=np.float128) 
    
    #Obs_seq splitting 
    initial_values = [ Obs_seq[0:order, :] ]
    data = [ Obs_seq[order:, :] ]
        
    for k in range(M):                 
        LL[:, k] = likelihood_k(coefficients[k], intercept[k], sigma[k], \
                                data, initial_values, s=0, order=order, \
                                distribution=innovation)   
    return LL


#/////////////////////////////////////////////////////////////////////////////
# Simulation - Forecasting: utils functions
#/////////////////////////////////////////////////////////////////////////////
    
## @fn compute_prediction_probs
#  @brief Compute probabilities P(Z_{T+h}=k | X_{1-p}^T) for h=1, ..., H
#  @param A 
#  @param Gamma_T K length array
#  @param H
#  @return HxM array
#
def compute_prediction_probs(A, Gamma_T, H):
    
    #nb regimes
    M = A.shape[0]
    
    #-----------VAR model case
    if M == 1:
        return np.ones(shape=(H, M), dtype=np.float64)
        
    #-----------Several regimes case
    #output
    pred_probs = np.zeros(shape=(H, M), dtype=np.float64)
    
    #normalization of Gamma_T
    Gamma_T = Gamma_T / np.sum(Gamma_T)
    
    #----initial case h = 0
    for k in range(M):
        pred_probs[0, k] = np.sum(A[:, k] * Gamma_T)
        
    pred_probs[0, :] = pred_probs[0, :] / np.sum(pred_probs[0, :])
               
    #----for h = 1, ..., H-1
    for h in range(1, H):
        for k in range(M):
            pred_probs[h, k] = np.sum(A[:, k] * pred_probs[h-1, :])
        
        pred_probs[h, :] = pred_probs[h, :] / np.sum(pred_probs[h, :])
                
    # assertions
    assert(np.sum(np.isnan(pred_probs)) == 0)
    assert(np.sum(pred_probs < 0.) == 0)
    assert(np.sum(pred_probs > 1.) == 0)
    
    
    return pred_probs


## @fn compute_means
#  @brief Compute the conditional mean of X_t within each state.
#  @param ar_coefficients 
#  @param ar_intercepts
#  @previous_vals The d nearest past values, orderxdim matrix.
#   They are stacked from X_{t-order} to X_{t-1}.
#  @return Kxdim-array where K is the number of regime
#
def compute_means(ar_coefficients, ar_intercepts, previous_vals, K, order, dim):
        
    assert((previous_vals.shape[0] == order) and (previous_vals.shape[1] == dim))
    
    #output
    means = np.zeros(shape=(K,dim), dtype=np.float64)
    
    for i in range(K):        
        for j in range(1, order+1):
            means[i,:] = means[i,:] + \
                 np.matmul(ar_coefficients[i][j-1], previous_vals[order-j, :])
            
        means[i,:] = means[i,:] + ar_intercepts[i]
        
    return means


## @fn rvs Random variables
#  @return dimension length array
#
def rvs(distribution, mean, covariance):

    if (distribution != "gaussian"):
        print("**************************************************************")
        print("Error: file regime_switching_ARM.py: given distribution" + \
              " is not supported!") 
        exit(1)
                    
    else:
        #assertion
        assert(mean.shape[0] == covariance.shape[0] == covariance.shape[1])
        
        den_x = multivariate_normal.rvs(mean=mean, cov=covariance, size=1)
        
        return den_x

     
## @fn cond_density_sampling
#
#  @brief Sample from the conditional density function at time-step t
#   (which is mixture of distributions).
#   Choose the state at time-step t from multinomial(states_probs_t) then
#   sample from the associated distribution.
#
#  @return 
#   * s int value: the selected state at time-step t
#   * sample length dimension array: the sample drawn from 
#
def cond_density_sampling(cond_means_t, sigmas, innovation, states_probs_t):
    
    """
    # assertions
    assert(np.sum(np.isnan(states_probs_t)) == 0)
    assert(np.sum(states_probs_t < 0.) == 0)
    assert(np.sum(states_probs_t > 1.) == 0)
    """
            
    # choose state, then sample X_t from the selected state distribution
    s = np.argmax(multinomial.rvs(1, states_probs_t)) 
    sample = rvs(innovation, cond_means_t[s, :], sigmas[s])  
    
    return (s, sample)


## @fn cond_density_quantiles
#
#  @brief Compute the quantiles of the conditional density function at 
#   time-step t (which is mixture of distributions)
#  
#  @NB: I did not found a method devoted to multi-dimensional quantiles 
#   estimate. 
#
#  @return nb_quantile x dimension array where each line corresponds to a 
#   specific quantile
#
def cond_density_quantiles(cond_means_t, sigmas, innovation, states_probs_t, \
                           list_q):
    pass  


## @fn sample_from_conf_interval_of_CD
#
#  @brief This function builds a confidence interval of the conditional density
#   function then return a sample from that confidence interval.
#   Quantiles are computed for each dimension independently from the other dimensions.
#   Then, an observation sampled from the predictive distribution and belong to
#   quantile intervals is returned.
#
#  @list_q Two elements list denoting the quantiles to be used in order to 
#   build confidence interval
#
#  @return dimension length array
#
def sample_from_conf_interval_of_CD(cond_means_t, sigmas, innovation, \
                                    states_probs_t, list_q):
    """
    # assertions
    assert(np.sum(np.isnan(states_probs_t)) == 0)
    assert(np.sum(states_probs_t < 0.) == 0)
    assert(np.sum(states_probs_t > 1.) == 0)
    """
            
    assert(len(list_q) == 2 and (list_q[0] < list_q[1]))
    
    #---generate the samples used in quantiles computing
    nb_samples = 200  
    samples = []
    
    for _ in range(nb_samples):
        # select state, then X_t from the selected state distribution
        s = np.argmax(multinomial.rvs(1, states_probs_t)) 
        tmp = rvs(innovation, cond_means_t[s, :], sigmas[s])   
        samples.append(tmp)
    
    #---Build confidence interval
    samples = np.vstack(samples)
    conf_interval = np.quantile(samples, list_q, axis=0)  
    inf_bound = conf_interval[0,:]
    sup_bound = conf_interval[1,:]
    
    #---looking for samples within that interval
    #indices of that samples
    within_inter = []
    for i in range(nb_samples):
        ok = np.all(samples[i, :] >= inf_bound) and np.all(samples[i, :] <= sup_bound)
        if(ok):
            within_inter.append(i)
    
    if(len(within_inter) > 0):   # at least one sample found within conf_interval
        chosen_index = np.random.choice(within_inter)
        return samples[chosen_index, :]
        
    else:  # no sample found, additionnal 10 samples are generated and tested 
        n = 0
        while n < 10:
            s = np.argmax(multinomial.rvs(1, states_probs_t)) 
            new_sample = rvs(innovation, cond_means_t[s, :], sigmas[s]) 
        
            if(np.all(new_sample >= inf_bound) and np.all(new_sample <= sup_bound)):
                break
            
            n = n + 1
                       
        return new_sample
    

#/////////////////////////////////////////////////////////////////////////////
# SIMULATION: Generate synthetic data from model or sample from PHMC-VAR
##/////////////////////////////////////////////////////////////////////////////
#
## @fn Simulation  
#
#  @brief
#
#  @param init_values The first d initial value where d is AR process order, 
#  dxdim matrix. They are stacked from X_{1-d} to X_0.
#  @param L Length of the simulated sequence.
#  @param coefficients RS-AR coefficients within each regime.
#  @param intercept RS-AR intercept within each regime.
#  @param sigma RS-AR standart deviation within each regime.
#  @param A
#  @param Pi
#  @param innovation 
#  
#  @return Computes a simulated time series from the given model
#
def simulation(init_values, L, coefficients, intercept, sigma, A, Pi, \
               innovation):
    
    #---------------output initialization
    #RS-AR order
    order = len(coefficients[0])
    #time series dimension
    dim = init_values.shape[1]      
    #number of classes
    K = A.shape[0]
    
    assert((init_values.shape[0] == order) and (init_values.shape[1] == dim))
    
    total_X = np.ones(shape=(L,dim), dtype=np.float64) * np.nan
    total_X = np.vstack((init_values, total_X))
    selec_states = np.ones(shape=L, dtype=np.int32) * (-1)
    
    #---initial state probabilities: t=0
    states_probs_t = Pi[0, :]
    
    #---------------simulation starts   
    for t in range(order, L+order, 1):  
        
        # conditional mean within each state
        cond_mean = compute_means(coefficients, intercept, \
                                  total_X[t-order:t, :], K, order, dim)   
        # sample the conditional
        (s, sample) = cond_density_sampling(cond_mean, sigma, innovation, states_probs_t)
        selec_states[t-order] = s
        total_X[t, :] = sample
        
        # state probabilities at t+1
        states_probs_t = A[selec_states[t-order], :]
       
                
    return (total_X, selec_states)



#/////////////////////////////////////////////////////////////////////////////
#       MODEL SELECTION CRITERIA
#/////////////////////////////////////////////////////////////////////////////
    
## @fun compute_BIC
#   Bayesian Information Criterion, equals -2*log_ll + ln(T)*n_par
#
#  @param log_ll Log_liikelihood of the trained model
#  @param nb_par Number of parameter of model
#  @param T Number of observation
# 
def compute_BIC(log_ll, nb_par, T):
    return -2*log_ll + math.log(T)*nb_par

## @fun compute_AIC
#   Bayesian Information Criterion, equals -2*log_ll + 2*n_par
#
#  @param log_ll Log_liikelihood of the trained model
#  @param nb_par Number of parameter of model
#  @param T Number of observation
# 
def compute_AIC(log_ll, nb_par):
    return -2*log_ll + 2*nb_par


#/////////////////////////////////////////////////////////////////////////////
#        ERROR METRICS: INFERENCE
#/////////////////////////////////////////////////////////////////////////////

## @fn 
#  @brief Compute the mean percentage error between two partitions
#
#  @param data_S1 List of N states sequences, each is a 1xT arrays.
#   Reference labels.
#  @param data_S2 List of N states sequences, each is a 1xT arrays.
#   Inferred labels.
#
#  @return N-length array
#
def mean_percentage_error(data_S1, data_S2):
    
    #nb sequences
    N1 = len(data_S1)
    N2 = len(data_S2)  
    assert(N1 == N2)  
    
    mean_errors = -1 * np.ones(shape=N1, dtype=np.float64)
    
    for s in range(N1):
        
        #sequence length
        T1 = data_S1[s].shape[1]
        T2 = data_S2[s].shape[1]  
        assert(T1 == T2)        
    
        assert(np.sum(data_S1[s][0,:] < 0) == 0)
        assert(np.sum(data_S2[s][0,:] < 0) == 0)
                
        mean_errors[s] =  np.mean(data_S1[s][0,:] != data_S2[s][0,:])
               
    return mean_errors

## @fn
#  @brief Hidden states are not accounted for  
#
def __mean_percentage_error(data_S1, data_S2):
    
    #nb sequences
    N1 = len(data_S1)
    N2 = len(data_S2)  
    assert(N1 == N2)  
    
    mean_errors = -1 * np.ones(shape=N1, dtype=np.float64)
    
    for s in range(N1):
        #sequence length
        T1 = data_S1[s].shape[1]
        T2 = data_S2[s].shape[1]  
        assert(T1 == T2)        
                    
        tmp = 0
        for t in range(T1):
            if(data_S1[s][0,t] >= 0 and data_S2[s][0,t] >= 0):
                tmp = tmp + (data_S1[s][0,t] != data_S2[s][0,t])
                
        mean_errors[s] =  tmp / T1 
        
    return mean_errors


## @fn 
#  @brief 
#
def adjusted_rand_index(data_S1, data_S2):
    pass


#/////////////////////////////////////////////////////////////////////////////
#        ERROR METRICS: FORECASTING
#/////////////////////////////////////////////////////////////////////////////
## @fn performance_metrics
#  @brief Computes MBias, RMSE, NRMSE and MAPE of H-ahead prediction over 
#   a single time series
#
#  @param Bias List of length N
#  @param NBias List of length N
#
#  @return Three arrays of length dimension
#
def performance_metrics(Bias, NBias):
    
    Bias = np.vstack(Bias)
    NBias = np.vstack(NBias)
    
    MBias = np.mean(Bias, axis=0)    
    RMSE = np.sqrt(np.mean(Bias**2, axis=0)) 
    
    NRMSE = np.sqrt(np.nanmean(NBias**2, axis=0))
    MAPE = np.nanmean(np.abs(NBias), axis=0)
    
    return (MBias, RMSE, NRMSE, MAPE)

      
