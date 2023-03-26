#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  3 19:11:11 2020

@author: dama-f
"""
from sys import exit 

import numpy as np
import pickle

from utils import compute_LL, compute_means, \
            cond_density_sampling, sample_from_conf_interval_of_CD, rvs
from scaled_backward_forward_backward_recursion import  BFB




#/////////////////////////////////////////////////////////////////////////////
# INFERENCE - GAMMA PROBABILITIES
#/////////////////////////////////////////////////////////////////////////////
    
## @fn
#  @brief
#
#  @param M
#  @param LL
#  @param A
#  @param Pi
#  @param partial_anno
#
#  @return TxM matrix of prbabilities Gamma.
#
def compute_gamma_BFB_algo(M, LL, A, Pi, partial_anno=[]):
                         
    #the modified FB algorithm reduces to the standard FB when weights equal 1
    (T, M) = LL.shape
    
    #---no partial annotations
    if(len(partial_anno) == 0):
        state_range = [i for i in range(M)]
        partial_anno = [ np.array(state_range, dtype=np.int32) for _ in range(T) ]
    elif(len(partial_anno) != T):
        print("ERROR: File forecasting_inference: inconsistent partial annotation! \n")
        exit(1)
    
    (_, _, Gamma, _) = BFB(M, LL, partial_anno, A, Pi)

    return Gamma


## @fn
#  @brief
#
#  @param A
#  @param Pi
#  @param coefficients
#  @param intercepts 
#  @param sigma
#  @param innovation Error term's law
#  @param Obs_seq Sequence to be labelled, Tx1 colomn vector
#
#  @return 1xT array of inferred states
#
def gamma_probs_based_inference(A, Pi, coefficients, intercepts, sigma, \
                                innovation, Obs_seq, partial_anno=[]):
        
    #compute LL, TxM matrix
    LL = compute_LL(coefficients, intercepts, sigma, innovation, Obs_seq)
    
    #number of states and timeseries length minus AR order
    (T, M) = LL.shape
    
    #compute gamma probabilities
    Gamma = compute_gamma_BFB_algo(M, LL, A, Pi, partial_anno)
    
    #at each time-step the step having the maximum marginal probability is chosen  
    opt_states = np.argmax(Gamma, axis=1)
        
    
    return (Gamma, opt_states.reshape((1,-1)))
    



#/////////////////////////////////////////////////////////////////////////////
#     INFERENCE - OUR VARIANT OF VITERBI
#/////////////////////////////////////////////////////////////////////////////
    
## @fun viterbi
#  Maximum A Posterio classification: P(Z,X) = P(Z|X)*P(X)
#
#  NB: With this implementation I think that np.argmax(D[T-1, :]) 
#  is different of max_S P(S|X) because of the calculation of probabilities 
#  P(S_1|sigma_1) and P(S_t|S_{t-1}, sigma_t, sigma_{t-1}) which are not
#  normalized here.
#  (compare this implementation to the equations of ML article). 
#  This remains true for the log version
#   
#  @param A
#  @param Pi
#  @param coefficients
#  @param intercepts 
#  @param sigma
#  @param innovation Error term's law
#  @param Obs_seq Sequence to be labelled, Tx1 colomn vector
#  @param partial_anno List of T 1D arrays where partial_anno[t] is the set
#   of possible states at time-step t. For instance partial_anno[t]=[0, 1] 
#   means that only states 0 and 4 are possible at time-step t.
#   Note that states are numbered from 0 to M-1.
#   If a empty array is given, the algorithm reduces to the standard Viterbi
#   algorithm
#
#  @return opt_states 1xT array
#
def viterbi(A, Pi, coefficients, intercepts, sigma, innovation, Obs_seq, \
            partial_anno):
   
    #number of states
    M = A.shape[0]     
    #compute LL, TxM matrix
    LL = compute_LL(coefficients, intercepts, sigma, innovation, Obs_seq)
    #effective length of observation sequence
    T = LL.shape[0]
                   
    #------no partial annotations
    if(len(partial_anno) == 0):
        state_range = [i for i in range(M)]
        partial_anno = [ np.array(state_range, dtype=np.int32) for _ in range(T) ]
    elif(len(partial_anno) != T):
        print("File forecasting_inference: inconsistent partial annotation! \n")
        exit(1)
        
    #------initialize probability matrix D
    D = np.zeros(shape=(T,M), dtype=np.float64)
    
    #------initial D probabilities
    possible_states_0 = partial_anno[0]
    for i in possible_states_0:
        D[0, i] = Pi[0, i] * LL[0, i]
               
    #------compute D for t=1,...,T-1
    for t in range(1, T):            
        possible_states_t = partial_anno[t]
        for i in possible_states_t:
                temp_product = A[:, i] * D[t-1, :]
                D[t, i] = np.max(temp_product) * LL[t, i]
                            
    assert(np.sum(D < 0) == 0)
    
    #------optimal state computing: backtracking
    opt_states = -1 * np.ones(shape=(1,T), dtype=np.int32)
    opt_states[0, T-1] = np.argmax(D[T-1, :])
    
    for t in range(T-2, -1, -1):
        opt_states[0, t] = np.argmax( D[t, :] * A[:, opt_states[0, t+1]] )
     
        
    return opt_states



## @fun viterbi_log: to be used when t > 30
#  @param A
#  @param Pi
#  @param coefficients
#  @param intercepts 
#  @param sigma 
#  @param innovation Error term's law
#  @param Obs_seq Sequence to be labelled, Tx1 colomn vector
#  @param partial_anno
#
#  @return opt_states 1xT array  
#
def viterbi_log(A, Pi, coefficients, intercepts, sigma, innovation, Obs_seq, \
                partial_anno):
    
    #number of states
    M = A.shape[0]     
    #compute LL, TxM matrix
    LL = compute_LL(coefficients, intercepts, sigma, innovation, Obs_seq)
    #effective length of observation sequence
    T = LL.shape[0]
            
    #------no partial annotations
    if(len(partial_anno) == 0):
        state_range = [i for i in range(M)]
        partial_anno = [ np.array(state_range, dtype=np.int32) for _ in range(T) ]
    elif(len(partial_anno) != T):
        print("File forecasting_inference: inconsistent partial annotation! \n")
        exit(1)
        
    #------compute log probabilities
    tiny = np.finfo(0.).tiny
    A_log = np.log(A + tiny)
    Pi_log = np.log(Pi + tiny)
    LL_log = np.log(LL + tiny)
    
    #------initialize log probability matrix D_log
    D_log = -np.inf * np.ones(shape=(T,M), dtype=np.float128)
    
    #------initial D probabilities
    possible_states_0 = partial_anno[0]
    for i in possible_states_0:
        D_log[0, i] = Pi_log[0, i] + LL_log[0, i]
        
    #------compute D for t=1,...,T-1
    for t in range(1, T):       
        possible_states_t = partial_anno[t]
        for i in possible_states_t:
                temp_sum = A_log[:, i] + D_log[t-1, :]
                D_log[t, i] = np.max(temp_sum) + LL_log[t, i]
                    
    #------optimal state computing: backtracking
    opt_states = -1 * np.ones(shape=(1,T), dtype=np.int32)
    opt_states[0, T-1] = np.argmax(D_log[T-1, :])
    
    for t in range(T-2, -1, -1):
        opt_states[0, t] = np.argmax( D_log[t, :] + A_log[:, opt_states[0, t+1]] )
     
        
    return opt_states



## @fn 
#  @brief
#
#  @param model_file
#  @param innovation
#  @param list_Obs_seq
#  @param list_partial_anno
#  @param list_scalers used for data preprocessing before inference
#  @param method Two possible values 'viterbi' or 'gammaProbs'
#
#  @return
#
def inference(model_file, list_Obs_seq, list_partial_anno=[], \
              list_scalers=None, method="viterbi"):
    
    #-----model loading
    infile = open(model_file, 'rb')
    phmc_var = pickle.load(infile)
    infile.close()
    
    #-----required phmc_var parameters
    innovation = "gaussian"
    A = phmc_var[1]
    Pi = phmc_var[2]
    ar_coefficients = phmc_var[5]
    ar_intercepts = phmc_var[7]  
    sigma = phmc_var[6]
    
    #assertion
    assert(np.sum(A < 0.0) == 0)
    assert(np.sum(Pi < 0.0) == 0)
        
    #nb sequence
    N = len(list_Obs_seq)
    
    #-----no observed state
    if(len(list_partial_anno) == 0):
        list_partial_anno = [ [] for _ in range(N) ] 
    else:
        assert(N == len(list_partial_anno))
        
    #data must be standardized
    standardization = True if (list_scalers != None) else False
            
    #-----Inference begins
    #output
    list_states = []
    list_Gamma = []
    
    for s in range(N):
        
        #data standardization
        if(standardization):
            stand_Obs_seq_s = list_scalers[s].transform(list_Obs_seq[s])
        else:
            stand_Obs_seq_s = list_Obs_seq[s]
    
        #inference
        if(method == "viterbi"):
            list_states.append( viterbi_log(A, Pi, ar_coefficients, \
                                            ar_intercepts, sigma, \
                                            innovation, stand_Obs_seq_s, \
                                            list_partial_anno[s]) )
        elif(method == "gammaProbs"):
            (Gamma, states) = gamma_probs_based_inference(A, Pi, \
                                    ar_coefficients, ar_intercepts, \
                                    sigma, innovation, stand_Obs_seq_s, \
                                    list_partial_anno[s])
            list_states.append(states)
            list_Gamma.append(Gamma)
        else:
            print("ERROR: file inference_forecasting.py: unknown inference method! \n")
            exit(1)   
            
    #-----outputs
    if(method == "viterbi"):
        return list_states
    else:
        return (list_states, list_Gamma)
      


#/////////////////////////////////////////////////////////////////////////////
#           FORECASTING 
#/////////////////////////////////////////////////////////////////////////////
    

## @fn
#  @brief perform H-steps ahead prediction over the given set of time series
#   
#  @param model_file 
#  @param set_of_time_series
#  @param H Forecast horizon 
#  @param set_partial_anno Partial annotations form timeseries defined as a 
#   list of T_s-order 1D arrays of possible states at each 
#   time-step 
#  @param f_type The forecast type, two possible values
#   * "optimalPoint" for optimal point forecast
#   * "monteCarlo" for Monte Carlo density forecast
#   * "quantile" for quantile forecast
#  
#  @return A list of N Hxdimension array of predicted values
#
def forecasting(model_file, set_of_time_series, H, set_partial_anno=None, \
                f_type="optimalPoint"):
                    
    #model loading
    infile = open(model_file, 'rb')
    phmc_var = pickle.load(infile)
    infile.close()
    
    #required phmc_var parameters
    innovation = "gaussian"
    A = phmc_var[1]
    Pi = phmc_var[2]
    ar_coefficients = phmc_var[5]
    ar_intercepts = phmc_var[7]     
    sigmas = phmc_var[6]
    
    #hyper-parameters 
    order = len(ar_coefficients[0])
    M = A.shape[0]
            
    #nb time series
    N = len(set_of_time_series)
        
    #assertions
    assert(np.sum(A < 0.0) == 0)
    assert(np.sum(np.isnan(A)) == 0)
    
    if(set_partial_anno != None):
        assert(len(set_partial_anno) == N)
        
    #output
    list_predictions = []
    
    #for each sequence
    for s in range(N):
                
        #coumpute Gamma_T
        LL = compute_LL(ar_coefficients, ar_intercepts, sigmas, innovation, \
                        set_of_time_series[s])
            
        if(set_partial_anno == None):
            Gamma_T = compute_gamma_BFB_algo(M, LL, A, Pi)[-1, :]
        else:
            Gamma_T = compute_gamma_BFB_algo(M, LL, A, Pi, set_partial_anno[s])[-1, :]
        
        #time series length
        T_s = set_of_time_series[s].shape[0]
        past_values = set_of_time_series[s][(T_s-order):T_s, :]
        
        #forecast
        if(f_type=="optimalPoint"):      
            list_predictions.append(forecasting_one_seq(A, ar_coefficients, \
                                                        ar_intercepts, \
                                                        Gamma_T, past_values, H)) 
        elif(f_type=="monteCarlo"):
            list_predictions.append( \
                    forecasting_one_seq_monte_carle(A, ar_coefficients, \
                                                    ar_intercepts, sigmas, \
                                                    innovation, Gamma_T, \
                                                    past_values, H) )
        elif(f_type=="quantile"):
            list_predictions.append( \
                    forecasting_one_seq_quantiles(A, ar_coefficients, \
                                                  ar_intercepts, sigmas, \
                                                  innovation, Gamma_T, \
                                                  past_values, H) )
        else:
            print("ERROR: file inference_forecasting.py: unknown forecast type! \n")
            exit(1)  
                          
    return list_predictions


#------------------------------------------
## @fn
#
def compute_state_probs_at_forecast(A, Gamma_T, H, partial_anno_at_forecast):
       
    #number of states
    M = Gamma_T.shape[0]
    
    #output
    state_probs = np.zeros(shape=(H, M), dtype=np.float64)
    
    #---no partial annotations
    if(len(partial_anno_at_forecast) == 0):
        state_range = [i for i in range(M)]
        partial_anno_at_forecast = [ np.array(state_range, dtype=np.int32) \
                                    for _ in range(H) ]
    elif(len(partial_anno_at_forecast) != H):
        print("ERROR: File forecasting_inference: inconsistent partial ", \
              "annotation at forecast horizons! \n")
        exit(1)
    
    #----compute state probabilities at forecast horizons
    #normalization of Gamma_T
    Gamma_T = Gamma_T / np.sum(Gamma_T)
    
    #----initial case h = 0
    for k in partial_anno_at_forecast[0]:
        state_probs[0, k] = np.sum(A[:, k] * Gamma_T)
        
    state_probs[0, :] = state_probs[0, :] / np.sum(state_probs[0, :])
               
    #----for h = 1, ..., H-1
    for h in range(1, H):
        for k in partial_anno_at_forecast[h]:
            state_probs[h, k] = np.sum(A[:, k] * state_probs[h-1, :])
        
        state_probs[h, :] = state_probs[h, :] / np.sum(state_probs[h, :])
                
    # assertions
    assert(np.sum(np.isnan(state_probs)) == 0)
    assert(np.sum(state_probs < 0.) == 0)
    assert(np.sum(state_probs > 1.) == 0)
    
    
    return state_probs


#------------------------------------------BEGIN OPTIMAL POINT FORECAST

## @forecasting_one_seq: OPTIMAL POINT FORECAST 
#  @brief Compute the H-steps ahead forecasting on the given sequence. 
#   At each time-step T+h, the expectation of X_t knowing X_{1-order}^{t-1} 
#   is computed (that is the predictive distribution mean).
#   NB: If order = 0, that is for PHMC model, the marginal distribution of 
#   X_t depends of no past values. This case is supported thanks to 
#   compute_means function.
#
#  @param A
#  @param ar_coefficients
#  @param ar_intercepts
#  @param Gamma_T K length array
#  @param past_values order x dimension array of initial values
#  @param H
#  @param partial_anno_at_forecast
#
#  @return A Hxdimension array of predicted values
# 
def forecasting_one_seq(A, ar_coefficients, ar_intercepts, Gamma_T, \
                        past_values, H, partial_anno_at_forecast=[]):
            
    #---hyper-parameters
    nb_regimes = len(ar_coefficients)
    order = len(ar_coefficients[0])
    dimension = ar_intercepts[0].shape[0]
    
    """
    #assertions
    assert(past_values.shape == (order, dimension)) 
    assert(nb_regimes == Gamma_T.shape[0])
    assert(np.sum(Gamma_T) != 0)
    """
        
    #---probabilities of states at forecast horizons: H x nb_regimes array
    pred_probs = compute_state_probs_at_forecast(A, Gamma_T, H,  \
                                                 partial_anno_at_forecast)     
 
    #---total X values 
    total_X = np.zeros(shape=(order+H, dimension), dtype=np.float64)    
    total_X[0:order, :] = past_values

    #---forecasting begins
    for t in range(order, H+order, 1):           
         
        #---the conditional means of X_t within each regime
        # nb_regimes x dimension array
        means = compute_means(ar_coefficients, ar_intercepts,\
                              total_X[(t-order):t, :], nb_regimes, \
                              order, dimension)
                
        #---prediction equals the weighted sum of conditional means
        for i in range(nb_regimes):
            means[i, :] = means[i, :] * pred_probs[t-order, i]
            
        total_X[t, :] = np.sum(means, axis=0)
                         
    
    return total_X[order:, :]


#-----------------------------------------------BEGIN MONTE CARLO FORECAST
## @fn forecasting_one_seq_monte_carle: 
#  @brief At each time-step T+h, one sample is generated from the predictive 
#   density
#
#  @return A Hxdimension array of predicted values
#
def forecasting_one_seq_monte_carle(A, ar_coefficients, ar_intercepts, \
                                    sigmas, innovation, Gamma_T, past_values, H):
            
    #---hyper-parameters
    nb_regimes = len(ar_coefficients)
    order = len(ar_coefficients[0])
    dimension = ar_intercepts[0].shape[0]
    
    """
    #assertions
    assert(past_values.shape == (order, dimension)) 
    assert(nb_regimes == Gamma_T.shape[0])
    assert(np.sum(Gamma_T) != 0)
    """
 
    #---total X values 
    total_X = np.zeros(shape=(order+H, dimension), dtype=np.float64)    
    total_X[0:order, :] = past_values   
        
    #---H x nb_regimes array
    pred_probs = compute_prediction_probs(A, Gamma_T, H) 
    

    #---forecasting begins
    for t in range(order, H+order, 1):           
         
        #conditional means of X_t within each regime
        # nb_regimes x dimension array
        cond_means = compute_means(ar_coefficients, ar_intercepts,\
                              total_X[(t-order):t, :], nb_regimes, \
                              order, dimension)
        
        #sample hat{X}_t from the conditional density of X_t 
        (_, sample) = cond_density_sampling(cond_means, sigmas, innovation,\
                                            pred_probs[t-order])
        total_X[t, :] = sample      
                        
            
    return total_X[order:, :]


#-----------------------------------------------BEGIN QUANTILES
## @fn forecasting_one_seq_quantiles: 
#  @brief At each time-step T+h, one sample is generated from a confidence
#   interval of the predictive density
#  @return A Hxdimension array of predicted values
#
def forecasting_one_seq_quantiles(A, ar_coefficients, ar_intercepts, \
                                  sigmas, innovation, Gamma_T, past_values, H):
            
    #---hyper-parameters
    nb_regimes = len(ar_coefficients)
    order = len(ar_coefficients[0])
    dimension = ar_intercepts[0].shape[0]
    
    """
    #assertions
    assert(past_values.shape == (order, dimension)) 
    assert(nb_regimes == Gamma_T.shape[0])
    assert(np.sum(Gamma_T) != 0)
    """
 
    #---total X values 
    total_X = np.zeros(shape=(order+H, dimension), dtype=np.float64)    
    total_X[0:order, :] = past_values   
        
    #---H x nb_regimes array
    pred_probs = compute_prediction_probs(A, Gamma_T, H) 
    

    #---forecasting begins
    for t in range(order, H+order, 1):           
         
        #conditional means of X_t within each regime
        # nb_regimes x dimension array
        cond_means = compute_means(ar_coefficients, ar_intercepts,\
                              total_X[(t-order):t, :], nb_regimes, \
                              order, dimension)
        
        # list_q=[0.25, 0.75]: use in CMAPSS HMP
        # list_q=[0.495, 0.505]: proposition for anesthesia data
        total_X[t, :] = sample_from_conf_interval_of_CD(cond_means, sigmas, \
                                                        innovation,\
                                                        pred_probs[t-order], \
                                                        list_q=[0.25, 0.75])
                    
    return total_X[order:, :]



#/////////////////////////////////////////////////////////////////////////////
#           H-STEP AHEAD OPTIMAL POINT FORECASTING
#/////////////////////////////////////////////////////////////////////////////
    
## @fn inverse_difference
#  @brief First order difference inverting
#
def inverse_difference(first_val, predictions):
    total_data = np.vstack((first_val, predictions))
    
    inver_diff = np.cumsum(total_data, axis=0)
    
    return inver_diff[1:, :]
    
    
## @fn
#
#  @param diff1 True if the data have been differentiate once before forecasting. 
#   If True scaler is ignored
#
def sliding_forecasting(model_file, time_series, H, partial_anno=None, \
                        scaler=None, diff1=False, use_anno_at_forecast=False):
        
    print("----------------LOG-INFO: sliding_forecasting_error")
    print("with partial_anno = ", True if (partial_anno != None) else False )
    print("with data scaling = ", True if (scaler != None) else False)
    print("with data differenced once = ", diff1)
    print("use_anno_at_forecast = ", use_anno_at_forecast)
    print()
    
    #model loading
    infile = open(model_file, 'rb')
    phmc_var = pickle.load(infile)
    infile.close()
    
    #required phmc_var parameters
    innovation = "gaussian"
    A = phmc_var[1]
    Pi = phmc_var[2]
    ar_coefficients = phmc_var[5]
    ar_intercepts = phmc_var[7]     
    sigma = phmc_var[6]
    
    #hyper-parameters 
    order = len(ar_coefficients[0])
    M = A.shape[0]  
    
    #assertions
    assert(np.sum(A < 0.0) == 0)
    assert(np.sum(np.isnan(A)) == 0)
    
    if(diff1):
        assert(scaler == None)
        
    if(partial_anno == None):
        assert(use_anno_at_forecast == False)
        
    #data must be standardized
    standardization = True if (scaler != None) else False
        
    #----------H-step sliding forecasts   
        
    #---data standardization or differentiation
    if(diff1):
        stand_timeseries = np.diff(time_series, n=1, axis=0)
    elif(standardization):
        stand_timeseries = scaler.transform(time_series)
    else:
        stand_timeseries = time_series
        
    #time series length
    T = stand_timeseries.shape[0]
    
    #first projection time-step
    projec_t = order + 1
    nb_projec = 0
    end_t = T - 1
    
    #outputs initialization
    if(not diff1):
        total_predictions = stand_timeseries[0:(order+2), :]
    else:
        total_predictions = time_series[0:(order+2), :]
        
    while(projec_t < end_t):
        
        #---data splitting
        all_past_values = stand_timeseries[0:(projec_t+1), :]
        order_past_values = stand_timeseries[(projec_t+1-order):(projec_t+1), :]     
        
        #---log-likelihood from 0 to forecast origine
        LL = compute_LL(ar_coefficients, ar_intercepts, sigma, innovation, \
                        all_past_values)
            
        #---compute Gamma_T
        if(partial_anno == None):
            Gamma_T = compute_gamma_BFB_algo(M, LL, A, Pi)[-1, :]
        else:
            Gamma_T = compute_gamma_BFB_algo(M, LL, A, Pi, \
                                partial_anno[0:(LL.shape[0])])[-1, :]
                    
        #---forecasting
        if(projec_t >= (T - H)):
            _H = T - 1 - projec_t
        else:
            _H = H
            
        if(use_anno_at_forecast):
            h_step_predictions = \
                    forecasting_one_seq(A, ar_coefficients, ar_intercepts, Gamma_T, \
                                        order_past_values, _H, \
                                    partial_anno[(LL.shape[0]):(LL.shape[0]+_H)])
        else:
            h_step_predictions = forecasting_one_seq(A, ar_coefficients, \
                                                     ar_intercepts, Gamma_T, \
                                                     order_past_values, _H)
        
        #assertion
        assert(h_step_predictions.shape[0] == _H)
        
        #---forecast error computing 
        if(not diff1):
            total_predictions = np.vstack((total_predictions, h_step_predictions))
        else:
            #invers differentiation
            tmp = inverse_difference(time_series[projec_t+1, :], h_step_predictions)
            total_predictions = np.vstack((total_predictions, tmp))
            
        #----next projection time, used in prognostic machine health
        projec_t = projec_t + H
        
        nb_projec += 1
             
    #assertion
    #assert(total_predictions.shape[0] == (nb_projec*H + order + 2))
    
    #---back to original scale
    if(diff1):
        original_scale_predictions = total_predictions                
    elif(standardization):
        original_scale_predictions = scaler.inverse_transform(total_predictions)
    else:
        original_scale_predictions = total_predictions
        
    
    return original_scale_predictions


## @fn
#  @brief Compute several estimates of H-step ahead forecast error. 
#
#  @param begin_t Must be strictly greater than order.
#  @param L Sliding windows size
#  @param set_of_scalers List of scalers to be used to standardize time series.
#   One scaler per time series.
#  @param set_partial_anno Partial annotations form timeseries defined as a 
#   list of T_s-order 1D arrays of possible states at each 
#
#  @param diff1 True if the data have been differentiate once before forecasting. 
#   If True set_of_scalers is ignored
#
#  @return Four arrays where lines correspond to estimations of forecast 
#   error metrics and columns correspond to data dimensions.
#
def sliding_forecasting_error(model_file, set_of_time_series, H, begin_t, L, \
                              set_partial_anno=None, set_of_scalers=None, \
                              diff1=False, use_anno_at_forecast=False):
    
    print("----------------LOG-INFO: sliding_forecasting_error")
    print("H={}, begin_t={}, L={}".format(H, begin_t, L))
    print("with partial_anno = ", True if (set_partial_anno != None) else False )
    print("with data scaling = ", True if (set_of_scalers != None) else False)
    print("with data differenced once = ", diff1)
    print("use_anno_at_forecast = ", use_anno_at_forecast)
    print()
    
    #model loading
    infile = open(model_file, 'rb')
    phmc_var = pickle.load(infile)
    infile.close()
    
    #required phmc_var parameters
    innovation = "gaussian"
    A = phmc_var[1]
    Pi = phmc_var[2]
    ar_coefficients = phmc_var[5]
    ar_intercepts = phmc_var[7]     
    sigma = phmc_var[6]
    
    #hyper-parameters 
    order = len(ar_coefficients[0])
    M = A.shape[0]  
    
    #nb time series
    N = len(set_of_time_series)
    
    #assertions
    assert(np.sum(A < 0.0) == 0)
    assert(np.sum(np.isnan(A)) == 0)
    assert(begin_t > order)
    
    if(diff1):
        assert(set_partial_anno == None)
    
    if(set_partial_anno != None):
        assert(len(set_partial_anno) == N)
    else:
        assert(use_anno_at_forecast == False)
            
    #data must be standardized
    standardization = True if (set_of_scalers != None) else False
    
    #outputs
    total_MBias = []
    total_RMSE = [] 
    total_NRMSE = [] 
    total_MAPE = []
    
    for s in range(N):
                
        #---data standardization or differentiation
        if(diff1):
            stand_timeseries_s = np.diff(set_of_time_series[s], n=1, axis=0)
        elif(standardization):
            stand_timeseries_s = set_of_scalers[s].transform(set_of_time_series[s])
        else:
            stand_timeseries_s = set_of_time_series[s]
            
        #time series length
        T_s = stand_timeseries_s.shape[0]
        
        #---first projection time-step
        projec_t = begin_t
                
        #---end projection time-step
        end_t = T_s - H 
        
        #---roling H-step prediction over the s^th time series
        Bias = []
        NBias = []    #normalized bias   
                
        while(projec_t < end_t):
            #---data splitting
            all_past_values = stand_timeseries_s[0:(projec_t+1), :]
            order_past_values = stand_timeseries_s[(projec_t+1-order):(projec_t+1), :]     
        
            #---log-likelihood from 0 to forecast origine
            LL = compute_LL(ar_coefficients, ar_intercepts, sigma, innovation, \
                            all_past_values)
                
            #---compute Gamma_T
            #no partial annotations provided
            if(set_partial_anno == None):  
                Gamma_T = compute_gamma_BFB_algo(M, LL, A, Pi)[-1, :]
            else:
                Gamma_T = compute_gamma_BFB_algo(M, LL, A, Pi, \
                                set_partial_anno[s][0:(LL.shape[0])])[-1, :]
            
            #---forecasting
            #use partial annotation at forecast horizon
            if(use_anno_at_forecast):
                predictions = \
                    forecasting_one_seq(A, ar_coefficients, ar_intercepts, \
                            Gamma_T, order_past_values, H, \
                            set_partial_anno[s][(LL.shape[0]):(LL.shape[0]+H)])
            else:
                predictions = forecasting_one_seq(A, ar_coefficients, ar_intercepts, \
                                                  Gamma_T, order_past_values, H)
            
            #assertion
            assert(predictions.shape[0] == H)
            
            #---back to original scale 
            # Define/compute observed and predicted values at t = projec_t+H
            if(diff1):
                obs_x =  set_of_time_series[s][(projec_t+H+1), :]
                
                original_scale_predictions = \
                        inverse_difference(set_of_time_series[s][projec_t+1, :], \
                                           predictions)
                pred_x = original_scale_predictions[-1, :]
            
            else:
                obs_x =  set_of_time_series[s][(projec_t+H), :]
                
                if(standardization):
                    original_scale_predictions = \
                            set_of_scalers[s].inverse_transform(predictions)
                else:
                    original_scale_predictions = predictions
                    
                pred_x = original_scale_predictions[-1, :]  
        
            #---forecast error computing 
            Bias.append( (obs_x - pred_x) )
            NBias.append( (obs_x - pred_x)/obs_x )         
            
            #----next projection time, used in prognostic machine health
            projec_t = projec_t + L
            
        #NB: from which number of projections it is pertinent to compute RMSE, 
        # etc. We chose 10 
        # use if(len(Bias) >= 0) when normalized scores are not used
        if(len(Bias)  >= 10):   
            #---forecast error of the s^th time series
            (MBias_, RMSE_, NRMSE_, MAPE_) = performance_metrics(Bias, NBias)
            total_MBias.append(MBias_)
            total_RMSE.append(RMSE_)
            total_NRMSE.append(NRMSE_)
            total_MAPE.append(MAPE_)   
        else:
            print("s={}, nb_projections = {}".format(s, len(Bias)))
                      
    total_MBias = np.vstack(total_MBias)
    total_RMSE = np.vstack(total_RMSE)
    total_NRMSE = np.vstack(total_NRMSE)
    total_MAPE = np.vstack(total_MAPE)   
                       
    return (total_MBias, total_RMSE, total_NRMSE, total_MAPE) 


## @fn performance_metrics
#  @brief Computes MBias, RMSE, NRMSE and MAPE of H-ahead prediction over 
#   a single time series
#
#  @param Bias List of length N
#  @param NBias List of length N
#
#  @return Four arrays of length dimension
#
def performance_metrics(Bias, NBias):
    
    Bias = np.vstack(Bias)
    NBias = np.vstack(NBias)
    
    MBias = np.mean(Bias, axis=0)    
    RMSE = np.sqrt(np.mean(Bias**2, axis=0))
    
    NRMSE = np.sqrt(np.nanmean(NBias**2, axis=0))
    MAPE = np.nanmean(np.abs(NBias), axis=0)
    
    return (MBias, RMSE, NRMSE, MAPE)




