import sys
import numpy as np
import concurrent.futures
from gaussian_innovation import Gaussian_X
from scaled_backward_forward_backward_recursion import BFB
from partially_hidden_MC import PHMC, update_F_from_Xi, update_MC_parameters



################################ BEGIN UTILS FUNCTIONS

## @fn compute_ll
#  @brief Another way to compute log-likelihood ll.
#   Likelihood of X_t can be computed as the weighted sum of the likehood within 
#   each regime:
#   P(X_t|past_values) = \sum_k P(X_t|past_values, Z_t=k) * P(Z_t=k).
#   Note that weights are regimes' probability at timestep t provided in
#   parameter list_Gamma.
#   In the end ll is equal to the sum of log( P(X_t|past_values) ) over all
#   timestep for all time series.  
#    
#  NB: Note that initial values are not included.
#    
def compute_ll(X_process, list_Gamma):
    
    log_ll = 0.0
    
    for s in range(len(list_Gamma)):
        
        LL_s = X_process.total_likelihood_s(s)
        
        for t in range(list_Gamma[s].shape[0]):
            
            log_ll = log_ll + np.log(np.sum( LL_s[t, :] * \
                                            list_Gamma[s][t, :]))
            
    return log_ll


## @fn run_BFB_s
#  @brief Run BFB algorithm on the s^{th} sequence.
#
#  @return BFB outputs and sequence index which is used is used in main 
#   process to properly collect the results of different child processes.
#
def run_BFB_s(M, LL, A, Pi, states_s, s):
        
    #run backward-forward-backward on the s^th sequence
    return (s,  BFB(M, LL, states_s, A, Pi)  ) 

## @fn M_Z_step
#  @brief Run the step M_Z substep of EM in which PHMC_process parameters are 
#   updated.
#
def M_Z_step(F, list_Gamma):
    return update_MC_parameters(F, list_Gamma)
    

## @fn compute_norm
#  @brief Compute the L1-norm of the difference between previous estimation and
#   current estumation of parameters.
#
def compute_norm(prev_estimated_param, PHMC_process, X_process):
        
    A = prev_estimated_param[1]
    Pi = prev_estimated_param[2]
    ar_coefficients = prev_estimated_param[5]
    ar_intercepts = prev_estimated_param[7]  
    Sigma = prev_estimated_param[6]
    
    norm = np.sum(np.abs(PHMC_process.A - A))
    norm = norm + np.sum(np.abs(PHMC_process.Pi - Pi))
    
    for k in range(X_process.nb_regime):
    
        norm = norm + np.sum(np.abs(X_process.intercept[k] - ar_intercepts[k]))
        norm = norm + np.sum(np.abs(X_process.sigma[k] - Sigma[k]))
    
        for i in range(X_process.order):
            norm = norm + np.sum(np.abs(X_process.coefficients[k][i] - \
                                        ar_coefficients[k][i]))
    return norm

################################ END UTILS FUNCTIONS
   
    

################################ BEGIN RANDOM INITIALIZATION OF EM

## @fn
#  @brief Initialize model parameters randomly then run nb_iters_per_init 
#   steps of EM and return the resulting estimation of parameters.
#
def run_EM_on_init_param(X_dim, X_order, nb_regimes, data, initial_values, \
                         states, innovation, var_init_method, hmc_init_method, \
                         nb_iters_per_init):
    
    # re-seed the pseudo-random number generator.
    # This guarantees that each process/thread runs this function with
    # a different seed.
    np.random.seed()
        
    assert(innovation == "gaussian")
    #----VAR process initialization  
    X_process = Gaussian_X(X_dim, X_order, nb_regimes, data, initial_values, \
                           init_method=var_init_method)              
            
    #----PHMC process initialization
    PHMC_process = PHMC(nb_regimes, init_method=hmc_init_method)
                
    #----run EM nb_iters_per_init times
    return EM(X_process, PHMC_process, states, nb_iters_per_init, \
              epsilon=1e-6, log_info=False)
 

## @fn
#  @brief EM initialization procedure.
#   Parameters are initialized nb_init times.
#   For each initialization, nb_iters_per_init of EM is executed.
#   Finally, the set of parameters that yield maximum likelihood is returned.
#
#  @param var_init_method
#  @param hmc_init_method
#  @param nb_init Number of initializations.
#  @param nb_iters_per_init Number of EM iterations.
#
#  @return The set of parameters having the highest likelihood value.
#   (ar_intercept, ar_coefficient, ar_sigma, Pi_, A_).
#
def random_init_EM(X_dim, X_order, nb_regimes, data, initial_values, states, \
                   innovation, var_init_method, hmc_init_method, \
                   nb_init, nb_iters_per_init):
        
    print("********************EM initialization begins**********************")        
    
    #------Runs EM nb_iters_per_init times on each initial values 
    output_params = []
    
    #### Begin multi-process parallel execution   
    with concurrent.futures.ProcessPoolExecutor(max_workers=20) as executor:           
        futures = []
        for _ in range(nb_init):
            futures.append( executor.submit(run_EM_on_init_param, X_dim, \
                                            X_order, nb_regimes, data, \
                                            initial_values, states, \
                                            innovation, var_init_method, \
                                            hmc_init_method, nb_iters_per_init) )
             
        for f in concurrent.futures.as_completed(futures):
            #print("Child process_X, I am done !")
            try:
                output_params.append(f.result())
            except Exception as exc:
                print("Warning: EM initialization generates an exception: ", exc)
    #### End multi-process parallel execution    
        
    #------Find parameters that yields maximum likelihood
    print("==================LIKELIHOODS====================")
    print("Number of EM initializations = ", nb_init)
    print("Number of SUCCESSFUL EM initializations = ", len(output_params))
    
    max_ll = -1e300  
    max_ind = -1
    nb_succ_init = len(output_params)
    
    for i in range(nb_succ_init):     
        print(output_params[i][0])

        if(output_params[i][0] > max_ll):
            max_ll = output_params[i][0]
            max_ind = i
        
    print("==================INITIAL VALUES OF PARAMETERS====================")
    print("------------------VAR process----------------------")
    print("total_log_ll= ", output_params[max_ind][0])
    print("ar_coefficients=", output_params[max_ind][5])
    print("intercept=", output_params[max_ind][7])
    print("sigma=", output_params[max_ind][6])
    print()
    print("------------------Markov chain----------------------")
    print("Pi=", output_params[max_ind][2])
    print("A=", output_params[max_ind][1])
    
    print("*********************EM initialization ends***********************")
    
    return (output_params[max_ind][7], output_params[max_ind][5], \
            output_params[max_ind][6], output_params[max_ind][2], \
            output_params[max_ind][1])
"""
((total_log_ll + X_process.init_val_ll()), PHMC_process.A, PHMC_process.Pi, \
list_Gamma, list_Alpha, X_process.coefficients, X_process.sigma, \
X_process.intercept, X_process.psi)
"""
################################ END RANDOM INITIALIZATION OF EM



################################ BEGIN  LEARNING 

## @fn
#  @brief Learn PHMC-LAR model parameters by EM algortihm.
#
#  @param X_order Autoregressive order, must be positive.
#  @param nb_regimes Number of switching regimes.
#
#  @param data List of length S, where S is the number of observed
#  time series. data[s], the s^th time series, is a T_s x dimension matrix
#  where T_s denotes its size starting at timestep t = order + 1 included.
#
#  @param initial_values List of length S. initial_values[s] is a column vector 
#   orderx1 of initial values associated with the s^th time series.
#
#  @param states List of S state sequences. states[s] is a line vector 1xT_s: \n
#    * if states[s][0,t] = -1 then Z^{(s)}_t is latent.
#    * otherwise states[s][0,t] is in {0, 1, 2, ..., M-1} and Z^{(s)}_t
#      is observed to be equal to states[s][0,t]. M is the number of regimes.
#
#  @param innovation Law of model error terms, only 'gaussian' noises are supported  
#  @param nb_iters Maximum number of EM iterations.
#  @param epsilon Convergence precision. EM will stops when the shift 
#   in parameters' estimate between two consecutive iterations is less than
#   epsilon. L1-norm was used.
#
#  @param var_init_method Initialization method to be considered for VAR 
#   component. Can take two values: "rand1", "rand2" or "datadriven".
#  @param hmc_init_method Initialization method to be considered for HMC 
#   component. Can take two values: "rand" or "uniform". 
#  @param nb_init, nb_iters_per_init Number of random initialization of EM and
#    number of EM iterations per initialization.
#
#  @return Parameters' estimation computed by EM algorithm.
#
def hmc_var_parameter_learning(X_dim, X_order, nb_regimes, data, \
                               initial_values, states, innovation, \
                               nb_iters=500, epsilon=1e-6, \
                               var_init_method='datadriven', \
                               hmc_init_method='uniform', \
                               nb_init=10, nb_iters_per_init=5):
        
    if(innovation != "gaussian"):
        print()
        print("ERROR: file EM_learning.py: the given distribution is not supported!")
        sys.exit(1)
        
    #---------------PHMC process initialization
    PHMC_process = PHMC(nb_regimes, hmc_init_method)
    
    #---------------VAR process initialization  
    X_process = Gaussian_X(X_dim, X_order, nb_regimes, data, initial_values, \
                           var_init_method)
    
    #----random initialization: run EM nb_init times then take the set of 
    #parameters that yields maximum likelihood
    (ar_inter, ar_coef, ar_sigma, Pi_, A_) = \
                                random_init_EM(X_dim, X_order, nb_regimes, \
                                               data, initial_values, states, \
                                               innovation, var_init_method, \
                                               hmc_init_method, \
                                               nb_init, nb_iters_per_init)
    X_process.set_parameters(ar_inter, ar_coef, ar_sigma)
    PHMC_process.set_parameters(Pi_, A_)
    
    #---------------psi parameters estimation
    X_process.estimate_psi_MLE()
           
    #---------------run EM
    return EM(X_process, PHMC_process, states, nb_iters, epsilon)
    
  
## @fn
#  @brief
#
#  @param X_process Object of class RSVARM.
#  @param PHMC_process Object of class PHMC.
#  @param states
#  @param nb_iters
#  @param epsilon
#  @param log_info
#
#  @return Parameters' estimation computed by EM algorithm. 
#
def EM (X_process, PHMC_process, states, nb_iters, epsilon, log_info=True):
    
    #nb observed sequences
    S = len(X_process.data)
    #nb regimes
    M = X_process.nb_regime
    
    #total_log_ll is in [-inf, 0], can be also greater than 0
    prec_total_log_ll = -1e300  
    
    #current/previous estimated parameters
    prev_estimated_param = (-np.inf,  PHMC_process.A.copy(), \
                            PHMC_process.Pi.copy(), [], [], \
                            X_process.coefficients.copy(), \
                            X_process.sigma.copy(), X_process.intercept.copy(),  \
                            X_process.psi)
    curr_estimated_param = {}
    
    #fix max_workers
    if(log_info):
        nb_workers=60   
    else:
        # initialization: 20*3 = 60 where 20 = max_workers_init
        nb_workers=3
           
    #--------------------------------------------nb_iters of EM algorithm
    for ind in range(nb_iters):
    
        #matrix of transition frequency
        F = np.zeros(dtype=np.float64, shape=(M, M))
        #
        list_Gamma = [{} for _ in range(S)]
        #
        list_Alpha = [{} for _ in range(S)]
        #
        total_log_ll = 0.0
            
        #----------------------------begin E-step  
        #### begin multi-process parallel execution
        #list of tasks, one per sequence   
        list_futures = []
        with concurrent.futures.ProcessPoolExecutor(max_workers=nb_workers) as executor:
            #run backward forward backward algorithm on each equence
            for s in range(S):             
                list_futures.append( executor.submit(run_BFB_s, M, \
                                     X_process.total_likelihood_s(s), \
                                     prev_estimated_param[1], \
                                     prev_estimated_param[2], states[s], s) )

            #collect the results as tasks complete
            for f in concurrent.futures.as_completed(list_futures):
                #results of the s^{th} sequence
                (s, (log_ll_s, Xi_s, Gamma_s, Alpha_s)) = f.result()
                #update transition frequency matrix
                F = update_F_from_Xi(F, Xi_s)           
                #Gamma and Alpha probabilities computed on the s^th sequence
                list_Gamma[s] = Gamma_s
                list_Alpha[s] = Alpha_s          
                #total log-likelihood over all observed sequence
                total_log_ll = total_log_ll + log_ll_s  
        #### end multi-process parallel execution
        #----------------------------end E-step
        
        #----------------------------begin M-steps 
        #### begin multi-process parallel execution
        with concurrent.futures.ProcessPoolExecutor(max_workers=1) as executor: 
            #-----M-Z substep
            task = executor.submit(M_Z_step, F, list_Gamma)
            
            #----M-X substep and update Theta_X
            X_process.update_parameters(list_Gamma)  
        
            #collect result of task and update Theta_Z
            (Pi_, A_) = task.result()
            PHMC_process.set_parameters(Pi_, A_)
        #### end multi-process parallel execution
        #----------------------------end M-steps 
        
        #-----------------------begin EM stopping condition        
        delta_log_ll = total_log_ll - prec_total_log_ll        
        log_ll_incr = delta_log_ll / np.abs(total_log_ll + prec_total_log_ll)
        
        abs_of_diff = compute_norm(prev_estimated_param, PHMC_process, X_process)      
        curr_estimated_param = (total_log_ll, PHMC_process.A.copy(), \
                                PHMC_process.Pi.copy(), list_Gamma, \
                                list_Alpha, X_process.coefficients.copy(), \
                                X_process.sigma.copy(), X_process.intercept.copy(),  \
                                X_process.psi)
            
        #EM stops with a warning, ADDED 2021/06/13
        if(np.isnan(abs_of_diff)):  
            print("--------------EM stops with a warning------------")
            print("At iteration {}, NAN values encountered in the estimated ",\
                  "parameters".format(ind+1))
            print("PARAMETERS AT ITERATION {} ARE RETURNED".format(ind))            
    
            return prev_estimated_param
                    
        #EM stops when log_ll decreases, added on 2022/07/20 
        if(delta_log_ll < 0.0):
            print("--------------EM CONVERGENCE------------")
            print("delta_log_ll = {}, is negative".format(delta_log_ll))
            print("PARAMETERS AT ITERATION {} ARE RETURNED".format(ind)) 
            
            return prev_estimated_param
            
        #convergence criterion
        # the criterion on delta_log_ll has been added on 2022/07/20
        if(abs_of_diff < epsilon) or (delta_log_ll < epsilon):             
            #LOG-info
            if(log_info):
                print("--------------EM CONVERGENCE------------")
                print("#EM_converges after {} iterations".format(ind+1))
                print("log_ll = {}".format(total_log_ll))
                print("delta_log_ll = {}".format(delta_log_ll))
                print("log_ll_increase = {}".format(log_ll_incr))
                print("shift in parameter = {}".format(abs_of_diff))
            break       
        else:
            #LOG-info
            if(log_info):
                print("iterations = {}".format(ind+1))
                print("log_ll_alpha = {}".format(total_log_ll))
                print("delta_log_ll = {}".format(delta_log_ll))
                print("log_ll_increase = {}".format(log_ll_incr))
                print("shift in parameter = {}".format(abs_of_diff))
        
            #update prec_total_log_ll
            prec_total_log_ll = total_log_ll
        
            #update prev_estimated_param
            prev_estimated_param = curr_estimated_param
        #-----------------------end EM stopping condition  
             
    #another way to compute log_likelihood
    print("another way to compute log_likelihood = ", compute_ll(X_process, \
                                                                 list_Gamma) )
           
    return curr_estimated_param
      
################################ END LEARNING  



