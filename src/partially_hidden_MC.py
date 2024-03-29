import numpy as np
from scipy.stats import dirichlet
#####################################################################################
##  @package PHMC
#   Partially Hidden Markov Chain
#
#  Let us consider a regime switching AR process X modeled by a probability 
#  distribution parameterized by \Theta^{(X)} = {Theta^{(X,z)}, z=1, ..., nb_regime}, 
#  where \Theta^{(X,z)} = {\phi_{0,z}, ..., \phi_{order,z}, sigma_z},
#  and X^{(1)}, ..., X^{(S)} S realizations of X.  \n
#  Regimes are modeled by a partially hidden Markov chain process 
#  Z having M possible states, parameterized by Theta^{(Z)} = (A, Pi) where 
#  A is the matrix of transition probabilities and Pi the initial state 
#  probability Pi.
#  Let Z^{(1)}, ..., Z^{(T)} the states associated with the realizations of X.\n
#  Let sigma^{(s)}_t in {0, 1, 2, ..., M-1} u {-1} with: \n
#       * sigma^{(s)}_t = -1  if Z^{(s)}_t is laten \n
#       * otherwise Z^{(s)}_t is observed 
#  
#  So for each couple (X^{(s)}, Z^{(s)}), one runs the Backward-Forward-Backward
#  recursion algorithm.
#  Which returns the time dependent transition probabilities
#  Xi^{(s)}_t = P(Z^{(s)}_{t+1}=j, Z^{(s)}_t=i | X^{(s)}, Sigma^{(s)}, Theta^{(X)}, Theta^{(Z)}),
#  as well as the probability to observe state i at time t 
#  gamma^{(s)}_t = P(Z^{(s)}_t=i | X^{(s)}, Sigma^{(s)}, Theta^{(X)}, Theta^{(Z)})
#  and the likelihood P(X^{(s)} | Theta^{(X)}, Theta^{(Z)}).
#
#  From the set of probabilities obtained for each couples (X^{(s)}, Z^{(s)}), 
#  one can update MC parameters A and Pi.
#
     

class PHMC():
    
    # Class attributes
               
    def __init__(self, M, init_method):   
        
        self.nb_states = M
        
        #-------random initialization
        if(init_method == "rand"):
            self.Pi = np.zeros(dtype=np.float64, shape=(1, M))
            concentration_param = [1 for i in range(M)]
            self.Pi[0, :] = dirichlet.rvs(concentration_param, 1)[0] 
        
            self.A = np.zeros(dtype=np.float64, shape=(M, M))
            for i in range(M):
                self.A[i, :] = dirichlet.rvs(concentration_param, 1)[0] 
        
        elif(init_method == "uniform"):
            #-------uniform initialization
            self.Pi = np.ones(dtype=np.float64, shape=(1, M)) / M
            self.A = np.ones(dtype=np.float64, shape=(M, M)) / M
                                                
        else:
            print("ERROR: in class PHMC, unkown initialization method!\n")
            exit(1)
        
        return 
    
        
    ## @brief
    #
    def update_parameters(self, F, list_Gamma):
                
        (Pi_, A_) = update_MC_parameters(F, list_Gamma)
        self.A = A_
        self.Pi = Pi_
                
        return 
    
    
    ## @brief
    #
    def set_parameters(self, Pi_, A_):
        self.Pi = Pi_
        self.A = A_
        
        return
    
    
## @fn update_MC_parameter
#
#  @param F MxM transition frequency matrix
#  
#  @param list_Gamma A list of matrices where list_Gamma[s] is a T_sxM 
#  matrix of time dependent transition probabilities for the s^th observed 
#  sequence. s in 1, ..., S, with S the number of observed sequences.
#  T_s denotes the specific size of sequence s (order initial values excluded).
#
#  @return (Pi, A):
#    * Pi Line vector 1xM, initial state probabilities, Pi[1,k] = P(Z_1 = k).
#    * Matrix MxM, A[i,j] = a_{i,j} = P(Z_t=j|Z_{t-1}=i).
#  
def update_MC_parameters(F, list_Gamma):
    
    #nb regimes
    M = np.shape(F)[0]
    #nb sequences
    S = len(list_Gamma)
    
    #--------------------Pi
    Pi = np.zeros(dtype=np.float64, shape=(1, M))
    
    # states 0 to M-2
    for i in range(M-1):
        aux = 0.0
        for s in range(S):
            aux = aux + list_Gamma[s][0, i]      
        Pi[0, i] = aux / S
         
    # state M-1
    Pi[0, M-1] = max(0, 1 - np.sum(Pi[0, 0:(M-1)]))
    #normalization: probs sum at one
    Pi[0, :] = Pi[0, :] / np.sum(Pi[0, :])
    
    #assertion: valid value domain
    assert(np.sum(np.isnan(Pi[0, :])) == 0)
    assert(np.sum(Pi[0, :] < 0.) == 0)
    assert(np.round(np.sum(Pi), 5) == 1.)
            
    #--------------------A: A equals F normalized
    for i in range(M):
        
        state_i_freq = 0.0
        for s in range(S):
            state_i_freq = state_i_freq + np.sum(list_Gamma[s][:, i])                   
            
        #NB: if state_i_freq = 0 then state i never reached
        # states 0 to M-2
        F[i, :] = F[i, :] / (state_i_freq + np.finfo(0.).tiny)
        # state M - 1
        F[i, M-1] = max(0, 1 - np.sum(F[i, 0:(M-1)]))
        # normalization: probs sum at one                     
        F[i, :] = F[i, :] / np.sum(F[i, :])
                        
        #assertion: valid value domain
        assert(np.sum(np.isnan(F[i, :])) == 0)
        assert(np.sum(F[i, :] < 0.) == 0)
        assert(np.round(np.sum(F[i, :]), 5) == 1.)
            
    return (Pi, F)


## @fn update_F_from_Xi
#
#  @param F MxM transition frequency matrix
#  @param Xi A 3D matrix where Xi[t, , ] is a MxM Matrix of
#  time dependent transition probabilities
#
#  @return Updates and returns F.
#  
def update_F_from_Xi(F, Xi):
    
    #dimension
    (T, _, _) = np.shape(Xi)
    
    for t in range(T):
        F = F + Xi[t, :, :]
    
    return F
