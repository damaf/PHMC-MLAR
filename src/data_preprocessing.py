#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 16 14:57:32 2021

@author: dama-f
"""

import numpy as np
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), \
                                        "../../model_validation/code")))
from preprocessing import build_event_categories 


#------------------------------------------------------------------------------
# SIMULATED ANESTHESIA PROFILES
#------------------------------------------------------------------------------    
################################### UTILS FUNCTIONS

## @fn event_category_definition
#  @brief 
#
#  @param nb_stimulus_level The number of stimulus levels to be considered.
#   Can take values 1, 2 or 3.  
#
#  @return
#   * events_to_be_ignored
#   * event_categories Dictonary of event categories where keys are 
#     indices associated with categories and values are lists of event names
#   * categories_name Dictionary of categories' name where keys are categories' 
#     index and values are associated names
#   * associative_dict Dictonary that associates event categories to states
#     where keys are state numbers and values are lists of event names.
#     Note that a single state can be associated with different event categories
#   * nb_classes_linked_to_evnt The number of states associated with the 
#     nb_categories of event categories
#
def associate_event_categories_to_states(nb_stimulus_level):
            
    if(nb_stimulus_level not in [1, 2, 3]):
        print("Error: file data_preprocessing.py: nb_stimulus_level must ", \
              "be in [1, 2, 3]")
        sys.exit(1)
        
    #-----------------------Build event categories
    (events_to_be_ignored, \
     event_categories, categories_name) = build_event_categories()
    
    minor_stimulus = event_categories[0]
    medium_stimulus = event_categories[1]
    major_stimulus = event_categories[2]
    hypnotic = event_categories[3]  
    morphinic = event_categories[4] 
    curare = event_categories[5]
    ventilation_control = event_categories[6]
    inflation_pneumo = event_categories[7]
    deflation_pneumo = event_categories[8]
        
    #----------------------Associate categories to states
    #---initialization of outputs
    associative_dict = dict()
    
    #---associated event categories to states
    #Note that a single state can be associated with different event categories
    
    #hidden states: events_to_be_ignored, events having no effects on
    #patient parameters and events whose effects are fuzzy/unknown
    tmp = events_to_be_ignored
    tmp.extend(curare)
    tmp.extend(ventilation_control)
    tmp.extend(inflation_pneumo)
    tmp.extend(deflation_pneumo)
    associative_dict[-1] = tmp
    
    if(nb_stimulus_level == 1):
        # 1 stimulus level 
        tmp = []
        tmp.extend(minor_stimulus)
        tmp.extend(medium_stimulus)
        tmp.extend(major_stimulus)
        associative_dict[0]  = tmp
        # hypnotic
        associative_dict[1] = hypnotic
        # morphinic
        associative_dict[2] = morphinic
        
        nb_classes_linked_to_evnt = 3
        
    elif(nb_stimulus_level == 2):
        # first stimulus level
        tmp = []
        tmp.extend(minor_stimulus)
        tmp.extend(medium_stimulus)
        associative_dict[0]  = tmp
        # second stimulus level 
        associative_dict[1] = major_stimulus
        # hypnotic
        associative_dict[2] = hypnotic
        # morphinic
        associative_dict[3] = morphinic
        
        nb_classes_linked_to_evnt = 4
        
    else:
        # 3 stimulus levels 
        associative_dict[0] = minor_stimulus
        associative_dict[1] = medium_stimulus
        associative_dict[2] = major_stimulus
        # hypnotic
        associative_dict[3] = hypnotic
        # morphinic
        associative_dict[4] = morphinic
        
        nb_classes_linked_to_evnt = 5
        
    return (associative_dict, nb_classes_linked_to_evnt)


## @fn state_of_event
#  @return The state associated with the category the given event belongs to. 
#   NB: by convension -1 is associated with non relevant event
# 
def state_of_event(associative_dict, event_name):
              
    #---research begins                          
    state = -2  
      
    for key in associative_dict.keys():
        if(event_name in associative_dict[key]):
            state = key
            break
        
    if((state == -2) and event_name != ''):
        print("Error: in function state_of_event: the given event {} is unknown".format(event_name))
        sys.exit(1)
        
    return state
    


################################### SET OF POSSIBLE STATES AT EACH TIME STEP
#  @fn set_of_possible_states
#  @brief 
#
#  @param sync_events List of (list of T_s lists). Each entry correponds
#   one event sequence in which events have been synchronized with 
#   time series sampling time. 
#   For the n^th sequence sync_events[n][t] is a list of events that have 
#   been synchronized with time step t.
#  @param nb_regimes number of possible states
#  @param one_stimulus If True a single stimulus level is used otherwise three 
#   are used.
#
#  @return
#   * states
#   * associative_dict
#   * nb_classes_linked_to_evnt   
#        
def set_of_possible_states_from_event_seq(sync_events, nb_regimes, one_stimulus):
    
    #---associate event categories to states
    (associative_dict, nb_classes_linked_to_evnt) = \
                            associate_event_categories_to_states(one_stimulus)
    
    #initialization of outputs
    states = []
           
    #number of event sequences
    N = len(sync_events)
    
    for n in range(N):
        states.append([])
        nb_timesteps = len(sync_events[n])
        
        for t in range(nb_timesteps):
            
            # no events have been synchronized with time-step t
            # therefore all states remain possible
            if(len(sync_events[n][t]) == 0):
                states[n].append( \
                     np.array([s for s in range(nb_regimes)], dtype=np.int32) )
                
            else:
                # at least one event have been synchronized with time-step t
                # find the states associated with these events              
                concomitant_states = [ state_of_event(associative_dict, evt) \
                                          for evt in sync_events[n][t] ]       
                    
                # states without duplicates and without -1
                # NB: -1 denotes the set of events to be ignored
                tmp = list(set(concomitant_states) - set([-1])) 
                
                if(len(tmp) == 0):  # all states are possible
                    states[n].append( \
                          np.array([s for s in range(nb_regimes)], dtype=np.int32) )
                else:
                    states[n].append(np.array(tmp, dtype=np.int32))
                                
    
    return (states, associative_dict, nb_classes_linked_to_evnt)

