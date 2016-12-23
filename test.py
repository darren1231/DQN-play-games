# -*- coding: utf-8 -*-
"""
Created on Thu Dec 22 14:50:10 2016

@author: darren
"""
import environment as env
import tensorflow as tf
env_state=env.environment("pygame")
env_state.refresh_scrren()



while True:
    env_state.refresh_scrren()
    next_state,reward,done =env_state.random_action()  # take a random action

    if done or (reward == -25):
        break















#while True:
#    env_state.refresh_scrren()
#    env_state.refresh_scrren()
#    #print i
#    #print env.action_space.sample()
##        now_life=env.ale.lives()  
#    next_state,reward,done =env_state.random_action()  # take a random action
#    #action_index = env.random_action()    
#    #print np.array(next_state).shape
#    print "reward",reward
#    #print "action_index",action_index
#    
#   #print "action_number",ACTIONS
##        next_life=env.ale.lives()  
#    
##        if next_life<now_life:
##            reward=-1
##        if reward!=0:
##            print reward
#    
#    #print next_state,reward,done
#    if done or (reward == -25):
#        break
#        
#    #print 'episdoe: ',episode,'  step: ',i
##env.monitor.close()
##print np.array(next_state)     
