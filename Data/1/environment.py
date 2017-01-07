# -*- coding: utf-8 -*-
"""
Created on Tue Dec 20 13:22:58 2016

@author: darren
"""
import random
import sys
import cv2
import numpy as np
sys.path.append("Wrapped Game Code/")
import pong_fun as game# whichever is imported "as game" will be used
import gym
import gym_ple

class environment():
    def __init__(self,game_name):
        if game_name=="pong":
            # open up a game state to communicate with emulator
            self.game_name="pong"
            self.game_state = game.GameState()
            self.action_number=3
            
        if game_name=="pygame":
            self.game_name="pygame"
            self.game_state = gym.make('FlappyBird-v0')
            self.action_number=self.game_state.action_space.n
            
    def random_action(self):
        # open up a game state to communicate with emulator
        if self.game_name=="pong":
            do_nothing = np.zeros(self.action_number)
            do_nothing[0] = 1
            x_t, r_0, terminal = self.game_state.frame_step(do_nothing)
            return  x_t, r_0, terminal
            
        if self.game_name=="pygame":
            next_state,reward,done,_ =self.game_state.step(self.game_state.action_space.sample())
            return next_state,reward,done
            
    def run_pick_action(self,action_index):
        if self.game_name=="pong":
            a_t = np.zeros([self.action_number])
            a_t[action_index] = 1
            x_t1_col, r_t, terminal = self.game_state.frame_step(a_t)
            return x_t1_col, r_t, terminal
        
        if self.game_name=="pygame":
            next_state,reward,done,_ =self.game_state.step(action_index)
            #self.game_state.render()
            
            return next_state,reward,done
            
    def pick_action(self,epsilon,readout_t):       
        
        a_t = np.zeros([self.action_number])
        action_index = 0
        if random.random() <= epsilon:
            action_index = random.randrange(self.action_number)
            a_t[action_index] = 1
        else:
            action_index = np.argmax(readout_t)
            a_t[action_index] = 1
        return a_t,action_index
       
        
    def preprocess(self,x_t1_col):
        
        x_t1 = cv2.cvtColor(cv2.resize(x_t1_col, (80, 80)), cv2.COLOR_BGR2GRAY)
        ret, x_t1 = cv2.threshold(x_t1,1,255,cv2.THRESH_BINARY)
        x_t1 = np.reshape(x_t1, (80, 80, 1))
        return x_t1
        
    def refresh_scrren(self):
        test=self.game_state.render()
        return 0
        
    def initialization(self):
        return self.game_state.reset()
"""function test """    
#env=environment("pygame")
#for i in range(1000):
#    env.refresh_scrren()
#    print i
#    #print env.action_space.sample()
##        now_life=env.ale.lives()  
#    next_state,reward,done =env.random_action()  # take a random action
#    #action_index = env.random_action()    
#    print np.array(next_state).shape
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
#print np.array(next_state)     
