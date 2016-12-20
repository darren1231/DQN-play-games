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

class environment():
    def __init__(self,game_name):
        if game_name=="pong":
            # open up a game state to communicate with emulator
            self.game_state = game.GameState()
            self.action_number=3
    
    def random_action(self):
        # open up a game state to communicate with emulator
        
        do_nothing = np.zeros(self.action_number)
        do_nothing[0] = 1
        x_t, r_0, terminal = self.game_state.frame_step(do_nothing)
        return  x_t, r_0, terminal
        
    def run_pick_action(self,a_t):
        
        x_t1_col, r_t, terminal = self.game_state.frame_step(a_t)
        return x_t1_col, r_t, terminal
      
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
#env=environment("pong")