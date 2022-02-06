# -*- coding: utf-8 -*-
"""
Created on Sun Nov 20 12:50:51 2016

@author: darren
"""

#!/usr/bin/env python


import cv2
import sys
import datetime
#sys.path.append("Wrapped Game Code/")
#import pong_fun as game# whichever is imported "as game" will be used
#import tetris_fun
import random
import numpy as np
import os
from collections import deque
import tensorflow.compat.v1 as tf
tf.disable_eager_execution()
import environment as env

"""This is important. You need to initialize your environment before tensorflow."""
env_state=env.environment("pong")   
#env_state.refresh_scrren()


# import tensorflow as tf
import brain as net

GAME = 'pong' # the name of the game being played for log files
ACTIONS = 2 # number of valid actions
GAMMA = 0.99 # decay rate of past observations
OBSERVE = 100000. # timesteps to observe before training
EXPLORE = 2000000. # frames over which to anneal epsilon
FINAL_EPSILON = 0.0001 # final value of epsilon
INITIAL_EPSILON = 0.0001 # starting value of epsilon
REPLAY_MEMORY = 50000 # number of previous transitions to remember
BATCH = 32 # size of minibatch
FRAME_PER_ACTION = 1
K=1


######################################################################
start = datetime.datetime.now()
store_network_path="temp/my_networks/"
tensorboard_path = "temp/logs/"
out_put_path = "temp/logs_" + GAME

if os.path.exists('temp'):
    pass
else:
    os.makedirs(store_network_path)
    os.makedirs(out_put_path)

pretrain_number=0


######################################################################

def sencond2time(senconds):

	if type(senconds)==type(1):
		h=senconds/3600
		sUp_h=senconds-3600*h
		m=sUp_h/60
		sUp_m=sUp_h-60*m
		s=sUp_m
		return ",".join(map(str,(h,m,s)))
	else:
		return "[InModuleError]:sencond2time(senconds) invalid argument type"

    
def check_load_status(checkpoint,saver,sess):
    
    if checkpoint and checkpoint.model_checkpoint_path:
        saver.restore(sess, checkpoint.model_checkpoint_path)
	#saver.restore(sess, "my_networks/pong-dqn-26000")
        print ("Successfully loaded:", checkpoint.model_checkpoint_path)
    else:
        print ("Could not find old network weights")
    
    print ("Press any key and Enter to continue:")
    # raw_input()



def trainNetwork(s, readout,sess,merged,writer,brain_net):

    
    a,y,train_step=brain_net.cost_function(readout)

    
    
    # store the previous observations in replay memory
    D = deque()

    # printing
    a_file = open(out_put_path  + "/readout.txt", 'w')
    e_file = open(out_put_path  + "/episode.txt", 'w')

    # get the first state by doing nothing and preprocess the image to 80x80x4
    
    x_t, r_0, terminal = env_state.random_action()    
    last_observation=x_t
    x_t = cv2.cvtColor(cv2.resize(x_t, (80, 80)), cv2.COLOR_BGR2GRAY)
    ret, x_t = cv2.threshold(x_t,1,255,cv2.THRESH_BINARY)
    s_t = np.stack((x_t, x_t, x_t, x_t), axis = 2)
    

    # saving and loading networks
    saver = tf.train.Saver()
    # sess.run(tf.initialize_all_variables())
    sess.run(tf.global_variables_initializer())
    checkpoint = tf.train.get_checkpoint_state(store_network_path)
    
    #saver.restore(sess, "new_networks/pong-dqn-"+str(pretrain_number))   
    
    check_load_status(checkpoint,saver,sess)
    


    epsilon = INITIAL_EPSILON
    t = 0
    total_score=0
    positive_score=0
    episode_score=0
    episode=0
    
    #envv = gym.make('Catcher-v0')
    
    while True:
        #envv.render()
        # choose an action epsilon greedily
        readout_t = readout.eval(feed_dict = {s : [s_t]})[0]
        
        #env_state.refresh_scrren()  
        
        a_t,action_index=env_state.pick_action(epsilon,readout_t)
        # scale down epsilon
        if epsilon > FINAL_EPSILON and t > OBSERVE:
            epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORE

        for i in range(0, K):
            # run the selected action and observe next state and reward
            
            x_t1_col, r_t, terminal = env_state.run_pick_action(action_index)
#            x_t1_col = np.maximum(x_t1_col,last_observation)
#            last_observation=x_t1_col
            x_t1 = env_state.preprocess(x_t1_col)
            s_t1 = np.append(x_t1, s_t[:,:,0:3], axis = 2)

#            # without pass reward            
#            if r_t==0.1:
#                train_r_t=0
#            else:
#                train_r_t=r_t
                
            # store the transition in D
            D.append((s_t, a_t, r_t, s_t1, terminal))
            if len(D) > REPLAY_MEMORY:
                D.popleft()
                
        if terminal==True:
            e_file.write(str(episode)+','+str(episode_score)+"\n")
            episode_score=0
            episode+=1
            
        if (terminal==True) and (env_state.game_name=="gym"):
            env_state.initialization()
        
        if r_t==1 or r_t==-1:
            total_score=total_score+r_t
            episode_score=episode_score+r_t
        if r_t==1:
            positive_score=positive_score+r_t

        # only train if done observing
        if t > OBSERVE:
            # sample a minibatch to train on
            minibatch = random.sample(D, BATCH)

            # get the batch variables
            s_j_batch = [d[0] for d in minibatch]
            a_batch = [d[1] for d in minibatch]
            r_batch = [d[2] for d in minibatch]
            s_j1_batch = [d[3] for d in minibatch]

            y_batch = []
            readout_j1_batch = readout.eval(feed_dict = {s : s_j1_batch})
            for i in range(0, len(minibatch)):
                # if terminal only equals reward
                if minibatch[i][4]:
                    y_batch.append(r_batch[i])
                else:
                    #y_batch.append(r_batch[i] + GAMMA * np.max(readout_j1_batch[i]))
                    y_batch.append(r_batch[i] + GAMMA * np.max(readout_j1_batch[i]))                   
                    

            # perform gradient step
            train_step.run(feed_dict = {
                y : y_batch,
                a : a_batch,
                s : s_j_batch})

        # update the old values
        s_t = s_t1
        t += 1

        # save progress every 10000 iterations
        if t % 10000 == 0:
            saver.save(sess, store_network_path + GAME + '-dqn', global_step = t+pretrain_number)
            
            #saver.save(sess, 'new_networks/' + GAME + '-dqn', global_step = t)

        if t % 500 == 0:  
            now=datetime.datetime.now()
            diff_seconds=(now-start).seconds
            time_text=sencond2time(diff_seconds)
            
            result = sess.run(merged,feed_dict = {s : [s_t]})
            writer.add_summary(result, t+pretrain_number)
            a_file.write(str(t+pretrain_number)+','+",".join([str(x) for x in readout_t]) + \
            ','+str(total_score)+ ','+str(positive_score) \
            +','+time_text+'\n')

        # print info
        state = ""
#        if t <= OBSERVE:
#            state = "observe"
#        elif t > OBSERVE and t <= OBSERVE + EXPLORE:
#            state = "explore"
#        else:
#            state = "train"
        print ("TIMESTEP:", t+pretrain_number, "/ ACTION:", action_index, "/ REWARD:", r_t, "/ Q_MAX: %e" % np.max(readout_t),'  time:(H,M,S):' \
        + sencond2time((datetime.datetime.now()-start).seconds))
        print ('Total score:',total_score,' Positive_score:',positive_score,"Epsilon:",epsilon)
        print ("Episode:",episode,"     score",episode_score)
        #print 'Total score:',total_score,' Positive_score:',positive_score,'   up:',readout_t[0],'    down:',readout_t[1],'  no:',readout_t[2]
       
        # write info to files
        
        #if t % 10000 <= 100:
            #a_file.write(",".join([str(x) for x in readout_t]) + '\n')
            #h_file.write(",".join([str(x) for x in h_fc1.eval(feed_dict={s:[s_t]})[0]]) + '\n')
            #cv2.imwrite("logs_pong/frame" + str(t) + ".png", x_t1)
        

def playGame():    
    
    sess = tf.InteractiveSession()
    brain_net = net.Brain(env_state.action_number)
    s, readout = brain_net.createNetwork()

    # merged = tf.merge_all_summaries()
    merged = tf.summary.merge_all()
    # writer = tf.train.SummaryWriter(tensorboard_path, sess.graph)
    writer = tf.summary.FileWriter(tensorboard_path, sess.graph)
    trainNetwork(s, readout,sess,merged,writer,brain_net)

def main():
    playGame()

if __name__ == "__main__":
    main()