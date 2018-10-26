'''
## Play ##
# Run a trained DDPG network on an Open AI gym environment, observe its performance on screen, and save to GIF file (optional)
@author: Mark Sinton (msinto93@gmail.com) 
'''

import os
import argparse
import gym
import tensorflow as tf
import numpy as np
import cv2
import imageio

from train import get_train_args
from utils.network import Actor, Actor_BN

    
def get_play_args(train_args):
    play_params = argparse.ArgumentParser()
    
    # Environment parameters (env must be same as used in training)
    play_params.add_argument("--env", type=str, default=train_args.env, help="Environment to use (must have low dimensional state space (i.e. not image) and continuous action space")
    play_params.add_argument("--random_seed", type=int, default=4321, help="Random seed for reproducability")
    
    # Game parameters
    play_params.add_argument("--num_eps", type=int, default=10, help="Number of episodes to run for")
    play_params.add_argument("--max_ep_length", type=int, default=1000, help="Maximum number of steps per episode")
    
    # Network parameters (these must be same as used in training)
    play_params.add_argument("--dense1_size", type=int, default=train_args.dense1_size, help="Size of first hidden layer in networks")
    play_params.add_argument("--dense2_size", type=int, default=train_args.dense2_size, help="Size of second hidden layer in networks")
    play_params.add_argument("--final_layer_init", type=float, default=train_args.final_layer_init, help="Initialise networks' final layer weights in range +/-final_layer_init")
    play_params.add_argument("--use_batch_norm", type=bool, default=train_args.use_batch_norm, help="Whether or not to use batch normalisation in the networks")
    
    # File/directories
    play_params.add_argument("--ckpt_dir", type=str, default='./ckpts', help="Directory for saving/loading checkpoints")
    play_params.add_argument("--ckpt_file", type=str, default=None, help="Checkpoint file to load (if None, load latest ckpt)")
    play_params.add_argument("--record_dir", type=str, default='./video', help="Directory to store recorded gif of gameplay (if None, do not record)")
    
    return play_params.parse_args()

    
def play(args):
    # Create environment
    env = gym.make(args.env)
    state_dims = env.observation_space.shape
    action_dims = env.action_space.shape
    action_bound_low = env.action_space.low
    action_bound_high = env.action_space.high
    
    # Define input placeholders    
    state_ph = tf.placeholder(tf.float32, ((None,) + state_dims))
    
    # Create policy (actor) network      
    if args.use_batch_norm:
        actor = Actor_BN(state_ph, state_dims, action_dims, action_bound_low, action_bound_high, args, is_training=False, scope='actor_main')
    else:
        actor = Actor(state_ph, state_dims, action_dims, action_bound_low, action_bound_high, args, scope='actor_main')
     
    # Create session
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)       
    
    # Load ckpt file
    loader = tf.train.Saver()    
    if args.ckpt_file is not None:
        ckpt = args.ckpt_dir + '/' + args.ckpt_file  
    else:
        ckpt = tf.train.latest_checkpoint(args.ckpt_dir)     
    
    loader.restore(sess, ckpt)
    print('%s restored.\n\n' % ckpt)
    
    # Create record directory
    if args.record_dir is not None:
        if not os.path.exists(args.record_dir):
            os.makedirs(args.record_dir)
        
    for ep in range(args.num_eps):
        state = env.reset()
        for step in range(args.max_ep_length):
            frame = env.render(mode='rgb_array')
            if args.record_dir is not None:
                filepath = args.record_dir + '/Ep%03d_Step%04d.jpg' % (ep, step)
                cv2.imwrite(filepath, frame)
            action = sess.run(actor.output, {state_ph:np.expand_dims(state, 0)})[0]     # Add batch dimension to single state input, and remove batch dimension from single action output
            state, _, terminal, _ = env.step(action)
        
            if terminal:
                break   
            
    env.close()
    
    # Convert saved frames to gif
    if args.record_dir is not None:
        images = []
        for file in sorted(os.listdir(args.record_dir)):
            # Load image
            filename = args.record_dir + '/' + file
            im = cv2.imread(filename)
            images.append(im)
            # Delete static image once loaded
            os.remove(filename)
             
        # Save as gif
        imageio.mimsave(args.record_dir + '/%s.gif' % args.env, images, duration=0.01)            
                       

if  __name__ == '__main__':
    train_args = get_train_args()
    play_args = get_play_args(train_args)
    play(play_args)            
            
        
    
    
        
        
