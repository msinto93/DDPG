'''
## Test ##
# Test a trained DDPG network. This can be run alongside training by running 'run_every_new_ckpt.sh'.
@author: Mark Sinton (msinto93@gmail.com) 
'''

import os
import sys
import argparse
import gym
import tensorflow as tf
import numpy as np
import scipy.stats as ss

from train import get_train_args
from utils.network import Actor, Actor_BN
    
def get_test_args(train_args):
    test_params = argparse.ArgumentParser()
    
    # Environment parameters (env must be same as used in training)
    test_params.add_argument("--env", type=str, default=train_args.env, help="Environment to use (must have low dimensional state space (i.e. not image) and continuous action space)")
    test_params.add_argument("--render", type=bool, default=False, help="Whether or not to display the environment on the screen during testing")
    test_params.add_argument("--random_seed", type=int, default=999999, help="Random seed for reproducability")
    
    # Testing parameters
    test_params.add_argument("--num_eps_test", type=int, default=100, help="Number of episodes to test for")
    test_params.add_argument("--max_ep_length", type=int, default=1000, help="Maximum number of steps per episode")
    
    # Network parameters (these must be same as used in training)
    test_params.add_argument("--dense1_size", type=int, default=train_args.dense1_size, help="Size of first hidden layer in networks")
    test_params.add_argument("--dense2_size", type=int, default=train_args.dense2_size, help="Size of second hidden layer in networks")
    test_params.add_argument("--final_layer_init", type=float, default=train_args.final_layer_init, help="Initialise networks' final layer weights in range +/-final_layer_init")
    test_params.add_argument("--use_batch_norm", type=bool, default=train_args.use_batch_norm, help="Whether or not to use batch normalisation in the networks")
    
    # Files/directories
    test_params.add_argument("--ckpt_dir", type=str, default='./ckpts', help="Directory for saving/loading checkpoints")
    test_params.add_argument("--ckpt_file", type=str, default=None, help="Checkpoint file to load and resume training from (if None, load latest ckpt)")
    test_params.add_argument("--results_dir", type=str, default='./test_results', help="Directory for saving txt file of results")
    test_params.add_argument("--results_file", type=str, default='results.txt', help="Text file of test results (if None, do not save results)")
    test_params.add_argument("--log_dir", type=str, default='./logs/test', help="Directory for saving Tensorboard logs")
    
    return test_params.parse_args()

def test(args):
    # Create environment
    env = gym.make(args.env)
    state_dims = env.observation_space.shape
    action_dims = env.action_space.shape
    action_bound_low = env.action_space.low
    action_bound_high = env.action_space.high
    
    # Set random seeds for reproducability
    env.seed(args.random_seed)
    np.random.seed(args.random_seed)
    tf.set_random_seed(args.random_seed)
    
    # Define input placeholder 
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
    sys.stdout.write('%s restored.\n\n' % ckpt)
    sys.stdout.flush() 
     
    ckpt_split = ckpt.split('-')
    train_ep = ckpt_split[-1]
    
    # Create summary writer to write summaries to disk
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)
    summary_writer = tf.summary.FileWriter(args.log_dir, sess.graph)
    
    # Create summary op to save episode reward to Tensorboard log
    reward_var = tf.Variable(0.0, trainable=False)
    tf.summary.scalar("Average Test Reward", reward_var)
    summary_op = tf.summary.merge_all()
    
    
    # Start testing
    
    rewards = [] 
    
    for test_ep in range(args.num_eps_test):
        state = env.reset()
        ep_reward = 0
        step = 0
        ep_done = False
        
        while not ep_done:
            if args.render:
                env.render()
            action = sess.run(actor.output, {state_ph:np.expand_dims(state, 0)})[0]     # Add batch dimension to single state input, and remove batch dimension from single action output
            state, reward, terminal, _ = env.step(action)
            
            ep_reward += reward
            step += 1
             
            # Episode can finish either by reaching terminal state or max episode steps
            if terminal or step == args.max_ep_length:
                sys.stdout.write('\x1b[2K\rTest episode {:d}/{:d}'.format(test_ep, args.num_eps_test))
                sys.stdout.flush()   
                rewards.append(ep_reward)
                ep_done = True   
            
    mean_reward = np.mean(rewards)
    error_reward = ss.sem(rewards)
            
    sys.stdout.write('\x1b[2K\rTesting complete \t Average reward = {:.2f} +/- {:.2f} /ep \n\n'.format(mean_reward, error_reward))
    sys.stdout.flush()  
    
    # Log average episode reward for Tensorboard visualisation
    summary_str = sess.run(summary_op, {reward_var: mean_reward})
    summary_writer.add_summary(summary_str, train_ep)
     
    # Write results to file        
    if args.results_file is not None:
        if not os.path.exists(args.results_dir):
            os.makedirs(args.results_dir)
        output_file = open(args.results_dir + '/' + args.results_file, 'a')
        output_file.write('Training Episode {}: \t Average reward = {:.2f} +/- {:.2f} /ep \n\n'.format(train_ep, mean_reward, error_reward))
        output_file.flush()
        sys.stdout.write('Results saved to file \n\n')
        sys.stdout.flush()      
    
    env.close()
    
    
if  __name__ == '__main__':
    train_args = get_train_args()
    test_args = get_test_args(train_args)
    test(test_args)    
    
    
    
    
    
    
    