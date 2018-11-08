'''
## Train ##
# Code to train DDPG Network on OpenAI Gym environments
@author: Mark Sinton (msinto93@gmail.com) 
'''

import os
import sys
import argparse
import gym
import tensorflow as tf
import numpy as np
import time

from utils.network import Critic, Actor, Critic_BN, Actor_BN
from utils.experience_replay import ReplayMemory   
from utils.ou_noise import OrnsteinUhlenbeckActionNoise
    
def get_train_args():
    train_params = argparse.ArgumentParser()
    
    # Environment parameters
    train_params.add_argument("--env", type=str, default='Pendulum-v0', help="Environment to use (must have low dimensional state space (i.e. not image) and continuous action space)")
    train_params.add_argument("--render", type=bool, default=False, help="Whether or not to display the environment on the screen during training")
    train_params.add_argument("--random_seed", type=int, default=99999999, help="Random seed for reproducability")
    
    # Training parameters
    train_params.add_argument("--batch_size", type=int, default=64)
    train_params.add_argument("--num_eps_train", type=int, default=50000, help="Number of episodes to train for")
    train_params.add_argument("--max_ep_length", type=int, default=1000, help="Maximum number of steps per episode")
    train_params.add_argument("--replay_mem_size", type=int, default=1000000, help="Maximum size of replay memory")
    train_params.add_argument("--initial_replay_mem_size", type=int, default=50000, help="Initial size of replay memory (populated by random actions) before learning starts")
    train_params.add_argument("--noise_scale", type=float, default=0.1, help="Scale of exploration noise range (as a fraction of action space range), e.g. for a noise_scale=0.1, the noise range is a tenth of the action space range")
    train_params.add_argument("--discount_rate", type=float, default=0.99, help="Discount rate (gamma) for future rewards.")
    
    # Network parameters
    train_params.add_argument("--critic_learning_rate", type=float, default=0.001)
    train_params.add_argument("--actor_learning_rate", type=float, default=0.0001)
    train_params.add_argument("--critic_l2_lambda", type=float, default=0.0, help="Coefficient for L2 weight regularisation in critic - if 0, no regularisation is performed")
    train_params.add_argument("--dense1_size", type=int, default=400, help="Size of first hidden layer in networks")
    train_params.add_argument("--dense2_size", type=int, default=300, help="Size of second hidden layer in networks")
    train_params.add_argument("--final_layer_init", type=float, default=0.003, help="Initialise networks' final layer weights in range +/-final_layer_init")
    train_params.add_argument("--tau", type=float, default=0.001, help="Parameter for soft target network updates")
    train_params.add_argument("--use_batch_norm", type=bool, default=False, help="Whether or not to use batch normalisation in the networks")
    
    # Files/Directories
    train_params.add_argument("--save_ckpt_step", type=float, default=200, help="Save checkpoint every N episodes")
    train_params.add_argument("--ckpt_dir", type=str, default='./ckpts', help="Directory for saving/loading checkpoints")
    train_params.add_argument("--ckpt_file", type=str, default=None, help="Checkpoint file to load and resume training from (if None, train from scratch)")
    train_params.add_argument("--log_dir", type=str, default='./logs/train', help="Directory for saving Tensorboard logs")
    
    return train_params.parse_args()


def update_target_network(network_params, target_network_params, tau=1.0):     
    # When tau=1.0, we perform a hard copy of parameters, otherwise a soft copy
     
    # Create ops which update target network parameters with (fraction of) main network parameters
    op_holder = []
    for from_var,to_var in zip(network_params, target_network_params):
        op_holder.append(to_var.assign((tf.multiply(from_var, tau) + tf.multiply(to_var, 1. - tau))))        
     
    return op_holder
            
    
def train(args):
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
    
    # Initialise replay memory
    replay_mem = ReplayMemory(args, state_dims, action_dims)
    
    # Initialise Ornstein-Uhlenbeck Noise generator
    exploration_noise = OrnsteinUhlenbeckActionNoise(mu=np.zeros(action_dims))
    noise_scaling = args.noise_scale * (action_bound_high - action_bound_low)
    
    # Define input placeholders    
    state_ph = tf.placeholder(tf.float32, ((None,) + state_dims))
    action_ph = tf.placeholder(tf.float32, ((None,) + action_dims))
    target_ph = tf.placeholder(tf.float32, (None, 1))  # Target Q-value - for critic training
    action_grads_ph = tf.placeholder(tf.float32, ((None,) + action_dims)) # Gradient of critic's value output wrt action input - for actor training
    is_training_ph = tf.placeholder_with_default(True, shape=None)
    
    # Create value (critic) network + target network
    if args.use_batch_norm:
        critic = Critic_BN(state_ph, action_ph, state_dims, action_dims, args, is_training=is_training_ph, scope='critic_main')
        critic_target = Critic_BN(state_ph, action_ph, state_dims, action_dims, args, is_training=is_training_ph, scope='critic_target')
    else:
        critic = Critic(state_ph, action_ph, state_dims, action_dims, args, scope='critic_main')
        critic_target = Critic(state_ph, action_ph, state_dims, action_dims, args, scope='critic_target')
    
    # Create policy (actor) network + target network
    if args.use_batch_norm:
        actor = Actor_BN(state_ph, state_dims, action_dims, action_bound_low, action_bound_high, args, is_training=is_training_ph, scope='actor_main')
        actor_target = Actor_BN(state_ph, state_dims, action_dims, action_bound_low, action_bound_high, args, is_training=is_training_ph, scope='actor_target')
    else:
        actor = Actor(state_ph, state_dims, action_dims, action_bound_low, action_bound_high, args, scope='actor_main')
        actor_target = Actor(state_ph, state_dims, action_dims, action_bound_low, action_bound_high, args, scope='actor_target')
    
    # Create training step ops
    critic_train_step = critic.train_step(target_ph)
    actor_train_step = actor.train_step(action_grads_ph)
    
    # Create ops to update target networks
    update_critic_target = update_target_network(critic.network_params, critic_target.network_params, args.tau)
    update_actor_target = update_target_network(actor.network_params, actor_target.network_params, args.tau)
           
    # Create session
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)    

    # Define saver for saving model ckpts
    model_name = args.env + '.ckpt'
    checkpoint_path = os.path.join(args.ckpt_dir, model_name)        
    if not os.path.exists(args.ckpt_dir):
        os.makedirs(args.ckpt_dir)
    saver = tf.train.Saver(max_to_keep=201)  
    
    # Load ckpt file if given
    if args.ckpt_file is not None:
        loader = tf.train.Saver()   #Restore all variables from ckpt
        ckpt = args.ckpt_dir + '/' + args.ckpt_file
        ckpt_split = ckpt.split('-')
        step_str = ckpt_split[-1]
        start_ep = int(step_str)    
        loader.restore(sess, ckpt)
    else:
        start_ep = 0
        sess.run(tf.global_variables_initializer())   
        # Perform hard copy (tau=1.0) of initial params to target networks
        sess.run(update_target_network(critic.network_params, critic_target.network_params))
        sess.run(update_target_network(actor.network_params, actor_target.network_params))
        
    # Create summary writer to write summaries to disk
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)
    summary_writer = tf.summary.FileWriter(args.log_dir, sess.graph)
    
    # Create summary op to save episode reward to Tensorboard log
    ep_reward_var = tf.Variable(0.0, trainable=False)
    tf.summary.scalar("Episode Reward", ep_reward_var)
    summary_op = tf.summary.merge_all()
    
    
    ## Training 
    
    # Initially populate replay memory by taking random actions 
    sys.stdout.write('\nPopulating replay memory with random actions...\n')   
    sys.stdout.flush()          
    env.reset()
     
    for random_step in range(1, args.initial_replay_mem_size+1):
        if args.render:
            env.render()
        action = env.action_space.sample()
        state, reward, terminal, _ = env.step(action)
        replay_mem.add(action, reward, state, terminal)
         
        if terminal:
            env.reset()
                         
        sys.stdout.write('\x1b[2K\rStep {:d}/{:d}'.format(random_step, args.initial_replay_mem_size))
        sys.stdout.flush() 
             
    sys.stdout.write('\n\nTraining...\n')   
    sys.stdout.flush()
    
    for train_ep in range(start_ep+1, args.num_eps_train+1):      
        # Reset environment and noise process
        state = env.reset()
        exploration_noise.reset()
        
        train_step = 0
        episode_reward = 0
        duration_values = []
        ep_done = False
        
        sys.stdout.write('\n')   
        sys.stdout.flush()
        
        while not ep_done:
            train_step += 1
            start_time = time.time()            
            ## Take action and store experience
            if args.render:
                env.render()
            if args.use_batch_norm:
                action = sess.run(actor.output, {state_ph:np.expand_dims(state, 0), is_training_ph:False})[0]     # Add batch dimension to single state input, and remove batch dimension from single action output
            else:
                action = sess.run(actor.output, {state_ph:np.expand_dims(state, 0)})[0]     
            action += exploration_noise() * noise_scaling
            state, reward, terminal, _ = env.step(action)
            replay_mem.add(action, reward, state, terminal)
            
            episode_reward += reward  
                        
            ## Train networks
            # Get minibatch
            states_batch, actions_batch, rewards_batch, next_states_batch, terminals_batch = replay_mem.getMinibatch() 
    
            # Critic training step    
            # Predict actions for next states by passing next states through policy target network
            future_action = sess.run(actor_target.output, {state_ph:next_states_batch})  
            # Predict target Q values by passing next states and actions through value target network
            future_Q = sess.run(critic_target.output, {state_ph:next_states_batch, action_ph:future_action})[:,0]   # future_Q is of shape [batch_size, 1], need to remove second dimension for ops with terminals_batch and rewards_batch which are of shape [batch_size]
            # Q values of the terminal states is 0 by definition
            future_Q[terminals_batch] = 0
            targets = rewards_batch + (future_Q*args.discount_rate)
            # Train critic
            sess.run(critic_train_step, {state_ph:states_batch, action_ph:actions_batch, target_ph:np.expand_dims(targets, 1)})   
            
            # Actor training step
            # Get policy network's action outputs for selected states
            actor_actions = sess.run(actor.output, {state_ph:states_batch})
            # Compute gradients of critic's value output wrt actions
            action_grads = sess.run(critic.action_grads, {state_ph:states_batch, action_ph:actor_actions})
            # Train actor
            sess.run(actor_train_step, {state_ph:states_batch, action_grads_ph:action_grads[0]})
            
            # Update target networks
            sess.run(update_critic_target)
            sess.run(update_actor_target)
            
            # Display progress            
            duration = time.time() - start_time
            duration_values.append(duration)
            ave_duration = sum(duration_values)/float(len(duration_values))
                        
            sys.stdout.write('\x1b[2K\rEpisode {:d}/{:d} \t Steps = {:d} \t Reward = {:.3f} \t ({:.3f} s/step)'.format(train_ep, args.num_eps_train, train_step, episode_reward, ave_duration))
            sys.stdout.flush()  
        
            if terminal or train_step == args.max_ep_length:
                # Log total episode reward and begin next episode
                summary_str = sess.run(summary_op, {ep_reward_var: episode_reward})
                summary_writer.add_summary(summary_str, train_ep)
                ep_done = True
        
        if train_ep % args.save_ckpt_step == 0:
            saver.save(sess, checkpoint_path, global_step=train_ep)
            sys.stdout.write('\n Checkpoint saved.')   
            sys.stdout.flush() 

    env.close()     
    

if  __name__ == '__main__':
    train_args = get_train_args()
    train(train_args)         
            
        
    
    
        
        
