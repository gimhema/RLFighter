from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
import tensorflow as tf

from tf_agents.environments import py_environment
from tf_agents.environments import tf_environment
from tf_agents.environments import tf_py_environment
from tf_agents.environments import utils
from tf_agents.specs import array_spec
from tf_agents.environments import wrappers
from tf_agents.trajectories import time_step as ts

from tf_agents.environments import suite_gym
from tf_agents.networks import q_network
from tf_agents.agents.dqn import dqn_agent
from tf_agents.utils import common
from tf_agents.policies import random_tf_policy
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.trajectories import trajectory

from tf_agents.agents.categorical_dqn import categorical_dqn_agent
from tf_agents.drivers import dynamic_step_driver
from tf_agents.eval import metric_utils
from tf_agents.metrics import tf_metrics
from tf_agents.networks import categorical_q_network

import RLServerModule as rls

tf.compat.v1.enable_v2_behavior()

global learning_iteration
global learning_iteration_start
import numpy as np

import threading
import time

# Write your property, isdone, state array, learning_iteration, learning_iteration_start
# state arr should create as np.array type
_learning_iteration = 2000 # example
_isDone = 0 # example
_stateArr = np.array([0, 0, 0, 0, 0, 0, 0], dtype=np.int32) # example

rlAgentServer = rls.RLAgentServer(_isDone, _stateArr, _learning_iteration)

class RLShooterEnvironment(py_environment.PyEnvironment):
    def __init__(self, _action, _observation, _state, _done):
        self._action_spec = _action
        self._observation_spec = _observation
        self._episode_ended = False
        self._reset_state = _state
        self._reset_done = _done
        self._state = _state
        self._done = _done
        
    
    def action_spec(self):
        return self._action_spec
    
    def observation_spec(self):
        return self._observation_spec

    def _reset(self):
        self._state = self._reset_state
        self._done = self._reset_done
        self._episode_ended = False
        return ts.restart(self._reset_state)
    
    
    # custom function
    def send_action(self, action):
        # send other action value to UE4 through using socket
        rlAgentServer.send(rlAgentServer.connectionSock,action) # send action value to UE4 through using socket
        self.set_state_done_fromUE4()
        return 0.0
    
    # custom function
    def set_state_done_fromUE4(self):
        # set self._state and self._done value through using data from UE4
        
        isReceived = False
        while True:
            if isReceived == True: break
            if rlAgentServer.isReceivedData == True:
                self._done, self._state = rlAgentServer.read_data()
                rlAgentServer.isReceivedData = False
                isReceived = True
        pass
    
    def _step(self, action):
        
        print("Action Value :", action)
        
        # This function is custom function
        # Write down your RL algorithm
        
        # when episode is ended
        if self._episode_ended:
            return self.reset()
        
        # these code are example when action value is seted 0~N
        self.send_action(str(action))
        
        
        return ts.transition(self._state, 0)
        """
        # these code are example when done value is seted 0~2
        if self._done == 2: # if you give a praise to your agent
            agent_reward = 0
            print("State Data : ", self._state)
            return ts.termination(self._state, agent_reward)
        elif self._done == 1: # if you give a penalty to your agent
            agemt_penalty = 0
            print("State Data : ", self._state)
            return ts.termination(self._state, agemt_penalty)
        elif self._done == 3:
            agent_small_reward = 0
            print("State Data : ", self._state)
            return ts.transition(
            self._state, reward= agent_small_reward, discount=0.4)
        else:
            print("State Data : ", self._state)
            return ts.transition(
            self._state, reward= 0.0, discount=0.1)
        """

_action_boundry_arr = array_spec.BoundedArraySpec(shape=(), dtype=np.int32, name='action', minimum=0, maximum=11)
_observation = array_spec.BoundedArraySpec(shape=(7,), dtype=np.int32, name='observation', 
                                                     minimum=[0, 0, 0, 0, 0, 0, 0], 
                                                     maximum=[0, 0, 0, 0, 0, 0, 0])

environment = RLShooterEnvironment(_action_boundry_arr, _observation, _stateArr, _isDone)

tf_env = tf_py_environment.TFPyEnvironment(environment)

# set block

train_env = tf_py_environment.TFPyEnvironment(environment)
eval_env = tf_py_environment.TFPyEnvironment(environment)

num_iterations = 100000 # @param {type:"integer"}

initial_collect_steps = 100  # @param {type:"integer"} 
collect_steps_per_iteration = 1  # @param {type:"integer"}
replay_buffer_max_length = 1000  # @param {type:"integer"}

batch_size = 64  # @param {type:"integer"}
learning_rate = 1e-3  # @param {type:"number"}
gamma = 0.99 # for Categorical DQN
log_interval = 200  # @param {type:"integer"}

# for Categorical DQN
num_atoms = 51  # @param {type:"integer"}
min_q_value = -20  # custom
max_q_value = 20  # custom
n_step_update = 2  # @param {type:"integer"}


num_eval_episodes = 10  # @param {type:"integer"}
eval_interval = 1000  # @param {type:"integer"}

fc_layer_params = (100, )

categorical_q_net = categorical_q_network.CategoricalQNetwork(
    train_env.observation_spec(),
    train_env.action_spec(),
    num_atoms=num_atoms,
    fc_layer_params=fc_layer_params)


optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate)

#train_step_counter = tf.Variable(0)
train_step_counter = tf.compat.v2.Variable(0)

agent = categorical_dqn_agent.CategoricalDqnAgent(
    train_env.time_step_spec(),
    train_env.action_spec(),
    categorical_q_network=categorical_q_net,
    optimizer=optimizer,
    min_q_value=min_q_value,
    max_q_value=max_q_value,
    n_step_update=n_step_update,
    td_errors_loss_fn=common.element_wise_squared_loss,
    gamma=gamma,
    train_step_counter=train_step_counter)

agent.initialize()

eval_policy = agent.policy
collect_policy = agent.collect_policy

random_policy = random_tf_policy.RandomTFPolicy(train_env.time_step_spec(),
                                                train_env.action_spec())

replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
    data_spec=agent.collect_data_spec,
    batch_size=train_env.batch_size,
    max_length=replay_buffer_max_length)

agent.collect_data_spec._fields


def collect_step(environment, policy, buffer):
    
    time_step = environment.current_time_step()
  #print("Time Step : ", time_step)
    action_step = policy.action(time_step)
  #print("Action Step : ", action_step)
    print("Selected Action Step : ", action_step.action)
    next_time_step = environment.step(action_step.action) 
    traj = trajectory.from_transition(time_step, action_step, next_time_step)

  # Add trajectory to the replay buffer
    buffer.add_batch(traj)

def collect_data(env, policy, buffer, steps):
    for _ in range(steps):
        collect_step(env, policy, buffer)
    
    
dataset = replay_buffer.as_dataset(
    num_parallel_calls=3, 
    sample_batch_size=batch_size, 
    num_steps=2).prefetch(3)

print("Dataset : ", dataset)

iterator = iter(dataset)

print("Iterator : ",iterator)


agent.train = common.function(agent.train)

agent.train_step_counter.assign(0)

import os
from tf_agents.policies import policy_saver

tempdir = "/Users/USER/Documents/RL/RLFighter/policy"
relearning_dir = "/Users/USER/Documents/RL/RLFighter/checkpoint"
global_step = tf.compat.v1.train.get_or_create_global_step()

# checkpoint_dir = os.path.join(tempdir, 'checkpoint')
checkpoint_dir = os.path.join(relearning_dir, 'checkpoint')
train_checkpointer = common.Checkpointer(
    ckpt_dir=checkpoint_dir,
    max_to_keep=1,
    agent=agent,
    policy=agent.policy,
    replay_buffer=replay_buffer,
    global_step=global_step
)

policy_dir = os.path.join(tempdir, 'policy')
tf_policy_saver = policy_saver.PolicySaver(agent.policy)

def saveModel():
    train_checkpointer.save(global_step)

# load model
def loadModel():
    train_checkpointer.initialize_or_restore()
    global_step = tf.compat.v1.train.get_global_step()
    tf_policy_saver.save(policy_dir)
    saved_policy = tf.compat.v2.saved_model.load(policy_dir)
    #evaluate_agent(saved_policy, eval_env, environment, rlAgentServer.learning_iteration)
    load_policy = saved_policy
    print("load model sucessful")
    return load_policy
    
loaded_policy = loadModel()

# Evaludate
def collect_step_load_model(environment, policy):
    # time_step = environment.reset()
    time_step = environment.current_time_step()
    print("Time Step : ", time_step)
    action_step = policy.action(time_step)
    time_step = environment.step(action_step.action)
    
def evaluate_agent(var):
            while True:
                #time_step = eval_env.current_time_step()
                collect_step_load_model(eval_env, loaded_policy)
                #time.sleep(0.05)

def create_load_model_thread(var):
    while True:
        print("RL Evaluate Ready : ", rlAgentServer.initIdx)
        time.sleep(1)
        if rlAgentServer.initIdx > 0:
            print("RL Evaluate Start")
            load_thread = threading.Thread(target=evaluate_agent, args=("1"))
            load_thread.start()
            break;

def rlServerStart(var):
    rlAgentServer.run_server()

load_model_thread = threading.Thread(target=create_load_model_thread, args=("1"))
load_model_thread.start()
#server_thread = threading.Thread(target=rlServerStart, args=("1"))
#server_thread.start()
rlAgentServer.run_server()