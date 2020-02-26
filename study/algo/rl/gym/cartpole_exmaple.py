import warnings
warnings.filterwarnings('ignore')

import tensorflow as tf
import keras
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.layers import Dense, Concatenate, Activation, Input, Flatten
from tensorflow.keras.models import Model, Sequential

import gym
import numpy as np

from tensorflow.python.framework.ops import disable_eager_execution
disable_eager_execution()

from rl.agents.dqn import DQNAgent
from rl.agents.sarsa import SarsaAgent
from rl.policy import EpsGreedyQPolicy, BoltzmannQPolicy, LinearAnnealedPolicy
from rl.memory import SequentialMemory


def build_model(shape, num_actions):
    inputs = Input(shape = (1, ) + shape)
    x = Flatten()(inputs)
    x = Dense(64, activation='relu')(x)
    x = Dense(64, activation='relu')(x)
    
    outputs = Dense(num_actions)(x)
    
    m = Model(inputs, outputs)

    return m


ENV_NAME = 'MountainCar-v0'


# Get the environment and extract the number of actions.
env = gym.make(ENV_NAME)
env.seed(0)
nb_actions = env.action_space.n

model = build_model(env.observation_space.shape, env.action_space.n)

memory = SequentialMemory(limit=20000, window_length=1)

policy = BoltzmannQPolicy()

# build model
dqn = DQNAgent(model=model, nb_actions=nb_actions, memory=memory, nb_steps_warmup=100,
               target_model_update=1e-2, policy=policy, gamma=0.9, train_interval=4, 
               enable_double_dqn=True, enable_dueling_network=True)
dqn.compile(optimizers.Adam(), metrics=['mae'])

# train one's agent
dqn.fit(env, nb_steps=50000)

# Finally, evaluate our algorithm for 5 episodes.
dqn.test(env, nb_episodes=5)