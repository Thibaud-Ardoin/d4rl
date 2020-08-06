import gym
import d4rl # Import required to register environments

# Create the environment
env = gym.make('flow-ring-random-v0')

# d4rl abides by the OpenAI gym interface
env.reset()
env.step(env.action_space.sample())

# Each task is associated with a dataset
dataset = env.get_dataset()
print(dataset['observations']) # An N x dim_observation Numpy array of observations
