import numpy as np
import gym

numAgents = 11

env = gym.make('CommonsGame:CommonsGame-v0', numAgents=numAgents, visualRadius=4)
env.reset()
for t in range(1000):
    nActions = np.random.randint(low=0, high=env.action_space.n, size=(numAgents,)).tolist()
    nObservations, nRewards, nDone, nInfo = env.step(nActions)
    env.render()
