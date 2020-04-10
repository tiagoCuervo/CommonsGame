import numpy as np
from env import CommonsGame

numAgents = 11  # Number of agents
visualRadius = 4

env = CommonsGame(numAgents, visualRadius)
env.reset()
for t in range(1000):
    nActions = np.random.randint(low=0, high=env.action_space.n, size=(numAgents,)).tolist()
    nObservations, nRewards, nDone, nInfo = env.step(nActions)
    env.render()
