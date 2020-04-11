import numpy as np
from pycolab.prefab_parts import sprites
from pycolab import things as pythings
from scipy.ndimage import convolve
from constants import respawnProbs


class PlayerSprite(sprites.MazeWalker):
    def __init__(self, corner, position, character, agentChars):
        super(PlayerSprite, self).__init__(
            corner, position, character, impassable=['='] + list(agentChars.replace(character, '')),
            confined_to_board=True)
        self.agentChars = agentChars
        self.orientation = np.random.choice(4)
        self.initPos = position
        self.timeout = 0

    def update(self, actions, board, layers, backdrop, things, the_plot):
        if actions is not None:
            a = actions[self.agentChars.index(self.character)]
        else:
            return
        if self._visible:
            if things['.'].curtain[self.position[0], self.position[1]]:
                self.timeout = 25
                self._visible = False
            else:
                if a == 0:  # go upward?
                    if self.orientation == 0:
                        self._north(board, the_plot)
                    elif self.orientation == 1:
                        self._east(board, the_plot)
                    elif self.orientation == 2:
                        self._south(board, the_plot)
                    elif self.orientation == 3:
                        self._west(board, the_plot)
                elif a == 1:  # go downward?
                    if self.orientation == 0:
                        self._south(board, the_plot)
                    elif self.orientation == 1:
                        self._west(board, the_plot)
                    elif self.orientation == 2:
                        self._north(board, the_plot)
                    elif self.orientation == 3:
                        self._east(board, the_plot)
                elif a == 2:  # go leftward?
                    if self.orientation == 0:
                        self._west(board, the_plot)
                    elif self.orientation == 1:
                        self._north(board, the_plot)
                    elif self.orientation == 2:
                        self._east(board, the_plot)
                    elif self.orientation == 3:
                        self._south(board, the_plot)
                elif a == 3:  # go rightward?
                    if self.orientation == 0:
                        self._east(board, the_plot)
                    elif self.orientation == 1:
                        self._south(board, the_plot)
                    elif self.orientation == 2:
                        self._west(board, the_plot)
                    elif self.orientation == 3:
                        self._north(board, the_plot)
                elif a == 4:  # turn right?
                    if self.orientation == 3:
                        self.orientation = 0
                    else:
                        self.orientation = self.orientation + 1
                elif a == 5:  # turn left?
                    if self.orientation == 0:
                        self.orientation = 3
                    else:
                        self.orientation = self.orientation - 1
                elif a == 6:  # do nothing?
                    self._stay(board, the_plot)
        else:
            if self.timeout == 0:
                self._teleport(self.initPos)
                self._visible = True
            else:
                self.timeout -= 1


class SightDrape(pythings.Drape):
    """Scope of agent Drap"""
    def __init__(self, curtain, character, agentChars, numPadPixels):
        super().__init__(curtain, character)
        self.agentChars = agentChars
        self.numPadPixels = numPadPixels
        self.h = curtain.shape[0] - (numPadPixels * 2 + 2)
        self.w = curtain.shape[1] - (numPadPixels * 2 + 2)

    def update(self, actions, board, layers, backdrop, things, the_plot):
        np.logical_and(self.curtain, False, self.curtain)
        ags = [things[c] for c in self.agentChars]
        for agent in ags:
            if agent.visible:
                pos = agent.position
                if agent.orientation == 0:
                    self.curtain[pos[0] - 1, pos[1]] = True
                elif agent.orientation == 1:
                    self.curtain[pos[0], pos[1] + 1] = True
                elif agent.orientation == 2:
                    self.curtain[pos[0] + 1, pos[1]] = True
                elif agent.orientation == 3:
                    self.curtain[pos[0], pos[1] - 1] = True
                self.curtain[:, :] = np.logical_and(self.curtain, np.logical_not(layers['=']))


class ShotDrape(pythings.Drape):
    """Tagging ray Drap"""
    def __init__(self, curtain, character, agentChars, numPadPixels):
        super().__init__(curtain, character)
        self.agentChars = agentChars
        self.numPadPixels = numPadPixels
        self.h = curtain.shape[0] - (numPadPixels * 2 + 2)
        self.w = curtain.shape[1] - (numPadPixels * 2 + 2)
        self.scopeHeight = numPadPixels + 1

    def update(self, actions, board, layers, backdrop, things, the_plot):
        beamWidth = 0
        beamHeight = self.scopeHeight
        np.logical_and(self.curtain, False, self.curtain)
        if actions is not None:
            for i, a in enumerate(actions):
                if a == 7:
                    agent = things[self.agentChars[i]]
                    if agent.visible:
                        pos = agent.position
                        if agent.orientation == 0:
                            if np.any(layers['='][pos[0] - beamHeight:pos[0],
                                      pos[1] - beamWidth:pos[1] + beamWidth + 1]):
                                collisionIdxs = np.argwhere(layers['='][pos[0] - beamHeight:pos[0],
                                                            pos[1] - beamWidth:pos[1] + beamWidth + 1])
                                beamHeight = beamHeight - (np.max(collisionIdxs) + 1)
                            self.curtain[pos[0] - beamHeight:pos[0],
                            pos[1] - beamWidth:pos[1] + beamWidth + 1] = True
                        elif agent.orientation == 1:
                            if np.any(layers['='][pos[0] - beamWidth:pos[0] + beamWidth + 1,
                            pos[1] + 1:pos[1] + beamHeight + 1]):
                                collisionIdxs = np.argwhere(layers['='][pos[0] - beamWidth:pos[0] + beamWidth + 1,
                            pos[1] + 1:pos[1] + beamHeight + 1])
                                beamHeight = np.min(collisionIdxs)
                            self.curtain[pos[0] - beamWidth:pos[0] + beamWidth + 1,
                            pos[1] + 1:pos[1] + beamHeight + 1] = True
                        elif agent.orientation == 2:
                            if np.any(layers['='][pos[0] + 1:pos[0] + beamHeight + 1,
                            pos[1] - beamWidth:pos[1] + beamWidth + 1]):
                                collisionIdxs = np.argwhere(layers['='][pos[0] + 1:pos[0] + beamHeight + 1,
                            pos[1] - beamWidth:pos[1] + beamWidth + 1])
                                beamHeight = np.min(collisionIdxs)
                            self.curtain[pos[0] + 1:pos[0] + beamHeight + 1,
                            pos[1] - beamWidth:pos[1] + beamWidth + 1] = True
                        elif agent.orientation == 3:
                            if np.any(layers['='][pos[0] - beamWidth:pos[0] + beamWidth + 1,
                                      pos[1] - beamHeight:pos[1]]):
                                collisionIdxs = np.argwhere(layers['='][pos[0] - beamWidth:pos[0] + beamWidth + 1,
                                                            pos[1] - beamHeight:pos[1]])
                                beamHeight = beamHeight - (np.max(collisionIdxs) + 1)
                            self.curtain[pos[0] - beamWidth:pos[0] + beamWidth + 1, pos[1] - beamHeight:pos[1]] = True
                        # self.curtain[:, :] = np.logical_and(self.curtain, np.logical_not(layers['=']))
        else:
            return


class AppleDrape(pythings.Drape):
    """Coins Drap"""
    def __init__(self, curtain, character, agentChars, numPadPixels):
        super().__init__(curtain, character)
        self.agentChars = agentChars
        self.numPadPixels = numPadPixels
        self.apples = np.copy(curtain)

    def update(self, actions, board, layers, backdrop, things, the_plot):
        rewards = []
        agentsMap = np.ones(self.curtain.shape, dtype=bool)
        for i in range(len(self.agentChars)):
            rew = self.curtain[things[self.agentChars[i]].position[0], things[self.agentChars[i]].position[1]]
            if rew:
                self.curtain[things[self.agentChars[i]].position[0], things[self.agentChars[i]].position[1]] = False
            rewards.append(rew * 1)
            agentsMap[things[self.agentChars[i]].position[0], things[self.agentChars[i]].position[1]] = False
        the_plot.add_reward(rewards)
        # Matrix of local stock of apples
        kernel = np.array([[1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [1, 1, 0, 1, 1], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1]])
        L = convolve(self.curtain[self.numPadPixels + 1:-self.numPadPixels - 1,
                     self.numPadPixels + 1:-self.numPadPixels - 1] * 1, kernel, mode='constant')
        probs = np.zeros(L.shape)
        probs[(L > 0) & (L <= 2)] = respawnProbs[0]
        probs[(L > 2) & (L <= 4)] = respawnProbs[1]
        probs[(L > 4)] = respawnProbs[2]
        appleIdxs = np.argwhere(np.logical_and(np.logical_xor(self.apples, self.curtain), agentsMap))
        for i, j in appleIdxs:
            self.curtain[i, j] = np.random.choice([True, False],
                                                  p=[probs[i - self.numPadPixels - 1, j - self.numPadPixels - 1],
                                                     1 - probs[i - self.numPadPixels - 1, j - self.numPadPixels - 1]])
