import random

import numpy as np
import matplotlib.pyplot as plt
import gym
from gym import spaces
from pycolab import ascii_art
from CommonsGame.constants import *
from CommonsGame.utils import buildMap, ObservationToArrayWithRGB
from CommonsGame.objects import PlayerSprite, AppleDrape, SightDrape, ShotDrape


class CommonsGame(gym.Env):
    """Custom Environment that follows gym interface"""
    metadata = {'render.modes': ['human']}

    def __init__(self, numAgents, visualRadius, mapSketch=bigMap, fullState=False):
        super(CommonsGame, self).__init__()
        self.fullState = fullState
        # Setup spaces
        self.action_space = spaces.Discrete(8)
        obHeight = obWidth = visualRadius * 2 + 1
        # Setup game
        self.numAgents = numAgents
        self.sightRadius = visualRadius
        self.agentChars = agentChars = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"[0:numAgents]
        self.mapHeight = len(mapSketch)
        self.mapWidth = len(mapSketch[0])
        if fullState:
            self.observation_space = spaces.Box(low=0, high=255, shape=(self.mapHeight + 2, self.mapWidth + 2, 3),
                                                dtype=np.uint8)
        else:
            self.observation_space = spaces.Box(low=0, high=255, shape=(obHeight, obWidth, 3), dtype=np.uint8)
        self.numPadPixels = numPadPixels = visualRadius - 1
        self.gameField = buildMap(mapSketch, numPadPixels=numPadPixels, agentChars=agentChars)
        self.state = None
        # Pycolab related setup:        
        self._game = self.buildGame()
        colourMap = dict([(a, (999, 0, 0)) for i, a in enumerate(agentChars)]  # Agents
                         + [('=', (705, 705, 705))]  # Steel Impassable wall
                         + [(' ', (0, 0, 0))]  # Black background
                         + [('@', (0, 999, 0))]  # Green Apples
                         + [('.', (750, 750, 0))]  # Yellow beam
                         + [('-', (200, 200, 200))])  # Grey scope
        self.obToImage = ObservationToArrayWithRGB(colour_mapping=colourMap)

    def buildGame(self):
        agentsOrder = list(self.agentChars)
        random.shuffle(agentsOrder)
        return ascii_art.ascii_art_to_game(
            self.gameField,
            what_lies_beneath=' ',
            sprites=dict(
                [(a, ascii_art.Partial(PlayerSprite, self.agentChars)) for a in self.agentChars]),
            drapes={'@': ascii_art.Partial(AppleDrape, self.agentChars, self.numPadPixels),
                    '-': ascii_art.Partial(SightDrape, self.agentChars, self.numPadPixels),
                    '.': ascii_art.Partial(ShotDrape, self.agentChars, self.numPadPixels)},
            # update_schedule=['.'] + agentsOrder + ['-'] + ['@'],
            update_schedule=['.'] + agentsOrder + ['-'] + ['@'],
            z_order=['-'] + ['@'] + agentsOrder + ['.']
        )

    def step(self, nActions):
        nInfo = {'n': []}
        self.state, nRewards, _ = self._game.play(nActions)
        nObservations, done = self.getObservation()
        nDone = [done] * self.numAgents
        return nObservations, nRewards, nDone, nInfo

    def reset(self):
        # Reset the state of the environment to an initial state
        self._game = self.buildGame()
        self.state, _, _ = self._game.its_showtime()
        nObservations, _ = self.getObservation()
        return nObservations

    def render(self, mode='human', close=False):
        # Render the environment to the screen
        board = self.obToImage(self.state)['RGB'].transpose([1, 2, 0])
        board = board[self.numPadPixels:self.numPadPixels + self.mapHeight + 2,
                self.numPadPixels:self.numPadPixels + self.mapWidth + 2, :]
        plt.figure(1)
        plt.imshow(board)
        plt.axis("off")
        plt.show(block=False)
        # plt.show()
        plt.pause(.05)
        plt.clf()

    def getObservation(self):
        done = not (np.logical_or.reduce(self.state.layers['@'], axis=None))
        ags = [self._game.things[c] for c in self.agentChars]
        obs = []
        board = self.obToImage(self.state)['RGB'].transpose([1, 2, 0])
        for a in ags:
            if a.visible or a.timeout == 25:
                if self.fullState:
                    ob = np.copy(board)
                    if a.visible:
                        ob[a.position[0], a.position[1], :] = [0, 0, 255]
                    ob = ob[self.numPadPixels:self.numPadPixels + self.mapHeight + 2,
                         self.numPadPixels:self.numPadPixels + self.mapWidth + 2, :]
                else:
                    ob = np.copy(board[
                                 a.position[0] - self.sightRadius:a.position[0] + self.sightRadius + 1,
                                 a.position[1] - self.sightRadius:a.position[1] + self.sightRadius + 1, :])
                    if a.visible:
                        ob[self.sightRadius, self.sightRadius, :] = [0, 0, 255]
                ob = ob / 255.0
            else:
                ob = None
            obs.append(ob)
        return obs, done
