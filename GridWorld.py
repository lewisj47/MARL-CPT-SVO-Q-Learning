#This program creates and visualizes a grid world, our ui.\

import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np
from itertools import product
import time

L = 12
n_agents = 4

totObs = []


Obs1 = [(r, c) for r in range(0,int(L/4)) for c in range(L)]
totObs.extend(Obs1)

Obs2 = [(r, c) for r in range(int(L/2), L) for c in range(int(L/2))]
totObs.extend(Obs2)

Obs3 = [(r, c) for r in range(int(L/2),L) for c in range(int(L*3/4),L)]
totObs.extend(Obs3)

B = []
for i in range(n_agents):
    B.append(L**2-1)

#Identifying corners and edges in the grid 
Corners = [0, L-1, L**2-L, L**2-1]
Left_Side = list(range(L,(L-1)*L,L))
Bottom = list(range(1,L-1))
Right_Side = list(range(2*L-1,L**2-1,L))
Top = list(range(L**2-L+1,L**2-1))

class FlatGridWorld:
    def __init__(self, size=L, start=(0,int((2/3) * L)), goal=(int((2/3) * L),int(L - (L/12))), obstacles=[]):
        self.size = size  # grid is size x size
        self.n_squares = size * size
        self.start = start
        self.goal = goal
        self.obstacles = [Obs1, Obs2, Obs3]
        self.reset()

    def reset(self):
        self.agent_pos = self.start
        return self.agent_pos

    def availableSqrs(self):
        self.valid = [(self.agent_pos[0] + 1, self.agent_pos[1]), 
        (self.agent_pos[0] - 1, self.agent_pos[1]), 
        (self.agent_pos[0], self.agent_pos[1] + 1),
        (self.agent_pos[0], self.agent_pos[1] - 1),
        (self.agent_pos[0] + 1, self.agent_pos[1] + 1),
        (self.agent_pos[0] - 1, self.agent_pos[1] + 1),
        (self.agent_pos[0] + 1, self.agent_pos[1] - 1),
        (self.agent_pos[0] - 1, self.agent_pos[1] - 1)]

        return self.valid

    def render(self):
        grid = np.zeros((self.size, self.size))

        # Fill in obstacle, agent, start, goal
        for coord in product(range(L), repeat=2):  # loops through (0,0), (0,1), ..., (11,11)
            if coord in totObs:
                grid[coord] = -1
        
        grid[0,0] = 0.2

        grid[self.agent_pos] = 1.5

        grid[self.goal] = 0.8

        cmap = colors.ListedColormap(['white', 'black', 'blue', 'green', 'red'])
        bounds = [-1.5, -0.5, 0.1, 0.5, 0.9, 1.5]
        norm = colors.BoundaryNorm(bounds, cmap.N)

        plt.imshow(grid, cmap=cmap, norm=norm)
        plt.xticks([])
        plt.yticks([])
        plt.grid(True)
        plt.ion() 
        plt.show()

    def updateWorld(self, new_pos):
        if new_pos not in totObs and new_pos in self.availableSqrs() and (0<new_pos[0]<12) and (0<new_pos[1]<12):
            self.agent_pos = new_pos

env = FlatGridWorld(size=L, start=(0,int((2/3) * L)), goal=(int((2/3) * L),L - L/12), obstacles=(Obs1,Obs2,Obs3))

state = env.reset()

for i in range(1,15):
    env.updateWorld((4,i))
    env.render()
    plt.pause(0.5)