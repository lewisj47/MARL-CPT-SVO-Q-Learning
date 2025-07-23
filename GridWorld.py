#This program creates and visualizes a grid world, our ui.\

import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.patches import Arc
import numpy as np
from itertools import product
import argparse

#The parse arguments allow for arguments to be passed to the program via the command line. size can be 12, 24 or 48.
parser = argparse.ArgumentParser()
parser.add_argument("size", type = int,  help="The size of the grid environment given as a length of one of the sides.")
parser.addargument("episodes", type=int, help="The number of episodes to undergo during training")
args = parser.parse_args()

#Right now there is 1 agent
n_agents = 1
num_episodes = args.episodes
all_sqrs = [(r,c) for r in range(args.size) for c in range(args.size)]

#Q-learning Definitions
alpha = 0.88
beta = 0.88
lr = 0.2
discount = 0.95
epsilon = 1.0
min_epsilon = 0.05
decay_rate = 0.995
# Note: will need to update epsilon per episode using
#       agent.epsilon = max(agent.min_epsilon, agent.epsilon * agent.epsilon_decay)

#An empty array to hold the coordinates of the obstacles. In our case, obstacles are spaces where the road is not. 
totObs = []

Obs1 = [(r, c) for r in range(0,(args.size//4)) for c in range(args.size)]
totObs.extend(Obs1)

Obs2 = [(r, c) for r in range((args.size//2), args.size) for c in range((args.size//2))]
totObs.extend(Obs2)

Obs3 = [(r, c) for r in range((args.size//2),args.size) for c in range(((args.size*3)//4),args.size)]
totObs.extend(Obs3)


#This class defines the environment in which the agent will learn. There is a corresponding size given as the length of
#one of the square worlds sides, the goal square, and an array containing the coordinates of each of the obstacles. 
class FlatGridWorld:
    def __init__(self, size, goal, obstacles=[]):
        self.size = args.size  # grid is size x size
        self.n_squares = size * size
        self.goal = goal
        self.obstacles = [Obs1, Obs2, Obs3]

    #Render sets a cmap for the obstacles, agents, and road, as well as creating lines for the road. It generates the
    #new world every time a change is made. 
    def render(self):
        plt.clf()

        grid = np.zeros((self.size, self.size))

        # Fill in obstacle, agent, start, goal
        for coord in product(range(args.size), repeat=2):  # loops through (0,0), (0,1), ..., (11,11)
            if coord in totObs:
                grid[coord] = -1

        for i in range(n_agents):
            grid[agents[i].agent_pos] = 1.0

        grid[self.goal] = 0.8

        cmap = colors.ListedColormap(['white', 'black', 'blue', 'green', 'red'])
        bounds = [-1.5, -0.5, 0.1, 0.5, 0.9, 1.5]
        norm = colors.BoundaryNorm(bounds, cmap.N)

        arc = Arc(
            xy=(args.size//2 - 0.5, args.size//2 - 0.5),             # center of the full ellipse
            width=args.size//4, height=args.size//4,     # diameter of the arc (radius*2)
            angle=270,               # rotation of the arc
            theta1=0, theta2=90,   # start and end angles in degrees
            color='white',
            linestyle=(0, (10, 5)),
            linewidth=2
        )

        plt.imshow(grid, cmap=cmap, norm=norm)
        plt.xticks([])
        plt.yticks([])
        plt.axvline(ymax = 0.50, x = args.size*5//8 - 0.5 if (args.size//12)%2 == 0 else args.size*5//8, color = 'yellow', linestyle='--') #((L/4)%2)*0.5)
        plt.axhline(xmin = 0.75, y= args.size*3//8 - 0.5 if (args.size//12)%2 == 0 else args.size//3, color='yellow', linestyle='--')
        plt.axhline(xmax = 0.5, y = args.size*3//8 - 0.5 if (args.size//12)%2 == 0 else args.size//3, color='yellow', linestyle='--')

        ax = plt.gca()
        ax.add_patch(arc)

        plt.grid(True)

    #Update world will take in an agent and a new position and update that agents position according to 
    #grid spaces where the agent is allowed to move to. 
    def updateWorld(self, agent, new_pos):
        for i in range(n_agents):
            if new_pos  in agent.availableSqrs():
                agent.agent_pos = new_pos

#The agent class holds the relevant information for each agent including starting location as a coordinate, 
#the number of the agent, the position, the speed, and the hyperparameter. 
class Agent:
    def __init__(self, agent_n, start, agent_pos, agent_v, phi, lamda, gamma, qtable):
        self.agent_n = agent_n
        self.start = start
        self.agent_pos = agent_pos
        self.agent_v = agent_v
        self.phi = phi
        self.lamda = lamda
        self.gamma = gamma
        self.qtable = {
            coord: {"up": 0, "down": 0, "left": 0, "right": 0}
            for coord in all_sqrs
        }
        self.reset()

    #Reset is called at the end of the initializing function to ensure the agents are at the right starting points
    def reset(self):
        self.agent_pos = self.start
        return self.agent_pos

    #Returns available squares which an agent may legally move to for any given agent. 
    def availableSqrs(self):
        self.valid = [(self.agent_pos[0] + 1, self.agent_pos[1]), 
        (self.agent_pos[0] - 1, self.agent_pos[1]), 
        (self.agent_pos[0], self.agent_pos[1] + 1),
        (self.agent_pos[0], self.agent_pos[1] - 1),
        (self.agent_pos[0] + 1, self.agent_pos[1] + 1),
        (self.agent_pos[0] - 1, self.agent_pos[1] + 1),
        (self.agent_pos[0] + 1, self.agent_pos[1] - 1),
        (self.agent_pos[0] - 1, self.agent_pos[1] - 1)]

        for i in self.valid:
            if i in totObs or (i[0]>args.size) and (i[1]>args.size):
                self.valid.remove(i)
        
        return self.valid
        
    
    def getQValue(self, state, action): #takes state as coord tuple and action as [up], [left]...
        """
            Returns Q(state,action)
            Note: need to make sure it returns zero if state is new
        """
        return self.qtable[state][action]
    
    def updateQ(self, state, action, next_state, reward):
        """
            Performs the CPT-based Q-value update
        """
        u_r = self.utility_function(reward)

        next_q = self.computeValueFromQValues(next_state)
        target = u_r + (discount * next_q)
        old_q = self.getQValue(state, action)
        new_q = ((1 - lr) * old_q) + (alpha * target)
    

    def utility_function(self, reward):
        return (reward ** alpha) if reward > 0 else (-self.lamda * (reward ** alpha))

env = FlatGridWorld(size=args.size, goal=(args.size - (2 * args.size//3), args.size - 1), obstacles=(Obs1,Obs2,Obs3))
agents = [Agent(agent_n = 1, start = (int(args.size - (2/3) * args.size),0), agent_pos = (int(args.size - (2/3) * args.size)), agent_v = 10, phi = 0, lamda = 0, gamma = 0)]

#Show the visualization
plt.ion()
plt.show()

for i in range(num_episodes):
    for i in range(args.size):
        env.updateWorld(agents[0], (args.size//3,i))
        env.render()
        plt.pause(0.01)