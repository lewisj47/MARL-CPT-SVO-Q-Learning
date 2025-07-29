#This program creates and visualizes a grid world, our ui.\

import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.patches import Arc
import numpy as np
from itertools import product
import argparse
import random
from copy import deepcopy # may need to pip install

#The parse arguments allow for arguments to be passed to the program via the command line. size can be 12, 24 or 48.
parser = argparse.ArgumentParser()
parser.add_argument("size", type = int,  help="The size of the grid environment given as a length of one of the sides.")
parser.add_argument("episodes", type=int, help="The number of episodes to undergo during training")
args = parser.parse_args()

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
    def updateWorld(self, agent, chosen_action):
        for i in range(n_agents):
            if chosen_action in agent.getLegalActions():
                chosen_action = random.choices(list(tp[self.agent_pos][chosen_action].keys()), weights=list(tp[self.agent_pos][chosen_action].values()), k=1)[0]
                agent.agent_pos = chosen_action
        t += 1


#The agent class holds the relevant information for each agent including starting location as a coordinate, 
#the number of the agent, the position, the speed, and the hyperparameter. 
class Agent:
    def __init__(self, agent_n, start, agent_pos, agent_v, phi, lamda, gamma_gain, gamma_loss):
        self.agent_n = agent_n
        self.start = start
        self.agent_pos = agent_pos
        self.agent_v = agent_v
        self.phi = phi
        self.lamda = lamda
        self.gamma_gain = gamma_gain
        self.gamma_loss = gamma_loss

        self.qtable = {(r,c): {} for r,c in product(range(12), repeat = 2)}
        for r,c in product(range(12), repeat = 2):
            for a in neighboringSqrs((r,c)):
                tp[(r,c)][(a[0] - r, a[1] - c)] = 0

        self.reset()

    #Reset is called at the end of the initializing function to ensure the agents are at the right starting points
    def reset(self):
        self.agent_pos = self.start
        return self.agent_pos
        
    def getLegalActions(self):
        legal_actions = []
        for i in self.neighboringSqrs():
            if i[0] > self.agent_pos[0]:
                legal_actions.append((1,0))
            if i[0] < self.agent_pos[0]:
                legal_actions.append((-1,0))
            if i[1] > self.agent_pos[1]:
                legal_actions.append((0,1))
            if i[1] < self.agent_pos[1]:
                legal_actions.append((0,-1))

        if self.agent_pos == env.goal:
            legal_actions = [0]
            return legal_actions
        else:
            return legal_actions
            
    def getQValue(self, action): #takes state as coord tuple and action as [up], [left]...
        """
            Returns Q(state, action)
            Note: need to make sure it returns zero if state is new
        """
        return self.qtable[self.agent_pos][action]
    
    def getAction(self):
        """
            Choose an action for a given state using the exploration rate
            When exploiting, use computeActionFromQValues
        """
        
        action = None

        #If at terminal state no legal actions can be taken
        if self.getLegalActions() == [0]:
            return None
        
        #Choose explore or exploit based on exploration rate epsilon
        explore = random.choices([True, False], weights=[epsilon, (1 - epsilon)], k=1)[0]
        if explore == True:
            action = random.choice(self.getLegalActions())
        else:
            action = self.getPolicy(self.agent_pos)

        return action

    def updateQ(self, action, q_old):
        """
            Performs the CPT-based Q-value update by using samples for the estimated future Q-value
        """
        samples = self.sample_outcomes(action, q_old)
        target = self.rho_cpt(samples)
        current_q = self.getQValue(action)
        new_q = ((1 - lr) * current_q) + (lr * target)
        self.qtable[self.agent_pos][action] = new_q 

    def sample_outcomes(self, action, q_old, n_samples=50):
        """
            Using the current state and passed action and dictionary of transition probabilities,
            compiles a list of samples for future Q-values to be modified using CPT and then used
            in the updateQ function
        """
        samples = []
        
        try:
            next_state_probs = tp[self.agent_pos][action]
        except KeyError:
            return [0.0] * n_samples # if (state, action) pair is not found

        next_states = list(next_state_probs.keys())
        probs = list(next_state_probs.values())

        for _ in range(n_samples):
            s_prime = random.choices(next_states, weights=probs, k=1)[0]

            reward = getReward(self.agent_pos, action)
            legal_actions = list(q_old[s_prime].keys()) if s_prime in q_old else []
            v_s_prime = max([q_old[s_prime][a] for a in legal_actions], default=0.0)

            full_return = reward + (self.gamma * v_s_prime) + random.gauss(0,1)

            samples.append(full_return)
        return samples

    def rho_cpt(self, samples):
        """
            Compute CPT-value of a discrete random variable X given samples
            'samples' is a list of outcomes (comprised of rewards + discounted future values)
        """ 

        X = np.array(samples)
        X_sort = np.sort(X, axis = None)
        N_max = len(X_sort)
       
        rho_plus = 0
        rho_minus = 0

        alpha = self.alpha
        lamda = self.lamda
        g_g = self.gamma_gain
        g_l = self.gamma_loss 


        for ii in range(0, N_max):
            z_1 = (N_max + ii - 1) / N_max
            z_2 = (N_max - ii) / N_max
            z_3 = ii / N_max
            z_4 = (ii-1) / N_max
            rho_plus = rho_plus + max(0, X_sort[ii])**alpha * (z_1**g_g / (z_1**g_g + (1 - z_1)**g_g)**(1 / g_g) - z_2**g_g / (z_2**g_g + (1 - z_2)**g_g)**(1 / g_g))
            rho_minus = rho_minus + (-lamda * max(0, -X_sort[ii])**alpha) * (z_3**g_l / (z_3**g_l + (1 - z_3)**g_l)**(1 / g_l) - z_4**g_l / (z_4**g_l + (1 - z_4)**g_l)**(1 / g_l))
        rho = rho_plus - rho_minus

        return rho
    
    """
    def value_function(self, reward):
        alpha = self.alpha
        if reward >= 0:
            return reward ** alpha
        else:
            return -self.lamda * ((-reward) ** alpha)

    def weight_function(self, p, mode):
        gamma = self.gamma_gain if mode == 'gain' else self.gamma_loss
        return (p** gamma) / (((p ** gamma) + (1 - p) ** gamma) ** (1 / gamma))
    
    """  
    
    def getPolicy(self):
        """
        Compute best action to take in a state. Will need to add 
        belief distribution for multi-agent CPT 
        """

        best_value = -float('inf') #may reduce to high int for speed?
        for action in self.getLegalActions():
            value = self.getQValue(self.agent_pos, action)
            best_value = max(best_value, value)
            if best_value == value:
                best_action = action

        return best_action
    

#Returns available squares which an agent may legally move to for any given agent. 
def neighboringSqrs(state):
    valid = [(state[0] + 1, state[1]), 
    (state[0] - 1, state[1]), 
    (state[0], state[1] + 1),
    (state[0], state[1] - 1)]
    #(state + 1, state + 1),
    #(state - 1, state + 1),
    #(state + 1, state - 1),
    #(state - 1, state - 1)]
    
    for i in valid:
        if (i[0] + state[0]>args.size) or (i[1] + state[1]>args.size):
            valid.remove(i)

    return valid

def rewardFunction(state):
    #The reward function will go in here

    const1 = 100
    const2 = 100
    const3 = 5

    return(const1 * Obs(state) - const2 * Goal(state) - const3 * t)

def Obs(state):
    if state in totObs:
        return 1
    else:
        return 0

def Goal(state):
    if state == env.goal:
        return 1
    else: 
        return 0
"""
Main function starts here
"""
#Right now there is 1 agent
n_agents = 1
num_episodes = args.episodes
all_sqrs = [(r,c) for r in range(args.size) for c in range(args.size)]
action_set = [(1,0),(0,1),(-1,0),(0,-1)]
n_actions = len(action_set)
n_states = len(all_sqrs)

t = 0

tp = {(r,c): {} for r,c in product(range(12), repeat = 2)}
for r,c in product(range(12), repeat = 2):
    for a in action_set:
        tp[(r,c)][a] = {}
        for n in neighboringSqrs((r,c)):
            if ((r + a[0], c + a[1]) == n):
                tp[(r,c)][a][n] = 0.95
            else:
                tp[(r,c)][a][n] = (1 - 0.95)/(len(neighboringSqrs((r,c))) - 1)

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

Obs2 = [(r, c) for r in range((args.size//2), args.size) for c in range(0,(args.size//2))]
totObs.extend(Obs2)

Obs3 = [(r, c) for r in range((args.size//2),args.size) for c in range(((args.size*3)//4),args.size)]
totObs.extend(Obs3)

env = FlatGridWorld(size=args.size, goal=(args.size - (2 * args.size//3), args.size - 1), obstacles=(Obs1,Obs2,Obs3))
agents = [Agent(agent_n = 1, start = (int(args.size - (2/3) * args.size),0), agent_pos = (int(args.size - (2/3) * args.size)), agent_v = 10, phi = 0, lamda = 0, gamma_gain = 0, gamma_loss = 0)]

#Show the visualization
plt.ion()
plt.show()


"""
Learning Flow

getAction() -> In terms of exploit/explore and transition probability. 

getReward() -> In terms of current state, action, and transition probability. 

updateQ() -> In terms of current state, action, future states and distorted transition probability. 

updateWorld() -> Update agent states and visualization. 

"""


for i in range(num_episodes):
    agents[0].reset()
    for i in range(args.size):
        env.updateWorld(agents[0], (args.size//3,i))
        env.render()
        plt.pause(0.01)