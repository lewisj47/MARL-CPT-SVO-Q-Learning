#This program creates and visualizes a grid world, our ui.\

import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.patches import Arc
import numpy as np
from itertools import product
import argparse
import random
import pprint
import math

#The parse arguments allow for arguments to be passed to the program via the command line. size can be 12, 24 or 48.
parser = argparse.ArgumentParser()
parser.add_argument("size", type = int,  help="The size of the grid environment given as a length of one of the sides.")
parser.add_argument("episodes", type=int, help="The number of episodes to undergo during training")
args = parser.parse_args()

end_goal = []
end_goal.extend([(r, c) for c in range(args.size - 1, args.size) for r in range((args.size*1)//4, args.size//2)])

#An empty array to hold the coordinates of the obstacles. In our case, obstacles are spaces where the road is not. 
totObs = []

Obs1 = [(r, c) for r in range(0,(args.size//4)) for c in range(args.size)]
totObs.extend(Obs1)

Obs2 = [(r, c) for r in range((args.size//2), args.size) for c in range(0,(args.size//2))]
totObs.extend(Obs2)

Obs3 = [(r, c) for r in range((args.size//2),args.size) for c in range(((args.size*3)//4),args.size)]
totObs.extend(Obs3)

#Environment Definitions
n_agents = 1
num_episodes = args.episodes
all_sqrs = [(r,c) for r in range(args.size) for c in range(args.size)]
corner_sqrs = [(0,0),(0,args.size), (args.size, 0), (args.size, args.size)]
action_set = [(1,0),(0,1),(-1,0),(0,-1)]
n_actions = len(action_set)
n_states = len(all_sqrs)


#Constants
C = 0.95

lr = 0.2
discount = 0.95
max_epsilon = 1.0
min_epsilon = 0.01

decay_rate = 0.001

#Global Variables
t = 0
t_e = 0
epsilon = 1

def main():
    global epsilon
    global t_e
    global t

    agents = [Agent(agent_n = 1, start = (int(args.size - (2/3) * args.size),0), agent_pos = (int(args.size - (2/3) * args.size),0), agent_v = 10, phi = 0, lamda = 2.5, gamma_gain = 0.61, gamma_loss = 0.69, alpha = 0.88, beta = 0.88)]
    env = FlatGridWorld(size=args.size, agents=agents, obstacles=(Obs1,Obs2,Obs3))
    
    for i in range(num_episodes):
        agents[0].reset()
        while True:
            
            action = agents[0].getAction(epsilon)

            agents[0].updateQ(action)

            env.updateWorld(agents[0], action)
            env.render()

            #Show the visualization
            plt.ion()
            plt.show()
            plt.pause(0.001)
           
            if (t > 250):
                t = 0
                break

            if agents[0].agent_pos in end_goal:
                t = 0
                print("finish")
                break
            
            if agents[0].agent_pos in totObs:
                t = 0
                break


        epsilon = min_epsilon + (max_epsilon - min_epsilon) * math.exp(-decay_rate * t_e)
        t_e += 1
        print(t_e)

    with open("qtable_output.txt", "w") as f:
        pprint.pprint(agents[0].qtable, stream=f)


def getLegalActions(state):
        legal_actions = []
        for i in neighboringSqrs(state):
            if i[0] > state[0]:
                legal_actions.append((1,0))
            if i[0] < state[0]:
                legal_actions.append((-1,0))
            if i[1] > state[1]:
                legal_actions.append((0,1))
            if i[1] < state[1]:
                legal_actions.append((0,-1))

        if Goal(state) == 1:
            legal_actions = [0]
            return legal_actions
        else:
            return legal_actions

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
    
    valid = [i for i in valid if 0 <= i[0] < args.size and 0 <= i[1] < args.size]

    return valid

def Goal(state):
    if state == end_goal:
        return 1
    else: 
        return 0
def Obs(state):
    if state in totObs:
        return 1
    else:
        return 0
    
def Dist(state):
    dist = min(math.sqrt((i[1] - state[1])**2 + (i[0] - state[0])**2) for i in end_goal)
    return dist / args.size

def rewardFunction(state):

    const1 = 100
    const2 = 10
    const3 = 100
    return(const1 * Goal(state) - const2 * Obs(state) + const3 * (1 / (1 + Dist(state))))

tp = {(r,c): {} for r,c in product(range(args.size), repeat = 2)}

for r,c in product(range(args.size), repeat = 2):
    for a in getLegalActions((r,c)):
        tp[(r,c)][a] = {}
        for n in neighboringSqrs((r,c)):
                tp[(r,c)][a][n] = 0

for r,c in product(range(args.size), repeat = 2):
    if (r,c) in corner_sqrs:
        for a in getLegalActions((r,c)):
            for n in neighboringSqrs((r,c)):
                if ((a[0] + r, a[1] + c) == n):
                    tp[(r,c)][a][n] = C
                else:
                    tp[(r,c)][a][n] = 1 - C
    elif ((r < 0) or (r >= args.size) or (c < 0) or (r >= args.size)):
        for a in getLegalActions((r,c)):
            for n in neighboringSqrs((r,c)):
                if ((a[0] + r, a[1] + c) == n):
                    tp[(r,c)][a][n] = C
                else:
                    tp[(r,c)][a][n] = (1 - C)/2
    elif ((r,c) == end_goal):
        for a in getLegalActions((r,c)):
            for n in neighboringSqrs((r,c)):
                tp[(r,c)][a][n] = 0
    else:
        for a in getLegalActions((r,c)):
            for n in neighboringSqrs((r,c)):
                if ((a[0] + r, a[1] + c) == n):
                    tp[(r,c)][a][n] = C
                else:
                    tp[(r,c)][a][n] = (1 - C)/3

#This class defines the environment in which the agent will learn. There is a corresponding size given as the length of
#one of the square worlds sides, the goal square, and an array containing the coordinates of each of the obstacles. 
class FlatGridWorld:
    def __init__(self, size, agents, obstacles=[]):
        self.size = args.size  # grid is size x size
        self.n_squares = size * size
        self.obstacles = [Obs1, Obs2, Obs3]
        self.agents = agents

    #Render sets a cmap for the obstacles, agents, and road, as well as creating lines for the road. It generates the
    #new world every time a change is made. 
    def render(self):
        plt.clf()

        grid = np.zeros((self.size, self.size))

        # Fill in obstacle, agent, start, goal
        for coord in product(range(args.size), repeat=2):  # loops through (0,0), (0,1), ..., (11,11)
            if coord in totObs:
                grid[coord] = -1

        for coord in product(range(args.size), repeat = 2):
            if coord in end_goal:
                grid[coord] = 0.8

        for i in range(n_agents):
            grid[self.agents[i].agent_pos] = 1.0
  
        grid[(0,1)] = 0.8

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

        plt.imshow(grid.T, cmap=cmap, norm=norm)
        plt.xticks([])
        plt.yticks([])
        plt.axhline(xmin = 0.50, y = args.size*5//8 - 0.5 if (args.size//12)%2 == 0 else args.size*5//8, color = 'yellow', linestyle='--') #((L/4)%2)*0.5)
        plt.axvline(ymin = 0.75, x= args.size*3//8 - 0.5 if (args.size//12)%2 == 0 else args.size//3, color='yellow', linestyle='--')
        plt.axvline(ymax = 0.5, x = args.size*3//8 - 0.5 if (args.size//12)%2 == 0 else args.size//3, color='yellow', linestyle='--')

        ax = plt.gca()
        ax.invert_yaxis()
        ax.add_patch(arc)

        plt.grid(True)

    #Update world will take in an agent and a new position and update that agents position according to 
    #grid spaces where the agent is allowed to move to. 
    def updateWorld(self, agent, chosen_action):
        if chosen_action in getLegalActions(agent.agent_pos):
            taken_action = random.choices(list(tp[agent.agent_pos][chosen_action].keys()), weights=list(tp[agent.agent_pos][chosen_action].values()), k=1)[0]
            agent.agent_pos = taken_action

        global t
        t += 1


#The agent class holds the relevant information for each agent including starting location as a coordinate, 
#the number of the agent, the position, the speed, and the hyperparameter. 
class Agent:
    def __init__(self, agent_n, start, agent_pos, agent_v, phi, lamda, gamma_gain, gamma_loss, alpha, beta):
        self.agent_n = agent_n
        self.start = start
        self.agent_pos = agent_pos
        self.agent_v = agent_v
        self.phi = phi
        self.lamda = lamda
        self.gamma_gain = gamma_gain
        self.gamma_loss = gamma_loss
        self.alpha = alpha
        self.beta = beta

        self.qtable = {(r,c): {} for r,c in product(range(args.size), repeat = 2)}

        for r,c in product(range(args.size), repeat = 2):
            for a in getLegalActions((r,c)):
                self.qtable[(r,c)][a] = 0

        self.reset()

    #Reset is called at the end of the initializing function to ensure the agents are at the right starting points
    def reset(self):
        self.agent_pos = self.start
        return self.agent_pos
            
    def getQValue(self, action): #takes state as coord tuple and action as [up], [left]...
        """
            Returns Q(state, action)
            Note: need to make sure it returns zero if state is new
        """
        return self.qtable[self.agent_pos][action]
    
    def getAction(self, epsilon):
        """
            Choose an action for a given state using the exploration rate
            When exploiting, use computeActionFromQValues
        """
        
        action = None

        #If at terminal state no legal actions can be taken
        if getLegalActions(self.agent_pos) == [0]:
            return None
        
        #Choose explore or exploit based on exploration rate epsilon
        explore = random.choices([True, False], weights=[epsilon, (1 - epsilon)], k=1)[0]
        if explore == True:
            action = random.choice(getLegalActions(self.agent_pos))
        else:
            action = self.getPolicy()

        return action

    def updateQ(self, action):
        """
            Performs the CPT-based Q-value update by using samples for the estimated future Q-value
        """
        samples = self.sample_outcomes(action)
        target = self.rho_cpt(samples)
        current_q = self.getQValue(action)
        new_q = ((1 - lr) * current_q) + (lr * target)
        self.qtable[self.agent_pos][action] = new_q 

    def sample_outcomes(self, action, n_samples=50):
        """
            Using the current state and passed action and dictionary of transition probabilities,
            compiles a list of samples for future Q-values to be modified using CPT and then used
            in the updateQ function
        """
        samples = []

        next_states = list(tp[self.agent_pos][action].keys())
        probs = list((tp[self.agent_pos][action]).values())

        for _ in range(n_samples):
            s_prime = random.choices(next_states, weights=probs, k=1)[0]
            reward = rewardFunction(s_prime)
            v_s_prime = max(self.qtable[s_prime].values(), default = 0.0)

            full_return = reward + (discount * v_s_prime) #+ random.gauss(0,1)

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

        g_g = self.gamma_gain
        g_l = self.gamma_loss 

        for ii in range(1, N_max):
            z_1 = (N_max - ii + 1) / N_max
            z_2 = (N_max - ii) / N_max
            z_3 = ii / N_max
            z_4 = (ii-1) / N_max
            rho_plus = rho_plus + max(0, X_sort[ii])**self.alpha * (z_1**g_g / (z_1**g_g + (1 - z_1)**g_g)**(1 / g_g) - z_2**g_g / (z_2**g_g + (1 - z_2)**g_g)**(1 / g_g))
            rho_minus = rho_minus + (-self.lamda * max(0, -X_sort[ii])**self.beta) * (z_3**g_l / (z_3**g_l + (1 - z_3)**g_l)**(1 / g_l) - z_4**g_l / (z_4**g_l + (1 - z_4)**g_l)**(1 / g_l))
        rho = rho_plus - rho_minus

        return rho
    
    # potentially unecessary
    def getReward(self, state, action):
        if action not in tp[state]:
            # action doesn't work with state (invalid state/action pair)
            print(f"Warning: action {action} not in tp[{state}]")
            return 0
        tot_reward = 0
        for next_state, prob in tp[state][action].items(): # note: items() returns dict key-value pairs as tuples
            tot_reward += prob * rewardFunction(next_state)
        
        return tot_reward  
    
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

        best_value = -float('inf')
        best_actions = []

        for action in getLegalActions(self.agent_pos):
            value = self.getQValue(action)
            if value > best_value:
                best_value = value
                best_actions = [action]
            elif value == best_value:
                best_actions.append(action)

        return random.choice(best_actions)

"""
Main function starts here
"""

main()


"""
Learning Flow

getAction() -> In terms of exploit/explore and transition probability. 

getReward() -> In terms of current state, action, and transition probability. 

updateQ() -> In terms of current state, action, future states and distorted transition probability. 

updateWorld() -> Update agent states and visualization. 

"""