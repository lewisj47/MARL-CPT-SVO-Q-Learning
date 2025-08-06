#This program creates and visualizes a grid world, our ui.\

import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np
from itertools import product
import argparse
import random
import pprint
import math
from tqdm import tqdm

#The parse arguments allow for arguments to be passed to the program via the command line. 
parser = argparse.ArgumentParser()
parser.add_argument("episodes", type=int, help="The number of episodes to undergo during training")
args = parser.parse_args()

end_goal = []
#end_goal.extend([(r, c) for c in range(23, 24) for r in range(9, 15)])
end_goal.extend([(r, c) for c in range(9,15) for r in range(23, 24)])

routes = {}
route_1 = [(13, r) for r in range(0, 23)]
lane_1 = [(r,c) for r in range(12,14) for c in range(0,23)]

route_2 = [(13,r) for r in range (0, 10)]
lane_2 = [(r,c) for r in range(12,15) for c in range(0,12)]
route_2.extend([(r,10) for r in range(13,23)])
lane_2.extend([(r,c) for r in range(13,23) for c in range(9,12)])

routes["1"] = {"Route": route_1, "Lane": lane_1}
routes["2"] = {"Route": route_2, "Lane": lane_2}


#An empty array to hold the coordinates of the obstacles. In our case, obstacles are spaces where the road is not. 
totObs = []

Obs1 = [(r, c) for r in range(0,9) for c in range(0,9)]
totObs.extend(Obs1)

Obs2 = [(r, c) for r in range(15, 24) for c in range(0,9)]
totObs.extend(Obs2)

Obs3 = [(r, c) for r in range(15,24) for c in range(15,24)]
totObs.extend(Obs3)

Obs4 = [(r, c) for r in range (0, 9) for c in range(15, 24)]
totObs.extend(Obs4)

#Environment Definitions
n_agents = 1
SIZE = 24
num_episodes = args.episodes

all_sqrs = [(r,c) for r in range(SIZE) for c in range(SIZE)]
corner_sqrs = [(0,0),(0,SIZE), (SIZE, 0), (SIZE, SIZE)]

action_set = [(1,0,-1),(1,0,0),(1,0,1),(0,1,-1),(0,1,0),(0,1,1),(-1,0,-1),(-1,0,0),(-1,0,1),(0,-1,-1),(0,-1,0),(0,-1,1),(0,0,-1),(0,0,0),(0,0,1)]
speed_set = [0,1,2]
dir_set = [1,2,3,4] # 1:right, 2:up, 3:left, 4:down

n_actions = len(action_set)
n_states = len(all_sqrs) * len(speed_set) * len(dir_set)


#Constants
C = 0.95

lr = 0.2
discount = 0.9
max_epsilon = 1.0
min_epsilon = 0.01
target_epsilon = 0.011

decay_rate = -math.log((target_epsilon - min_epsilon) / (max_epsilon - min_epsilon)) / num_episodes


#Global Variables
t = 0
t_e = 0
epsilon = 1
finish_n = 0


def main():
    global epsilon
    global t_e
    global t
    global finish_n
    
    
    
    agents = [Agent(agent_n = 1, start_state = (13, 0, 1, 2), state = (0, 0, 1, 2), phi = 0, lamda = 2.5, gamma_gain = 0.61, gamma_loss = 0.69, alpha = 0.88, beta = 0.88)]
    env = FlatGridWorld(size=SIZE, agents=agents, obstacles=(Obs1,Obs2,Obs3))

    
    for i in tqdm(range(num_episodes)):
        agents[0].reset()
        while True:
            action = agents[0].getAction(epsilon)

            agents[0].updateQ(action)

            env.updateWorld(agents[0], action)
            env.render()

            #Show the visualization
            #if (t_e > args.episodes - 10):
            plt.ion()
            plt.show()
            plt.pause(0.0001)
           
            if (t > 250):
                t = 0
                break

            if (agents[0].state[0], agents[0].state[1]) in end_goal:
                t = 0
                finish_n += 1
                break
            
            if (agents[0].state[0], agents[0].state[1]) in totObs:
                t = 0
                break


        epsilon = min_epsilon + (max_epsilon - min_epsilon) * math.exp(-decay_rate * t_e)
        t_e += 1

    print(f"Agent reached the goal {finish_n} times, {(finish_n / args.episodes) * 100}% of all episodes.")

    with open("qtable_output.txt", "w") as f:
        pprint.pprint(agents[0].qtable, stream=f)

gen = (
    (r, c, s, d, a, n)
    for r, c in product(range(24), repeat=2)
    for s in speed_set
    for d in dir_set
    for a in legal_actions_cache[(r, c, s, d)]
    for n in neighbor_cache[(r, c, s, d)]
)

def Goal(state):
    if (state[0],state[1]) in end_goal:
        return 1
    else: 
        return 0
def Obs(state):
    if (state[0],state[1]) in totObs:
        return 1
    else:
        return 0

#Returns available squares (not states) which an agent may legally move to for any given agent. 
def neighboringStates(state):
    valid = []
    if (state[2] == 0):
        for s in (0,1):
            for dir in dir_set:
                valid.extend([(state[0] + 1, state[1], s, d), 
                (state[0] - 1, state[1], s, d), 
                (state[0], state[1] + 1, s, d),
                (state[0], state[1] - 1, s, d)])
    if (state[2] == 1):
        for s in (0,1,2):
            if state[3] == 1:
                for d in (1,2,4):
                    valid.extend([(state[0] + 1, state[1], s, d),
                    (state[0], state[1] + 1, s, d),
                    (state[0], state[1] - 1, s, d)])
            if state[3] == 2:
                for d in (1,2,3):
                    valid.extend([(state[0] + 1, state[1], s, d), 
                    (state[0] - 1, state[1], s, d), 
                    (state[0], state[1] + 1, s, d)])
            if state[3] == 3:
                for d in (2,3,4):
                    valid.extend([ (state[0] - 1, state[1], s, d), 
                    (state[0], state[1] + 1, s, d),
                    (state[0], state[1] - 1, s, d)])
            if state[3] == 4:
                for d in (3,4,1):
                    valid.extend([(state[0] + 1, state[1], s, d), 
                    (state[0] - 1, state[1], s, d),
                    (state[0], state[1] - 1, s, d)])
    if state[2] == 2:
        for s in (1,2):
            if state[3] == 1:
                valid.extend((state[0] + 2, state[1], s, 1))
            if state[3] == 2:
                valid.extend((state[0], state[1] + 2, s, 2))
            if state[3] == 3:
                valid.extend((state[0] - 2, state[1], s, 3))
            if state[3] == 4:
                valid.extend((state[0], state[1] - 2, s, 4))
    
    valid = [i for i in valid if 0 <= i[0] < SIZE and 0 <= i[1] < SIZE]

    return valid


neighbor_cache = {
    (r, c, s, d): neighboringStates((r,c,s,d)) for r, c in product(range(SIZE), repeat=2) for s in speed_set for d in dir_set
}

def getLegalActions(state):
        speed = state[2]
        dir = state[3]
        legal_actions = []
        if dir == 1:
            if speed == 0:
                for acc in range(0,2):
                    for i in neighboringStates(state):
                        if i[0] > state[0]:
                            legal_actions.append((1,0,acc))
                        if i[0] < state[0]:
                            legal_actions.append((-1,0,acc))
                        if i[1] > state[1]:
                            legal_actions.append((0,1,acc))
                        if i[1] < state[1]:
                            legal_actions.append((0,-1,acc))
                        legal_actions.append((0,0,acc))
            elif speed == 1:
                for acc in range(-1,2):
                    for i in neighboringStates(state):
                        if i[0] > state[0]:
                            legal_actions.append((1,0,acc))
                        if i[1] > state[1]:
                            legal_actions.append((0,1,acc))
                        if i[1] < state[1]:
                            legal_actions.append((0,-1,acc))
            elif speed == 2:
                for acc in range(-1,1):
                    for i in neighboringStates(state):
                        if i[0] > state[0]:
                            legal_actions.append((1,0,acc))
        elif dir == 2:
            if speed == 0:
                for acc in range(0,2):
                    for i in neighboringStates(state):
                        if i[0] > state[0]:
                            legal_actions.append((1,0,acc))
                        if i[0] < state[0]:
                            legal_actions.append((-1,0,acc))
                        if i[1] > state[1]:
                            legal_actions.append((0,1,acc))
                        if i[1] < state[1]:
                            legal_actions.append((0,-1,acc))
                        legal_actions.append((0,0,acc))
            elif speed == 1:
                for acc in range(-1,2):
                    for i in neighboringStates(state):
                        if i[0] > state[0]:
                            legal_actions.append((1,0,acc))
                        if i[0] < state[0]:
                            legal_actions.append((-1,0,acc))
                        if i[1] > state[1]:
                            legal_actions.append((0,1,acc))
            elif speed == 2:
                for acc in range(-1,1):
                    for i in neighboringStates(state):
                        if i[1] > state[1]:
                            legal_actions.append((0,1,acc))
        elif dir == 3:
            if speed == 0:
                for acc in range(0,2):
                    for i in neighboringStates(state):
                        if i[0] > state[0]:
                            legal_actions.append((1,0,acc))
                        if i[0] < state[0]:
                            legal_actions.append((-1,0,acc))
                        if i[1] > state[1]:
                            legal_actions.append((0,1,acc))
                        if i[1] < state[1]:
                            legal_actions.append((0,-1,acc))

            elif speed == 1:
                for acc in range(-1,2):
                    for i in neighboringStates(state):
                        if i[0] < state[0]:
                            legal_actions.append((-1,0,acc))
                        if i[1] > state[1]:
                            legal_actions.append((0,1,acc))
                        if i[1] < state[1]:
                            legal_actions.append((0,-1,acc))
            elif speed == 2:
                for acc in range(-1,1):
                    for i in neighboringStates(state):
                        if i[1] > state[1]:
                            legal_actions.append((0,1,acc))
        elif dir == 4:
            if speed == 0:
                for acc in range(0,2):
                    for i in neighboringStates(state):
                        if i[0] > state[0]:
                            legal_actions.append((1,0,acc))
                        if i[0] < state[0]:
                            legal_actions.append((-1,0,acc))
                        if i[1] > state[1]:
                            legal_actions.append((0,1,acc))
                        if i[1] < state[1]:
                            legal_actions.append((0,-1,acc))
                        legal_actions.append((0,0,acc))
            elif speed == 1:
                for acc in range(-1,2):
                    for i in neighboringStates(state):
                        if i[0] > state[0]:
                            legal_actions.append((1,0,acc))
                        if i[0] < state[0]:
                            legal_actions.append((-1,0,acc))
                        if i[1] < state[1]:
                            legal_actions.append((0,-1,acc))
            elif speed == 2:
                for acc in range(-1,1):
                    for i in neighboringStates(state):
                        if i[1] < state[1]:
                            legal_actions.append((0,-1,acc))    

        if Goal(state) == 1:
            legal_actions = [0]
            return legal_actions
        else:
            return legal_actions

legal_actions_cache = {
    (r, c, s, d): getLegalActions((r, c, s, d)) for r, c in product(range(SIZE), repeat=2) for s in speed_set for d in dir_set
}

def actionDir(action):
    if action[0] >= 1:
        return 1
    if action[1] >= 1:
        return 2
    if action[0] <= 1:
        return 3
    if action[1] <= 1:
        return 4

def onRoute(state, route):

    if (state[0], state[1]) in route["Route"]:
        return (route["Route"].index(state)/SIZE)
    elif (state[0], state[1]) in route["Lane"]:
        return 0
    else:
        return -0.7


#Still need to add reward for stop sign stoppage and collision avoidance
def rewardFunction(state, action):
    const1 = 1000   # Reward for reaching the goal
    const2 = 100    # Penalty for hitting an obstacle
    const3 = 10     # Reward for being on the route
    const4 = 1      # Penalty for accelerating or decelerating
    return(const1 * Goal(state) - const2 * Obs(state) + const3 * onRoute(state, routes['2']) - const4 * abs(action[2]))
  

tp = {(r,c,s,d)[a]: {} for r,c in product(range(SIZE), repeat = 2) for s in speed_set for d in dir_set for a in legal_actions_cache(r, c, s, d)}

for r,c,s,d,a,n in gen:
    tp[(r,c,s,d)][a][n] = 0

for r,c,s,d,a,n in gen:
    if ((r,c) in end_goal):
        tp[(r,c,s,d)][a][n] = 0 
    
    elif (s == 0):
        if ((s * a[0] + r, s * a[1] + c) == (n[0],n[1]) and (s + a == n[2]) and (actionDir((a[0],a[1])) == n[3])):
            tp[(r,c,s,d)][a][n] = 1
        else:
            tp[(r,c,s,d)][a][n] = 0

    elif (d == n[3]):
        if (a[2] == 0):
            if((s * a[0] + r, s * a[1] + c) == (n[0],n[1]) and (s + a == n[2]) and (actionDir((a[0],a[1])) == n[3])):
                tp[(r,c,s,d)][a][n] = 0.99
            else:
                tp[(r,c,s,d)][a][n] = 0.01
        elif not (a[2] == 0):
            if((s * a[0] + r, s * a[1] + c) == (n[0],n[1]) and (s + a == n[2]) and (actionDir((a[0],a[1])) == n[3])):
                tp[(r,c,s,d)][a][n] = 0.95
            else:
                tp[(r,c,s,d)][a][n] = 0.05
                
    elif not (d == n[3]):
        if (a[2] == 0):
            if((s * a[0] + r, s * a[1] + c) == (n[0],n[1]) and (s + a == n[2]) and (actionDir((a[0],a[1])) == n[3])):
                tp[(r,c,s,d)][a][n] = 0.95
            else:
                tp[(r,c,s,d)][a][n] = 0.05
        elif not (a[2] == 0):
            if((s * a[0] + r, s * a[1] + c) == (n[0],n[1]) and (s + a == n[2]) and (actionDir((a[0],a[1])) == n[3])):
                tp[(r,c,s,d)][a][n] = 0.85
            else:
                tp[(r,c,s,d)][a][n] = 0.15


#This class defines the environment in which the agent will learn. There is a corresponding size given as the length of
#one of the square worlds sides, the goal square, and an array containing the coordinates of each of the obstacles. 
class FlatGridWorld:
    def __init__(self, size, agents, obstacles=[]):
        self.size = SIZE  # grid is size x size
        self.n_squares = size * size
        self.obstacles = [Obs1, Obs2, Obs3]
        self.agents = agents

    #Render sets a cmap for the obstacles, agents, and road, as well as creating lines for the road. It generates the
    #new world every time a change is made. 
    def render(self):
        plt.clf()

        grid = np.zeros((self.size, self.size))

        # Fill in obstacle, agent, start, goal
        for coord in product(range(SIZE), repeat=2):  # loops through (0,0), (0,1), ..., (11,11)
            if coord in totObs:
                grid[coord] = -1

        for coord in product(range(SIZE), repeat = 2):
            if coord in end_goal:
                grid[coord] = 0.8

        for i in routes["2"]["Route"]:
            grid[i] = 0.2

        for i in range(n_agents):
            grid[(self.agents[i].state[0], self.agents[i].state[1])] = 1.0


        cmap = colors.ListedColormap(['white', 'black', 'blue', 'green', 'red'])
        bounds = [-1.5, -0.5, 0.1, 0.5, 0.9, 1.5]
        norm = colors.BoundaryNorm(bounds, cmap.N)

        plt.imshow(grid.T, cmap=cmap, norm=norm)
        plt.xticks([])
        plt.yticks([])
        plt.axhline(xmin = 0.65, y = 11.5, color = 'yellow', linestyle='--')
        plt.axhline(xmax = 0.35, y = 11.5, color = 'yellow', linestyle='--')
        plt.axvline(ymin = 0.65, x= 11.5, color='yellow', linestyle='--')
        plt.axvline(ymax = 0.35, x = 11.5, color='yellow', linestyle='--')

        ax = plt.gca()
        ax.invert_yaxis()
        plt.grid(True)

    #Update world will take in an agent and a new position and update that agents position according to 
    #grid spaces where the agent is allowed to move to. 
    def updateWorld(self, agent, chosen_action):
        if chosen_action in legal_actions_cache[agent.state]:
            next_state = random.choices(list(tp[agent.state][chosen_action].keys()), weights=list(tp[agent.state][chosen_action].values()), k=1)[0]
            agent.state = next_state
        global t
        t += 1


#The agent class holds the relevant information for each agent including starting location as a coordinate, 
#the number of the agent, the position, the speed, and the hyperparameter. 
class Agent:
    def __init__(self, agent_n, start, state, phi, lamda, gamma_gain, gamma_loss, alpha, beta):
        self.agent_n = agent_n
        self.start = start
        self.state = state
        self.phi = phi
        self.lamda = lamda
        self.gamma_gain = gamma_gain
        self.gamma_loss = gamma_loss
        self.alpha = alpha
        self.beta = beta


        self.qtable = {(r,c,s,d): {} for r,c in product(range(SIZE), repeat = 2) for s in (0,1,2) for d in (1,2,3,4)}

        for r,c in product(range(24), repeat = 2):
            for s in (0,1,2):
                for d in (1,2,3,4):
                    for a in legal_actions_cache[(r,c,s,d)]:
                        self.qtable[(r,c,s,d)][a] = 0

        self.reset()

    #Reset is called at the end of the initializing function to ensure the agents are at the right starting points
    def reset(self):
        self.state = self.start
        return self.state
            
    def getQValue(self, action):
        """
            Returns Q(state, action)
            Note: need to make sure it returns zero if state is new
        """
        return self.qtable[self.state][action]
    
    def getAction(self, epsilon):
        """
            Choose an action for a given state using the exploration rate
            When exploiting, use computeActionFromQValues
        """
        action = None

        #If at terminal state no legal actions can be taken
        if legal_actions_cache[self.state] == [0]:
            return None
        
        #Choose explore or exploit based on exploration rate epsilon
        explore = random.choices([True, False], weights=[epsilon, (1 - epsilon)], k=1)[0]
        if explore == True:
            action = random.choice(legal_actions_cache[self.state])
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
        self.qtable[self.state][action] = new_q 

    def sample_outcomes(self, action, n_samples=50):
        """
            Using the current state and passed action and dictionary of transition probabilities,
            compiles a list of samples for future Q-values to be modified using CPT and then used
            in the updateQ function
        """
        samples = []
        next_states = list(tp[self.state][action].keys())
        probs = list((tp[self.state][action]).values())

        for _ in range(n_samples):
            s_prime = random.choices(next_states, weights=probs, k=1)[0]
            reward = rewardFunction(s_prime, action)
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
            rho_minus = rho_minus + (self.lamda * max(0, -X_sort[ii])**self.beta) * (z_3**g_l / (z_3**g_l + (1 - z_3)**g_l)**(1 / g_l) - z_4**g_l / (z_4**g_l + (1 - z_4)**g_l)**(1 / g_l))
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

        best_value = -float('inf')
        best_actions = []

        for action in legal_actions_cache[self.state]:
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

