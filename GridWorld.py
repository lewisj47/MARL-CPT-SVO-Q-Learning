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
from tqdm import tqdm

#The parse arguments allow for arguments to be passed to the program via the command line. size can be 12, 24 or 48.
parser = argparse.ArgumentParser()
parser.add_argument("size", type = int,  help="The size of the grid environment given as a length of one of the sides.")
parser.add_argument("episodes", type=int, help="The number of episodes to undergo during training")
args = parser.parse_args()

end_goal = []
end_goal.extend([(r, c) for c in range(args.size - 1, args.size) for r in range((args.size*1)//4, args.size//2)])
route_1 = [(10, r) for r in range(0, 23)]

#end_goal = [(r, c) for r in range(args.size - 1, args.size) for c in range(args.size//2, (args.size*3)//4)]
route_2 = [(r, 8) for r in range(0,13)]
route_2.append((13, c) for c in range(10, 23))

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

    agents = [Agent(agent_n = 1, start = (9, 0), agent_pos = (int(args.size - (2/3) * args.size),0), agent_speed = 1, agent_dir = 1, phi = 0, lamda = 2.5, gamma_gain = 0.61, gamma_loss = 0.69, alpha = 0.88, beta = 0.88)]
    env = FlatGridWorld(size=args.size, agents=agents, obstacles=(Obs1,Obs2,Obs3))
    
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

            if agents[0].agent_pos in end_goal:
                t = 0
                finish_n += 1
                break
            
            if agents[0].agent_pos in totObs:
                t = 0
                break


        epsilon = min_epsilon + (max_epsilon - min_epsilon) * math.exp(-decay_rate * t_e)
        print(epsilon)
        t_e += 1

    print(f"Agent reached the goal {finish_n} times, {(finish_n / args.episodes) * 100}% of all episodes.")

    with open("qtable_output.txt", "w") as f:
        pprint.pprint(agents[0].qtable, stream=f)


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
def neighboringSqrs(state):
    valid = []
    valid.append((state[0] + 1, state[1]), 
    (state[0] - 1, state[1]), 
    (state[0], state[1] + 1),
    (state[0], state[1] - 1))
    if state[2] == 2:
        valid.append((state[0] + 2, state[1]), 
        (state[0] - 2, state[1]), 
        (state[0], state[1] + 2),
        (state[0], state[1] - 2))
    
    valid = [i for i in valid if 0 <= i[0] < args.size and 0 <= i[1] < args.size]

    return valid

neighbor_cache = {
    (r, c): neighboringSqrs((r, c)) for r, c in product(range(args.size), repeat=2) 
}

def getLegalActions(state):
        speed = state[2]
        dir = state[3]
        legal_actions = []
        if dir == 1:
            if speed == 0:
                for acc in range(0,2):
                    for i in neighboringSqrs(state):
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
                    for i in neighboringSqrs(state):
                        if i[0] > state[0]:
                            legal_actions.append((1,0,acc))
                        if i[1] > state[1]:
                            legal_actions.append((0,1,acc))
                        if i[1] < state[1]:
                            legal_actions.append((0,-1,acc))
            elif speed == 2:
                for acc in range(-1,1):
                    for i in neighboringSqrs(state):
                        if i[0] > state[0]:
                            legal_actions.append((1,0,acc))
        elif dir == 2:
            if speed == 0:
                for acc in range(0,2):
                    for i in neighboringSqrs(state):
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
                    for i in neighboringSqrs(state):
                        if i[0] > state[0]:
                            legal_actions.append((1,0,acc))
                        if i[0] < state[0]:
                            legal_actions.append((-1,0,acc))
                        if i[1] > state[1]:
                            legal_actions.append((0,1,acc))
            elif speed == 2:
                for acc in range(-1,1):
                    for i in neighboringSqrs(state):
                        if i[1] > state[1]:
                            legal_actions.append((0,1,acc))
        elif dir == 3:
            if speed == 0:
                for acc in range(0,2):
                    for i in neighboringSqrs(state):
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
                    for i in neighboringSqrs(state):
                        if i[0] < state[0]:
                            legal_actions.append((-1,0,acc))
                        if i[1] > state[1]:
                            legal_actions.append((0,1,acc))
                        if i[1] < state[1]:
                            legal_actions.append((0,-1,acc))
            elif speed == 2:
                for acc in range(-1,1):
                    for i in neighboringSqrs(state):
                        if i[1] > state[1]:
                            legal_actions.append((0,1,acc))
        elif dir == 4:
            if speed == 0:
                for acc in range(0,2):
                    for i in neighboringSqrs(state):
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
                    for i in neighboringSqrs(state):
                        if i[0] > state[0]:
                            legal_actions.append((1,0,acc))
                        if i[0] < state[0]:
                            legal_actions.append((-1,0,acc))
                        if i[1] < state[1]:
                            legal_actions.append((0,-1,acc))
            elif speed == 2:
                for acc in range(-1,1):
                    for i in neighboringSqrs(state):
                        if i[1] < state[1]:
                            legal_actions.append((0,-1,acc))    

        if Goal(state) == 1:
            legal_actions = [0]
            return legal_actions
        else:
            return legal_actions

legal_actions_cache = {
    (r, c, s, d): getLegalActions((r, c, s, d)) for r, c in product(range(args.size), repeat=2) for s in speed_set for d in dir_set
}


def onRoute(state, route):
    if (state[0],state[1]) in route:
        return (route.index((state[0],state[1])) / 24)
    else:
        return 0

def rewardFunction(state):
    const1 = 100
    const2 = 10
    const4 = 100
    
    return(const1 * Goal(state) - const2 * Obs(state) + const4 * onRoute(state, route_1))

tp = {(r,c): {} for r,c in product(range(args.size), repeat = 2)}

for r,c in product(range(args.size), repeat = 2):
    for a in legal_actions_cache[(r,c)]:
        tp[(r,c)][a] = {}
        for n in neighbor_cache[(r,c)]:
                tp[(r,c)][a][n] = 0

for r,c in product(range(args.size), repeat = 2):
    if (r,c) in corner_sqrs:
        for a in legal_actions_cache[(r,c)]:
            for n in neighbor_cache[(r,c)]:
                if ((a[0] + r, a[1] + c) == n):
                    tp[(r,c)][a][n] = C
                else:
                    tp[(r,c)][a][n] = 1 - C
    elif ((r < 0) or (r >= args.size) or (c < 0) or (r >= args.size)):
        for a in legal_actions_cache[(r,c)]:
            for n in neighbor_cache[(r,c)]:
                if ((a[0] + r, a[1] + c) == n):
                    tp[(r,c)][a][n] = C
                else:
                    tp[(r,c)][a][n] = (1 - C)/2
    elif ((r,c) in end_goal):
        for a in legal_actions_cache[(r,c)]:
            for n in neighbor_cache[(r,c)]:
                tp[(r,c)][a][n] = 0
    else:
        for a in legal_actions_cache[(r,c)]:
            for n in neighbor_cache[(r,c)]:
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

        for i in route_1:
            grid[i] = 0.2

        for i in range(n_agents):
            grid[self.agents[i].agent_pos] = 1.0


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
        if chosen_action in legal_actions_cache[agent.agent_pos[0],agent.agent_pos[1], agent.agent_speed, agent.agent_dir]:
            taken_action = random.choices(list(tp[agent.agent_pos][chosen_action].keys()), weights=list(tp[agent.agent_pos][chosen_action].values()), k=1)[0]
            agent.agent_pos = taken_action

        global t
        t += 1


#The agent class holds the relevant information for each agent including starting location as a coordinate, 
#the number of the agent, the position, the speed, and the hyperparameter. 
class Agent:
    def __init__(self, agent_n, start, agent_pos, agent_speed, agent_dir, phi, lamda, gamma_gain, gamma_loss, alpha, beta):
        self.agent_n = agent_n
        self.start = start
        self.agent_pos = agent_pos
        self.agent_speed = agent_speed
        self.agent_dir = agent_dir
        self.phi = phi
        self.lamda = lamda
        self.gamma_gain = gamma_gain
        self.gamma_loss = gamma_loss
        self.alpha = alpha
        self.beta = beta

        self.qtable = {(r,c): {} for r,c in product(range(args.size), repeat = 2)}

        for r,c in product(range(args.size), repeat = 2):
            for a in legal_actions_cache[(r,c)]:
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
        if legal_actions_cache[self.agent_pos] == [0]:
            return None
        
        #Choose explore or exploit based on exploration rate epsilon
        explore = random.choices([True, False], weights=[epsilon, (1 - epsilon)], k=1)[0]
        if explore == True:
            action = random.choice(legal_actions_cache[self.agent_pos])
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
            rho_minus = rho_minus + (self.lamda * max(0, -X_sort[ii])**self.beta) * (z_3**g_l / (z_3**g_l + (1 - z_3)**g_l)**(1 / g_l) - z_4**g_l / (z_4**g_l + (1 - z_4)**g_l)**(1 / g_l))
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

        for action in legal_actions_cache[self.agent_pos]:
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