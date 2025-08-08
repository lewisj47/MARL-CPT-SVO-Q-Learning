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
#End goal at the top
#end_goal.extend([(r, c) for c in range(23, 24) for r in range(9, 15)])

#End goal on the right
end_goal.extend([(r, c) for c in range(9,12) for r in range(22, 24)])


#Routes are used in the reward function to reward the agent for making progress towards the goal
routes = {}

#From bottom to top
route_1 = [(13, r) for r in range(0, 24)]
lane_1 = [(r,c) for r in range(12,14) for c in range(0,24)]

#From bottom to right
route_2 = [(13,r) for r in range (0, 11)]
lane_2 = [(r,c) for r in range(12,15) for c in range(0,12)]
route_2.extend([(r,10) for r in range(13,24)])
lane_2.extend([(r,c) for r in range(13,24) for c in range(9,12)])


#Populating route dictionary
routes["1"] = {"Route": route_1, "Lane": lane_1}
routes["2"] = {"Route": route_2, "Lane": lane_2}


#An empty array to hold the coordinates of the obstacles. In our case, obstacles are spaces where the road is not. 
totObs = []

#Bottom left obstacle
Obs1 = [(r, c) for r in range(0,9) for c in range(0,9)]
totObs.extend(Obs1)
#Bottom right obstacle
Obs2 = [(r, c) for r in range(15, 24) for c in range(0,9)]
totObs.extend(Obs2)
#Top right Obstacle
Obs3 = [(r, c) for r in range(15,24) for c in range(15,24)]
totObs.extend(Obs3)
#Top left obstacle
Obs4 = [(r, c) for r in range (0, 9) for c in range(15, 24)]
totObs.extend(Obs4)

#Environment Definitions

#Number of agents
n_agents = 1
#Size of grid
SIZE = 24
#Number of episodes passed as an argument in the command line
num_episodes = args.episodes

#All squares in the grid world
all_sqrs = [(r,c) for r in range(SIZE) for c in range(SIZE)]
#Corner sqrs in the grid world
corner_sqrs = [(0,0),(0,SIZE), (SIZE, 0), (SIZE, SIZE)]

#All available actions to the agent
action_set = [(1,0,-1),(1,0,0),(1,0,1),(0,1,-1),(0,1,0),(0,1,1),(-1,0,-1),(-1,0,0),(-1,0,1),(0,-1,-1),(0,-1,0),(0,-1,1)]
#All speeds the agent can be driving at

speed_set = [0,1,2]
#All directions the agent can be driving in
dir_set = [1,2,3,4] # 1:right, 2:up, 3:left, 4:down

#Length of the action set
n_actions = len(action_set)
#Length of the state set
n_states = len(all_sqrs) * len(speed_set) * len(dir_set)


#Constants
lr = 0.5
discount = 0.9
max_epsilon = 1.0
min_epsilon = 0.01
target_epsilon = 0.011
decay_rate = -math.log((target_epsilon - min_epsilon) / (max_epsilon - min_epsilon)) / num_episodes

#T used to measure the number of ticks in a single episode
t = 0


def main():
    global t

    epsilon = 1

    #Number of episodes where the agent makes it to the finish line
    finish_n = 0
    
    
    #List containing all agent objects
    agents = [Agent(agent_n = 1, start = (13, 0, 1, 2), state = (13, 0, 1, 2), phi = 0, lamda = 2.5, gamma_gain = 0.61, gamma_loss = 0.69, alpha = 0.88, beta = 0.88)]
    #Environment object that is updated and rendered
    env = FlatGridWorld(size=SIZE, agents=agents, obstacles=(Obs1,Obs2,Obs3))

    #Contains all datapoints for cumulative reward per episode which is then graphed once the training session is over
    rewardGraph = []

    for i in tqdm(range(num_episodes)):
        agents[0].reset()                               #Reset agent states

        while True:
            action = agents[0].getAction(epsilon)       #Get an action for agent i

            agents[0].updateQ(action)                   #Update q-value for agent i having taken action at state

            totReward = 0                               #Total cumulative reward per episode
            totReward += rewardFunction(agents[0].state, action)

            env.updateWorld(agents[0], action)          #Update agent positions and speeds
            env.render()                                #Render in visualization

            #Show the visualization
            #if (t_e > args.episodes - 10):
            plt.ion()                                   #Activate interactive mode
            plt.show()                                  #Show visualization
            plt.pause(0.0001)                           #Pause between episodes in seconds
           

            x, y, s, d = agents[0].state

            if (x, y) in end_goal or (x, y) in totObs or not neighbor_cache.get((x, y, s, d), False):
                t = 0
                if (x, y) in end_goal:
                    finish_n += 1
                break

        rewardGraph.append(totReward) #Extend list of cumulative rewards per episode

        epsilon = min_epsilon + (max_epsilon - min_epsilon) * math.exp(-decay_rate * i) #Update epsilon according to decay rate

    plt.ioff()  # Turn off interactive mode if it was on
    plt.figure()  # ✅ Start a new figure window
    plt.plot(range(1, num_episodes + 1), rewardGraph, label='Reward per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Reward per Episode')
    plt.grid(True)
    plt.legend()
    plt.show()
    plt.savefig("reward_plot.png")  # Save as image file
    
    print(f"Agent reached the goal {finish_n} times, {(finish_n / args.episodes) * 100}% of all episodes.")

    with open("qtable_output.txt", "w") as f:
        pprint.pprint(agents[0].qtable, stream=f)

"""
Goal(state):

This function returns one if the state passed is in the end_goal and zero if not
"""
def Goal(state):
    if ((state[0],state[1]) in end_goal):
        return 1
    else: 
        return 0
    
"""
Obs(state):

This function returns one if the state passed is in the totObs and zero if not
"""
def Obs(state):
    if (state[0],state[1]) in totObs:
        return 1
    else:
        return 0

"""
neighboringStates(state):

This function returns all possible next states for an agent including available grid squares, speeds, and directions as a list of tuples
"""
def neighboringStates(state):
    valid = []
    if (state[2] == 0):
        for s in (0,1):
            for d in dir_set:
                valid.extend([(state[0], state[1], s, d)])
    if (state[2] == 1):
        for s in (0,1,2):
            if state[3] == 1:
                valid.extend([(state[0] + 1, state[1], s, 1),
                (state[0], state[1] + 1, s, 2),
                (state[0], state[1] - 1, s, 4)])
            if state[3] == 2:
                valid.extend([(state[0] + 1, state[1], s, 1), 
                (state[0] - 1, state[1], s, 3), 
                (state[0], state[1] + 1, s, 2)])
            if state[3] == 3:
                valid.extend([ (state[0] - 1, state[1], s, 3), 
                (state[0], state[1] + 1, s, 2),
                (state[0], state[1] - 1, s, 4)])
            if state[3] == 4:
                valid.extend([(state[0] + 1, state[1], s, 1), 
                (state[0] - 1, state[1], s, 3),
                (state[0], state[1] - 1, s, 4)])
    if state[2] == 2:
        for s in (1,2):
            if state[3] == 1:
                valid.extend([(state[0] + 2, state[1], s, 1)])
            if state[3] == 2:
                valid.extend([(state[0], state[1] + 2, s, 2)])
            if state[3] == 3:
                valid.extend([(state[0] - 2, state[1], s, 3)])
            if state[3] == 4:
                valid.extend([(state[0], state[1] - 2, s, 4)])
    
    valid = [i for i in valid if 0 <= i[0] < SIZE and 0 <= i[1] < SIZE]
    #print(valid)
    return valid

#neighbor_cache is used so the neighboringStates function doesnt need to be called every time
neighbor_cache = {
    (c, r, s, d): neighboringStates((c,r,s,d)) for c in range(SIZE) for r in range(SIZE) for s in speed_set for d in dir_set
}

"""
dirToAction(dir)

This function takes a direction as an integer from 1 to 4 and returns the vector corresponding to that direction as a tuple
"""
def dirToAction(dir):
    dir_map = {
        1: (1, 0),
        2: (0, 1),
        3: (-1, 0),
        4: (0, -1),
    }
    return dir_map.get(dir)

"""
actionToDir(action)

This function takes a tuple and returns the integer corresponding to that direction
"""
def actionToDir(action):
    action_map = {
        (1, 0): 1,
        (0, 1): 2,
        (-1, 0): 3,
        (0, -1): 4,
    }
    return action_map.get(action)

"""
getLegalActions(state)

This function takes a state as a tuple and returns a list of tuples which represents all actions that agent may take at that state including movement 
and acceleration. 
"""
def getLegalActions(state):
    c, r, s, d = state
    possible_actions = [(1,0), (-1,0), (0,1), (0,-1)]

    dir_vec = dirToAction(d)
    dir_backwards = (-dir_vec[0], -dir_vec[1])

    if Goal(state):
        return [(0, 0, 0)]
    
    legal_actions = []
    neighbors = set(neighbor_cache[state])

    if s == 0:
        for acc in (0, 1):
            for dx, dy in possible_actions:
                new_c = c + dx
                new_r = r + dy
                new_d = actionToDir((dx, dy))
                new_s = s + acc

                new_state = (c, r, new_s, new_d)
                
                if new_state in neighbors:
                    legal_actions.append((dx, dy, acc))
    
    elif s == 1:
        for acc in (-1, 0, 1):
            for dx, dy in possible_actions:
                if (dx, dy) == dir_backwards:
                    continue
                new_c = c + dx
                new_r = r + dy
                new_d = actionToDir((dx, dy))
                new_s = s + acc
                new_state = (new_c, new_r, new_s, new_d)
                if new_state in neighbors:
                    legal_actions.append((dx, dy, acc))

    elif s == 2:
        for acc in (-1, 0):
            dx, dy = dir_vec
            new_c = c + s * dx
            new_r = r + s * dy
            new_s = s + acc
            new_state = (new_c, new_r, new_s, d)
            if new_state in neighbors:
                legal_actions.append((dx, dy, acc))
    #print(legal_actions)
    return legal_actions    


#legal_actions_cache is used so that getLegalActions does not need to be called. 
legal_actions_cache = {
    (c, r, s, d): getLegalActions((c, r, s, d)) for c in range(SIZE) for r in range(SIZE) for s in speed_set for d in dir_set
}

"""
onRoute(state,route)

This function returns the value to be used in the reward function. The value returned if the agent is not on the route is -0.7 and this times some
constant is deducted from the agents reward. If the agent is on the route, the index of the grid square that the agent is on is returned, 
multiplied by a constant, and given to the agent as a reward. The route lists are initialized such that later indices are closer to the goal, 
so the agent will recieve a higher reward for being on a route square closer to the goal. 
"""
def onRoute(state, route):

    if (state[0], state[1]) in route["Route"]:
        return (route["Route"].index((state[0],state[1]))/SIZE)
    elif (state[0], state[1]) in route["Lane"]:
        return (route["Lane"].index((state[0],state[1]))/(5 * SIZE))
    else:
        return 0

def notMoving(state, action):
    if state[2] == 0 and action[2] == 0:
        return 1
    else:
        return 0

"""
rewardFunction(state, action)

This function takes the state and action of an agent and returns the reward produced by the environment. In our case, things like being on a goal square, 
hitting an obstacle, and being on the route are important to the reward function. 
"""
def rewardFunction(state, action):
    const1 = 1000   # Reward for reaching the goal
    const2 = 100    # Penalty for hitting an obstacle
    const3 = 1000     # Reward for being on the route
    const4 = 0.5    # Penalty for accelerating or decelerating
    const5 = 1      # Penalty for not moving
    return(const1 * Goal(state) - const2 * Obs(state) + const3 * onRoute(state, routes['2']) - const4 * abs(action[2]) - const5 * notMoving(state, action))
  

tp = {(c,r,s,d): {a: {} for a in legal_actions_cache[(c, r, s, d)]} 
    for c in range(SIZE)
    for r in range(SIZE)
    for s in speed_set
    for d in dir_set}

for c, r, s, d, a in [
    (c, r, s, d, a)
    for c in range(SIZE)
    for r in range(SIZE)
    for s in speed_set
    for d in dir_set
    for a in legal_actions_cache[(c, r, s, d)]
]:
    neighbors = neighbor_cache[(c, r, s, d)]
    probs = {}
    
    # Identify the "correct" neighbor (the intended move)
    correct_neighbor = None
    for n in neighbors:
        if (s == 0):
            if ((c, r) == (n[0], n[1])) and (a[2] == n[2]) and (actionToDir((a[0], a[1])) == n[3]):
                correct_neighbor = n
                break
        else:
            if ((s * a[0] + c, s * a[1] + r) == (n[0], n[1]) 
                and (s + a[2] == n[2]) 
                and (actionToDir((a[0], a[1])) == n[3])):
                correct_neighbor = n
                break

    # Assign main probability based on direction/speed match
    if (c, r) in end_goal:
        probs = {n: 1.0 if n == (c, r, s, d) else 0.0 for n in neighbors}
        tp[(c, r, s, d)][a] = probs
        continue
    elif s == 0:
        main_prob = 1.0
    elif correct_neighbor and d == correct_neighbor[3]:
        main_prob = 0.99 if a[2] == 0 else 0.95
    else:
        main_prob = 0.95 if a[2] == 0 else 0.85

    # Distribute leftover probability to others
    if correct_neighbor:
        num_others = len(neighbors) - 1
        if num_others > 0:
            other_prob = (1.0 - main_prob) / num_others
        else:
            other_prob = 0.0

        for n in neighbors:
            if n == correct_neighbor:
                probs[n] = main_prob
            else:
                probs[n] = other_prob
    else:
        # No correct neighbor (edge case) -> uniform distribution
        uniform_prob = 1.0 / len(neighbors)
        probs = {n: uniform_prob for n in neighbors}

    tp[(c, r, s, d)][a] = probs

"""
FlatGridWorld

The FlatGridWorld class defines the environment in which the agent learns. It contains methods such as render and update world. It mainly contains 
logic for displaying the visualization. 
"""

class FlatGridWorld:
    def __init__(self, size, agents, obstacles=[]):
        self.size = SIZE  # grid is size x size
        self.n_squares = size * size
        self.obstacles = [Obs1, Obs2, Obs3]
        self.agents = agents

    """
    render()

    The render method sets the cmap for agents, obstacles, and road to be displayed in a matplotlib figure. 
    """
    def render(self):
        plt.clf()

        grid = np.zeros((self.size, self.size))

        # Fill in obstacle, agent, start, goal
        for coord in totObs:
            grid[coord] = -1

        for coord in end_goal:
            grid[coord] = 0.8

        grid[(1,0)] = 0.8

        for i in routes["2"]["Route"]:
            grid[i] = 0.2

        speed_color_map = {
            0: 'green',
            1: 'yellow',
            2: 'red'
        }

        for i in range(n_agents):
            grid[(self.agents[i].state[0], self.agents[i].state[1])] = 1.0
        

            dx, dy = dirToAction(self.agents[i].state[3])
            arrow_color = speed_color_map.get(self.agents[i].state[2], 'black')

            plt.arrow(self.agents[i].state[0], self.agents[i].state[1], dx * self.agents[i].state[2], dy * self.agents[i].state[2],
            head_width=0.5, head_length=0.5, fc=arrow_color, ec=arrow_color)

        plt.text(0.05, 0.05, f"Ticks: {t}", 
         transform=plt.gca().transAxes,  # position relative to axes (0-1)
         fontsize=10, color='black', 
         verticalalignment='bottom', horizontalalignment='left')

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

    """
    updateWorld(agent, chosen_action)

    updateWorld() takes an agent and a chosen action and runs it past the environment before updating the agents state. Importantly, 
    it takes the probabilities from tp and updates the agent state accordingly. 
    """
    def updateWorld(self, agent, chosen_action):
        if chosen_action in legal_actions_cache[agent.state]:
            next_state = random.choices(list(tp[agent.state][chosen_action].keys()), weights=list(tp[agent.state][chosen_action].values()), k=1)[0]
            agent.state = next_state
        global t
        t += 1


"""
Agent

The agent class holds the logic for q-learning, action fetching, and other methods like sampling which are important to the agent. 
"""
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


        self.qtable = {(c,r,s,d): {} for c,r in product(range(SIZE), repeat = 2) for s in (0,1,2) for d in (1,2,3,4)}

        for r,c in product(range(24), repeat = 2):
            for s in (0,1,2):
                for d in (1,2,3,4):
                    for a in legal_actions_cache[(c,r,s,d)]:
                        self.qtable[(c,r,s,d)][a] = 0

        self.reset()

    #Reset is called at the end of the initializing function to ensure the agents are at the right starting points
    def reset(self):
        self.state = self.start
        return self.state
    
    """
    getQvalue(action)

    getQvalue retrieves the qvalue for a state and action from the qtable. 
    """
    def getQValue(self, action):
        """
            Returns Q(state, action)
            Note: need to make sure it returns zero if state is new
        """
        return self.qtable[self.state][action]
    
    """
    getAction(epsilon)

    The getAction method retrieves a chosen action based on the probability of exploring vs exploiting.
    """
    def getAction(self, epsilon):
        action = None
        if legal_actions_cache[self.state] == [0]:
            return None

        explore = random.choices([True, False], weights=[epsilon, (1 - epsilon)], k=1)[0]
        if explore == True:
            action = random.choice(legal_actions_cache[self.state])
        else:
            action = self.getPolicy()
        return action

    """
    updateQ(action)

    updateQ performs the cpt-based q-value updating required for the agent to learn. It contains sampling, the rho-cpt function, and updating
    the q-value based on the learning rate before it updates the q-table. 
    """
    def updateQ(self, action):
        samples = self.sample_outcomes(action)
        target = self.rho_cpt(samples)
        current_q = self.getQValue(action)
        new_q = ((1 - lr) * current_q) + (lr * target)
        self.qtable[self.state][action] = new_q 


    """
    sample_outcomes(action, n_samples)

    Using the current state, passed action, and dictionary of transition probabilities,
    compiles a list of samples for future Q-values to be modified using CPT and then used
    in the updateQ function
    """
    def sample_outcomes(self, action, n_samples=50):
        samples = []
        next_states = list(tp[self.state][action].keys())
        probs = list(tp[self.state][action].values())
            
        for _ in range(n_samples):
            s_prime = random.choices(next_states, weights=probs, k=1)[0]
            reward = rewardFunction(s_prime, action)
            v_s_prime = max(self.qtable[s_prime].values(), default = 0.0)

            full_return = reward + (discount * v_s_prime) #+ random.gauss(0,1)

            samples.append(full_return)
        return samples


    """
    rho_cpt(samples)

    Compute CPT-value of a discrete random variable X given samples
    'samples' is a list of outcomes (comprised of rewards + discounted future values)
    """
    def rho_cpt(self, samples):
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
    getPolicy()

    Compute best action to take in a state. Will need to add 
    belief distribution for multi-agent CPT
    """
    def getPolicy(self):

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

main()

