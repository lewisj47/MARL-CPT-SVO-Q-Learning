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
from tqdm import trange

parser = argparse.ArgumentParser()
parser.add_argument("episodes", type=int, help="The number of episodes to undergo during training")
args = parser.parse_args()

start_state_1 = (13, 0, 1)
start_state_2 = (13, 0, 1)
start_state_3 = (0, 10, 1)

#End Goals
end_goal_1 = []
end_goal_2 = []
end_goal_1.extend([(c, r) for r in range(22, 24) for c in range(12, 15)])
end_goal_2.extend([(c, r) for r in range(9, 12) for c in range(22, 24)])

#Routes are used in the reward function to reward the agent for making progress towards the goal
routes = {}

route_1 = [(13, r) for r in range(0, 24)]
route_2 = [(13, r) for r in range (0, 11)]
route_2.extend([(c, 10) for c in range(14, 24)])
route_3 = [(c, 10) for c in range (0,24)]

routes["1"] = {"Route": route_1, "End Goal": end_goal_1, "Start State": start_state_1}
routes["2"] = {"Route": route_2, "End Goal": end_goal_2, "Start State": start_state_2}
routes["3"] = {"Route": route_3, "End Goal": end_goal_2, "Start State": start_state_3}

allRoutes = route_1 + route_2 + route_3
allGoals = end_goal_1 + end_goal_2

allStates = []
for r in allRoutes:
    x,y = r
    for s in (0, 1, 2):
        allStates.append((x, y, s))

#Stop sign regions
stop_region = [(c, 8) for c in range(12, 15)]
slow_region = [(c, 7) for c in range(12, 15)]

#Obstacles 
totObs = []

Obs1 = [(c, r) for c in range(0, 9) for r in range(0, 9)]
totObs.extend(Obs1)
Obs2 = [(c, r) for c in range(15, 24) for r in range(0, 9)]
totObs.extend(Obs2)
Obs3 = [(c, r) for c in range(15, 24) for r in range(15, 24)]
totObs.extend(Obs3)
Obs4 = [(c, r) for c in range (0, 9) for r in range(15, 24)]
totObs.extend(Obs4)

#Environment Definitions
n_agents = 2
SIZE = 24
num_episodes = args.episodes

all_sqrs = [(r,c) for r in range(SIZE) for c in range(SIZE)]

speed_set = [0, 1, 2]

#Constants
lr = 0.2
discount = 0.9
max_epsilon = 1.0
min_epsilon = 0.05
target_epsilon = 0.051
decay_rate = -math.log((target_epsilon - min_epsilon) / (max_epsilon - min_epsilon)) / num_episodes

#T used to measure the number of ticks in a single episode
t = 0

def main():
    global env
    global t

    epsilon = 1

    #Environment object that is updated and rendered
    env = FlatGridWorld(size=SIZE, agents=[], obstacles=(Obs1,Obs2,Obs3))

    #List containing all agent objects
    global agents
    agents = [Agent(agent_n = 1, route = routes['2'], state = (13, 0, 1), phi = 0, lamda = 2.5, gamma_gain = 0.61, gamma_loss = 0.69, alpha = 0.88, beta = 0.88, env=env),
              Agent(agent_n = 2, route = routes['3'], state = (13, 0, 1), phi = 0, lamda = 0.5, gamma_gain = 0.61, gamma_loss = 0.69, alpha = 0.88, beta = 0.88, env=env)]
    
    env.agents = agents

    #Initializing global state
    env.rebuildGlobalState()

    for i in tqdm(range(num_episodes)):
        for agent in agents:
            agent.reset()                               #Reset agent states
        
        while True:
            for agent in agents:
                if ((agent.state[0], agent.state[1]) in agent.route["End Goal"]):
                    continue
                action = agent.getAction(env.global_state, epsilon)       #Get an action for agent i
                if action is None:
                    continue

                agent.updateQ(env.global_state, action)                   #Update q-value for agent i having taken action at state

                env.updateWorld(agent, action)          #Update agent positions and speeds

            env.rebuildGlobalState()          

            env.render()                                #Render in visualization

            #Show the visualization
            plt.ion()                                   #Activate interactive mode
            plt.show()                                  #Show visualization
            plt.pause(0.001)                           #Pause between episodes in seconds

            all_finished = all((agent.state[0], agent.state[1]) in agent.route["End Goal"] for agent in agents)

            for l in env.global_state:
                x,y,s = l
                if (x >= SIZE or y >= SIZE):
                    break

            if hasCollided(env.global_state) == 1:
                tqdm.write("Collision detected, stopping episode.")
                break

            if all_finished or hasCollided(env.global_state):
                t = 0
                break

        epsilon = min_epsilon + (max_epsilon - min_epsilon) * math.exp(-decay_rate * i) #Update epsilon according to decay rate
    
    for agent in agents:
        with open(f"qtable_output{agent.agent_n}.txt", "w") as f:
            pprint.pprint(agent.qtable, stream=f)

"""
runTest()

This function will run the agent for a few episodes with no learning, no exploration, and no transition probability.
"""
def runTest(iter_n, env):

    global t

    tqdm.write(f"Now running {iter_n} iterations of test episodes.")

    total_rewards = [0 for _ in range(len(agents))]
    collision_num = 0

    env.rebuildGlobalState()

    for i in tqdm(range(num_episodes)):
        for agent in env.agents:
            agent.reset()                               #Reset agent states
        
        while True:
            for agent in env.agents:
                if ((agent.state[0], agent.state[1]) in agent.route["End Goal"]):
                    continue
                action = agent.getAction(env.global_state, 0)       #Get an action for agent i
                if action is None:
                    continue
                
                total_rewards[agent.agent_n - 1] += rewardFunction(agent.state, agent.route, action, env) #Get the reward for agent i

                agent.updateQ(env.global_state, action)                   #Update q-value for agent i having taken action at state

                env.updateWorld(agent, action)          #Update agent positions and speeds

            env.rebuildGlobalState()          

            env.render()                                #Render in visualization

            #Show the visualization
            plt.ion()                                   #Activate interactive mode
            plt.show()                                  #Show visualization
            plt.pause(0.01)                           #Pause between episodes in seconds

            if hasCollided(env.global_state) == 1:
                tqdm.write("Collision detected, stopping episode.")
                collision_num += 1
                break
            
            all_finished = all((agent.state[0], agent.state[1]) in agent.route["End Goal"] for agent in env.agents)

            if all_finished or hasCollided(env.global_state):
                t = 0
                break

    trange_desc = f"Collisions: {collision_num}"
    trange.set_description(trange_desc)  # dynamic description

    print(f"Total number of collisions: {collision_num}")
    print(f"Total rewards for each agent: {total_rewards}")

"""
Goal(state):

This function returns one if the state passed is in the end_goal and zero if not
"""
def Goal(state, route):
    if ((state[0],state[1]) in route["End Goal"]):
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

def hasCollided(global_state):
    positions = [(state[0], state[1]) for state in global_state]

    for pos in positions[:]:
        for r in routes:
            if pos in routes[r]["End Goal"] and pos in positions:
                positions.remove(pos)
    
    for state in global_state:
        for r, path in routes.items():
            pos = (state[0], state[1])
            if pos in path:
                idx = path.index(pos)
                if idx + 1 < len(path):
                    next_tile = path[idx + 1]
                    for s in global_state:
                        if s[0] == next_tile[0] and s[1] == next_tile[1]:
                            third_entry = s[2]
                            break
                    if state[2] == 2 and next_tile in [(s[0], s[1]) for s in global_state] and third_entry == 2:
                        print("Edge case")
                        return 1

    if len(positions) != len(set(positions)):
        print("regular")
        return 1
    else:
        return 0

"""
neighboringStates(state):

This function returns all possible next states for an agent including available grid squares, speeds, and directions as a list of tuples
"""
def neighboringStates(state, route):
    valid = []
    route_list = route["Route"]

    # Precompute coordinate -> index map
    route_index = {pos: i for i, pos in enumerate(route_list)}

    # If in goal, no next states
    if (state[0], state[1]) in route["End Goal"]:
        return valid

    # Only process if current position is on the route
    if (state[0], state[1]) in route_index:
        idx = route_index[(state[0], state[1])]
        speed = state[2]

        # Speed 0: stay or start moving
        if speed == 0:
            for s in (0, 1):
                valid.append((state[0], state[1], s))

        # Speed > 0: move forward that many steps
        else:
            for s in range(max(0, speed - 1), min(3, speed + 2)):  
                next_idx = idx + speed
                if next_idx < len(route_list):
                    next_r, next_c = route_list[next_idx]
                    valid.append((next_r, next_c, s))

    # Filter valid states: must be on the route
    route_coords = set(route_list)
    valid = [i for i in valid if (i[0], i[1]) in route_coords]

    return valid


"""
getLegalActions(state)

This function takes a state as a tuple and returns a list of tuples which represents all actions that agent may take at that state including movement 
and acceleration. 
"""
def getLegalActions(state):
    x, y, s = state

    legal_actions = []

    if s == 0:
        legal_actions = [0,1]
    
    elif s == 1:
        legal_actions = [-1, 0, 1]

    elif s == 2:
        legal_actions = [-1,0]

    return legal_actions    

def notMoving(state, action):
    if state[2] == 0 and action == 0:
        return 1
    else:
        return 0

def stopArea(state, action):
    global stopped
    x,y,s = state
    if (x,y) in stop_region:
        if s == 0 and action == 1:
            return 5
        elif s == 0 and action == 0:
            return -1
        else:
            return -3
    else:
        return 0

def slowArea(state, action):
    x,y,s = state
    if (x,y) in slow_region:
        if s == 1 and action == -1:
            return 3
        else:
            return -1
    else:
        return 0

"""
rewardFunction(state, action)

This function takes the state and action of an agent and returns the reward produced by the environment. In our case, things like being on a goal square, 
hitting an obstacle, and being on the route are important to the reward function. 
"""
def rewardFunction(state, route, action, env):
    const1 = 2000   # Reward for reaching the goal
    const2 = 600
    const4 = 50    # Penalty for accelerating or decelerating
    const5 = 50     # Penalty for not moving
    const6 = 1000
    return(const1 * Goal(state, route) - 
           const4 * abs(action) - 
           const2 * hasCollided(env.global_state) -
           const5 * notMoving(state, action) + 
           const6 * stopArea(state, action) + 
           const6 * slowArea(state, action))
  
tp = {}
tp = {r: {} for r in routes}
for r in routes:
    for (x, y) in routes[r]["Route"]:
        for s in (0, 1, 2):
            if (x, y, s) not in tp[r]:
                tp[r][(x, y, s)] = {}
            for action in getLegalActions((x, y, s)):
                if action not in tp[r][(x, y, s)]:
                    tp[r][(x, y, s)][action] = {}
                for (n_x, n_y, n_s) in neighboringStates((x, y, s), routes[f"{r}"]):
                    if s == 0:
                        if ((n_x, n_y, n_s) == (x + s, y + s, s + action)):
                            tp[r][(x, y, s)][action][(n_x, n_y, n_s)] = 0.99
                        else:
                            tp[r][(x, y, s)][action][(n_x, n_y, n_s)] = 0.01
                    elif s == 1:
                        if ((n_x, n_y, n_s) == (x + s, y + s, s + action)):
                            tp[r][(x, y, s)][action][(n_x, n_y, n_s)] = 0.95
                        else:
                            tp[r][(x, y, s)][action][(n_x, n_y, n_s)] = 0.05
                    elif s == 2:
                        if ((n_x, n_y, n_s) == (x + s, y + s, s + action)):
                            tp[r][(x, y, s)][action][(n_x, n_y, n_s)] = 0.9
                        else:
                            tp[r][(x, y, s)][action][(n_x, n_y, n_s)] = 0.1
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
        self.global_state = []
        for agent in agents:
            self.global_state.append(agent.state)

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

        for agent in self.agents:
            for coord in agent.route["End Goal"]:
                grid[coord] = 0.8

            for coord in agent.route["Route"]:
                grid[coord] = 0.2

        for i in range(n_agents):
            if ((self.agents[i].state[0], self.agents[i].state[1])) not in allGoals:
                grid[(self.agents[i].state[0], self.agents[i].state[1])] = 1.0
                for m, agent in enumerate(self.agents):
                    x, y = agent.state[0], agent.state[1]
                    plt.text(x, y, agent.agent_n,   # agent index as the number
                            ha='center', va='center',
                            fontsize=8, color='white')
            else:
                grid[(self.agents[i].state[0], self.agents[i].state[1])] = 0.2

        plt.text(0.05, 0.05, f"Ticks: {t}", 
                 transform=plt.gca().transAxes,  # position relative to axes (0-1)
                 fontsize=10, color='black', 
                 verticalalignment='bottom', horizontalalignment='left')

        cmap = colors.ListedColormap(['white', 'black', 'blue', 'green', 'red', (1,1,0,0.5), (1,0,0,0.5)])
        bounds = [-1.5, -0.5, 0.1, 0.5, 0.9, 1.5, 2, 2.5]
        norm = colors.BoundaryNorm(bounds, cmap.N)

        plt.imshow(grid.T, cmap=cmap, norm=norm)
        plt.xticks([])
        plt.yticks([])
        plt.axhline(xmin = 0.65, y = 11.5, color = 'yellow', linestyle='--')
        plt.axhline(xmax = 0.35, y = 11.5, color = 'yellow', linestyle='--')
        plt.axvline(ymin = 0.65, x= 11.5, color='yellow', linestyle='--')
        plt.axvline(ymax = 0.35, x = 11.5, color='yellow', linestyle='--')
        plt.axhline(y=8, xmin=0.5, xmax=0.65, color='white', linestyle='-')

        ax = plt.gca()
        ax.invert_yaxis()
        plt.grid(True)

    """
    updateWorld(agent, chosen_action)

    updateWorld() takes an agent and a chosen action and runs it past the environment before updating the agents state. Importantly, 
    it takes the probabilities from tp and updates the agent state accordingly. 
    """
    def updateWorld(self, agent, chosen_action):
        for key, value in routes.items():
            if value is agent.route:
                route_num = key
        if chosen_action in getLegalActions(agent.state):
            next_state = random.choices(list(tp[route_num][agent.state][chosen_action].keys()), weights=list(tp[route_num][agent.state][chosen_action].values()), k=1)[0]
            agent.state = next_state
        global t
        t += 1

    def rebuildGlobalState(self):
        self.global_state = [agent.state for agent in sorted(self.agents, key = lambda ag: ag.agent_n)]

"""
Agent

The agent class holds the logic for q-learning, action fetching, and other methods like sampling which are important to the agent. 
"""
class Agent:
    def __init__(self, agent_n, route, state, phi, lamda, gamma_gain, gamma_loss, alpha, beta, env):
        self.agent_n = agent_n
        self.route = route
        self.state = state
        self.phi = phi
        self.lamda = lamda
        self.gamma_gain = gamma_gain
        self.gamma_loss = gamma_loss
        self.alpha = alpha
        self.beta = beta
        self.env = env

        self.qtable = {}

        coord = product(allStates, repeat=n_agents)
        for combo in coord:
            self.qtable[combo] = {}
            for a in (-1,0,1):
                self.qtable[combo][a] = 0

        self.reset()

    def reset(self):
        self.state = self.route["Start State"]
        return self.state
    
    """
    getQvalue(action)

    getQvalue retrieves the qvalue for a state and action from the qtable. 
    """
    def getQValue(self, global_state, action):
        """
            Returns Q(state, action)
            Note: need to make sure it returns zero if state is new
        """
        state_key = tuple(global_state)

        return self.qtable[state_key][action]
    
    """
    getAction(epsilon)

    The getAction method retrieves a chosen action based on the probability of exploring vs exploiting.
    """
    def getAction(self, global_state, epsilon):
        action = None
        if getLegalActions(self.state) == [0]:
            return None

        explore = random.choices([True, False], weights=[epsilon, (1 - epsilon)], k=1)[0]
        if explore == True:
            action = random.choice(getLegalActions(self.state))
        else:
            action = self.getPolicy(global_state)
        return action

    """
    updateQ(action)

    updateQ performs the cpt-based q-value updating required for the agent to learn. It contains sampling, the rho-cpt function, and updating
    the q-value based on the learning rate before it updates the q-table. 
    """
    def updateQ(self, global_state, action):
        samples = self.sample_outcomes(action)
        target = self.rho_cpt(samples)
        current_q = self.getQValue(global_state, action)
        new_q = ((1 - lr) * current_q) + (lr * target)

        state_key = tuple(global_state)

        self.qtable[state_key][action] = new_q 


    """
    sample_outcomes(action, n_samples)

    Using the current state, passed action, and dictionary of transition probabilities,
    compiles a list of samples for future Q-values to be modified using CPT and then used
    in the updateQ function
    """
    def sample_outcomes(self, action, n_samples=50):
        """
        Using the current state, passed action, and dictionary of transition probabilities,
        compiles a list of samples for future Q-values to be modified using CPT and then used
        in the updateQ function. This version also predicts other agents' next states.
        """
        samples = []
        for key, value in routes.items():
            if value is self.route:
                route_num = key
        next_states = list(tp[route_num][self.state][action].keys())
        probs = list(tp[route_num][self.state][action].values())
            
        for _ in range(n_samples):
            # Sample the next state for the current agent
            s_prime = random.choices(next_states, weights=probs, k=1)[0]
            predicted_global_state = []
            
            for other_agent in self.env.agents:
                for key, value in routes.items():
                    if value is other_agent.route:
                        route_num = key
                if other_agent == self:
                    predicted_global_state.append(s_prime)
                else:
                    other_actions = getLegalActions(other_agent.state)
                    if not other_actions:
                        predicted_global_state.append(other_agent.state)
                        continue
                    weights = [sum(tp[route_num][other_agent.state][a].values()) for a in other_actions]
                    if sum(weights) == 0:
                        other_action = random.choice(other_actions)
                    else:
                        other_action = random.choices(other_actions, weights=weights, k=1)[0]
                    if other_agent.state not in tp or other_action not in tp[route_num][other_agent.state]:
                        other_s_prime = other_agent.state
                    else:
                        other_next_states = list(tp[other_agent.state][other_action].keys())
                        otherprobs = list(tp[route_num][other_agent.state][other_action].values())
                        other_s_prime = random.choices(other_next_states, weights=otherprobs, k=1)[0]
                    predicted_global_state.append(other_s_prime)
            
            reward = rewardFunction(s_prime, self.route, action, self.env)

            legal_actions = getLegalActions(s_prime)
            if not legal_actions:
                v_s_prime = 0
            else:
                v_s_prime = max(self.getQValue(predicted_global_state, a) for a in legal_actions)
            
            full_return = reward + (discount * v_s_prime)
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
            rho_plus = rho_plus + max(0, X_sort[ii])**self.alpha * (
                z_1**g_g / (z_1**g_g + (1 - z_1)**g_g)**(1 / g_g) 
                - z_2**g_g / (z_2**g_g + (1 - z_2)**g_g)**(1 / g_g)
            )
            rho_minus = rho_minus + (self.lamda * max(0, -X_sort[ii])**self.beta) * (
                z_3**g_l / (z_3**g_l + (1 - z_3)**g_l)**(1 / g_l) 
                - z_4**g_l / (z_4**g_l + (1 - z_4)**g_l)**(1 / g_l)
            )
        rho = rho_plus - rho_minus

        return rho
    
    """
    getPolicy()

    Compute best action to take in a state. Will need to add 
    belief distribution for multi-agent CPT
    """
    def getPolicy(self, global_state):

        best_value = -float('inf')
        best_actions = []

        for action in getLegalActions(self.state):
            value = self.getQValue(global_state, action)
            if value > best_value:
                best_value = value
                best_actions = [action]
            elif value == best_value:
                best_actions.append(action)

        return random.choice(best_actions)

main()

lr = 0

runTest(25, env)
