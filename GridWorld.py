#This program creates and visualizes a grid world, our ui.\

import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np
import argparse
import random
import pprint
import math
from tqdm import tqdm
from colorama import Fore, Style

parser = argparse.ArgumentParser()
parser.add_argument("episodes", type=int, help="The number of episodes to undergo during training")
parser.add_argument("testepisodes", type=int, help="The number of episodes to undergo during testing")
args = parser.parse_args()

start_state_1 = (13, 3, 1)
start_state_2 = (13, 8, 0)
start_state_3 = (0, 10, 2)
start_state_4 = (6, 10, 2)

#End Goals
end_goal_1 = []
end_goal_2 = []
end_goal_1.extend([(c, r) for r in range(22, 24) for c in range(12, 15)])
end_goal_2.extend([(c, r) for r in range(9, 12) for c in range(22, 24)])

#Routes are used in the reward function to reward the agent for making progress towards the goal
routes = {}

route_1 = [(13, r) for r in range(3, 24)]
route_2 = [(13, r) for r in range (0, 11)]
route_2.extend([(c, 10) for c in range(14, 24)])

turn_2 = [(13, 8), (13, 9), (13,10), (14, 10), (15, 10)]

route_3 = [(c, 10) for c in range(0, 24)]

route_4 = [(c, 10) for c in range(4, 24)]

#Route 1: straight up
routes["1"] = {"Route": route_1, "End Goal": end_goal_1, "Start State": start_state_1}

#Route 2: up and right
routes["2"] = {"Route": route_2, "End Goal": end_goal_2, "Start State": start_state_2, "Turn": turn_2}

#Route 3: straight right
routes["3"] = {"Route": route_3, "End Goal": end_goal_2, "Start State": start_state_3}

#Route 3: straight right later later starting position
routes["4"] = {"Route": route_4, "End Goal": end_goal_2, "Start State": start_state_4}

allRoutes = route_1 + route_2 + route_3
allGoals = end_goal_1 + end_goal_2


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
SIZE = 24
num_episodes = args.episodes
num_test = args.testepisodes


#Constants
discount = 0.95
max_epsilon = 1.0
min_epsilon = 0.05
target_epsilon = 0.051
decay_rate = -math.log((target_epsilon - min_epsilon) / (max_epsilon - min_epsilon)) / num_episodes

#Global Variables
t = 0           #T used to measure the number of ticks in a single episode
lr = 0.2
n_agents = 2

def main():
  
    global env
    global t
    global lr
    global n_agents
    epsilon = 1

    collisions = 0

    #Environment object that is updated and rendered
    env = FlatGridWorld(size=SIZE, agents=[])
    
    #List containing all agent objects
    # Timid agent: lamda = 2.5, gamma_gain = 0.61, gamma_loss = 0.69, beta, alpha = 0.88
    # Expectation agent: lamda = 1, gamma_gain = 1, gamma_loss = 1, beta, alpha = 1
    # Purely Altruistic: phi = pi/2
    # Mid Altruistic: phi = pi/4
    # Purely Egoistic: phi = 0
    global agents

    agents = [Agent(agent_n = 1, route = routes['2'], phi = 0, lamda = 1, gamma_gain = 1, gamma_loss = 1, alpha = 1, beta = 1, env=env),
              #Agent(agent_n = 2, route = routes['3'], phi = 0, lamda = 1, gamma_gain = 0.69, gamma_loss = 0.69, alpha = 0.88, beta = 0.88, env=env)
              Agent(agent_n = 2, route = routes['4'], phi = 0, lamda = 1, gamma_gain = 1, gamma_loss = 1, alpha = 1, beta = 1, env=env)
              ]
    
    env.agents = agents
    n_agents = len(agents)

    #Running windows for policy stabiliy and q-deltas
    entropy_window = np.zeros((n_agents, 100))
    qdelta_window = np.zeros((n_agents, 100))
    reward_window = np.zeros((n_agents, 100))

    #Initializing global state
    env.rebuildGlobalState()

    #Running windows for policy stabiliy and q-deltas
    entropy_window = np.zeros((n_agents, 100))
    qdelta_window = np.zeros((n_agents, 100))
    reward_window = np.zeros((n_agents, 100))

    tot_reward = np.zeros(n_agents)
    window_index = 0
    
    prev_rewards = [None] * n_agents
    prev_entropy = [None] * n_agents
    prev_qdelta  = [None] * n_agents

    for i in tqdm(range(num_episodes + num_test)):
        tot_reward[:] = 0
        for agent in agents:
            agent.reset()                               #Reset agent states
        
        env.rebuildGlobalState()

        entropy_ep = np.zeros(n_agents)
        qdelta_ep = np.zeros(n_agents)
        counts = np.zeros(n_agents)
        if i >= num_episodes:
            if i == num_episodes:
                tqdm.write(f"Agents collided {collisions} times in {i} episodes.")            
                tqdm.write("Training complete. Starting testing...")
                collisions = 0
            lr = 0
            epsilon = 0
        
 
        while True:
            for agent in agents:
                
                if ((agent.state[0], agent.state[1]) in agent.route["End Goal"]):
                    continue

                action = agent.getAction(env.global_state, epsilon)       #Get an action for agent i
                if action is None:
                    continue

                delta = agent.updateQ(env.global_state, action)   # <-- ΔQ from updateQ
                qdelta_ep[agent.agent_n - 1] += delta
                counts[agent.agent_n - 1] += 1

                s_prime = env.updateWorld(agent, action)

                predicted_global_state = [a.state for a in sorted(env.agents, key=lambda ag: ag.agent_n)]

                tot_reward[agent.agent_n - 1] += rewardFunction(agent, s_prime, action, predicted_global_state, log=True)

                entropy_ep[agent.agent_n - 1] += policy_entropy(agent, env.global_state, epsilon)

            env.rebuildGlobalState()          

            if i > num_episodes:
                env.render()                                #Render in visualization
                #Show the visualization
                plt.ion()                                   #Activate interactive mode
                plt.show()                                  #Show visualization
                plt.pause(0.2)                           #Pause between episodes in seconds


            all_finished = all((agent.state[0], agent.state[1]) in agent.route["End Goal"] for agent in agents)      

            if hasCollided(env.global_state):
                collisions += 1
                t = 0
                break

            if all_finished:
                t = 0
                break
                
        if i < num_episodes:
            epsilon = min_epsilon + (max_epsilon - min_epsilon) * math.exp(-decay_rate * i) #Update epsilon according to decay rate

        for idx in range(n_agents):
            if counts[idx] > 0:
                avg_entropy = entropy_ep[idx] / counts[idx]
                avg_qdelta = qdelta_ep[idx] / counts[idx]
            else:
                avg_entropy, avg_qdelta = 0, 0

            entropy_window[idx, window_index] = avg_entropy
            qdelta_window[idx, window_index] = avg_qdelta

        reward_window[:, window_index] = tot_reward
        window_index = (window_index + 1) % 100
        if ((i + 1) % 100) == 0:
            avg_rewards = reward_window.mean(axis=1)
            avg_entropy = entropy_window.mean(axis=1)
            avg_qdelta = qdelta_window.mean(axis=1)
            tqdm.write(f"Episode {i + 1}:")

            """
            for agent in agents:
                for coord in agent.route["Route"]:
                    if coord in stop_region:
                        tqdm.write(f"Agent {agent.agent_n} stopped at stop sign {stop_counter[agent.agent_n - 1]}% of the time")
            """

            stop_counter = [0] * n_agents

            for idx in range(n_agents):
                arrow_r = trend_arrow(avg_rewards[idx], prev_rewards[idx], higher_is_better=True)
                arrow_e = trend_arrow(avg_entropy[idx], prev_entropy[idx], higher_is_better=False)  # usually lower entropy = more confident
                arrow_q = trend_arrow(avg_qdelta[idx], prev_qdelta[idx], higher_is_better=False)   # smaller ΔQ means more stable

                tqdm.write(
                    f"Agent {idx+1} | "
                    f"reward={avg_rewards[idx]:.2f}{arrow_r}, "
                    f"entropy={avg_entropy[idx]:.3f}{arrow_e}, "
                    f"ΔQ={avg_qdelta[idx]:.4f}{arrow_q}"
                )
                prev_rewards[idx] = avg_rewards[idx]
                prev_entropy[idx] = avg_entropy[idx]
                prev_qdelta[idx]  = avg_qdelta[idx]

    print(f"Agents collided {collisions} times in {num_test} episodes.")
    

def trend_arrow(current, previous, higher_is_better=True):
    if previous is None:
        return ""  # no arrow for first measurement
    if current > previous:
        return Fore.GREEN + "↑" + Style.RESET_ALL if higher_is_better else Fore.RED + "↑" + Style.RESET_ALL
    elif current < previous:
        return Fore.RED + "↓" + Style.RESET_ALL if higher_is_better else Fore.GREEN + "↓" + Style.RESET_ALL
    else:
        return Fore.YELLOW + "→" + Style.RESET_ALL

def policy_entropy(agent, global_state, epsilon):
    q_values = [agent.getQValue(global_state, a) for a in getLegalActions(agent.state, agent.route)]
    if not q_values:
        return 0.0

    n_actions = len(q_values)
    best = max(q_values)
    probs = []
    for q in q_values:
        if q == best:
            probs.append((1 - epsilon) + epsilon / n_actions)
        else:
            probs.append(epsilon / n_actions)

    return -sum(p * math.log(p + 1e-12) for p in probs)


"""
Goal(state):

This function returns one if the state passed is in the end_goal and zero if not
"""
def Goal(state, route):
    if ((state[0],state[1]) in route["End Goal"]):
        return 1
    else:
        return 0


def hasCollided(global_state):
    positions = [(state[0], state[1]) for state in global_state]

    for pos in positions[:]:
        for rid in routes:
            if pos in routes[rid]["End Goal"] and pos in positions:
                positions.remove(pos)
    
    for state in global_state:

        for r, route in routes.items():
            if (state[0], state[1]) in route["Route"]:
                idx = route["Route"].index((state[0], state[1]))
                if idx + 1 < len(route["Route"]):
                    if (state[2] == 2 and any((route["Route"][idx + 1][0], route["Route"][idx + 1][1], s) in global_state for s in (0,1))):
                        return 1
            
    if len(positions) != len(set(positions)):
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
            for s in range(max(0, speed - 1), 3):  
                next_idx = idx + speed
                if next_idx < len(route_list):
                    next_r, next_c = route_list[next_idx]
                    valid.append((next_r, next_c, s))

    if "Turn" in route and (state[0], state[1]) in route["Turn"]:
        valid = [v for v in valid if v[2] != 2]

    # Filter valid states: must be on the route
    route_coords = set(route_list)
    valid = [i for i in valid if (i[0], i[1]) in route_coords]

    return valid


"""
getLegalActions(state)

This function takes a state as a tuple and returns a list of tuples which represents all actions that agent may take at that state including movement 
and acceleration. 
"""
def getLegalActions(state, route):
    x, y, s = state
    if "Turn" in route and (x, y) in route["Turn"]:
        if s == 0:
            return [0, 1]
        if s == 1:
            return [-1, 0]
        if s == 2:
            return [-1]

    if s == 0:
        return [0, 1]
    
    elif s == 1:
        return [-1, 0, 1]

    elif s == 2:
       return [-1, 0]
    

def notMoving(state, action):
    if state[2] == 0 and action == 0:
        return 1
    else:
        return 0

def stopArea(state, action):
    x, y, s = state
    if (x, y) in stop_region:
        if s == 0 and action == 1:
            return 2
        elif s == 0 and action == 0: 
            return 0.2
        else:
            return 0
    else:
        return 0

def slowArea(state, action):
    x, y, s = state
    if (x, y) in slow_region:
        if s == 1 and action == -1:
            return 1
        else:
            return -1
    else:
        return 0
    
def proximityCheck(agent, state, global_state):
    route = agent.route["Route"]
    x, y, _ = state
    for idx, entry in enumerate(route):
        if entry == (x, y):
            adj_one = route[idx + 1] if idx + 1 < len(route) else None
            adj_two = route[idx + 2] if idx + 2 < len(route) else None
            break

    found = False
    penalty = 0
    for s in global_state:
        x, y, _ = s
        if not found and s == state:
            found = True
            continue
        if (x, y) == adj_one:
            penalty = max(penalty, 1)
        elif (x, y) == adj_two:
            penalty = max(penalty, 0.25)
    return penalty 

def bubbleCheck(agent, state, global_state):
    if Goal(state, agent.route):
        return 0
    x, y, _ = state

    bubble_1 = [(x+i, y+j, s) for i in range(-1, 2) for j in range(-1, 2) for s in (0, 1, 2)]

    bubble_2 = [(x+i, y+j, s) for i in range(-2, 3) for j in range(-2, 3) for s in (0, 1, 2)]    

    penalty = 0
    for idx, other_state in enumerate(global_state):
        if idx == agent.agent_n - 1:
            continue
        if other_state in bubble_2 and not Goal(other_state, agent.route):
            if other_state in bubble_1:
                penalty += 1
            else:
                penalty += 0.5
    return penalty

def collisionCheck(agent, global_state):
    agent_state = agent.state
    agent_pos = (agent_state[0], agent_state[1])
    agent_speed = agent_state[2]
    route = agent.route["Route"]

    if agent_pos in agent.route["End Goal"]:
        return 0
    
    other_states = [s for s in global_state if s != agent_state]

    if agent_pos in [(s[0], s[1]) for s in other_states]:
        return 1

    if agent_pos in route:
        idx = route.index(agent_pos)
        if idx + 1 < len(route):
            next_pos = route[idx + 1]
            for s in other_states:
                if (s[0], s[1]) == next_pos and agent_speed == 2 and s[2] in (0, 1):
                    return 1
                
    return 0


"""
rewardFunction(state, action)

This function takes the state and action of an agent and returns the reward produced by the environment. In our case, things like being on a goal square, 
hitting an obstacle, and being on the route are important to the reward function. 
"""


def rewardFunction(agent, state, action, global_state, log = False):
    global t
    route = agent.route
    
    const1 = 30     # Reward for reaching the goal
    const2 = 25     # Penalty for colliding with another agent
    const3 = 0.05   # Penalty per move
    const4 = 2      # Penalty for tailing another agent
    const5 = 0.5    # Penalty for being within 2 squares of another agent
    const6 = 0     # Reward for stopping at the stop sign
    const7 = 0      # Reward for slowing before the stop sign 
    const8 = 0.5    # Penalty for not moving


    goal_reward = const1 * Goal(state, route)
    collision_penalty = const2 * collisionCheck(agent, global_state)
    move_penalty = const3 * t
    tailing_penalty = const4 * proximityCheck(agent, state, global_state)
    bubble_penalty = const5 * bubbleCheck(agent, state, global_state)
    stop_reward = const6 * stopArea(state, action)
    slow_reward = const7 * slowArea(state, action)
    not_moving_penalty = const8 * notMoving(state, action)

    total_reward = (goal_reward - collision_penalty - move_penalty - tailing_penalty - bubble_penalty + stop_reward + slow_reward - not_moving_penalty)

    if log:
        """
        print(f"Agent {agent.agent_n} | State: {state} | Action: {action} \nGlobal State: {global_state} \n Rewards: {{\n"
              f"  Goal: {goal_reward},\n"
              f"  Collision: -{collision_penalty},\n"
              f"  Move: -{move_penalty},\n"
              f"  Tailing: -{tailing_penalty},\n"
              f"  Bubble: -{bubble_penalty},\n"
              f"  Stop: {stop_reward},\n"
              f"  Slow: {slow_reward},\n"
              f"  Not Moving: -{not_moving_penalty}\n"
              f"}} | Total Reward: {total_reward} \n")
        """
        if collision_penalty != 0:
            print(f"collision penalty: {collision_penalty}")
        if stop_reward != 0:
            print(f"stop reward: {stop_reward}")
        if slow_reward != 0:
            print(f"slow reward: {slow_reward}")
    return total_reward


# Precompute for speed
route_idx = {rid: {pos: i for i, pos in enumerate(info["Route"])}
             for rid, info in routes.items()}

tp = {rid: {} for rid in routes}
for rid, info in routes.items():
    rlist = info["Route"]
    idx_of = route_idx[rid]
    for (x, y) in rlist:
        i = idx_of[(x, y)]
        for s in (0, 1, 2):
            tp[rid].setdefault((x, y, s), {})
            candidates = list(neighboringStates((x, y, s), info))
            for action in getLegalActions((x, y, s), info):
                tp[rid][(x, y, s)].setdefault(action, {})
                if not candidates:
                    continue
                if action == 0: 

                    p_intended = 0.99
                else:
                    p_intended = 0.90
                
                n_cand = len(candidates)

                p_other = (1.0 - p_intended) / max(1, n_cand - 1)
                    
                for (nx, ny, ns) in candidates:

                    # Intended position is progress along the route by 's'
                    intended_pos_ok = (i + s < len(rlist) and (nx, ny) == rlist[i + s])
                    # Intended speed is s + action
                    intended_speed_ok = (ns == s + action)
                    is_intended = intended_pos_ok and intended_speed_ok
                    
                    tp[rid][(x, y, s)][action][(nx, ny, ns)] = p_intended if is_intended else p_other

"""
FlatGridWorld

The FlatGridWorld class defines the environment in which the agent learns. It contains methods such as render and update world. It mainly contains 
logic for displaying the visualization. 
"""

class FlatGridWorld:
    def __init__(self, size, agents):
        self.size = SIZE  # grid is size x size
        self.n_squares = size * size
        self.agents = agents
        self.global_state = []
        for agent in agents:
            self.global_state.append(agent.state)

    """
    render()

    The render method sets the cmap for agents, obstacles, and road to be displayed in a matplotlib figure. 
    """
    def render(self):
        global n_agents
        plt.clf()

        grid = np.zeros((self.size, self.size))

        # Fill in obstacle, agent, start, goal
        for coord in totObs:
            grid[coord] = -1

        for agent in self.agents:
            for coord in agent.route["End Goal"]:
                grid[coord] = 0.8

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
        if chosen_action in getLegalActions(agent.state, agent.route):
            next_state = random.choices(list(tp[route_num][agent.state][chosen_action].keys()), weights=list(tp[route_num][agent.state][chosen_action].values()), k=1)[0]
            agent.state = next_state
        global t
        t += 1
        return next_state

    def rebuildGlobalState(self):
        self.global_state = [agent.state for agent in sorted(self.agents, key = lambda ag: ag.agent_n)]

"""
Agent

The agent class holds the logic for q-learning, action fetching, and other methods like sampling which are important to the agent. 
"""
class Agent:
    def __init__(self, agent_n, route, phi, lamda, gamma_gain, gamma_loss, alpha, beta, env):
        self.agent_n = agent_n
        self.route = route
        self.phi = phi
        self.lamda = lamda
        self.gamma_gain = gamma_gain
        self.gamma_loss = gamma_loss
        self.alpha = alpha
        self.beta = beta
        self.env = env

        self.qtable = {}

        global n_agents

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
        """
        state_key = tuple(global_state)

        if state_key not in self.qtable:
            return  0.0
        
        return self.qtable[state_key].get(action, 0.0)
    
    """
    getAction(epsilon)

    The getAction method retrieves a chosen action based on the probability of exploring vs exploiting.
    """
    def getAction(self, global_state, epsilon):

        explore = random.choices([True, False], weights=[epsilon, (1 - epsilon)], k=1)[0]
        if explore:
            legal_actions = getLegalActions(self.state, self.route)
            if not legal_actions:
                return None
            return random.choice(legal_actions)
        else:
            action = self.getPolicy(global_state)
            #legal_actions = getLegalActions(self.state)
            #print(f"[DEBUG] Agent {self.agent_n} | Current state: {self.state} | Legal actions: {legal_actions} | Chosen action: {action}")

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

        delta = new_q - current_q

        state_key = tuple(global_state)
        if state_key not in self.qtable:
            self.qtable[state_key] = {}
        self.qtable[state_key][action] = new_q

        return delta



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
        route_num = None
        for key, value in routes.items():
            if value is self.route:
                route_num = key
                break


        next_states = list(tp[route_num][self.state][action].keys())
        probs = list(tp[route_num][self.state][action].values())
        
        ordered_agents = sorted(self.env.agents, key=lambda a: a.agent_n)

        for _ in range(n_samples):
            # Sample the next state for the current agent
            states_by_id = {}
            actions_by_id = {}
            
            for ag in ordered_agents:
                if ag is self:
                    s_prime = random.choices(next_states, weights=probs, k=1)[0]
                    states_by_id[ag.agent_n] = s_prime
                else:
                    other_rid = None
                    for key, value in routes.items():
                        if value is ag.route:
                            other_rid = key
                            break
                    other_actions = getLegalActions(ag.state, ag.route)
                    if not other_actions:
                        other_s_prime = ag.state
                        other_action = None
                    else:
                        other_action = random.choice(other_actions)
                        if (ag.state not in tp[other_rid] or other_action not in tp[other_rid][ag.state] or not tp[other_rid][ag.state][other_action]):
                            other_s_prime = ag.state
                        else:
                            other_next_states = list(tp[other_rid][ag.state][other_action].keys())
                            other_probs = list(tp[other_rid][ag.state][other_action].values())                    
                            other_s_prime = random.choices(other_next_states, weights=other_probs, k=1)[0]
                    states_by_id[ag.agent_n] = other_s_prime
                    actions_by_id[ag.agent_n] = other_action

            predicted_global_state = [states_by_id[i] for i in range(1, len(ordered_agents) + 1)]

            self_reward = rewardFunction(self, predicted_global_state[self.agent_n - 1], action, predicted_global_state)

            other_rewards = 0.0
            for ag in ordered_agents:
                if ag is self:
                    continue
                r = rewardFunction(ag, predicted_global_state[ag.agent_n - 1], actions_by_id.get(ag.agent_n), predicted_global_state)
                other_rewards += r
            
            if len(self.env.agents) > 1:
                avg_other_reward = other_rewards / (len(self.env.agents) - 1)
            else:
                avg_other_reward = 0.0
            weighted_joint_reward = math.cos(self.phi) * self_reward + math.sin(self.phi) * avg_other_reward

            legal_actions = getLegalActions(s_prime, self.route)

            if not legal_actions:
                v_s_prime = 0.0
            else:
                v_s_prime = max(self.getQValue(predicted_global_state, a) for a in legal_actions)
            
            full_return = weighted_joint_reward + (discount * v_s_prime)
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
        legal_actions = getLegalActions(self.state, self.route)
        if not legal_actions:
            return None
        
        best_value = -float('inf')
        best_actions = []

        for action in legal_actions:
            value = self.getQValue(global_state, action)
            if value > best_value:
                best_value = value
                best_actions = [action]
            elif value == best_value:
                best_actions.append(action)

        return random.choice(best_actions)

main()