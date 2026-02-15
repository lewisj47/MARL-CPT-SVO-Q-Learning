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

end_goal_1 = []
end_goal_2 = []

#End goal at the top
end_goal_1.extend([(c, r) for c in range(12, 15) for r in range(22, 24)])

#End goal on the right
end_goal_2.extend([(c, r) for c in range(22, 24) for r in range(9, 12)])

#Routes are used in the reward function to reward the agent for making progress towards the goal
routes = {}

#From bottom to top
route_1 = [(13, r) for r in range(0, 24)]
lane_1 = [(c, r) for r in range(0, 24) for c in range(12, 15)]

#From bottom to right
route_2 = [(13, r) for r in range (0, 11)]
lane_2 = [(c, r) for r in range(0, 12) for c in range(12, 15)]
route_2.extend([(c, 10) for c in range(13, 24)])
lane_2.extend([(c, r) for c in range(13, 24) for r in range(9, 12)])

#From left to right
route_3 = [(c, 10) for c in range(0, 24)]
lane_3 = [(c, r) for c in range(0, 24) for r in range(9, 12)]

#Populating route dictionary
routes["1"] = {"Route": route_1, "Lane": lane_1, "End Goal": end_goal_1}
routes["2"] = {"Route": route_2, "Lane": lane_2, "End Goal": end_goal_2}
routes["3"] = {"Route": route_3, "Lane": lane_3, "End Goal": end_goal_2}

#Stop sign regions
stop_region = [(c, 8) for c in range(12, 15)]
slow_region = [(c,7) for c in range(12, 15)]

#An empty array to hold the coordinates of the obstacles. In our case, obstacles are spaces where the road is not. 
totObs = []

#Bottom left obstacle
Obs1 = [(c, r) for c in range(0, 9) for r in range(0, 9)]
totObs.extend(Obs1)
#Bottom right obstacle
Obs2 = [(c, r) for c in range(15, 24) for r in range(0, 9)]
totObs.extend(Obs2)
#Top right Obstacle
Obs3 = [(c, r) for c in range(15, 24) for r in range(15, 24)]
totObs.extend(Obs3)
#Top left obstacle
Obs4 = [(c, r) for c in range (0, 9) for r in range(15, 24)]
totObs.extend(Obs4)

#Environment Definitions

#Number of agents
n_agents = 2
#Size of grid
SIZE = 24
#Number of episodes passed as an argument in the command line
num_episodes = args.episodes

#All squares in the grid world
all_sqrs = [(r,c) for r in range(SIZE) for c in range(SIZE)]
#Corner sqrs in the grid world
corner_sqrs = [(0,0),(0,SIZE), (SIZE, 0), (SIZE, SIZE)]


speed_set = [0, 1, 2]
#All directions the agent can be driving in
dir_set = [1, 2, 3, 4] # 1:right, 2:up, 3:left, 4:down



#Constants
alpha0 = 0.2
alpha_pow = 0.5
discount = 0.98
max_epsilon = 1.0
min_epsilon = 0.05
target_epsilon = 0.051
decay_rate = -math.log((target_epsilon - min_epsilon) / (max_epsilon - min_epsilon)) / num_episodes

#T used to measure the number of ticks in a single episode
t = 0



def main():

    global t

    epsilon = 1

    #Number of episodes where the agents make it to the finish line
    finish_n = 0

    #Number of collisions that occur
    collision_n = 0

    # Environment object that is updated and rendered
    env = FlatGridWorld(size=SIZE, agents=[], obstacles=(Obs1, Obs2, Obs3, Obs4))

    # List containing all agent objects
    # Timid agent: lamda = 2.5, gamma_gain = 0.61, gamma_loss = 0.69, beta = 0.88, alpha = 0.88
    agents = [
        Agent(agent_n=1, route=routes['1'], start=(13, 0, 1, 2), state=(13, 0, 1, 2), phi=0, lamda=1, gamma_gain=1, gamma_loss=1, alpha=1, beta=1, env=env)
        , Agent(agent_n=2, route=routes['3'], start=(0, 10, 1, 1), state=(0, 10, 1, 1), phi=0, lamda=1, gamma_gain=1, gamma_loss=1, alpha=1, beta=1, env=env)
    ]

    # Add agents to the environment
    env.agents = agents
    
    # Initialize the global state as a list of all agents' states
    env.rebuildGlobalState()

    tot_reward = np.zeros(n_agents)  
    reward_window = np.zeros((n_agents, 50)) 
    window_index = 0

    for j in tqdm(range(num_episodes)):
        for agent in agents:
            agent.reset()  # Reset agent states
            tot_reward[agent.agent_n - 1] = 0  # Reset total cumulative reward per episode

        while True:
            # check if need lines for state_before
            env.rebuildGlobalState()
            state_before = env.global_state.copy()

            for agent in agents:
                if (agent.state[0], agent.state[1]) in agent.route["End Goal"] or (agent.state[0], agent.state[1]) in totObs:
                    continue
                action = agent.getAction(state_before, epsilon)  # Get an action for the agent
                if action is None:
                    continue
                agent.updateQ(state_before, action)  # Update q-value for the agent having taken action at state

                tot_reward[agent.agent_n - 1] += rewardFunction(agent.state, agent.route, action, env)

                env.updateWorld(agent, action)  # Update agent positions and speeds

            env.rebuildGlobalState()

            # Check for collisions
            if hasCollided(env.global_state):
                print("Collision detected!")
                collision_n += 1
                break  # Exit the loop if a collision occurs

            env.render()  # Render in visualization

            # Show the visualization
            plt.ion()  # Activate interactive mode
            plt.show()  # Show visualization
            plt.pause(0.0001)  # Pause between episodes in seconds

            # Check for termination conditions
            all_finished = all((agent.state[0], agent.state[1]) in agent.route["End Goal"] for agent in agents)
            obs_hit = any((agent.state[0], agent.state[1]) in totObs for agent in agents)
            all_stuck_or_finished = all(len(neighbor_cache[agent.state]) == 0 or (agent.state[0], agent.state[1]) in agent.route["End Goal"] for agent in agents)

            if all_finished or obs_hit or all_stuck_or_finished:
                t = 0
                if all_finished:
                    finish_n += 1
                    print(f"All agents reached the goal in episode {j + 1}!")
                break  # Break out of the loop if any termination condition is met
        reward_window[:, window_index] = tot_reward  # Store the total reward for each agent in the reward window
        window_index = (window_index + 1) % 50  # Update the window index to cycle through the last 50 episodes

        epsilon = min_epsilon + (max_epsilon - min_epsilon) * math.exp(-decay_rate * j) #Update epsilon according to decay rate
        if (j % 50) == 0:
            avg_rewards = reward_window.mean(axis=1)
            print(f"Episode {j + 1}/{num_episodes} completed. Epsilon: {epsilon:.4f}")
            for idx, avg in enumerate(avg_rewards, start=1):
                print(f" Agent {idx} average reward (last 50 episodes): {avg:.2f}")

    print(f"Agents reached the goal {finish_n} times, {(finish_n / args.episodes) * 100}% of all episodes.")
    print(f"Collisions occurred {collision_n} times, {(collision_n / args.episodes) * 100}% of all episodes.")

    for agent in agents:
        with open(f"qtable_output{agent.agent_n}.txt", "w") as f:
            pprint.pprint(agent.qtable, stream=f)


def hasCollided(global_state):
    """
    Checks if any two agents occupy the same position.
    """
    positions = [(state[0], state[1]) for state in global_state]  # Extract (x, y) positions

    
    return len(positions) != len(set(positions))  # Check for duplicates

"""
Goal(state):

This function returns one if the state passed is in the end_goal and zero if not
"""
def Goal(state, route):
    """
    This function returns 1 if the state passed is in the agent's end goal and 0 otherwise.
    """
    if (state[0], state[1]) in route["End Goal"]:
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
        for s in (0, 1):
            for d in dir_set:
                valid.extend([(state[0], state[1], s, d)])
    elif (state[2] == 1):
        for s in (0, 1, 2):
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
    elif state[2] == 2:
        for s in (1, 2):
            if state[3] == 1:
                valid.extend([(state[0] + 2, state[1], s, 1)])
            if state[3] == 2:
                valid.extend([(state[0], state[1] + 2, s, 2)])
            if state[3] == 3:
                valid.extend([(state[0] - 2, state[1], s, 3)])
            if state[3] == 4:
                valid.extend([(state[0], state[1] - 2, s, 4)])
    
    valid = [i for i in valid if 0 <= i[0] < SIZE and 0 <= i[1] < SIZE]
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
        return ((1 + route["Route"].index((state[0],state[1])))/len(route["Route"]))
    elif (state[0], state[1]) in route["Lane"]:
        return ((1 + route["Lane"].index((state[0],state[1])))/(5 * len(route["Route"])))
    else:
        return 0


def notMoving(state, action):
    if state[2] == 0 and action[2] == 0:
        return 1
    else:
        return 0

def stopArea(state, action):
    if (state[0], state[1]) in stop_region:
        if state[2] == 0 and action[2] == 1:
            return 1
        elif state[2] == 0 and action[2] == 0:
            return 0
        else:
            return -5
    else:
        return 0

def slowArea(state, action):
    if (state[0], state[1]) in slow_region:
        if state[2] == 1 and action[2] == -1:
            return 1
        else:
            return 0
    else:
        return 0
    
def collisionCheck(global_state):
    if hasCollided(global_state):
        return 1
    else:
        return 0

"""
rewardFunction(state, action)

This function takes the state and action of an agent and returns the reward produced by the environment. In our case, things like being on a goal square, 
hitting an obstacle, and being on the route are important to the reward function. 
"""
def rewardFunction(state, route, action, env, global_state=None):
    """
    This function takes the state, route, and action, and returns the reward produced by the environment.
    """
    const1 = 1000   # Reward for reaching the goal
    const2 = 100    # Penalty for hitting an obstacle
    const3 = 250    # Reward for being on the route
    const4 = 5      # Penalty for accelerating or decelerating
    const5 = 10     # Penalty for not moving
    const6 = 100    # Penalty for collision

    if global_state is None:
        global_state = env.global_state

    raw_reward = (
        const1 * Goal(state, route) -
        const2 * Obs(state) +
        const3 * onRoute(state, route) -
        const4 * abs(action[2]) -
        const5 * notMoving(state, action) -
        const6 * collisionCheck(global_state)
    )
    return raw_reward / 1000.0
  

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
    if s == 0:
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
        print(f"Warning: No correct neighbor found for state {(c, r, s, d)} with action {a}. Using uniform distribution.")

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
        self.global_state = [agent.state for agent in agents]

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

            for i in agent.route["Route"]:
                grid[i] = 0.2

        speed_color_map = {
            0: 'green',
            1: 'yellow',
            2: 'red'
        }

        for agent in self.agents:
            grid[(agent.state[0], agent.state[1])] = 1.0

            dx, dy = dirToAction(agent.state[3])
            arrow_color = speed_color_map.get(agent.state[2], 'black')

            plt.arrow(agent.state[0], agent.state[1], dx * agent.state[2], dy * agent.state[2],
                      head_width=0.5, head_length=0.5, fc=arrow_color, ec=arrow_color)

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
        if chosen_action in legal_actions_cache[agent.state]:
            next_state = random.choices(list(tp[agent.state][chosen_action].keys()), weights=list(tp[agent.state][chosen_action].values()), k=1)[0]
            agent.state = next_state
        global t
        t += 1

    def rebuildGlobalState(self):
        """
        Rebuilds the global state in a fixed order.
        """
        self.global_state = [agent.state for agent in sorted(self.agents, key=lambda ag: ag.agent_n)]
"""
Agent

The agent class holds the logic for q-learning, action fetching, and other methods like sampling which are important to the agent. 
"""
class Agent:
    def __init__(self, agent_n, route, start, state, phi, lamda, gamma_gain, gamma_loss, alpha, beta, env):
        self.agent_n = agent_n
        self.route = route
        self.start = start
        self.state = state
        self.phi = phi
        self.lamda = lamda
        self.gamma_gain = gamma_gain
        self.gamma_loss = gamma_loss
        self.alpha = alpha
        self.beta = beta
        self.env = env

        self.visits = {} # dict[(global_state, action)] -> int
        self.qtable = {}
        # Update Q-table to use global state and individual action as the key
        """
        self.qtable = {}
        for global_state in product(
            range(SIZE), repeat=2 * len(env.agents)
        ):  # Global state includes all agents' positions
            for action in legal_actions_cache[self.state]:
                self.qtable[global_state] = {action: 0 for action in legal_actions_cache[self.state]}
        """
        self.reset()

    def reset(self):
        self.state = self.start
        return self.state

    def get_alpha(self, global_state, action):
        state_key = tuple(global_state)
        k = (state_key, action)
        n_visits = self.visits.get(k, 0)
        alpha = alpha0 / ((n_visits + 1) ** alpha_pow)

        return alpha

    def getQValue(self, global_state, action):
        """
        Returns Q(global_state, action)
        """
        # Convert to tuple-of-tuples for dictionary key
        state_key = tuple(global_state)
    
        if state_key not in self.qtable:
            return 0.0  # default Q-value for unseen state-action
    
        return self.qtable[state_key].get(action, 0.0)

    def getAction(self, global_state, epsilon):
        """
        Select an action for the agent based on the global state.
        """
        explore = random.choices([True, False], weights=[epsilon, (1 - epsilon)], k=1)[0]
        if explore:
            legal_actions = legal_actions_cache.get(self.state, [])
            if not legal_actions:
                return None
            return random.choice(legal_actions)
        else:
            action = self.getPolicy(global_state)
            return action

    """
    updateQ(action)

    updateQ performs the cpt-based q-value updating required for the agent to learn. It contains sampling, the rho-cpt function, and updating
    the q-value based on the learning rate before it updates the q-table. 
    """
    def updateQ(self, global_state, action):
        state_key = tuple(global_state)
        k = (state_key, action)

        alpha = self.get_alpha(global_state, action)
        self.visits[k] = self.visits.get(k, 0) + 1

        samples = self.sample_outcomes(action)
        target = self.rho_cpt(samples)
        current_q = self.getQValue(global_state, action)
        new_q = ((1 - alpha) * current_q) + (alpha * target)

        if state_key not in self.qtable:
            self.qtable[state_key] = {}
        self.qtable[state_key][action] = new_q 



    def sample_outcomes(self, action, n_samples=50):
        """
        Using the current state, passed action, and dictionary of transition probabilities,
        compiles a list of samples for future Q-values to be modified using CPT and then used
        in the updateQ function. This version also predicts other agents' next states.
        """
        samples = []
        next_states = list(tp[self.state][action].keys())
        probs = list(tp[self.state][action].values())

        for _ in range(n_samples):
            # Sample the next state for the current agent
            s_prime = random.choices(next_states, weights=probs, k=1)[0]

            # Predict next states for other agents
            predicted_global_state = []
            for other_agent in self.env.agents:
                if other_agent == self:
                    predicted_global_state.append(s_prime)
                else:
                    other_actions = legal_actions_cache.get(other_agent.state, [])
                    if not other_actions:
                        predicted_global_state.append(other_agent.state)
                        continue
                    weights = [sum(tp[other_agent.state][a].values()) for a in other_actions]
                    if sum(weights) == 0:
                        other_action = random.choice(other_actions)  # fallback
                    else:
                        other_action = random.choices(other_actions, weights=weights, k=1)[0]

                    if other_agent.state not in tp or other_action not in tp[other_agent.state]:
                        other_s_prime = other_agent.state  # No transition available, stay in current state
                    else:
                        other_next_states = list(tp[other_agent.state][other_action].keys())
                        other_probs = list(tp[other_agent.state][other_action].values())
                        other_s_prime = random.choices(other_next_states, weights=other_probs, k=1)[0]
                    predicted_global_state.append(other_s_prime)

            # Calculate the reward for the current agent based on the predicted global state
            reward = rewardFunction(s_prime, self.route, action, self.env, global_state=predicted_global_state)

            # Get the maximum Q-value for the predicted global state
            legal_actions = legal_actions_cache.get(s_prime, [])
            if not legal_actions:
                v_s_prime = 0.0
            else:
                v_s_prime = max(self.getQValue(predicted_global_state, a) for a in legal_actions)

            # Compute the full return
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
            rho_plus += max(0, X_sort[ii])**self.alpha * (
                z_1**g_g / (z_1**g_g + (1 - z_1)**g_g)**(1 / g_g)
                - z_2**g_g / (z_2**g_g + (1 - z_2)**g_g)**(1 / g_g)
            )
            rho_minus += (self.lamda * max(0, -X_sort[ii])**self.beta) * (
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
        legal_actions = legal_actions_cache.get(self.state, [])

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

