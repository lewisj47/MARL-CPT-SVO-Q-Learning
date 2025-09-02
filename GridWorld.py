import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np
import argparse
import random
import math
from tqdm import tqdm
from colorama import Fore, Style

# Arguments passed from command terminal
parser = argparse.ArgumentParser()
parser.add_argument("episodes", type=int, help="The number of episodes to undergo during training")
parser.add_argument("testepisodes", type=int, help="The number of episodes to undergo during testing")
parser.add_argument("scenario", type = str, help="A string which is the name of the scenario which is being run")
args = parser.parse_args()

# Start States
start_state_1 = (13, 3, 1)
start_state_2 = (13, 7, 0)
start_state_3 = (3, 10, 2)
start_state_4 = (5, 10, 2)

# End Goals
end_goal_1 = []
end_goal_2 = []
end_goal_1.extend([(c, r) for r in range(22, 24) for c in range(12, 15)])
end_goal_2.extend([(c, r) for r in range(9, 12) for c in range(22, 24)])

# Routes
route_1 = [(13, r) for r in range(3, 24)]
route_2 = [(13, r) for r in range (0, 11)]
route_2.extend([(c, 10) for c in range(14, 24)])

turn_2 = [(13, 7), (13, 8), (13, 9), (13, 10), (14, 10)]


route_3 = [(c, 10) for c in range(0, 24)]

route_4 = [(c, 10) for c in range(4, 24)]

# Dictionary indexed by route number containing the route, end goal, and start state
routes = {}

# Route 1: straight up
routes["1"] = {"Route": route_1, "End Goal": end_goal_1, "Start State": start_state_1}

# Route 2: up and right
routes["2"] = {"Route": route_2, "End Goal": end_goal_2, "Start State": start_state_2, "Turn": turn_2}

# Route 3: straight right
routes["3"] = {"Route": route_3, "End Goal": end_goal_2, "Start State": start_state_3}

# Route 4: straight right later later starting position
routes["4"] = {"Route": route_4, "End Goal": end_goal_2, "Start State": start_state_4}

allRoutes = route_1 + route_2 + route_3
allGoals = end_goal_1 + end_goal_2

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
decay_rate = -math.log((target_epsilon - min_epsilon) / (max_epsilon - min_epsilon)) / num_episodes # Exponential decay constant for epsilon

#Global Variables
t = 0       # t used to measure the number of ticks in a single episode
lr = 0.2    # learning rate (alpha)


def main() -> None:
    """Initializes agents and environment and then runs a series of training episodes where agents develop 
    a Q-learning policy to reach their goal state quickly while avoiding collisions. Finally, the program 
    runs and visualizes a series of testing episodes where agents operate using their developed policy.

    Episode structure:
        1. Initialize agents.
        2. Given current state, each agent chooses an action based on an epsilon-greedy policy.
        3. Each agent updates its Q-table for the global state and chosen action.
        4. The environment and true global state are updated. 
        5. Repeat steps 2-4 until agents reach terminal state (collision or all in goals).
        6. Repeat steps 1-5 for each episode.
    
    Args:
        None

    Returns:
        None
    """
  
    # Global variables
    global env
    global t
    global lr
    global n_agents

    # Local Variables
    epsilon = 1
    collisions = 0

    # Environment object that is updated and rendered
    env = FlatGridWorld(size=SIZE, agents=[])
    global agents

    # Agent profile characteristics
    # Timid agent: lamda = 2.5, gamma_gain = 0.61, gamma_loss = 0.69, beta, alpha = 0.88
    # Expectation agent: lamda = 1, gamma_gain = 1, gamma_loss = 1, beta, alpha = 1
    # Purely Altruistic: phi = pi/2
    # Mid Altruistic: phi = pi/4
    # Purely Egoistic: phi = 0

    global agents

    agents = scenario_init(args.scenario)
    
    env.agents = agents
    n_agents = len(agents)


    #Initializing global state
    env.global_state = [agent.state for agent in sorted(agents, key = lambda ag: ag.agent_n)]

    #Running windows for policy stabiliy and q-deltas
    entropy_window = np.zeros((n_agents, 100))
    qdelta_window = np.zeros((n_agents, 100))
    reward_window = np.zeros((n_agents, 100))
    other_window = np.zeros((n_agents, 100))

    avg_rewards = [0] * n_agents
    avg_entropy = [0] * n_agents 
    avg_qdelta = [0] * n_agents
    avg_other = [0] * n_agents
    svo_reward = [0] * n_agents

    window_index = 0
    
    prev_rewards = [0] * n_agents
    prev_entropy = [0] * n_agents
    prev_qdelta  = [0] * n_agents
    prev_svo_reward = [0] * n_agents

    for i in tqdm(range(num_episodes + num_test), mininterval = 5.0):
        for agent in agents:
            agent.reset()                               #Reset agent states
        
        entropy_ep = np.zeros(n_agents)
        qdelta_ep = np.zeros(n_agents)
        reward_ep = np.zeros(n_agents)

        counts = np.zeros(n_agents)
        if i >= num_episodes:
            if i == num_episodes:
                tqdm.write(f"Agents collided {collisions} times in {i} episodes.")            
                tqdm.write("Training complete. Starting testing...")
                collisions = 0
            lr = 0
            epsilon = 0
        
 
        while True:
            actions = {}
            for agent in agents:
                
                if ((agent.state[0], agent.state[1]) in agent.route["End Goal"]):
                    continue

                action = agent.getAction(env.global_state, epsilon)       #Get an action for agent i
                if action is None:
                    continue
                
                actions[agent] = action

            for agent, action in actions.items():
                delta = agent.updateQ(env.global_state, action)   # <-- ΔQ from updateQ
                qdelta_ep[agent.agent_n - 1] += delta

            next_states = env.updateWorld(actions)

            for agent, action in actions.items():
                s_prime = next_states[agent]
                predicted_global_state = env.global_state
                if i <= num_episodes:
                    reward_ep[agent.agent_n - 1] += rewardFunction(agent, s_prime, action, predicted_global_state)
                else:
                    reward_ep[agent.agent_n - 1] += rewardFunction(agent, s_prime, action, predicted_global_state, log=True)
                    tqdm.write(f"SVO Reward for Agent {agent.agent_n}: {svo_reward[agent.agent_n - 1]} \n")
                entropy_ep[agent.agent_n - 1] += policy_entropy(agent, env.global_state, epsilon)

            counts[agent.agent_n - 1] += 1

            if i > num_episodes:
                env.render()                #Render in visualization

                #Show the visualization
                plt.ion()                   #Activate interactive mode
                plt.show()                  #Show visualization
                plt.pause(0.2)              #Pause between episodes in seconds

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
                avg_entropy[idx] = entropy_ep[idx] / counts[idx]
                avg_qdelta[idx] = qdelta_ep[idx] / counts[idx]
                avg_rewards[idx] = reward_ep[idx] / counts[idx]

        for idx in range(n_agents):
            entropy_window[idx, window_index] = avg_entropy[idx]
            qdelta_window[idx, window_index] = avg_qdelta[idx]
            reward_window[idx, window_index] = avg_rewards[idx]
            other_window[idx, window_index] = avg_other[idx]

        window_index = (window_index + 1) % 100

        if ((i + 1) % 100) == 0 or i > num_episodes:
            avg_rewards = reward_window.mean(axis=1)
            avg_entropy = entropy_window.mean(axis=1)
            avg_qdelta = qdelta_window.mean(axis=1)

            if n_agents > 1:
                total_avg = sum(avg_rewards)
                avg_other = (total_avg - avg_rewards) / (n_agents - 1)
            else:
                avg_other = np.zeros(n_agents)

            for idx in range(n_agents):
                svo_reward[idx] = (avg_rewards[idx] * math.cos(agents[idx].phi) + avg_other[idx] * math.sin(agents[idx].phi))

            tqdm.write(f"Episode {i + 1}:")

            for idx in range(n_agents):
                arrow_r = trend_arrow(avg_rewards[idx], prev_rewards[idx], higher_is_better=True)
                arrow_e = trend_arrow(avg_entropy[idx], prev_entropy[idx], higher_is_better=False)  # usually lower entropy = more confident
                arrow_q = trend_arrow(avg_qdelta[idx], prev_qdelta[idx], higher_is_better=False)   # smaller ΔQ means more stable
                arrow_svo = trend_arrow(svo_reward[idx], prev_svo_reward[idx], higher_is_better=True)

                tqdm.write(
                    f"Agent {idx+1} | "
                    f"reward={avg_rewards[idx]:.2f}{arrow_r}, "
                    f"entropy={avg_entropy[idx]:.3f}{arrow_e}, "
                    f"|ΔQ|={avg_qdelta[idx]:.4f}{arrow_q}, \n "
                    f"Other Reward: {avg_other[idx]:.2f} | "
                    f"Total Weighted Reward: {avg_rewards[idx]:.2f} * cos({(agents[idx].phi):.2f}) + {avg_other[idx]:.2f} * sin({(agents[idx].phi):.2f}) = {(avg_rewards[idx] * math.cos(agents[idx].phi) + avg_other[idx] * math.sin(agents[idx].phi)):.2f}{arrow_svo}"
                )
                prev_rewards[idx] = avg_rewards[idx]
                prev_entropy[idx] = avg_entropy[idx]
                prev_qdelta[idx]  = avg_qdelta[idx]
                prev_svo_reward[idx] = svo_reward[idx]

    print(f"Agents collided {collisions} times in {num_test} episodes.")

    for agent in agents:
        tqdm.write(f"Agent {agent.agent_n}: \n Phi: {agent.phi} \n Lambda: {agent.lamda} \n Gamma+ : {agent.gamma_gain} \n Gamma- : {agent.gamma_loss} ")
    

def scenario_init(scenario):
    if scenario == "2_agent_right_turn":
        return([Agent(agent_n = 1, route = routes['2'], phi = 0, lamda = 1, gamma_gain = 1, gamma_loss = 1, alpha = 1, beta = 1, env=env),
                Agent(agent_n = 2, route = routes['4'], phi = math.pi / 3, lamda = 1, gamma_gain = 1, gamma_loss = 1, alpha = 1, beta = 1, env=env)
                ])
    if scenario == "3_agent_right_turn":
        return([Agent(agent_n = 1, route = routes['2'], phi = 0, lamda = 1, gamma_gain = 1, gamma_loss = 1, alpha = 0.88, beta = 0.88, env=env),
                Agent(agent_n = 2, route = routes['4'], phi = math.pi / 3, lamda = 1, gamma_gain = 1, gamma_loss = 1, alpha = 0.88, beta = 0.88, env=env),
                Agent(agent_n = 3, route = routes['3'], phi = 0, lamda = 1, gamma_gain = 1, gamma_loss = 1, alpha = 0.88, beta = 0.88, env=env)
                ])

def trend_arrow(current, previous, higher_is_better=True):
    """Uses the colorama library to create an arrow with direction and color determined by the relationship between two passed values.

    Args: 
        current (float): New value to compare with the old.
        previous (float): Old value.
        higher_is_better (bool): Used to map the arrow color to the sign of the change in value.
            Defaults to False.
        
    Returns:
        str: String containing ANSI escape codes that appears in the terminal as a colored arrow.
    """
    
    if previous is None:
        return ""  # no arrow for first measurement
    if current > previous:
        return Fore.GREEN + "↑" + Style.RESET_ALL if higher_is_better else Fore.RED + "↑" + Style.RESET_ALL
    elif current < previous:
        return Fore.RED + "↓" + Style.RESET_ALL if higher_is_better else Fore.GREEN + "↓" + Style.RESET_ALL
    else:
        return Fore.YELLOW + "→" + Style.RESET_ALL


def policy_entropy(agent, global_state, epsilon) -> float:
    """Compute the entropy of the agent's epsilon-greedy policy given its Q-values at the current state.

    Args:
        agent (class): Specific to the agent.
        global_state (list of tuples): The state (xpos, ypos, speed) of each agent.
        epsilon (float): The current probability weight for exploration.

    Returns:
        float: The entropy of the policy. Should get smaller per episode.
    """
    
    # Collect Q-values for all legal actions in this state.
    q_values = [agent.getQValue(global_state, a) for a in getLegalActions(agent.state, agent.route)]
    if not q_values:
        return 0.0

    n_actions = len(q_values)
    best = max(q_values)
    probs = []

    # Construct action probabilities under epsilon-greedy policy
    for q in q_values:
        if q == best:
            # Best action gets - epsilon plus its share of exploration mass
          # Entropy = -∑ p log p (small offset avoids log(0))
            probs.append((1 - epsilon) + epsilon / n_actions)
        else:
            # Non-greedy actions get only the exploration mass
            probs.append(epsilon / n_actions)

    # Entropy = -∑ p log p (small offset avoids log(0))
    return -sum(p * math.log(p + 1e-12) for p in probs)


def Goal(state, route) -> int:
    """Checks if the agent is in its route's goal state.

    Args:
        state (tuple): State (xpos, ypos, speed) of the agent.
        route (dictionary of a list of tuples): Coordinates of agent start state, route, and end state. 

    Returns:
        int: 1 if agent in goal, 0 otherwise.
    """

    if ((state[0],state[1]) in route["End Goal"]):
        return 1
    else:
        return 0


def hasCollided(global_state) -> bool:
    """Checks if there has been any collision between agents.

    Args:
        global_state (list of tuples): The state (xpos, ypos, speed) of each agent.

    Returns:
        bool: True if a collision has occurred, False otherwise
    """

    # Do not check for collision in goal states
    goal_cells = set().union(*[set(info["End Goal"]) for info in routes.values()])
    positions = [(state[0], state[1]) for state in global_state if (state[0], state[1]) not in goal_cells]
    
    # Check edge case for when one agent 'jumps' over another without ever occupying the same space
    for state in global_state:
        for _ , route in routes.items():
            if (state[0], state[1]) in route["Route"]:
                idx = route["Route"].index((state[0], state[1]))
                if idx + 1 < len(route["Route"]):
                    if (state[2] == 2 and any((route["Route"][idx + 1][0], route["Route"][idx + 1][1], s) in global_state for s in (0, 1))):
                        return True
    
    # Collision has occurred if more than one agent have the same position
    if len(positions) != len(set(positions)):
        return True
    else:
        return False


def neighboringStates(state, route) -> list:
    """Returns all possible next states for an agent given their current state and route.

    Args:
        state (tuple): State (xpos, ypos, speed) of the agent.
        route (dictionary of a list of tuples): Coordinates of agent start state, route, and end state. 
    
    Returns:
        valid (list): List of all possible next states.
    """

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
            step = min(speed, len(route_list) - 1 - idx)
            for next_s in range(max(0, speed - 1), 3):  
                next_idx = idx + step
                if next_idx < len(route_list):
                    next_r, next_c = route_list[next_idx]
                    valid.append((next_r, next_c, next_s))

    # If state is within a turn within its route, restrict next speed to 0 or 1
    if "Turn" in route and (state[0], state[1]) in route["Turn"]:
        valid = [v for v in valid if v[2] != 2]

    # Filter valid states: must be on the route
    route_coords = set(route_list)
    valid = [i for i in valid if (i[0], i[1]) in route_coords]

    return valid


def getLegalActions(state, route) -> list:
    """Returns a list of all legal actions depending on a state and a route.

    Args:
        state (tuple): State (xpos, ypos, speed) of the agent.
        route (dictionary of a list of tuples): Coordinates of agent start state, route, and end state. 

    Returns:
        list: The set of all legal actions for the state. Can be slow down (-1), speed up (1), or stay the same speed (0).
    """
    
    x, y, s = state
    # If the state exists within a turn on the passed route, don't allow the agent to speed up
    if "Turn" in route and (x, y) in route["Turn"]:
        if s == 0:
            return [0, 1]
        if s == 1:
            return [-1, 0]
        if s == 2:
            return [-1]

    if s == 0:
        return [0, 1]   # Cannot slow down if stopped
    
    elif s == 1:
        return [-1, 0, 1]

    elif s == 2:
       return [-1, 0]   # Cannot speed up if going max speed
    

def notMoving(state, action) -> int:
    """Checks if the agent is not moving (has speed 0 and chooses action 0).

    Args:
        state (tuple): State (xpos, ypos, speed) of the agent.
        action (int): The action the agent has chosen to take (-1, 0, 1).
    
    Returns:
        int: 1 if the agent is not moving, 0 otherwise.
    """

    if state[2] == 0 and action == 0:
        return 1
    else:
        return 0

    
def proximityCheck(agent, state, global_state):
    """Checks if the agent is directly behind another agent (tailgating behavior).
    
    Args:
        agent (class): Specific to the agent.
        state (tuple): State (xpos, ypos, speed) of that agent.
        global_state (list of tuples): the state of each agent.
    
    Returns:
        penalty (float): 1.0 for within 1 square, 0.25 for within 2 squares.
        
    """
    
    # Ensure no penalty if agent already in goal
    if Goal(state, agent.route):
        return 0.0
    
    route = agent.route["Route"]
    x, y, _ = state
    for idx, entry in enumerate(route):
        if entry == (x, y):
            # Define next two coord pairs on agent's route
            adj_one = route[idx + 1] if idx + 1 < len(route) else None
            adj_two = route[idx + 2] if idx + 2 < len(route) else None
            break

    penalty = 0.0

    # Check if another agent is right in front of the agent
    for idx, s in enumerate(global_state):
        if idx == agent.agent_n - 1:
            continue    # Skip self
        ox, oy, _ = s
        other_agent = agent.env.agents[idx]
        if Goal(s, other_agent.route):
            continue    # Don't count as tailing if other agent is in their goal
        if (ox, oy) == adj_one:
            penalty = max(penalty, 1.0)
        elif (ox, oy) == adj_two:
            penalty = max(penalty, 0.25)

    return penalty 


def bubbleCheck(agent, state, global_state) -> float:
    """Checks if any other agents are within 1 or 2 squares in any direction.

    Args:
        agent (class): Specific to the agent.
        state (tuple): State (xpos, ypos, speed) of that agent.
        global_state (list of tuples): the state of each agent.
    
    Returns:
        penalty (float): 1.0 for within 1 square, 0.5 for within 2 squares.
    
    """
    
    # Ensure no penalty if agent already in goal
    if Goal(state, agent.route):
        return 0.0
    
    x, y, _ = state

    # Capture all states within 1 or 2 xpos or ypos steps from the agent
    bubble_1 = [(x+i, y+j, s) for i in range(-1, 2) for j in range(-1, 2) for s in (0, 1, 2)]
    bubble_2 = [(x+i, y+j, s) for i in range(-2, 3) for j in range(-2, 3) for s in (0, 1, 2)]    

    penalty = 0.0

    # Increase penalty for each agent in bubble with more severity for being within the smaller bubble
    for idx, other_state in enumerate(global_state):
        if idx == agent.agent_n - 1:
            continue    # Skip self
        other_agent = agent.env.agents[idx]
        if Goal(other_state, other_agent.route):
            continue    # Don't count agent as within bubble if they are in their goal
        if other_state in bubble_2:
            if other_state in bubble_1:
                penalty += 1.0
            else:
                penalty += 0.5

    return penalty

  
def collisionCheck(agent, state, global_state) -> int:
    """Checks if the agent has collided with another agent.
    
    Args:
        agent (class): Specific to the agent.
        state (tuple): State (xpos, ypos, speed) of that agent.
        global_state (list of tuples): the state of each agent.
    
    Returns:
        int: 1 if collision has occurred, 0 otherwise.
    """


    x, y, sp = state
    route = agent.route["Route"]

    # Ensure multiple agents in same end goal aren't considered to have collided
    if (x, y) in agent.route["End Goal"]:
        return 0
    
    idx_self = agent.agent_n - 1
    other_states = [s for i, s in enumerate(global_state) if i != idx_self]

    # Check if occupies same state as another agent
    if (x, y) in [(s[0], s[1]) for s in other_states]:
        return 1

    # Ensure that a jump over another agent moving at a slower speed counts as collision
    if (x, y) in route:
        i = route.index((x, y))
        if i + 1 < len(route):
            next_pos = route[i + 1]
            for s in other_states:
                if (s[0], s[1]) == next_pos and sp == 2 and s[2] in (0, 1):
                    return 1      
                  
    # Ensures that when an agent is jumped over, it too receives the collision deduction
    for _, r in routes.items():
        other_route = r["Route"]
        for os in other_states:
            if (os[0], os[1]) in other_route:
                j = other_route.index((os[0], os[1]))
                if j + 1 < len(other_route):
                    next_pos = other_route[j + 1]
                    if ((x, y) == next_pos and os[2] == 2 and sp in (0, 1)):
                        return 1
    return 0


def rewardFunction(agent, state, action, global_state, log = False) -> float:
    """Returns the total reward given by the environment according to an agent's route, state, action, and the global state. 

    Args:
        agent (class): Specific to the agent.
        state (tuple): State (xpos, ypos, speed) of that agent.
        action (int): The action the agent has chosen to take.
        global_state (list of tuples): The state of each agent.
        log (bool): True for print reward information. 
            Defaults to False.
    
    Returns:
        total_reward (float): The summation of all penalties and rewards associated with the agent's action and the global state.
    """
    
    global t
    route = agent.route
    
    # Reward weights
    const1 = 40     # Reward for reaching the goal
    const2 = 100     # Penalty for colliding with another agent
    const3 = 0.25    # Penalty per move
    const4 = 5      # Penalty for tailing another agent
    const5 = 0.5    # Penalty for being within 2 squares of another agent 
    const6 = 2    # Penalty for not moving

    # Reward weighting
    goal_reward = const1 * Goal(state, route)
    collision_penalty = const2 * collisionCheck(agent, state, global_state)
    move_penalty = round(const3 * t, 2)
    tailing_penalty = const4 * proximityCheck(agent, state, global_state)
    bubble_penalty = const5 * bubbleCheck(agent, state, global_state)
    not_moving_penalty = const6 * notMoving(state, action)

    total_reward = (goal_reward - collision_penalty - move_penalty - tailing_penalty - bubble_penalty - not_moving_penalty)


    if log:  
        tqdm.write(f"Agent {agent.agent_n} | State: {state} | Action: {action} \nGlobal State: {global_state} \n Rewards: {{\n"
              f"  Goal: {goal_reward},\n"
              f"  Collision: -{collision_penalty},\n"
              f"  Move: -{move_penalty},\n"
              f"  Tailing: -{tailing_penalty},\n"
              f"  Bubble: -{bubble_penalty},\n"
              f"  Not Moving: -{not_moving_penalty}\n"
              f"}} | Total Reward: {total_reward} \n")
        
    return total_reward


# Map each route ID to a lookup table that assigns an index to every (x, y) position
# Example: route_idx[rid][(x, y)] = position index along that route
route_idx = {rid: {pos: i for i, pos in enumerate(info["Route"])}
             for rid, info in routes.items()}

# tp = transition probabilities
# Structure: tp[route_id][(x, y, s)][action][(nx, ny, ns)] = probability
# Meaning: given current state (position, speed) and an action, what's the probability
#          of transitioning to the next state (nx, ny, ns).tp = {rid: {} for rid in routes}
tp = {rid: {} for rid in routes}

# For each route, loop through all coord pairs in that route and valid speeds for those coord pairs
for rid, info in routes.items():
    rlist = info["Route"]
    idx_of = route_idx[rid]
    for (x, y) in rlist:
        i = idx_of[(x, y)]
        for s in (0, 1, 2):
            tp[rid].setdefault((x, y, s), {})
            candidates = list(neighboringStates((x, y, s), info))

            # Loop through all legal actions for a given next state and assign probabilities
            for action in getLegalActions((x, y, s), info):
                tp[rid][(x, y, s)].setdefault(action, {})
                if not candidates:  # Only add to tp if a valid next state exists
                    continue
                
                # Action = 0 (keep speed) is more reliable than speed changes
                if action == 0:
                    p_intended = 0.99
                else:
                    p_intended = 0.90
                
                n_cand = len(candidates)

                # Distribute leftover probability across unintended outcomes
                p_other = (1.0 - p_intended) / max(1, n_cand - 1)

                # How far an agent can move forward given speed s so that it stays on route
                step = min(s, len(rlist) - 1 - i)

                for (nx, ny, ns) in candidates:

                    # Intended position is progress along the route by 'step'
                    intended_pos_ok = ((nx, ny) == rlist[i + step])
                    
                    # Intended next speed is s + action
                    intended_speed_ok = (ns == s + action)

                    # If both next position and next speed align, assign intended probability
                    is_intended = intended_pos_ok and intended_speed_ok
                    tp[rid][(x, y, s)][action][(nx, ny, ns)] = p_intended if is_intended else p_other


class FlatGridWorld:

    def __init__(self, size, agents) -> None:
        """Initializes the world size, agents, and global state.

        Args:
            self: The environment.
            size (int): The length of one side of the world.
            agents (list of classes): Each agent class.

        Returns:
            None
        
        """

        self.size = size  # grid is size x size
        self.agents = agents
        self.global_state = [agent.state for agent in sorted(agents, key = lambda ag: ag.agent_n)]


    def render(self) -> None:
        """Sets the cmap for agents, obstacles, and road to be displayed in a matplotlib figure.

        Args:
            self: The environment.

        Returns:
            None
        """

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
                for _ , agent in enumerate(self.agents):
                    x, y = agent.state[0], agent.state[1]
                    plt.text(x, y, agent.agent_n,   # agent index as the number
                            ha='center', va='center',
                            fontsize=8, color='white')
            else:
                grid[(self.agents[i].state[0], self.agents[i].state[1])] = 0.2

        # Display the number of ticks occurring in an episode
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
        plt.axhline(y=8, xmin=0.5, xmax=0.65, color='white', linestyle='-')

        ax = plt.gca()
        ax.invert_yaxis()
        plt.grid(True)


    def updateWorld(self, actions) -> dict:
        """Takes an ordered dictionary of actions and updates each agent state accordingly using transition probabilities.

        Args:
            self: The environment.
            actions (dict of ints): Actions to be attempted by each agent.

        Returns: 
            new_states (dict of tuples): Dictionary pairing each agent to their new state. 
        """

        new_states = {}

        # find next state for each agent according to their action
        for agent, action in actions.items():

            # Derive agent route id
            route_num = None
            for key, value in routes.items():
                if value is agent.route:
                    route_num = key
                    break

            # Create list of possible next states and associated transition probabilities
            next_states = list(tp[route_num][agent.state][action].keys())
            probs = list(tp[route_num][agent.state][action].values())
            
            if not next_states:
                # If no next state exists, agent state stays static
                s_prime = agent.state
            else:
                # Otherwise, choose next state according to tp
                s_prime = random.choices(next_states, weights=probs, k=1)[0]
            
            new_states[agent] = s_prime
        
        for agent, s_prime in new_states.items():
            agent.state = s_prime

        # Update the global state with the true next states
        self.global_state = [agent.state for agent in sorted(self.agents, key = lambda ag: ag.agent_n)]
        
        global t
        t += 1
        return new_states



class Agent:

    def __init__(self, agent_n, route, phi, lamda, gamma_gain, gamma_loss, alpha, beta, env) -> None:
        """Initialize agent parameters, Q-table, and state.

        Args:
            self: Specific to the agent.
            agent_n (int): Agent index.
            route (dictionary of a list of tuples): Coordinates of agent start state, route, and end state. 
            phi (float): SVO angular alignment.
            lamda (float): Loss aversion coefficient (CPT).
            gamma_gain (float): Decision weight for gains (CPT).
            gamma_loss (float): Decision weight for losses (CPT).
            alpha (float): Sensitivity to gains (CPT).
            beta (float): Sensitivity to losses (CPT).
            env (class): The environment (contains agents, obstacles, etc.).

        Returns:
            None
        
        """

        self.agent_n = agent_n
        self.route = route
        self.phi = phi
        self.lamda = lamda
        self.gamma_gain = gamma_gain
        self.gamma_loss = gamma_loss
        self.alpha = alpha
        self.beta = beta
        self.env = env

        # Lazy Q-table approach (populated as new global states appear)
        self.qtable = {}

        global n_agents

        self.reset()


    def reset(self) -> tuple:
        """Sets agent state to its start state as defined in its route.
        
        Args:
            self: Specific to the agent.

        Returns:
            tuple: The agent's state (reset to its start state).
        """

        self.state = self.route["Start State"]
        return self.state
    

    def getQValue(self, global_state, action) -> float:
        """Looks up the Q-value for a given global state and action.

        Args:
            self: Spcific to the agent.
            global_state (list of tuples): The state (xpos, ypos, speed) of each agent.
            action (int): The action taken by the agent.

        Returns:
            float: the corresponding Q-value read from the Q-table.
        """

        # Convert global_state to tuple to match with Q-table entry data type
        state_key = tuple(global_state)

        if state_key not in self.qtable:
            return  0.0     # Default to 0 if global_state does not yet exist in Q-table
        
        return self.qtable[state_key].get(action, 0.0)
    

    def getAction(self, global_state, epsilon) -> int:
        """Returns an action to be taken by the agent at a global state that randomly explores or exploits based on epsilon.
        
        Args:
            self: Specific to the agent.
            global_state (list of tuples): The state (xpos, ypos, speed) of each agent.
            epsilon (float): An exponentially decreasing parameter that is used to weigh the likelihood of exploring vs exploiting.

        Returns:
            action (int): The suggested action to be taken.
        """

        # Explore with probability of epsilon
        explore = random.choices([True, False], weights=[epsilon, (1 - epsilon)], k=1)[0]
        if explore:
            # Choose random legal action
            legal_actions = getLegalActions(self.state, self.route)
            if not legal_actions:
                return None
            return random.choice(legal_actions)
        else:
            # Follow policy based on Q-values
            action = self.getPolicy(global_state)

            return action


    def updateQ(self, global_state, action) -> float:
        """Uses the Bellman equation to update the Q-value for a given global state and action.

        Args:
            self: Specific to agent.
            global_state (list of tuples): The state (xpos, ypos, speed) of each agent.
            action (int): The action (-1, 0, 1) chosen by the agent.

        Returns:
            delta (float): Absolute value of the change between the previous Q-value and new Q-value.
        """

        # Create a list of potential total returns for the taken action and predicted next global states
        samples = self.sample_outcomes(action)

        # Average and distort the list of sampled returns using CPT (return = reward + discounted lookahead value)
        target = self.rho_cpt(samples)

        current_q = self.getQValue(global_state, action)

        # Blend predicted return (target) with current Q-value to update the Q-value for the global state and action
        new_q = ((1 - lr) * current_q) + (lr * target)
        delta = abs(new_q - current_q)
        state_key = tuple(global_state)
        if state_key not in self.qtable:
            self.qtable[state_key] = {}
        self.qtable[state_key][action] = new_q

        # Return the absolute value of the difference between old and new Q-values to be used to observe agent learning trends
        return delta


    def sample_outcomes(self, action, n_samples=50) -> list:
        """Compiles a list of returns (reward + discounted lookahead value) across sampled state/action pairs.

        Args: 
            self: Specific to agent.
            action (int): The action taken by the agent. 
            n_samples (int): Number of samples to be taken (defaults to 50).

        Returns:
            samples (list of floats): List of returns for a state/action pair according to transition probabilty.
        """

        samples = []

        # Derive agent route id
        route_num = None
        for key, value in routes.items():
            if value is self.route:
                route_num = key
                break

        # Create list of possible next states and associated transition probabilities
        next_states = list(tp[route_num][self.state][action].keys())
        probs = list(tp[route_num][self.state][action].values())
        
        # Sort agents by agent_n to ensure ordering consistency
        ordered_agents = sorted(self.env.agents, key=lambda a: a.agent_n)

        # Take samples of possible next global states and collect discounted rewards in list
        for _ in range(n_samples):
            states_by_id = {}   # Ordered dict of all next states
            actions_by_id = {}  # Ordered dict of all next actions
            
            # Loop through all agents and fill dicts with possible next states and actions
            for ag in ordered_agents:
                # Choose possible next state for self
                if ag is self:
                    s_prime = random.choices(next_states, weights=probs, k=1)[0]
                    states_by_id[ag.agent_n] = s_prime
                
                # For all other agents, use a legal action to determine a possible next state and add both to the dicts
                else:
                    # Derive other agent route id
                    other_rid = None
                    for key, value in routes.items():
                        if value is ag.route:
                            other_rid = key
                            break
                    other_actions = getLegalActions(ag.state, ag.route)
                    if not other_actions:
                        other_s_prime = ag.state    # If no legal action available, agent state stays static
                        other_action = None
                    else:
                        other_action = random.choice(other_actions)
                        if (ag.state not in tp[other_rid] or other_action not in tp[other_rid][ag.state] or not tp[other_rid][ag.state][other_action]):
                            other_s_prime = ag.state    # If state/action pair not present in tp, agent state stays static
                        else:
                            other_next_states = list(tp[other_rid][ag.state][other_action].keys())
                            other_probs = list(tp[other_rid][ag.state][other_action].values())                    
                            other_s_prime = random.choices(other_next_states, weights=other_probs, k=1)[0]
                    states_by_id[ag.agent_n] = other_s_prime
                    actions_by_id[ag.agent_n] = other_action

            # Ordered list of all predicted next states
            predicted_global_state = [states_by_id[i] for i in range(1, len(ordered_agents) + 1)]

            # Compute own reward
            self_reward = rewardFunction(self, predicted_global_state[self.agent_n - 1], action, predicted_global_state)

            other_rewards = 0.0
            # Compute total rewards for all other agents
            for ag in ordered_agents:
                if ag is self:
                    continue    # Only look at other agents
                r = rewardFunction(ag, predicted_global_state[ag.agent_n - 1], actions_by_id.get(ag.agent_n), predicted_global_state)
                other_rewards += r
            
            # Compute average other agent reward
            if len(self.env.agents) > 1:
                avg_other_reward = other_rewards / (len(self.env.agents) - 1)
            else:
                avg_other_reward = 0.0
            
            # Compute total weighted utility using SVO
            weighted_joint_reward = math.cos(self.phi) * self_reward + math.sin(self.phi) * avg_other_reward

            legal_actions = getLegalActions(s_prime, self.route)

            # Calculate the best Q-value for the predicted global state across all legal actions (lookahead value)
            if not legal_actions:
                v_s_prime = 0.0
            else:
                v_s_prime = max(self.getQValue(predicted_global_state, a) for a in legal_actions)
            
            full_return = weighted_joint_reward + (discount * v_s_prime) # Multiply v_s_prime by discount factor to balance immediate reward with long-term planning
            samples.append(full_return)
        
        return samples


    def rho_cpt(self, samples) -> float:
        """Uses CPT to transform a list of sampled rewards to a single estimated reward for a state/action pair.
            CPT distorts low and high probabilities and scales the overall reward based on loss sensitivity.
            CPT formula from Tversky and Kahneman 1992 paper
            
            Args:
                self: Specific to agent.
                samples (list of floats): List of possible rewards for a given state/action pair.
 
            Returns:
                rho (float): The CPT-distorted average reward.
        """

        # Convert samples to a numpy array and sort them in ascending order
        X = np.array(samples)
        X_sort = np.sort(X, axis = None)
        N_max = len(X_sort)

        rho_plus = 0
        rho_minus = 0

        # Decision weights for gains (gamma+) and losses  (gamma-)
        g_g = self.gamma_gain
        g_l = self.gamma_loss 

        # Iterate through ordered outcomes to compute decision weights
        for ii in range(1, N_max):
            z_1 = (N_max - ii + 1) / N_max  # upper trail prob (gain side)
            z_2 = (N_max - ii) / N_max      # lower trail prob (gain side)
            z_3 = ii / N_max                # upper trail prob (loss side)
            z_4 = (ii-1) / N_max            # lower trail prob (loss side)

            # Contribution of positive outcomes (gains)
            rho_plus = rho_plus + max(0, X_sort[ii])**self.alpha * (
                z_1**g_g / (z_1**g_g + (1 - z_1)**g_g)**(1 / g_g) 
                - z_2**g_g / (z_2**g_g + (1 - z_2)**g_g)**(1 / g_g)
            )

            # contribution of negative outcomes (losses), weighted by lamda (loss aversion)
            rho_minus = rho_minus + (self.lamda * max(0, -X_sort[ii])**self.beta) * (
                z_3**g_l / (z_3**g_l + (1 - z_3)**g_l)**(1 / g_l) 
                - z_4**g_l / (z_4**g_l + (1 - z_4)**g_l)**(1 / g_l)
            )

        # Overall prospect value = weighted gains - weighted losses
        rho = rho_plus - rho_minus

        return rho
    

    def getPolicy(self, global_state) -> int:
        """Uses an agent's Q-table to find its best action for a given global state.

        Args:
            self: Specific to agent.
            global_state (list of tuples): The state (xpos, ypos, speed) of each agent.

        Returns:
            int: The best action for the agent according to its Q-table given the global state.
        """

        legal_actions = getLegalActions(self.state, self.route)
        if not legal_actions:
            return None # if no legal action exists, return None
        
        best_value = -float('inf')
        best_actions = []

        # Find the action that returns the best Q-value
        for action in legal_actions:
            value = self.getQValue(global_state, action)
            if value > best_value:
                best_value = value
                best_actions = [action]
            elif value == best_value:
                best_actions.append(action)

        return random.choice(best_actions) # random choice in case of Q-value tie


main()
