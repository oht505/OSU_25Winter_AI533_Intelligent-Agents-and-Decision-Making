# Mini Project 2
# Name: Hyuntaek Oh
# Email: ohhyun@oregonstate.edu

import numpy as np
import matplotlib
matplotlib.use('TkAgg',force=True)
from matplotlib import pyplot as plt
import random

np.set_printoptions(precision=2, suppress=True)

#######################################################################################
# 0. Set up the environment                                                           #
#######################################################################################
# Environmental parameters
ENV_ROWS, ENV_COLS = 4, 4
ACTIONS = [0, 1, 2, 3]  # left: 0, up: 1, right: 2, down: 3
ACTIONS_MAPPING = [(0, -1), (-1, 0), (0, 1), (1, 0)]
TEXT_ACTIONS = ['left', 'up', 'right', 'down']
SLIDES = {0: [(-1, 0), (1, 0)],
          1: [(0, -1), (0, 1)],
          2: [(-1, 0), (1, 0)],
          3: [(0, -1), (0, 1)]}
SLIDES_IDX = {0: [1, 3],
              1: [0, 2],
              2: [1, 3],
              3: [0, 2]}
START = (0, 0, 0, 0)
GOAL = (3, 3, 0, 0)
WATER = [(2, 1), (2, 2)]
FIRE = [(0, 1), (0, 2)]
PROB_INTENDED = 0.8
PROB_SLIDE = 0.1
Q_table = np.zeros((ENV_ROWS, ENV_COLS, len(ACTIONS)))

# Define feature vector dimension for linear approximator
feature_dim = 5

# Initialize w and theta arbitrarily for linear approximation
theta = np.random.uniform(-0.01, 0.01, (feature_dim, len(ACTIONS)))
w = np.random.uniform(-0.01, 0.01, feature_dim)

# Hyperparameters
GAMMA = 0.95
ALPHA = 0.01
EPSILON = 0.1
BETA = 0.01
LAMBDA = 0.8
MAX_EPISODES = 100
NUM_TRIALS = 100

# Rewards setting
REWARD = -np.ones((ENV_ROWS, ENV_COLS)) # all other states
REWARD[WATER[0]], REWARD[WATER[1]] = -5, -5     # Water
REWARD[FIRE[0]], REWARD[FIRE[1]] = -10, -10   # Fire
REWARD[GOAL[:2]] = 100  # Goal
#######################################################################################


def is_terminal_state(state):
    """
    If the agent arrives at GOAL, return True.
    if not, return False (meaning that it is not terminal state)

    :param state: current state <row, col, checkWater, checkFire>
    :return: True if it is in terminal state, False if not.
    """
    if state == GOAL:
        return True
    else:
        return False


def epsilon_greedy_action_selection(q_table, state, epsilon):
    """
    Select next action based on Epsilon-greedy policy

    :param q_table: Q-value table
    :param state: Current state and information about obstacles (x, y, check_water, check_fire)
    :param epsilon: Random factor for exploration
    :return: Epsilon greedy action based on the probability
    """
    curr_row, curr_col, check_water, check_fire = state
    curr_pos = (curr_row, curr_col)

    # The total number of actions |A|
    num_actions = len(ACTIONS)

    # Best action A*
    A_star = np.argmax(q_table[curr_pos])

    # Calculate the probability of the distribution
    action_probs = np.ones(num_actions) * (epsilon / num_actions)
    action_probs[A_star] += (1.0 - epsilon)

    # Select random action based on the probability
    return np.random.choice(ACTIONS, p=action_probs)


def get_next_state(state, action):
    """
    Obtain next possible state based on Transition probability

    :param state: Current state (x, y, check_water, check_fire)
    :param action: Current action [0:left, 1:up, 2:right, 3:down]
    :return: Next state and reward for one step
    """
    curr_row, curr_col, check_water, check_fire = state
    curr_pos = (curr_row, curr_col)

    # Transitional function
    possible_moves = [action, (action + 1) % 4, (action - 1) % 4]
    transition_probs = [PROB_INTENDED, PROB_SLIDE, PROB_SLIDE]

    # Actual action
    chosen_action = np.random.choice(possible_moves, p=transition_probs)
    dRow, dCol = ACTIONS_MAPPING[chosen_action]

    # Calculate next position
    new_row = min(max(curr_row + dRow, 0), ENV_ROWS - 1)
    new_col = min(max(curr_col + dCol, 0), ENV_COLS - 1)
    new_pos = (new_row, new_col)

    # Check and update obstacles
    check_water = 1 if new_pos in WATER else 0
    check_fire = 1 if new_pos in FIRE else 0

    # Reward based on the current position
    reward = REWARD[curr_pos]

    # Update new state
    next_state = (new_row, new_col, check_water, check_fire)

    return next_state, reward


def SARSA(q_table=Q_table, max_episodes=100, gamma=GAMMA, alpha=ALPHA, epsilon=EPSILON):
    """
    SARSA algorithm with ϵ-greedy policy

    :param q_table: Init Q-value table
    :param max_episodes: The maximum number of episodes
    :param gamma: Discount factor (0.95)
    :param alpha: Learning rate (0.4689)
    :param epsilon: Random factor for exploration (0.4589)

     step_reward: reward for each step
     episode_reward: reward for each episode
     total_rewards: rewards for whole episodes

    :return: optimal Q-value table, Total Rewards for all episodes
    """
    Q = np.copy(q_table)
    total_rewards = []

    for episode in range(max_episodes):
        # Initialize current state (0, 0, 0, 0)
        curr_state = START

        # Reward for each episode
        episode_reward = 0

        # Action select
        curr_action = epsilon_greedy_action_selection(Q, curr_state, epsilon)

        while True:
            # Get next state and reward for one step
            next_state, step_reward = get_next_state(curr_state, curr_action)

            # Get next action
            next_action = epsilon_greedy_action_selection(Q, next_state, epsilon)

            # Update Q-value
            Q[curr_state[:2]][curr_action] += alpha * (step_reward + gamma * Q[next_state[:2]][next_action]
                                                       - Q[curr_state[:2]][curr_action])

            # Terminate Process if terminal state
            if is_terminal_state(curr_state):
                episode_reward += REWARD[GOAL[:2]]
                break

            # Accumulate reward for current episode
            episode_reward += step_reward

            # Update state, action
            curr_state, curr_action = next_state, next_action

        # Book-keeping the rewards for each episode
        total_rewards.append(episode_reward)

    # Convert into numpy array
    total_rewards = np.array(total_rewards)

    return Q, total_rewards


def Q_Learning(q_table=Q_table, max_episodes=100, gamma=GAMMA, alpha=ALPHA, epsilon=EPSILON):
    """
    Q-learning algorithm with ϵ-greedy policy

    :param q_table: init Q-value table initialized to zeros
    :param max_episodes: The maximum number of episodes
    :param gamma: Discount factor (0.95)
    :param alpha: Learning rate (0.4908)
    :param epsilon: Random factor for exploration (0.7492)

     step_reward: reward for each step
     episode_reward: reward for each episode
     total_rewards: rewards for whole episodes

    :return: optimal Q-value table, Rewards for all episodes
    """

    # Copy init Q-value table
    Q = np.copy(q_table)

    # Rewards for entire episodes
    total_rewards = []

    for episode in range(max_episodes):
        # Initialize current state (0, 0, 0, 0)
        curr_state = START

        # Reward for each episode
        episode_reward = 0

        while True:
            # Choose action from current state
            curr_action = epsilon_greedy_action_selection(Q, curr_state, epsilon)

            # Get next state and reward for one step
            next_state, step_reward = get_next_state(curr_state, curr_action)

            # Find maximum value and index of state-action pair
            max_Q_value = max(Q[next_state[:2]])

            # Update Q-value
            Q[curr_state[:2]][curr_action] += alpha * (step_reward + gamma * max_Q_value
                                                       - Q[curr_state[:2]][curr_action])

            # Terminate Process if terminal state
            if is_terminal_state(curr_state):
                episode_reward += REWARD[GOAL[:2]]
                break

            # Accumulate reward for current episode
            episode_reward += step_reward

            # Update state
            curr_state = next_state

        # Book-keeping the rewards for each episode
        total_rewards.append(episode_reward)

    # Convert into numpy array
    total_rewards = np.array(total_rewards)

    return Q, total_rewards


def SARSA_lambda(q_table=Q_table, max_episodes=100, gamma=GAMMA, alpha=ALPHA, epsilon=EPSILON, lambda_val=LAMBDA):
    """
        SARSA(λ) algorithm with ϵ-greedy policy

        :param q_table: init Q-value table
        :param max_episodes: The maximum number of episodes
        :param gamma: Discount factor (0.95)
        :param alpha: Learning rate (0.4306)
        :param epsilon: Random factor for exploration (0.2573)
        :param lambda_val: Improving convergence factor (0.8)

         step_reward: reward for each step
         episode_reward: reward for each episode
         total_rewards: rewards for all episodes
         et: a list for eligibility trace

        :return: optimal Q-value table, Rewards for all episodes
        """
    # Copy init Q-value table
    Q = np.copy(q_table)

    # Rewards for entire episodes
    total_rewards = []

    for episode in range(max_episodes):
        # Initialize current state (0, 0, 0, 0)
        curr_state = START

        # Reward for each episode
        episode_reward = 0

        # Select action
        curr_action = epsilon_greedy_action_selection(Q, curr_state, EPSILON)

        # Eligibility trace for Q(s,a)
        et = np.zeros((ENV_ROWS, ENV_COLS, len(ACTIONS)))

        while True:
            # Get next state and reward for one step
            next_state, step_reward = get_next_state(curr_state, curr_action)

            # Get next action
            next_action = epsilon_greedy_action_selection(Q, next_state, epsilon)

            # Calculate TD
            temporal_diff = step_reward + gamma * Q[next_state[:2]][next_action] - Q[curr_state[:2]][curr_action]

            # Count Eligibility trace
            et[curr_state[:2]][curr_action] += 1

            # Update Q-value table with Eligibility trace
            # Update Eligibility trace with lambda value
            for row in range(ENV_ROWS):
                for col in range(ENV_COLS):
                    for action in ACTIONS:
                        Q[row][col][action] += alpha * temporal_diff * et[row][col][action]
                        et[row][col][action] = gamma * lambda_val * et[row][col][action]

            # Terminate Process if terminal state
            if is_terminal_state(curr_state):
                episode_reward += REWARD[GOAL[:2]]
                break

            # Update state, action
            curr_state, curr_action = next_state, next_action

            # Accumulate reward
            episode_reward += step_reward

        # Book-keeping the rewards for each episode
        total_rewards.append(episode_reward)

    # Convert into numpy array
    total_rewards = np.array(total_rewards)

    return Q, total_rewards


def manhattan_distance(pos1, pos2):
    """
    Manhattan distance between position 1 and position 2

    :param pos1: current position
    :param pos2: target position (Water, Fire, and Goal)
    :return: the value of the distance between two different positions
    """
    return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])


def state_to_feature_vector(state):
    """
    Convert the state into feature vector for functional approximation
    :param state: the given state (current or next)
    :return: (numpy array format) feature vector including 5 features
    """
    # Current position
    row, col = state[:2]

    # Manhattan Distance between obstacles (Water, Fire) and Goal
    d_water = min(manhattan_distance((row, col), water_pos) for water_pos in WATER)
    d_fire = min(manhattan_distance((row, col), fire_pos) for fire_pos in FIRE)
    d_goal = manhattan_distance((row, col), GOAL[:2])
    return np.array([row / (ENV_ROWS - 1), col / (ENV_COLS - 1), d_water, d_fire, d_goal])


def softmax_action_selection(state, softmax_theta=theta, tau=1.0):
    """
    Softmax action selection with tau

    :param state: current state
    :param softmax_theta: the given Actor parameter
    :param tau: control the distribution factor
    :return: action, probability of each action
    """
    # Calculate softmax probabilities
    state_feature = state_to_feature_vector(state)
    preference = state_feature @ softmax_theta
    scaled_preferences = preference / tau
    exp_preferences = np.exp(scaled_preferences - np.max(scaled_preferences))
    action_probs = exp_preferences / np.sum(exp_preferences)

    # Choose action
    action = np.random.choice(ACTIONS, p=action_probs)

    return action, action_probs


def policy_gradient(probs, action):
    """
    Policy gradient Calculation

    :param probs: current action probabilities
    :param action:  current action
    :return: policy gradient
    """
    # calculate gradient log
    grad = -probs
    grad[action] += 1
    return grad


def get_adaptive_lr(initial_lr, decay_rate, episode):
    """
    Adaptively manipulate learning rate by using episodic decaying

    :param initial_lr: current learning rate
    :param decay_rate: the speed of decaying
    :param episode: the number of episode
    :return: an adaptive learning rate based on episodic decaying
    """
    return max(initial_lr / (1 + decay_rate * episode), 1e-4)


def Actor_Critic(max_episodes=200, gamma=GAMMA, alpha=0.0548, beta=0.1643):
    """
    TD Actor-Critic algorithm

    :param max_episodes: the maximum number of episodes
    :param gamma: Discount factor (0.95)
    :param alpha: Learning rate for Actor (0.0548 = 0.06)
    :param beta: Learning rate for Critic (0.1643 = 0.17)

     tau: Softmax temperature to control preference
     v_s: value function of current feature vector
     v_s_next: value function of next feature vector
     w: Critic parameter
     theta: Actor parameter
     grad_log_policy: log gradient for policy gradient

    :return: weights, theta, rewards for all episodes
    """
    global w, theta

    # Rewards for max episodes
    total_rewards = []

    # Initialize alpha, beta for adaptive learning rate and beta
    init_alpha = alpha
    init_beta = beta

    for episode in range(max_episodes):
        # START position
        curr_state = START

        # Initialize reward for each episode
        episode_reward = 0

        # Adaptively decaying learning rate alpha and beta
        alpha = get_adaptive_lr(init_alpha, decay_rate=0.005, episode=episode)
        beta = get_adaptive_lr(init_beta, decay_rate=0.005, episode=episode)

        # Softmax temperature
        tau = max(0.1, 1.0 * np.exp(-0.03 * episode))

        while True:
            # Select action by using Softmax policy
            curr_action, probs = softmax_action_selection(curr_state, theta, tau)

            # Get next state
            next_state, step_reward = get_next_state(curr_state, curr_action)

            # Calculate state feature vector
            curr_state_features = state_to_feature_vector(curr_state)
            next_state_features = state_to_feature_vector(next_state)

            # Calculate TD error
            v_s = w @ curr_state_features
            v_s_next = w @ next_state_features
            temporal_difference = step_reward + gamma * v_s_next - v_s
            temporal_difference = np.clip(temporal_difference, -1.0, 1.0)   # Avoid extremely large or small values

            # Update Critic with moving average method
            w = w * (1 - beta) + beta * temporal_difference * curr_state_features

            # Update Actor by using a limited max gradient
            grad_log_policy = policy_gradient(probs, curr_action)
            max_grad = 0.2
            theta += alpha * temporal_difference * np.outer(curr_state_features, np.clip(grad_log_policy, -max_grad, max_grad))

            # Terminate Process if terminal state
            if is_terminal_state(curr_state):
                episode_reward += REWARD[GOAL[:2]]
                break

            # Accumulate reward for each step
            episode_reward += step_reward

            # Update state
            curr_state = next_state

        # Book-keeping the rewards for each episode
        total_rewards.append(episode_reward)

    # Convert into numpy array
    total_rewards = np.array(total_rewards)

    return w, theta, total_rewards


def run_experiments(set_algorithm, trials=100, episodes=100,
                    sarsa_alpha=0.4689, sarsa_epsilon=0.4589,
                    ql_alpha=0.4908, ql_epsilon=0.7492,
                    sarsa_lambda_alpha=0.4306, sarsa_lambda_epsilon=0.2573,
                    actor_critic_alpha=0.06, actor_critic_beta=0.17):

    # a set of rewards for all trials
    trial_rewards = []

    # Run 100 trials
    for i in range(trials):

        # Run the algorithm 100 episodes for each trial
        if set_algorithm == "SARSA":
            _, episodes_reward = SARSA(Q_table, episodes, sarsa_alpha, sarsa_epsilon)
        elif set_algorithm == "Q-Learning":
            _, episodes_reward = Q_Learning(Q_table, episodes, ql_alpha, ql_epsilon)
        elif set_algorithm == "SARSA(λ)":
            _, episodes_reward = SARSA_lambda(Q_table, episodes, sarsa_lambda_alpha, sarsa_lambda_epsilon)
        elif set_algorithm == "Actor-Critic":
            _, _, episodes_reward = Actor_Critic(episodes, GAMMA, actor_critic_alpha, actor_critic_beta)

        # Book-keeping the result of the algorithm selected
        trial_rewards.append(episodes_reward)

    # Convert rewards for mean and standard deviation
    trial_rewards = np.array(trial_rewards)
    mean_rewards = np.mean(trial_rewards, axis=0)
    std_rewards = np.std(trial_rewards, axis=0)

    return mean_rewards, std_rewards


def plot_learning_curve(rewards, name_algorithm):

    # Plot learning curve for the given algorithm
    num_episodes = np.arange(len(rewards))
    plt.figure(figsize=(10, 6))
    plt.plot(num_episodes, rewards, label=name_algorithm, color='red')

    set_title = name_algorithm + " Learning Curve"
    plt.title(set_title)
    plt.xlabel("Episodes")
    plt.ylabel("Accumulated Reward")
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_error_bar(mean_rewards, std_rewards, name_algorithm):

    # Plot error bar for the given algorithm
    num_trials = np.arange(len(mean_rewards))
    plt.figure(figsize=(10, 6))
    plt.errorbar(num_trials, mean_rewards, yerr=std_rewards, fmt='-o',
                 capsize=3, capthick=1, markersize=3, label=name_algorithm,
                 color='b')

    set_title = name_algorithm + " Error Bars"
    plt.title(set_title)
    plt.xlabel("Trials")
    plt.ylabel("Average Reward")
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_all_algorithms(sarsa, ql, sarsa_lambda, ac):

    # Plot all algorithm at once
    num_episodes = np.arange(len(sarsa))
    plt.figure(figsize=(10, 6))
    plt.plot(num_episodes, sarsa, label='SARSA', color='red')
    plt.plot(num_episodes, ql, label='Q-learning', color='blue')
    plt.plot(num_episodes, sarsa_lambda, label='SARSA(λ)', color='green')
    plt.plot(num_episodes, ac, label='Actor-Critic', color='purple')
    plt.title("Comparison the Actor-Critic accumulated reward vs. SARSA, Q-learning, and SARSA(λ)")
    plt.xlabel("Episodes")
    plt.ylabel("Accumulated Reward")
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_accumulated_rewards(sarsa, ql, sarsa_lambda, ac):

    # Accumlated Rewards for each algorithm
    sarsa_cumsum = np.cumsum(sarsa)
    q_learning_cumsum = np.cumsum(ql)
    sarsa_lambda_cumsum = np.cumsum(sarsa_lambda)
    actor_critic_cumsum = np.cumsum(ac)

    plt.figure(figsize=(10, 6))
    plt.plot(sarsa_cumsum, label='SARSA', linestyle="-", color='red')
    plt.plot(q_learning_cumsum, label='Q-learning', linestyle="--", color='blue')
    plt.plot(sarsa_lambda_cumsum, label='SARSA(λ)', linestyle=":",  color='green')
    plt.plot(actor_critic_cumsum, label='Actor-Critic', linestyle="-.", color='purple')

    plt.title("Accumulated Reward Comparison Over 100 Episodes")
    plt.xlabel("Episodes")
    plt.ylabel("Accumulated Rewards")
    plt.legend()
    plt.grid(True)
    plt.show()


def random_search_hyperparams_tuning(num_samples=20, algorithm=None):
    """
    Random search hyperparameter tuning
    :param num_samples: the number of sampling
    :param algorithm: the given algorithm
    :return: None
    """

    # Set the range of Actor-Critic hyperparameters
    if algorithm == "Actor-Critic":
        alpha_range = (0.001, 0.07)
        epsilon_range = (0.1, 0.9)

    # Set the range of SARSA, Q-learning, SARSA(λ) hyperparameters
    else:
        alpha_range = (0.01, 0.5)
        epsilon_range = (0.1, 0.9)

    # Variables for best parameters
    best_alpha, best_epsilon, best_beta = None, None, None
    best_reward = -float("inf")

    for _ in range(num_samples):
        alpha = random.uniform(*alpha_range)
        beta = np.clip(alpha * np.random.randint(2, 5), alpha, 0.8)
        epsilon = random.uniform(*epsilon_range)

        # Test the given algorithm
        if algorithm == "SARSA":
            mean_reward, _ = run_experiments(algorithm, trials=20, episodes=100, sarsa_alpha=alpha, sarsa_epsilon=epsilon)
        elif algorithm == "Q-Learning":
            mean_reward, _ = run_experiments(algorithm, trials=20, episodes=100, ql_alpha=alpha, ql_epsilon=epsilon)
        elif algorithm == "SARSA(λ)":
            mean_reward, _ = run_experiments(algorithm, trials=20, episodes=100, sarsa_lambda_alpha=alpha, sarsa_lambda_epsilon=epsilon)
        elif algorithm == "Actor-Critic":
            mean_reward, std_reward = run_experiments(algorithm, trials=100, episodes=100, actor_critic_alpha=alpha, actor_critic_beta=beta)

        # Find the best alpha, epsilon by using average reward
        if np.mean(mean_reward) > best_reward:
            best_alpha, best_epsilon, best_beta = alpha, epsilon, beta
            best_reward = np.mean(mean_reward)

    # Print out the result
    if algorithm == "Actor-Critic":
        print(f"{algorithm} -> Best Alpha: {best_alpha:.4f}, Best Beta: {best_beta:.4f}, Best Reward: {best_reward:.3f}")
    else:
        print(f"{algorithm} -> Best Alpha: {best_alpha:.4f}, Best Epsilon: {best_epsilon:.4f}, Best Reward: {best_reward:.3f}")


def main():

    print("Part A - 1. tabular SARSA\n")
    sarsa_q_table, sarsa_rewards = SARSA(Q_table, 100)
    sarsa_mean, sarsa_std = run_experiments("SARSA")
    plot_learning_curve(sarsa_rewards, "SARSA")
    plot_error_bar(sarsa_mean, sarsa_std, "SARSA")
    print(f'100 Episodes SARSA Q-table: \n{sarsa_q_table}, \n\n SARSA Rewards: \n{sarsa_rewards}')

    print("\n----------------------------------------------------------------------\n")

    print("Part A - 2. tabular Q-Learning\n")
    ql_q_table, ql_rewards = Q_Learning(Q_table, 100)
    ql_mean, ql_std = run_experiments("Q-Learning")
    plot_learning_curve(ql_rewards, "Q-Learning")
    plot_error_bar(ql_mean, ql_std, "Q-Learning")
    print(f'100 Episodes Q-Learning Q-table: \n{ql_q_table}, \n\n Q-Learning Rewards: \n{ql_rewards}')

    print("\n----------------------------------------------------------------------\n")

    print("Part A - 3. tabular SARSA(λ)\n")
    sarsa_lambda_q_table, sarsa_lambda_rewards = SARSA_lambda(Q_table, 100)
    sarsa_lambda_mean, sarsa_lambda_std = run_experiments("SARSA(λ)")
    plot_learning_curve(sarsa_lambda_rewards, "SARSA(λ)")
    plot_error_bar(sarsa_lambda_mean, sarsa_lambda_std, "SARSA(λ)")
    print(f'100 Episodes SARSA(λ) Q-table: \n{sarsa_lambda_q_table}, \n\n SARSA(λ) Rewards: \n{sarsa_lambda_rewards}')

    print("\n----------------------------------------------------------------------\n")

    print("Part B - 1. Actor-Critic\n")
    actor_critic_w, actor_critic_theta, actor_critic_rewards = Actor_Critic(max_episodes=100)
    actor_critic_mean, actor_critic_std = run_experiments("Actor-Critic")
    plot_learning_curve(actor_critic_rewards, "Actor-Critic")
    plot_error_bar(actor_critic_mean, actor_critic_std, "Actor-Critic")
    print(f'100 Episodes weight: \n{actor_critic_w}, \n\n theta: \n{actor_critic_theta}, \n\n reward: \n{actor_critic_rewards}')

    print("\n----------------------------------------------------------------------\n")

    print("Part B - 2. Comparison the accumulated reward\n")
    print(f'SARSA: {np.sum(sarsa_rewards)}, Q-learning: {np.sum(ql_rewards)}, '
          f'SARSA(λ): {np.sum(sarsa_lambda_rewards)}, Actor-Critic: {np.sum(actor_critic_rewards)}')
    plot_all_algorithms(sarsa_rewards, ql_rewards, sarsa_lambda_rewards, actor_critic_rewards)
    plot_accumulated_rewards(sarsa_rewards, ql_rewards, sarsa_lambda_rewards, actor_critic_rewards)



if __name__ == "__main__":
    # (Independently) Tuning Hyperparameters
    # random_search_hyperparams_tuning(algorithm="SARSA")           # Alpha: 0.4689 / Epsilon: 0.4589
    # random_search_hyperparams_tuning(algorithm="Q-Learning")      # Alpha: 0.4908 / Epsilon: 0.7492
    # random_search_hyperparams_tuning(algorithm="SARSA(λ)")        # Alpha: 0.4306 / Epsilon: 0.2573
    # random_search_hyperparams_tuning(algorithm="Actor-Critic")    # alpha: 0.0548(0.06) / beta: 0.1643(0.17)

    # main function for assignment
    main()





