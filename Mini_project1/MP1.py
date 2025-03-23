import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
import matplotlib.pyplot as plt

np.set_printoptions(precision=2, suppress=True)

# Part A - 1. Set up the environment

# Environment parameter
ENV_ROWS, ENV_COLS = 4, 4
ACTIONS = [(0, -1), (-1, 0), (0, 1), (1, 0)]    # left: 0, up: 1, right: 2, down: 3
TEXT_ACTIONS = ['left', 'up', 'right', 'down']
SLIDES = {0: [(-1, 0), (1, 0)],
          1: [(0, -1), (0, 1)],
          2: [(-1, 0), (1, 0)],
          3: [(0, -1), (0, 1)]}
SLIDES_IDX = {0: [1, 3],
              1: [0, 2],
              2: [1, 3],
              3: [0, 2]}
START = (0, 0)
GOAL = (3, 3)
WATER_1, WATER_2 = (2, 1), (2, 2)
FIRE_1, FIRE_2 = (0, 1), (0, 2)
PROB_SUCCESS = 0.8
PROB_SLIDE = 0.1
GAMMA = [0.3, 0.95]

# Rewards setting
REWARD = -np.ones((ENV_ROWS, ENV_COLS)) # all other states
REWARD[WATER_1], REWARD[WATER_2] = -5, -5     # Water
REWARD[FIRE_1], REWARD[FIRE_2] = -10, -10   # Fire
REWARD[GOAL] = 100  # Goal

def update_state_value(curr_row, curr_col, gamma, v_table):

    next_values = []
    for i, action in enumerate(ACTIONS):
        next_row, next_col = curr_row + action[0], curr_col + action[1]
        if next_row < 0 or next_row >= ENV_ROWS:
            next_row = curr_row

        if next_col < 0 or next_col >= ENV_COLS:
            next_col = curr_col

        intended_value = PROB_SUCCESS * v_table[next_row][next_col]

        # Slide Case
        slide_values = 0
        for slide in SLIDES[i]:
            next_slide_row, next_slide_col = curr_row + slide[0], curr_col + slide[1]
            if next_slide_row < 0 or next_slide_row >= ENV_ROWS:
                next_slide_row = curr_row

            if next_slide_col < 0 or next_slide_col >= ENV_COLS:
                next_slide_col = curr_col

            slide_values += PROB_SLIDE * v_table[next_slide_row][next_slide_col]

        # Value calculation for each action
        next_value = REWARD[curr_row][curr_col] + gamma * (intended_value + slide_values)
        next_values.append(next_value)

    # Update maximum value
    update_value = max(next_values)
    max_value_action_idx = np.argmax(next_values)

    return update_value, max_value_action_idx

def find_policy(policy_table, v_table):

    for row in range(ENV_ROWS):
        for col in range(ENV_COLS):

            best_action = None
            best_value = float('-inf')
            for action_idx, action in enumerate(ACTIONS):
                next_row, next_col = row + action[0], col + action[1]

                if next_row < 0 or next_row >= ENV_ROWS:
                    next_row = row
                if next_col < 0 or next_col >= ENV_COLS:
                    next_col = col

                value = v_table[next_row, next_col]

                if value > best_value:
                    best_value = value
                    best_action = action_idx

            policy_table[row][col] = TEXT_ACTIONS[best_action]

    return policy_table

def value_iteration(gamma):

    # Value function table
    v_table = np.zeros((ENV_ROWS, ENV_COLS))
    v_table[GOAL] = 100

    while True:
        next_v_table = np.copy(v_table)

        # Update All state values and Find max_action
        for row in range(ENV_ROWS):
            for col in range(ENV_COLS):
                next_v_table[row][col], _ = update_state_value(row, col, gamma, v_table)

        # Terminate process
        if np.array_equal(v_table, next_v_table):
            break

        v_table = next_v_table

    final_v_table = next_v_table

    # Policy table
    policy_table = [[''] * ENV_COLS for _ in range(ENV_ROWS)]

    # Find optimal policy based on the value table
    final_policy_table = find_policy(policy_table, v_table)

    return final_v_table, final_policy_table

def policy_improvement(policy, v, gamma):

    new_policy_table = np.copy(policy)

    # Find best action
    for row in range(ENV_ROWS):
        for col in range(ENV_COLS):
            _, best_action_idx = update_state_value(row, col, gamma, v)
            new_policy_table[row][col] = best_action_idx

    return new_policy_table

def policy_evaluation(policy, v, gamma):

    v_table = v

    while True:
        next_v_table = np.copy(v_table)

        for row in range(ENV_ROWS):
            for col in range(ENV_COLS):

                action = policy[row][col]
                next_row, next_col = row + ACTIONS[action][0], col + ACTIONS[action][1]

                if next_row < 0 or next_row >= ENV_ROWS:
                    next_row = row
                if next_col < 0 or next_col >= ENV_COLS:
                    next_col = col

                # Slide
                slide_value = 0
                for slide in SLIDES[action]:
                    next_slide_row, next_slide_col = row + slide[0], col + slide[1]

                    if next_slide_row < 0 or next_slide_row >= ENV_ROWS:
                        next_slide_row = row
                    if next_slide_col < 0 or next_slide_col >= ENV_COLS:
                        next_slide_col = col

                    slide_value += PROB_SLIDE * v_table[next_slide_row][next_slide_col]

                # Update state value
                next_v_table[row][col] = REWARD[row][col] + gamma * (PROB_SUCCESS * v_table[next_row][next_col] +
                                          slide_value)

        if np.array_equal(v_table, next_v_table):
            break

        v_table = next_v_table

    return v_table

def policy_iteration(gamma):
    # Return: policy and objective value

    # Initialize random policy
    policy_table = np.random.choice([0, 1, 2, 3], size=(ENV_ROWS, ENV_COLS))

    # Convert the random policy as text
    text_policy_table = [[''] * ENV_COLS for _ in range(ENV_ROWS)]
    for row in range(ENV_ROWS):
        for col in range(ENV_COLS):
            text_policy_table[row][col] = TEXT_ACTIONS[policy_table[row][col]]

    # print(f'Initial policy table: \n {np.array(text_policy_table)} \n')

    while True:
        # Initialize random value table
        v_table = np.random.rand(ENV_ROWS, ENV_COLS) * 0.1
        v_table[GOAL] = 100

        # Policy Evaluation
        new_v_table = policy_evaluation(policy_table, v_table, gamma)

        # Policy Improvement
        new_policy_table = policy_improvement(policy_table, new_v_table, gamma)

        # Terminate process condition
        if np.array_equal(new_policy_table, policy_table):
            break

        policy_table = new_policy_table

    # Text policy table
    final_policy_table = [[''] * ENV_COLS for _ in range(ENV_ROWS)]
    for row in range(ENV_ROWS):
        for col in range(ENV_COLS):
            final_policy_table[row][col] = TEXT_ACTIONS[new_policy_table[row][col]]

    final_policy_table = np.array(final_policy_table)

    return final_policy_table, new_v_table

def generate_episode(policy):

    trajectory = []
    accumulative_rewards = 0
    agent_pos = START   # (0, 0)

    while True:

        # Interpret and Find action from policy table
        curr_policy_dir = policy[agent_pos]
        action_idx = TEXT_ACTIONS.index(curr_policy_dir)
        slide_action_1, slide_action_2 = SLIDES_IDX[action_idx]
        actions = [action_idx, slide_action_1, slide_action_2]
        selected_action = np.random.choice(actions, p=[0.8, 0.1, 0.1])
        text_selected_action = TEXT_ACTIONS[selected_action]
        next_row, next_col = ACTIONS[selected_action]

        # remain condition
        new_agent_row, new_agent_col = agent_pos[0] + next_row, agent_pos[1] + next_col
        if (agent_pos[0] + next_row) < 0 or (agent_pos[0] + next_row) >= ENV_ROWS:
            new_agent_row = agent_pos[0]
        if (agent_pos[1] + next_col) < 0 or (agent_pos[1] + next_col) >= ENV_COLS:
            new_agent_col = agent_pos[1]

        # New agent's position
        new_agent_pos = (new_agent_row, new_agent_col)

        # Reward
        reward = REWARD[agent_pos]
        accumulative_rewards += REWARD[agent_pos]

        # Record trajectory
        trajectory.append([agent_pos, text_selected_action, reward])

        if agent_pos == GOAL:
            break

        # Update agent's position
        agent_pos = new_agent_pos

    return trajectory, accumulative_rewards

def DAGGER(optimal_policy, N):

    # Initialize a random policy π_hat, which is not optimal policy
    policy_hat = [[3, 1, 2, 0] for _ in range(ENV_ROWS)]
    # policy_hat = np.random.choice([0, 1, 2, 3], size=(ENV_ROWS, ENV_COLS))
    # for i in range(ENV_ROWS):
    #     policy_hat[i][0], policy_hat[i][3] = 3, 3   # Avoid inefficiency

    # test set
    test_set = []

    # Convert the random policy as text
    text_policy_hat = [[''] * ENV_COLS for _ in range(ENV_ROWS)]
    for row in range(ENV_ROWS):
        for col in range(ENV_COLS):
            test_set.append([row, col])
            text_policy_hat[row][col] = TEXT_ACTIONS[policy_hat[row][col]]

    # Create a data set D
    D_data = []
    D_labels = []

    # Create a model
    # model = RandomForestClassifier()
    model = DecisionTreeClassifier(max_depth=5)
    # model = SVC(kernel='rbf')

    for i in range(N):
        # Generate a trajectory tau_i using policy_hat and B.1.
        tau_i, _ = generate_episode(np.array(text_policy_hat))

        for D_i in tau_i:
            # Store state
            state = D_i[0]
            D_data.append(list(state))

            # Store expert action
            expert_action = optimal_policy[state]
            D_labels.append(TEXT_ACTIONS.index(expert_action))

        # Train model
        model.fit(D_data, D_labels)

        # Predict the action and Produce new policy
        for row in range(ENV_ROWS):
            for col in range(ENV_COLS):
                action_idx = model.predict([[row, col]])[0]
                text_policy_hat[row][col] = TEXT_ACTIONS[action_idx]

    predicted_actions = [model.predict([state])[0] for state in test_set]
    expert_actions = [TEXT_ACTIONS.index(optimal_policy[tuple(state)]) for state in test_set]

    accuracy = accuracy_score(expert_actions, predicted_actions)

    return accuracy

def plot_accuracy(N, set_accuracy):

    plt.figure(figsize=(10,6))

    plt.plot(N, set_accuracy, label='Accuracy', color='red')
    plt.title("Variances in model Accuracy with N value")
    plt.xlabel("N")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)
    plt.show()


def main():
    # Part A - 2. Value Iteration for each gamma (0.3, 0.95)
    print(f'\n Value iteration ')
    for gamma_val in GAMMA:
        print(f'\nCurrent Gamma value is < {gamma_val} > ')
        v, policy = value_iteration(gamma_val)
        print(f'output policy: \n {np.array(policy)} \n')
        print(f'value table: \n {v}')

    print(f'\n------------------------------------------------')

    # Part A - 3. Policy Iteration with gamma 0.95
    print(f'\n Policy Iteration ')
    gamma = GAMMA[1]
    print(f'\nCurrent Gamma value is < {gamma} > ')
    output_policy, value_table = policy_iteration(gamma)
    print(f'output policy: \n {output_policy} \n')
    print(f'value table: \n {value_table}')

    print(f'\n------------------------------------------------')

    # Part B - 1. Generate an episode
    print(f'\n Generate an episode\n')
    episode, rewards = generate_episode(output_policy)
    for row in episode:
        print(row)

    print(f'\n------------------------------------------------')

    # Part B - 2. Implement DAGGER Algorithm
    print(f'\n Implement DAGGER Algorithm\n')
    N = [5, 10, 20, 30, 40, 50]
    set_accuracy = []
    for n in N:
        accuracy = DAGGER(output_policy, n)
        print(f"N:{n},  accuracy: {accuracy}")
        set_accuracy.append(accuracy)

    plot_accuracy(N, set_accuracy)

if __name__ == "__main__":
    print(f'REWARD: \n {REWARD}')
    main()
