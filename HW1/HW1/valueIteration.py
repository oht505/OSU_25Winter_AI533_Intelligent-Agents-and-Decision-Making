import pandas as pd

# Given problem parameters
gamma = 1  # Discount factor
rewards = {
    "s1": 0,
    "s2": -1,
    "s3": 2,
    "s4": 0
}

# Adjusted initial values to induce a_1 -> a_2 -> a_2 behavior
V_adjusted = {
    "s1": 0,  # Initial value for s1
    "s2": 5,  # Initial value for s2
    "s3": 2,  # Initial value for s3
    "s4": 3  # Initial value for s4
}



# Store calculations per iteration
value_iteration_steps_adjusted = []

# Perform value iteration for a few iterations to analyze policy changes
for t in range(6):
    new_V = {}
    # Value function updates based on Bellman equation
    new_V["s3"] = rewards["s3"] + gamma * V_adjusted["s3"]
    new_V["s2"] = rewards["s2"] + gamma * V_adjusted["s3"]
    new_V["s4"] = rewards["s4"] + gamma * V_adjusted["s4"]
    new_V["s1"] = max(rewards["s1"] + gamma * V_adjusted["s2"], rewards["s1"] + gamma * V_adjusted["s4"])

    # Save step-by-step calculations
    value_iteration_steps_adjusted.append({
        "Iteration": t,
        "V(s1)": new_V["s1"],
        "V(s2)": new_V["s2"],
        "V(s3)": new_V["s3"],
        "V(s4)": new_V["s4"],
        "π(s1)": "a1" if new_V["s1"] == rewards["s1"] + gamma * V_adjusted["s2"] else "a2"
    })

    V_adjusted = new_V  # Update values for the next iteration

# Convert to DataFrame for better visualization
df_value_iteration_steps_adjusted = pd.DataFrame(value_iteration_steps_adjusted)
print(df_value_iteration_steps_adjusted)
