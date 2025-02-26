from BanditRNN import HybridAgent_opt, UCBAgent, ThompsonAgent
import numpy as np
import pandas as pd


# -----------------------------
# Compute Pseudo R^2 Measure for Each Agent
# -----------------------------
def compute_total_log_likelihood(agent_class, data, optimized_params, n_states):
    """
    Compute the total log likelihood for the provided data given an agent.
    
    Parameters:
        agent_class: The class of the agent (e.g., HybridAgent_opt, UCBAgent, or ThompsonAgent).
        data: A DataFrame containing simulation data. Assumes columns 'State', 'Action', and 'Reward'.
        optimized_params: The optimized parameters for the agent. For ThompsonAgent, pass None.
        n_states: Total number of states for the simulation.
        
    Returns:
        total_log_likelihood: The summed log likelihood over all trials.
    """
    total_log_likelihood = 0.0

    # Instantiate the agent using the optimized parameters if required
    if agent_class == HybridAgent_opt:
        beta, gamma = optimized_params
        agent = agent_class(n_states=n_states, beta=beta, gamma=gamma)
    elif agent_class == UCBAgent:
        lamda, gamma = optimized_params
        agent = agent_class(n_states=n_states, lamda=lamda, gamma=gamma)
    elif agent_class == ThompsonAgent:
        agent = agent_class(n_states=n_states)
    else:
        raise ValueError("Agent class not recognized.")

    # Loop over each trial in the data
    for idx, row in data.iterrows():
        state = int(row['Global_Trial'])  # For simulation data, we use capitalized column names.
        choice = int(row['Action'])
        # Get the probability for action 0 at the current state.
        prob_0 = agent.get_choice_probs(state)
        # If the chosen action is 0, use prob_0; if action is 1, use (1 - prob_0)
        prob = prob_0 if choice == 0 else (1 - prob_0)
        total_log_likelihood += np.log(prob + 1e-10)
        reward = row['Reward']
        agent.update(choice, reward, state)
    
    return total_log_likelihood

def compute_pseudo_r2(agent_class, data, optimized_params, n_states):
    """
    Compute the pseudo R^2 (McFadden's pseudo R^2) for the provided agent and data.
    
    Parameters:
        agent_class: The agent class (HybridAgent_opt, UCBAgent, or ThompsonAgent).
        data: A DataFrame with columns 'State', 'Action', and 'Reward'.
        optimized_params: The optimized parameters for the agent. For ThompsonAgent, pass None.
        n_states: Total number of states for the simulation.
        
    Returns:
        pseudo_r2: The pseudo R^2 value.
    """
    total_ll_model = compute_total_log_likelihood(agent_class, data, optimized_params, n_states)
    # Null model: assumes a constant choice probability of 0.5 for every trial.
    total_ll_null = len(data) * np.log(0.5)
    pseudo_r2 = 1 - (total_ll_model / total_ll_null)
    return pseudo_r2

# -----------------------------
# Compute and Report Pseudo R^2 for Simulation Data
# -----------------------------

df_RNN_hybrid = pd.read_csv('data/simulation_RNN_hybrid_26021359_smallerbatch.csv')
# Total number of states in the simulation (as used above)
n_states_sim = n_participants * n_block_per_p * n_trials_per_block

# For HybridAgent_opt, use the optimized parameters from earlier.
pseudo_r2_hybrid = compute_pseudo_r2(HybridAgent_opt, df_hybrid, optimized_params["HybridAgent_opt"], n_states_sim)
print(f"Pseudo R^2 for HybridAgent_opt: {pseudo_r2_hybrid:.4f}")

# For UCBAgent, we used a fixed parameter set in simulation ([lamda, gamma] = [0.1, 0]).
pseudo_r2_ucb = compute_pseudo_r2(UCBAgent, df_ucb, [0.1, 0], n_states_sim)
print(f"Pseudo R^2 for UCBAgent: {pseudo_r2_ucb:.4f}")

# For ThompsonAgent, no optimization parameters are needed.
pseudo_r2_thompson = compute_pseudo_r2(ThompsonAgent, df_thompson, None, n_states_sim)
print(f"Pseudo R^2 for ThompsonAgent: {pseudo_r2_thompson:.4f}")