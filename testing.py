from BanditRNN import HybridAgent_opt, UCBAgent, ThompsonAgent
import numpy as np
import os
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from RNN import GRUModel, num_classes, input_size, device
from data_utils import load_preprocess_data

# load data
filename = 'data/results_hybrid.csv'
date = '27021320_nopenalty'
xs, ys, xs_test, ys_test = load_preprocess_data(filename)

optimized_params = {
    "HybridAgent_opt": [1, 1],
    "UCBAgent": [2, 2]}

#load optimized hyperparameters
csv_path = os.path.join("optimized_parameters", f"optimized_hyperparams_{date}.csv")
df_params = pd.read_csv(csv_path)
best_hidden_size = int(df_params['hidden_size'].iloc[0])
best_num_layers = int(df_params['num_layers'].iloc[0])

# Load datasets
df_hybrid = pd.read_csv('data/results_hybrid.csv')
df_thompson = pd.read_csv('data/results_thompson.csv')
df_ucb = pd.read_csv('data/results_ucb.csv')
df_rnn_human = pd.read_csv('data/simulation_RNN_human_optuna_150k_layerHP_second_try.csv')
df_rnn_thompson = pd.read_csv('data/simulation_trained_network_thompson2.csv')
df_rnn_ucb = pd.read_csv('data/simulation_trained_network_ucb.csv')
df_rnn_hybrid = pd.read_csv('data/simulation_RNN_hybrid_26022135.csv')
df_human = pd.read_csv('kalman_human_data.csv')
df_human = df_human.rename(columns={"choice": "Action"})

model_final = GRUModel(input_size, best_hidden_size, best_num_layers, num_classes).to(device)
checkpoint_path = f"checkpoints/{date}model_epoch_16.pth"  # Replace with your checkpoint
model_final.load_state_dict(torch.load(checkpoint_path))
model_final.eval()


with torch.no_grad():
    # Get the logits from the model for the training data
    outputs_train = model_final(xs)
    # Convert logits to predicted class indices (0 or 1)
    # Transform logits to probabilities using softmax
    probs_train = F.softmax(outputs_train, dim=-1)  # same shape as outputs_train
    
    # Sample an action from the probability distribution for each trial and sequence
    # Reshape probabilities to 2D: (seq_length*n_sequences, num_classes)
    probs_2d = probs_train.reshape(-1, probs_train.shape[-1])
    # Sample one action for each trial
    predicted_train = torch.multinomial(probs_2d, num_samples=1).reshape(outputs_train.shape[0], outputs_train.shape[1])
    # Get the true labels (squeeze the last dim)
    actual_train = ys.squeeze(-1).long()  # shape: (seq_length, n_sequences)

# Print out a few predictions vs. the actual values
print("Sanity Check: Predictions on Training Data")
print("Predicted labels (first 5 sequences):")
print(predicted_train[:, :5].cpu().numpy())
print("Actual labels (first 5 sequences):")
print(actual_train[:, :5].cpu().numpy())
print(f"Predicted shape: {predicted_train.shape}, Actual shape: {actual_train.shape}")

# Compute accuracy across all training trials
accuracy = (predicted_train == actual_train).float().mean().item()
print(f"Training Prediction Accuracy: {accuracy:.4f}")


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

df_RNN_hybrid = pd.read_csv('data/simulation_RNN_hybrid_27021145.csv')
# Total number of states in the simulation (as used above)
n_states_sim = len(df_RNN_hybrid)

# For HybridAgent_opt, use the optimized parameters from earlier.
pseudo_r2_hybrid = compute_pseudo_r2(HybridAgent_opt, df_RNN_hybrid, optimized_params["HybridAgent_opt"], n_states_sim)
print(f"Pseudo R^2 for HybridAgent_opt: {pseudo_r2_hybrid:.4f}")

# For UCBAgent, we used a fixed parameter set in simulation ([lamda, gamma] = [0.1, 0]).
pseudo_r2_ucb = compute_pseudo_r2(UCBAgent, df_RNN_hybrid, optimized_params["UCBAgent"], n_states_sim)
print(f"Pseudo R^2 for UCBAgent: {pseudo_r2_ucb:.4f}")

# For ThompsonAgent, no optimization parameters are needed.
pseudo_r2_thompson = compute_pseudo_r2(ThompsonAgent, df_RNN_hybrid, None, n_states_sim)
print(f"Pseudo R^2 for ThompsonAgent: {pseudo_r2_thompson:.4f}")