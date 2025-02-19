import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import torch.nn.functional as F  # for softmax

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyper parameters
hidden_size = 20
num_classes = 2
num_epochs = 30000
batch_size = 800
learning_rate = 1e-3

### NEW/UPDATED: Changed input_size to 2 (action-reward encoded in a 2D vector)
input_size = 2
sequence_length = 10
num_layers = 1

# Load data
filename = 'data/results_hybrid.csv'
df = pd.read_csv(filename)

if filename == 'human_data.csv':
    df = df.rename(columns={"choice": "Action", "reward": "Reward"})
    df['Action'] = df['Action'].astype(int) - 1

n_trials_per_block = 10
n_blocks_pp = 20
n_participants = len(df) // (n_trials_per_block * n_blocks_pp)
n_blocks = n_participants * n_blocks_pp

choices = df['Action'].to_numpy().reshape(n_participants, n_blocks_pp, n_trials_per_block)
rewards = df['Reward'].to_numpy().reshape(n_participants, n_blocks_pp, n_trials_per_block)
data = np.stack((choices, rewards), axis=-1)  # Shape: (n_participants, n_blocks_pp, n_trials_per_block, 2)

data = data.reshape(-1, n_trials_per_block, 2)  # Shape: (600, 10, 2)
data = np.transpose(data, (1, 0, 2))  # Shape: (10, 600, 2)

data_tensor = torch.tensor(data, dtype=torch.float32).to(device)

split_idx = int(0.8 * data_tensor.shape[1])
train_tensor = data_tensor[:, :split_idx, :]  # (10, 480, 2)
test_tensor = data_tensor[:, split_idx:, :]   # (10, 120, 2)

n_sequences = train_tensor.shape[1]

# Prepare labels and inputs for next choice prediction
### NEW/UPDATED: Build inputs such that the reward is multiplied with the one-hot action vector.
### Instead of [action_onehot, reward] (3 dimensions), we now get [reward, 0] if action 0, or [0, reward] if action 1.
xs = np.zeros((train_tensor.shape[0], train_tensor.shape[1], input_size), dtype=np.float32)
ys = np.zeros((train_tensor.shape[0], train_tensor.shape[1], 1), dtype=np.float32)

train_choices = train_tensor[:, :, 0].cpu().numpy()  # shape: (10, n_sequences)
train_rewards = train_tensor[:, :, 1].cpu().numpy()    # shape: (10, n_sequences)

for sess_i in range(n_sequences):
    # For each session, build the previous action-reward vector.
    # Use a dummy vector [0,0] for the first trial.
    prev_vectors = []
    # Dummy input for first trial
    prev_vectors.append([0, 0])
    for t in range(1, train_tensor.shape[0]):
        choice = train_choices[t-1, sess_i]
        reward  = train_rewards[t-1, sess_i]
        # Multiply reward by one-hot action vector:
        if choice == 0:
            vector = [reward, 0]
        else:
            vector = [0, reward]
        prev_vectors.append(vector)
    prev_vectors = np.array(prev_vectors)  # shape: (10, 2)
    xs[:, sess_i, :] = prev_vectors
    ys[:, sess_i, 0] = train_choices[:, sess_i]

xs = torch.from_numpy(xs).to(device)
ys = torch.from_numpy(ys).to(device)

print(f'xs shape: {xs.shape}')  # Expected: (seq_length, n_sequences, 2)
print(f'ys shape: {ys.shape}')  # Expected: (seq_length, n_sequences, 1)

# Define the LSTM model
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(LSTMModel, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=False)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(1), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, x.size(1), self.hidden_size).to(device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out)
        return out

model = LSTMModel(input_size, hidden_size, num_layers, num_classes).to(device)

def repetition_penalty(logits, penalty_weight=0.1):
    """
    Computes a penalty that discourages repeated actions.
    
    Args:
        logits: Tensor of shape (seq_len, batch_size, num_classes)
                representing the raw output (logits) from your model.
        penalty_weight: A scaling factor for how much to penalize repetition.
    
    Returns:
        A scalar penalty value.
    """
    # Convert logits to probabilities with softmax along the class dimension.
    probs = F.softmax(logits, dim=-1)  # Shape: (seq_len, batch_size, num_classes)
    seq_len = probs.shape[0]
    
    penalty = 0.0
    # For each consecutive pair of time steps, compute a measure of similarity.
    for t in range(1, seq_len):
        # Element-wise product over the class dimension gives a similarity measure.
        # For each sequence in the batch, sum over classes.
        sim = torch.sum(probs[t] * probs[t-1], dim=-1)  # Shape: (batch_size,)
        # The higher the similarity, the more repeated the behavior.
        # Average over the batch and add to the penalty.
        penalty += sim.mean()
    
    # Multiply by a weight factor to control the impact of the penalty.
    return penalty_weight * penalty

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

n_sequences = xs.shape[1]
for epoch in range(num_epochs):
    permutation = torch.randperm(n_sequences)
    xs_shuffled = xs[:, permutation, :]
    ys_shuffled = ys[:, permutation, :]
    
    for i in range(0, n_sequences, batch_size):
        batch_x = xs_shuffled[:, i:i+batch_size, :].to(device)  # (seq_length, batch_size, 2)
        batch_y = ys_shuffled[:, i:i+batch_size, :].to(device)  # (seq_length, batch_size, 1)
        batch_y = batch_y.squeeze(-1).long()  # (seq_length, batch_size)
        
        outputs = model(batch_x)  # (seq_length, batch_size, num_classes)
        loss_ce = criterion(outputs.view(-1, num_classes), batch_y.view(-1))
        # Compute the repetition penalty using the entire output sequence.
        penalty = repetition_penalty(outputs)

        # Total loss is a combination of cross-entropy and the penalty.
        loss = loss_ce + penalty
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    if epoch % 10 == 0:
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

# Prepare test data using the new encoding
test_choices = test_tensor[:, :, 0].cpu().numpy()  # (10, n_test_sequences)
test_rewards = test_tensor[:, :, 1].cpu().numpy()    # (10, n_test_sequences)

### NEW/UPDATED: Build test inputs with the same encoding ([reward, 0] or [0, reward])
xs_test = np.zeros((test_tensor.shape[0], test_tensor.shape[1], input_size), dtype=np.float32)
ys_test = np.zeros((test_tensor.shape[0], test_tensor.shape[1], 1), dtype=np.float32)

for sess_i in range(test_tensor.shape[1]):
    prev_vectors = []
    prev_vectors.append([0, 0])
    for t in range(1, test_tensor.shape[0]):
        choice = test_choices[t-1, sess_i]
        reward  = test_rewards[t-1, sess_i]
        if choice == 0:
            vector = [reward, 0]
        else:
            vector = [0, reward]
        prev_vectors.append(vector)
    prev_vectors = np.array(prev_vectors)
    xs_test[:, sess_i, :] = prev_vectors
    ys_test[:, sess_i, 0] = test_choices[:, sess_i]

xs_test = torch.from_numpy(xs_test).to(device)
ys_test = torch.from_numpy(ys_test).to(device)

# --- Forward Simulation (with updated input encoding) ---
n_participants = 44
n_blocks_pp = 20      
n_trials_per_block = 10  

### NEW/UPDATED: Set simulation input size to 2.
model.eval()
simulation_data = []
global_trial = 0

with torch.no_grad():
    for participant in range(n_participants):
        for block in range(n_blocks_pp):
            mean_reward_block = np.random.normal(0, np.sqrt(100), 2)
            # Create an input sequence for simulation with shape (n_trials, 1, input_size)
            input_seq = torch.zeros(n_trials_per_block, 1, input_size, device=device)
            
            n_states = n_trials_per_block
            V_t = np.zeros(n_states)
            RU = np.zeros(n_states)  
            TU = np.zeros(n_states) 
            post_mean = np.zeros((2, n_states))
            post_variance = np.ones((2, n_states)) * 5
            kalman_gain = np.zeros((2, n_states))
            
            for trial in range(n_trials_per_block):
                outputs = model(input_seq)  # (n_trials, 1, num_classes)
                logits_t = outputs[trial, 0, :]  # (num_classes,)
                probs_t = F.softmax(logits_t, dim=0).detach().cpu().numpy()
                rnn_action = np.random.choice([0, 1], p=probs_t)
                reward_vector = np.random.normal(mean_reward_block, np.sqrt(10), 2)
                rnn_reward = reward_vector[rnn_action]
                
                ### NEW/UPDATED: Encode the input as reward * one-hot vector.
                if rnn_action == 0:
                    encoded_input = [rnn_reward, 0]
                else:
                    encoded_input = [0, rnn_reward]
                input_seq[trial, 0, :] = torch.tensor(encoded_input, dtype=torch.float32, device=device)
                
                for i in range(2):
                    if rnn_action == i:
                        if trial == 0:
                            prev_variance = 5
                            prev_mean = 0
                        else:
                            prev_variance = post_variance[i][trial - 1]
                            prev_mean = post_mean[i][trial - 1]
                        kalman_gain[i][trial] = prev_variance / (prev_variance + 10)
                        post_variance[i][trial] = (1 - kalman_gain[i][trial]) * prev_variance
                        post_mean[i][trial] = prev_mean + kalman_gain[i][trial] * (rnn_reward - prev_mean)
                    else:
                        if trial == 0:
                            post_variance[i][trial] = 5
                            post_mean[i][trial] = 0
                        else:
                            post_variance[i][trial] = post_variance[i][trial - 1]
                            post_mean[i][trial] = post_mean[i][trial - 1]
                            
                V_t[trial] = post_mean[0][trial] - post_mean[1][trial]
                sigma1 = post_variance[0][trial]
                sigma2 = post_variance[1][trial]
                TU[trial] = np.sqrt(sigma1 + sigma2)
                RU[trial] = np.sqrt(sigma1) - np.sqrt(sigma2)
                
                simulation_data.append({
                    "Participant": participant,
                    "Block": block,
                    "Trial": trial,
                    "Global_Trial": global_trial,
                    "Mean_Reward_Arm0": mean_reward_block[0],
                    "Mean_Reward_Arm1": mean_reward_block[1],
                    "Action": rnn_action,
                    "Reward": rnn_reward,
                    "RNN_Prob_Action0": probs_t[0],
                    "V_t": V_t[trial],
                    "Kalman_post_mean_0": post_mean[0][trial],
                    "Kalman_post_mean_1": post_mean[1][trial],
                    "Kalman_post_variance_0": post_variance[0][trial],
                    "Kalman_post_variance_1": post_variance[1][trial],
                    "Kalman_kalman_gain_0": kalman_gain[0][trial],
                    "Kalman_kalman_gain_1": kalman_gain[1][trial],
                    "RU": RU[trial],
                    "TU": TU[trial]
                })
                global_trial += 1

df_simulation = pd.DataFrame(simulation_data)
df_simulation.to_csv("data/simulation_trained_network_hybrid.csv", index=False)
print("Simulation complete. Data saved to data/simulation_trained_network_hybrid.csv")
