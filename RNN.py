import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


#device config
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# hyper parameters
#input_size = 784 # 28x28
hidden_size = 20
num_classes = 2
num_epochs = 100000
batch_size = 800
learning_rate = 1e-3

input_size = 2
sequence_length = 10
num_layers = 6

# load data
filename = 'data/results_hybrid.csv'
df = pd.read_csv(filename)

if filename == 'human_data.csv':
    #rename columns "choice" and "reward" to "Action" and "Reward"
    df = df.rename(columns={"choice": "Action", "reward": "Reward"})
    #convert "Action" to 0 and 1
    df['Action'] = df['Action'].astype(int) - 1  # Convert choice to 0, 1

n_trials_per_block = 10
n_blocks_pp = 20
n_participants = len(df) // (n_trials_per_block * n_blocks_pp)
n_blocks = n_participants * n_blocks_pp



choices = df['Action'].to_numpy().reshape(n_participants, n_blocks_pp, n_trials_per_block)
rewards = df['Reward'].to_numpy().reshape(n_participants, n_blocks_pp, n_trials_per_block)
data = np.stack((choices, rewards), axis=-1)  # Shape: (30, 20, 10, 2)

# Reshape to (600, 10, 2) and transpose to get (10, 600, 2) for RNN
data = data.reshape(-1, n_trials_per_block, 2)  # Shape: (600, 10, 2)
data = np.transpose(data, (1, 0, 2))  # Shape: (10, 600, 2)

data_tensor = torch.tensor(data, dtype=torch.float32).to(device)

# Define the split index
split_idx = int(0.8 * data_tensor.shape[1])  # 80% for training

# Split data into training and testing sets
train_tensor = data_tensor[:, :split_idx, :]  # Shape: (10, 480, 2)
test_tensor = data_tensor[:, split_idx:, :]   # Shape: (10, 120, 2)

n_sequences = train_tensor.shape[1]


# Prepare labels for next choice prediction
# Shift the choice data by one step for each sequence in train_tensor
train_choices = train_tensor[:, :, 0]  # Assuming the choice is the first feature
train_rewards = train_tensor[:, :, 1]  # Assuming the reward is the second feature
#train_labels = train_choices[1:, :]    # Remove the first step for labels
#train_inputs = train_tensor[:-1, :, :]  # Remove the last step for inputs to align

xs = np.zeros((train_tensor.shape[0], train_tensor.shape[1], 2), dtype=np.float32)
ys = np.zeros((train_tensor.shape[0], train_tensor.shape[1], 1), dtype=np.float32)

for sess_i in range(n_sequences):
    prev_choices = np.concatenate(([0], train_choices[0:-1, sess_i]))
    prev_rewards = np.concatenate(([0], train_rewards[0:-1, sess_i]))
    xs[:, sess_i] = np.swapaxes(
    np.concatenate(([prev_choices], [prev_rewards]), axis=0), 0, 1)
    ys[:, sess_i] = np.expand_dims(train_choices[:, sess_i], 1)
    if sess_i == 0:
        print(f'previous choices shape: {prev_choices.shape}')
        print(f'current xs: {xs[:, sess_i]}')
        print(f'current ys: {ys[:, sess_i]}')
        print(f'xs shape: {xs.shape}')
        print(f'ys shape: {ys.shape}')

"""for sess_i in range(n_sequences):
    prev_choices = train_choices[:, sess_i].unsqueeze(1)
    prev_rewards = train_rewards[:, sess_i].unsqueeze(1)
    #print(prev_rewards.shape)
    #print(f'previous choices: {prev_choices.shape}')
    # Reshape tensors to (10, 1)

    # Concatenate along the last dimension
    xs[:, sess_i] = torch.cat((prev_choices, prev_rewards), dim=1)  # Shape: (10, 2)

    #xs[:, sess_i] = np.concatenate(prev_choices, prev_rewards)
    ys[:, sess_i] = np.expand_dims(train_choices[:, sess_i], 1)
    if sess_i == 0:
        print(f'previous choices shape: {prev_choices.shape}')
        print(f'current xs: {xs[:, sess_i]}')
        print(f'current ys: {ys[:, sess_i]}')
        print(f'xs shape: {xs.shape}')
        print(f'ys shape: {ys.shape}')"""

xs = torch.from_numpy(xs)
ys = torch.from_numpy(ys)
print(f'xs shape: {xs.shape}')
print(f'ys shape: {ys.shape}')


class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(RNN, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=False, nonlinearity='relu')
        self.fc = nn.Linear(hidden_size, num_classes)
        self.softmax = nn.LogSoftmax(dim=2)

    def forward(self, x):
        # set initial hidden states
        h0 = torch.zeros(self.num_layers, x.size(1), self.hidden_size).to(device)

        # forward
        out, _ = self.rnn(x, h0)

        out = self.fc(out)
        out = self.softmax(out)

        return out

model = RNN(input_size, hidden_size, num_layers, num_classes)

# Assuming:
# xs: shape (n_trials_per_block, n_sequences, input_size)  e.g., (10, 600, 2)
# ys: shape (n_trials_per_block, n_sequences, 1) 
# and hyperparameters: batch_size, num_epochs, etc.

criterion = nn.CrossEntropyLoss()
#optimizer = torch.optim.RMSprop(model.parameters(), lr=learning_rate)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


n_sequences = xs.shape[1]

for epoch in range(num_epochs):
    # Optionally shuffle the sequences for each epoch:
    permutation = torch.randperm(n_sequences)
    xs_shuffled = xs[:, permutation, :]
    ys_shuffled = ys[:, permutation, :]

    # Process in mini-batches:
    for i in range(0, n_sequences, batch_size):
        # Create the mini-batch
        batch_x = xs_shuffled[:, i:i+batch_size, :].to(device)  # Shape: (seq_length, batch_size, input_size)
        batch_y = ys_shuffled[:, i:i+batch_size, :].to(device)  # Shape: (seq_length, batch_size, 1)
        
        # Remove the last singleton dimension from labels and convert to Long for CrossEntropyLoss
        batch_y = batch_y.squeeze(-1).long()  # Now shape: (seq_length, batch_size)
        
        # Forward pass through the model
        outputs = model(batch_x)  # Expected shape: (seq_length, batch_size, num_classes)
        #print(outputs.shape)
        
        # Exclude the first time step from loss calculation
        outputs_valid = outputs[:]  # Shape: ((seq_length-1), batch_size, num_classes)
        labels_valid = batch_y[:]   # Shape: ((seq_length-1), batch_size)
        
        # Flatten outputs and labels for the loss function:
        loss = criterion(outputs_valid.view(-1, num_classes), labels_valid.view(-1))
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Print progress periodically
        if ((i // batch_size)) % 10 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")




# test

# Prepare labels for next choice prediction
test_choices = test_tensor[:, :, 0]    # Extract the choice feature
test_rewards = test_tensor[:, :, 1]    # Extract the reward feature
#test_labels = test_choices[:, :]      # Shift by one for the labels
#test_inputs = test_tensor[:-1, :, :]   # Shift inputs to align with labels

xs_test = np.zeros((test_tensor.shape[0], test_tensor.shape[1], 2), dtype=np.float32)
ys_test = np.zeros((test_tensor.shape[0], test_tensor.shape[1], 1), dtype=np.float32)

for sess_i in range(test_tensor.shape[1]):
    prev_choices = np.concatenate(([0], test_choices[0:-1, sess_i]))
    if sess_i == 0:
        print(f'previous choices shape: {prev_choices.shape}')
    prev_rewards = np.concatenate(([0], test_rewards[0:-1, sess_i]))
    xs_test[:, sess_i] = np.swapaxes(
    np.concatenate(([prev_choices], [prev_rewards]), axis=0), 0, 1)
    ys_test[:, sess_i] = np.expand_dims(test_choices[:, sess_i], 1)

"""for sess_i in range(test_tensor.shape[1]):
    prev_choices_test = test_choices[:, sess_i].unsqueeze(1)
    prev_rewards_test = test_rewards[:, sess_i].unsqueeze(1)
    # print(prev_rewards.shape)
    # print(f'previous choices: {prev_choices.shape}')
    # Reshape tensors to (10, 1)

    # Concatenate along the last dimension
    xs_test[:, sess_i] = torch.cat((prev_choices_test, prev_rewards_test), dim=1)  # Shape: (10, 2)

    # xs[:, sess_i] = np.concatenate(prev_choices, prev_rewards)
    ys_test[:, sess_i] = np.expand_dims(test_choices[:, sess_i], 1)"""

xs_test = torch.from_numpy(xs_test)
ys_test = torch.from_numpy(ys_test)

# Assume xs_test and ys_test have shapes:
# xs_test: (seq_length, n_test_sequences, input_size)
# ys_test: (seq_length, n_test_sequences, 1)

n_test_sequences = xs_test.shape[1]

all_predictions = []
all_labels = []

"""with torch.no_grad():
    for i in range(0, n_test_sequences, batch_size):
        # Create a mini-batch from test data
        batch_x = xs_test[:, i:i+batch_size, :].to(device)  # Shape: (seq_length, batch_size, input_size)
        batch_y = ys_test[:, i:i+batch_size, :].to(device)  # Shape: (seq_length, batch_size, 1)
        
        # Convert labels to proper shape and type for evaluation
        batch_y = batch_y.squeeze(-1).long()  # Now: (seq_length, batch_size)
        
        # Forward pass
        outputs = model(batch_x)  # Expected shape: (seq_length, batch_size, num_classes)
        
        # Exclude the first time step (dummy input) from evaluation
        valid_outputs = outputs[:]
        valid_labels = batch_y[:]
        
        # Get predictions: if using CrossEntropyLoss, model outputs logits
        if valid_outputs.ndim > 2 and valid_outputs.size(2) > 1:
            _, predictions = torch.softmax(valid_outputs, dim=2)  # shape: (seq_length-1, batch_size)
            #sample from predictions
            sampled_classes = torch.multinomial(predictions.view(-1, num_classes), 1).view(batch_x.shape[1], batch_x[0])  # Sample classes
        else:
            # For binary classification with a single logit output, apply threshold
            predictions = (valid_outputs > 0).int()
        
        all_predictions.append(predictions.cpu())
        all_labels.append(valid_labels.cpu())"""

"""    # Concatenate results from all batches
    predictions_tensor = torch.cat(all_predictions, dim=1)  # shape: (seq_length-1, total_test_sequences)
    labels_tensor = torch.cat(all_labels, dim=1)

    # Compute overall accuracy
    n_correct = (predictions_tensor == labels_tensor).sum().item()
    n_samples = labels_tensor.numel()
    accuracy = 100.0 * n_correct / n_samples

    print(f'Accuracy = {accuracy:.2f}%')"""


    ### forward simulation ###

import numpy as np
import torch
import torch.nn.functional as F  # for softmax
import pandas as pd

# Assume these hyperparameters (as in your training/simulation code)
n_participants = 44
n_blocks_pp = 20      # number of blocks per participant
n_trials_per_block = 10  # sequence length
input_size = 2        # [action, reward]
num_classes = 2       # two-armed bandit
innov_variance = 100  # as used in your agents
noise_variance = 10

# Ensure the model is in evaluation mode
model.eval()

# List to store simulation results
simulation_data = []

# Global state counter (optional)
global_trial = 0
with torch.no_grad():
    for participant in range(n_participants):
        for block in range(n_blocks_pp):
            # For each block, sample a new set of mean rewards (one per action) from N(0, innov_variance)
            mean_reward_block = np.random.normal(0, np.sqrt(innov_variance), 2)
            
            input_seq = torch.zeros(n_trials_per_block, 1, input_size, device=device)

            # Initialize Kalman filter variables for this sequence.
            # We treat each trial index as a "state" (0 to n_trials_per_block-1)
            n_states = n_trials_per_block
            V_t            = np.zeros(n_states)         # Difference between post_mean of arm 0 and 1
            RU             = np.zeros(n_states)  
            TU             = np.zeros(n_states) 
            post_mean      = np.zeros((2, n_states))      # Posterior mean for each arm
            post_variance  = np.ones((2, n_states)) * 5   # Posterior variance for each arm
            kalman_gain    = np.zeros((2, n_states))      # Kalman gain for each arm
            
            # Loop over trials in the block.
            # We start at trial 1 because trial 0 is the dummy input.
            for trial in range(n_trials_per_block):

                outputs = model(input_seq)  # shape: (n_trials_per_block, 1, num_classes)
                
                # Extract the logits at the current trial.
                logits_t = outputs[trial, 0, :]  # shape: (num_classes,)
                probs_t = F.softmax(logits_t, dim=0).detach().cpu().numpy()
                
                # Sample an action according to the predicted probability distribution.
                rnn_action = np.random.choice([0, 1], p=probs_t)
                
                # For each trial, rewards for both actions are drawn from N(mean_reward, noise_variance)
                reward_vector = np.random.normal(mean_reward_block, np.sqrt(noise_variance), 2)
                rnn_reward = reward_vector[rnn_action]
                                
                # Update the input sequence at the current trial with the observed [action, reward].
                input_seq[trial, 0, 0] = float(rnn_action)
                input_seq[trial, 0, 1] = float(rnn_reward)

                # --- Kalman Filter Update ---
                # (Assume the filter uses the previous trial's state to update the current one.)
                for i in range(2):  # for each arm (0 and 1)
                    if rnn_action == i:
                        # Update using the innovation:
                        kalman_gain[i][trial] = post_variance[i][trial - 1] / (post_variance[i][trial - 1] + noise_variance)
                        post_variance[i][trial] = (1 - kalman_gain[i][trial]) * post_variance[i][trial - 1]
                        post_mean[i][trial] = post_mean[i][trial - 1] + kalman_gain[i][trial] * (rnn_reward - post_mean[i][trial - 1])
                    else:
                        # For the unchosen action, carry the previous state forward.
                        post_variance[i][trial] = post_variance[i][trial - 1]
                        post_mean[i][trial] = post_mean[i][trial - 1]

                            # Compute derived Kalman filter variables.
                V_t[trial] = post_mean[0][trial] - post_mean[1][trial]
                sigma1 = post_variance[0][trial]
                sigma2 = post_variance[1][trial]
                std_dev = np.sqrt(sigma1 + sigma2)
                TU[trial] = np.sqrt(post_variance[0][trial] + post_variance[1][trial])
                RU[trial] = np.sqrt(post_variance[0][trial]) - np.sqrt(post_variance[1][trial])                

                # --- Record data ---
                simulation_data.append({
                    "Participant": participant,
                    "Block": block,
                    "Trial": trial,  # trial index (trial 0 is dummy)
                    "Global_Trial": global_trial,
                    "Mean_Reward_Arm0": mean_reward_block[0],
                    "Mean_Reward_Arm1": mean_reward_block[1],
                    # RNN outputs:
                    "Action": rnn_action,
                    "Reward": rnn_reward,
                    "RNN_Prob_Action0": probs_t[0],
                    # Kalman filter variables:
                    "V_t": V_t[trial],
                    "Kalman_post_mean_0": post_mean[0][trial],
                    "Kalman_post_mean_1": post_mean[1][trial],
                    "Kalman_post_variance_0": post_variance[0][trial],
                    "Kalman_post_variance_1": post_variance[1][trial],
                    "Kalman_kalman_gain_0": kalman_gain[0][trial],
                    "Kalman_kalman_gain_1": kalman_gain[1][trial],
                    "RU": RU[trial],
                    "TU": TU[trial]})
                
                global_trial += 1

                
    # Convert the simulation results to a DataFrame and (optionally) save to CSV.
    df_simulation = pd.DataFrame(simulation_data)
    df_simulation.to_csv("data/simulation_trained_network_hybrid.csv", index=False)

    print("Simulation complete. Data saved to data/simulation_trained_network.csv")





