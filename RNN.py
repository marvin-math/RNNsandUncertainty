import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import torch.nn.functional as F  # for softmax
from sklearn.model_selection import KFold  # for k-fold cross validation
import copy

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyper parameters
hidden_size = 20
num_classes = 2
# Use a smaller number of epochs during CV to save time:
num_epochs_cv = 200  
num_epochs_full = 50000  # full training epochs for final training
batch_size = 80
learning_rate = 1e-3

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

# Split into training and test tensors
split_idx = int(0.8 * data_tensor.shape[1])
train_tensor = data_tensor[:, :split_idx, :]  # (10, 480, 2)
test_tensor = data_tensor[:, split_idx:, :]   # (10, 120, 2)

n_sequences = train_tensor.shape[1]

# Prepare labels and inputs for next choice prediction
# Build inputs such that the reward is multiplied with the one-hot action vector.
xs = np.zeros((train_tensor.shape[0], train_tensor.shape[1], input_size), dtype=np.float32)
ys = np.zeros((train_tensor.shape[0], train_tensor.shape[1], 1), dtype=np.float32)

train_choices = train_tensor[:, :, 0].cpu().numpy()  # shape: (10, n_sequences)
train_rewards = train_tensor[:, :, 1].cpu().numpy()    # shape: (10, n_sequences)

for sess_i in range(n_sequences):
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

# Define the repetition penalty function.
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
    probs = F.softmax(logits, dim=-1)  # Shape: (seq_len, batch_size, num_classes)
    seq_len = probs.shape[0]
    
    penalty = 0.0
    # For each consecutive pair of time steps, compute a measure of similarity.
    for t in range(1, seq_len):
        sim = torch.sum(probs[t] * probs[t-1], dim=-1)  # (batch_size,)
        penalty += sim.mean()
    
    return penalty_weight * penalty

criterion = nn.CrossEntropyLoss()

# ======== K-Fold Cross Validation to select penalty_weight ========
candidate_penalties = np.linspace(0, 1, 100)
k = 5
kf = KFold(n_splits=k, shuffle=True, random_state=42)
penalty_results = {}

# For each candidate penalty weight, we will average the validation loss over the k folds.
for penalty_weight in candidate_penalties:
    val_losses = []
    print(f"\nEvaluating penalty_weight = {penalty_weight}")
    for fold, (train_idx, val_idx) in enumerate(kf.split(range(n_sequences))):
        # Create fold-specific training and validation sets
        xs_train = xs[:, train_idx, :]
        ys_train = ys[:, train_idx, :]
        xs_val = xs[:, val_idx, :]
        ys_val = ys[:, val_idx, :]

        # Create a new model instance for each fold
        model_cv = LSTMModel(input_size, hidden_size, num_layers, num_classes).to(device)
        optimizer_cv = torch.optim.Adam(model_cv.parameters(), lr=learning_rate)
        
        # (Optionally) record loss for each epoch in this fold
        # Here we train for a smaller number of epochs for speed.
        for epoch in range(num_epochs_cv):
            permutation = torch.randperm(xs_train.shape[1])
            xs_shuffled = xs_train[:, permutation, :]
            ys_shuffled = ys_train[:, permutation, :]
            
            model_cv.train()
            for i in range(0, xs_train.shape[1], batch_size):
                batch_x = xs_shuffled[:, i:i+batch_size, :]
                batch_y = ys_shuffled[:, i:i+batch_size, :]
                batch_y = batch_y.squeeze(-1).long()  # (seq_length, batch_size)
                
                outputs = model_cv(batch_x)  # (seq_length, batch_size, num_classes)
                loss_ce = criterion(outputs.view(-1, num_classes), batch_y.view(-1))
                penalty = repetition_penalty(outputs, penalty_weight=penalty_weight)
                loss = loss_ce + penalty
                
                optimizer_cv.zero_grad()
                loss.backward()
                optimizer_cv.step()
        
        # After training, evaluate on the validation fold.
        model_cv.eval()
        with torch.no_grad():
            outputs_val = model_cv(xs_val)
            val_batch_y = ys_val.squeeze(-1).long()
            loss_ce_val = criterion(outputs_val.view(-1, num_classes), val_batch_y.view(-1))
            penalty_val = repetition_penalty(outputs_val, penalty_weight=penalty_weight)
            val_loss = loss_ce_val + penalty_val
            val_losses.append(val_loss.item())
            print(f"  Fold {fold+1}: Validation Loss = {val_loss.item():.4f}")
    
    avg_val_loss = np.mean(val_losses)
    penalty_results[penalty_weight] = avg_val_loss
    print(f"Average validation loss for penalty_weight {penalty_weight}: {avg_val_loss:.4f}")

# Select the penalty weight with the lowest average validation loss.
best_penalty = min(penalty_results, key=penalty_results.get)
print(f"\nBest penalty_weight selected: {best_penalty}")

# ======== Train final model on the full training set using best_penalty ========
model = LSTMModel(input_size, hidden_size, num_layers, num_classes).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# We will store training loss for plotting.
loss_history = []

n_sequences_full = xs.shape[1]
for epoch in range(num_epochs_full):
    permutation = torch.randperm(n_sequences_full)
    xs_shuffled = xs[:, permutation, :]
    ys_shuffled = ys[:, permutation, :]
    
    model.train()
    epoch_loss = 0.0
    n_batches = 0
    for i in range(0, n_sequences_full, batch_size):
        batch_x = xs_shuffled[:, i:i+batch_size, :].to(device)
        batch_y = ys_shuffled[:, i:i+batch_size, :].to(device)
        batch_y = batch_y.squeeze(-1).long()  # (seq_length, batch_size)
        
        outputs = model(batch_x)  # (seq_length, batch_size, num_classes)
        loss_ce = criterion(outputs.view(-1, num_classes), batch_y.view(-1))
        penalty = repetition_penalty(outputs, penalty_weight=best_penalty)
        loss = loss_ce + penalty
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
        n_batches += 1
    
    avg_epoch_loss = epoch_loss / n_batches
    loss_history.append(avg_epoch_loss)
    
    if epoch % 10 == 0:
        print(f"Epoch [{epoch+1}/{num_epochs_full}], Loss: {avg_epoch_loss:.4f}")

# Plot the training loss curve.
plt.figure(figsize=(8, 5))
plt.plot(loss_history, label='Training Loss')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title(f"Training Loss Curve (penalty_weight = {best_penalty})")
plt.legend()
plt.show()

# Sort the keys so that the line plot connects points in order.
penalty_weights = sorted(penalty_results.keys())
avg_losses = [penalty_results[p] for p in penalty_weights]

plt.figure(figsize=(8, 6))
plt.plot(penalty_weights, avg_losses, marker='o', linestyle='-', color='blue')
plt.xlabel("Penalty Weight")
plt.ylabel("Average CV Loss")
plt.title("Cross Validation Loss per Penalty Weight")
plt.grid(True)
plt.show()
