import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# -----------------------------
# Data Loading & Preprocessing (same as your original code)
# -----------------------------
hidden_size = 20
num_classes = 2
# Use a smaller number of epochs for cross validation:
num_epochs_cv = 1000  
num_epochs_full = 50000  # final training
batch_size = 240
learning_rate = 1e-3

input_size = 2
sequence_length = 10
num_layers = 1

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
data = np.stack((choices, rewards), axis=-1)  # (n_participants, n_blocks_pp, n_trials_per_block, 2)

data = data.reshape(-1, n_trials_per_block, 2)  # (600, 10, 2)
data = np.transpose(data, (1, 0, 2))  # (10, 600, 2)

data_tensor = torch.tensor(data, dtype=torch.float32).to(device)

# Split into training and test (we use training for CV)
split_idx = int(0.8 * data_tensor.shape[1])
train_tensor = data_tensor[:, :split_idx, :]  # (10, 480, 2)
test_tensor  = data_tensor[:, split_idx:, :]   # (10, 120, 2)

n_sequences = train_tensor.shape[1]

# Prepare training inputs (xs) and labels (ys)
xs = np.zeros((train_tensor.shape[0], train_tensor.shape[1], input_size), dtype=np.float32)
ys = np.zeros((train_tensor.shape[0], train_tensor.shape[1], 1), dtype=np.float32)

train_choices = train_tensor[:, :, 0].cpu().numpy()  # (10, n_sequences)
train_rewards = train_tensor[:, :, 1].cpu().numpy()    # (10, n_sequences)

for sess_i in range(n_sequences):
    prev_vectors = []
    # Dummy input for first trial
    prev_vectors.append([0, 0])
    for t in range(1, train_tensor.shape[0]):
        choice = train_choices[t-1, sess_i]
        reward = train_rewards[t-1, sess_i]
        if choice == 0:
            vector = [reward, 0]
        else:
            vector = [0, reward]
        prev_vectors.append(vector)
    xs[:, sess_i, :] = np.array(prev_vectors)
    ys[:, sess_i, 0] = train_choices[:, sess_i]

xs = torch.from_numpy(xs).to(device)
ys = torch.from_numpy(ys).to(device)

print(f'xs shape: {xs.shape}')  # Expected: (seq_length, n_sequences, 2)
print(f'ys shape: {ys.shape}')  # Expected: (seq_length, n_sequences, 1)

# -----------------------------
# 1. Compute Target Repetition Rate from Data
# -----------------------------
def compute_target_repetition(ys):
    """
    Compute the fraction of consecutive trials in which the same action occurred.
    """
    ys_np = ys.squeeze(-1).cpu().numpy().astype(int)  # shape: (seq_length, n_sequences)
    total_same = 0
    total_pairs = 0
    seq_length, n_seq = ys_np.shape
    for seq in range(n_seq):
        for t in range(1, seq_length):
            total_pairs += 1
            if ys_np[t, seq] == ys_np[t-1, seq]:
                total_same += 1
    return total_same / total_pairs if total_pairs > 0 else 0.0

target_repetition = compute_target_repetition(ys)
print(f"Target repetition rate in the data: {target_repetition:.3f}")

# -----------------------------
# 2. Define the Repetition Incentive Loss Function
# -----------------------------
def repetition_incentive_loss(logits, incentive_weight=0.1, target_repetition=target_repetition):
    """
    Computes a loss term that encourages the model's repetition (computed via softmax similarities)
    to match the target repetition rate observed in the data.
    """
    probs = F.softmax(logits, dim=-1)  # shape: (seq_len, batch_size, num_classes)
    seq_len = probs.shape[0]
    rep_sum = 0.0
    for t in range(1, seq_len):
        sim = torch.sum(probs[t] * probs[t-1], dim=-1)  # similarity per sequence
        rep_sum += sim.mean()
    avg_repetition = rep_sum / (seq_len - 1)
    loss_incentive = incentive_weight * (avg_repetition - target_repetition) ** 2
    return loss_incentive

# -----------------------------
# 3. Define the LSTM Model (unchanged)
# -----------------------------
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

criterion = nn.CrossEntropyLoss()

# -----------------------------
# 4. Cross Validation to Select the Best Incentive Weight
# -----------------------------
# Define candidate incentive weight values. (For example, 100 evenly spaced between 0 and 1)
candidate_incentive_weights = np.linspace(0, 0.5, 10)

k = 5  # number of folds
kf = KFold(n_splits=k, shuffle=True, random_state=42)
incentive_results = {}  # maps candidate_incentive_weight -> average CV loss

print("Starting cross validation...")
for incentive_weight in candidate_incentive_weights:
    fold_val_losses = []
    for train_idx, val_idx in kf.split(range(n_sequences)):
        # Create fold-specific datasets
        xs_train = xs[:, train_idx, :]
        ys_train = ys[:, train_idx, :]
        xs_val = xs[:, val_idx, :]
        ys_val = ys[:, val_idx, :]
        
        # Instantiate a fresh model for this fold
        model_cv = LSTMModel(input_size, hidden_size, num_layers, num_classes).to(device)
        optimizer_cv = torch.optim.Adam(model_cv.parameters(), lr=learning_rate)
        
        # Train for a reduced number of epochs for speed
        for epoch in range(num_epochs_cv):
            permutation = torch.randperm(xs_train.shape[1])
            xs_shuffled = xs_train[:, permutation, :]
            ys_shuffled = ys_train[:, permutation, :]
            model_cv.train()
            for i in range(0, xs_train.shape[1], batch_size):
                batch_x = xs_shuffled[:, i:i+batch_size, :].to(device)
                batch_y = ys_shuffled[:, i:i+batch_size, :].to(device)
                batch_y = batch_y.squeeze(-1).long()  # shape: (seq_length, batch_size)
                
                outputs = model_cv(batch_x)
                loss_ce = criterion(outputs.view(-1, num_classes), batch_y.view(-1))
                loss_incentive = repetition_incentive_loss(outputs,
                                                           incentive_weight=incentive_weight,
                                                           target_repetition=target_repetition)
                loss = loss_ce + loss_incentive
                
                optimizer_cv.zero_grad()
                loss.backward()
                optimizer_cv.step()
        
        # Evaluate on the validation fold
        model_cv.eval()
        with torch.no_grad():
            outputs_val = model_cv(xs_val)
            val_batch_y = ys_val.squeeze(-1).long()
            loss_ce_val = criterion(outputs_val.view(-1, num_classes), val_batch_y.view(-1))
            loss_incentive_val = repetition_incentive_loss(outputs_val,
                                                           incentive_weight=incentive_weight,
                                                           target_repetition=target_repetition)
            val_loss = loss_ce_val + loss_incentive_val
            fold_val_losses.append(val_loss.item())
    avg_val_loss = np.mean(fold_val_losses)
    incentive_results[incentive_weight] = avg_val_loss
    print(f"Incentive Weight: {incentive_weight:.3f} - Avg. CV Loss: {avg_val_loss:.4f}")

# Plot the CV loss as a function of incentive weight (line plot)
weights_sorted = sorted(incentive_results.keys())
losses_sorted = [incentive_results[w] for w in weights_sorted]

plt.figure(figsize=(8, 6))
plt.plot(weights_sorted, losses_sorted, marker='o', linestyle='-', color='blue')
plt.xlabel("Incentive Weight")
plt.ylabel("Average CV Loss")
plt.title("CV Loss vs. Incentive Weight")
plt.grid(True)
plt.show()

# Select the best incentive weight (lowest CV loss)
best_incentive_weight = min(incentive_results, key=incentive_results.get)
print(f"Selected Best Incentive Weight: {best_incentive_weight:.3f}")

# -----------------------------
# 5. Final Training on Full Training Set Using the Best Incentive Weight
# -----------------------------
model_final = LSTMModel(input_size, hidden_size, num_layers, num_classes).to(device)
optimizer_final = torch.optim.Adam(model_final.parameters(), lr=learning_rate)
loss_history = []
n_sequences_full = xs.shape[1]

for epoch in range(num_epochs_full):
    permutation = torch.randperm(n_sequences_full)
    xs_shuffled = xs[:, permutation, :]
    ys_shuffled = ys[:, permutation, :]
    model_final.train()
    epoch_loss = 0.0
    n_batches = 0
    for i in range(0, n_sequences_full, batch_size):
        batch_x = xs_shuffled[:, i:i+batch_size, :].to(device)
        batch_y = ys_shuffled[:, i:i+batch_size, :].to(device)
        batch_y = batch_y.squeeze(-1).long()
        
        outputs = model_final(batch_x)
        loss_ce = criterion(outputs.view(-1, num_classes), batch_y.view(-1))
        loss_incentive = repetition_incentive_loss(outputs,
                                                   incentive_weight=best_incentive_weight,
                                                   target_repetition=target_repetition)
        loss = loss_ce + loss_incentive
        
        optimizer_final.zero_grad()
        loss.backward()
        optimizer_final.step()
        
        epoch_loss += loss.item()
        n_batches += 1
    avg_epoch_loss = epoch_loss / n_batches
    loss_history.append(avg_epoch_loss)
    
    if epoch % 10 == 0:
        print(f"Final Training Epoch [{epoch+1}/{num_epochs_full}], Loss: {avg_epoch_loss:.4f}")

# Plot final training loss
plt.figure(figsize=(8, 5))
plt.plot(loss_history, label='Training Loss')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title(f"Training Loss (Best Incentive Weight = {best_incentive_weight:.3f})")
plt.legend()
plt.show()

# Sort the keys so that the line plot connects points in order.
penalty_weights = sorted(incentive_results.keys())
avg_losses = [incentive_results[p] for p in penalty_weights]

plt.figure(figsize=(8, 6))
plt.plot(penalty_weights, avg_losses, marker='o', linestyle='-', color='blue')
plt.xlabel("Penalty Weight")
plt.ylabel("Average CV Loss")
plt.title("Cross Validation Loss per Penalty Weight")
plt.grid(True)
plt.show()