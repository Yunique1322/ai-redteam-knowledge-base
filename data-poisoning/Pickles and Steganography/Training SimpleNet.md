# Training SimpleNet

To demonstrate the pickle/steganography attack you first need a legitimate model to poison. This section defines one, trains it briefly on throwaway data (just so the weights hold real values), and saves it. **The one attack-relevant trick is hidden in the architecture: a deliberately oversized layer whose only job is to be a big place to hide the payload later.**

## Define and build SimpleNet

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import os

# Seed for reproducibility
SEED = 1337
np.random.seed(SEED)
torch.manual_seed(SEED)


# Define a simple Neural Network
class SimpleNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)
        # Add a larger layer potentially suitable for steganography later
        self.large_layer = nn.Linear(hidden_size, hidden_size * 5)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        # Note: large_layer is defined but not used in forward pass for simplicity
        # In a real model, all layers would typically be used.
        return x


# Model parameters
input_dim = 10
hidden_dim = 64  # Increased hidden size for larger layers
output_dim = 1
target_model = SimpleNet(input_dim, hidden_dim, output_dim)
```

The real network is just `fc1 -> relu -> fc2`. **The `large_layer` (64 to 320) is defined but never called in `forward()`, so it does nothing to the model's output. It exists only to add a large parameter tensor to hide data in.** In a real attack you would use whatever big layer already exists in a genuine model; here they added one on purpose so there is room.

## Inspect the state_dict to find the best hiding spot

```python
print("SimpleNet model structure:")
print(target_model)
print("\nModel parameters (state_dict keys and initial values):")
for name, param in target_model.state_dict().items():
    print(f"  {name}: shape={param.shape}, numel={param.numel()}, dtype={param.dtype}")
    if param.numel() > 0:
        print(f"    Initial values (first 3): {param.flatten()[:3].tolist()}")
```

This lists every parameter tensor with its size. **You are shopping for the biggest tensor, because more numbers means more LSBs to hide bytes in.**
![Inspect the state dict to find the best hiding spot](/assets/Pasted%20image%2020260926015536.png)

**`large_layer.weight` has 20480 elements, by far the biggest, so that is the target tensor for hiding the payload.**

## Minimal training (just to populate the weights)

```python
# Generate dummy data
num_samples = 100
X_train = torch.randn(num_samples, input_dim)
true_weights = torch.randn(input_dim, output_dim)
y_train = X_train @ true_weights + torch.randn(num_samples, output_dim) * 0.5

# Prepare DataLoader
dataset = TensorDataset(X_train, y_train)
dataloader = DataLoader(dataset, batch_size=16)

# Loss and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(target_model.parameters(), lr=0.01)

# Simple training loop
num_epochs = 5 # Minimal training
print(f"\n'Training' the model for {num_epochs} epochs...")
target_model.train() # Set model to training mode
for epoch in range(num_epochs):
    epoch_loss = 0.0
    for inputs, targets in dataloader:
        optimizer.zero_grad()
        outputs = target_model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    print(f"  Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss/len(dataloader):.4f}")

print("Training complete.")
```

Nothing important here. **It is a throwaway regression on random data whose only purpose is to fill the weights with realistic trained values instead of the initial ones.** The task itself does not matter. Same five-step loop you already know (`zero_grad`, forward, loss, backward, step).

## Save the "legitimate" file

```python
legitimate_state_dict_file = "target_model.pth"

try:
    # Save the model's state dictionary. torch.save uses pickle internally.
    torch.save(target_model.state_dict(), legitimate_state_dict_file)
    print(f"\nLegitimate model state_dict saved to '{legitimate_state_dict_file}'.")
except Exception as e:
    print(f"\nError saving legitimate state_dict: {e}")
```

**This clean `target_model.pth` is the "before" file. The next section tampers with it: hides the payload in `large_layer.weight`'s LSBs and wraps it with a malicious `__reduce__` so it runs code on load.** Remember `torch.save` uses pickle, which is exactly the mechanism that makes the tampered version dangerous.

## How much can you hide (capacity)

The intuition, no need for the formula: **each float lends you `n` of its least significant bits, so total capacity is (number of floats × n) bits, divided by 8 for bytes.**

For `large_layer.weight` with 20480 elements and `NUM_LSB = 2` (the setting used next section): 20480 × 2 = 40960 bits, which is **5120 bytes, about 5 kB**. Plenty of room for a real payload. More LSBs per weight means more capacity but more numerical damage, so it is the same stealth-versus-strength dial as everywhere else.

