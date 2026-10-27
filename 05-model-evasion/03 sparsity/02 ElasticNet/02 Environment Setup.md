# Environment Setup

## The idea in one breath

Pure setup, no new attack math. You install the HTB library, fix the random seed, pick GPU or CPU, and build the victim model - the **same `MNISTClassifierWithDropout`** you attacked with DeepFool, trained to **98.89%** and cached to `output/mnist_target.pth` so you never retrain. The one wrinkle worth noticing: the `save_model` / `load_model` calls here are used **differently** from the caching you set up in the DeepFool cheatsheet.

> **Heads-up:** nothing mangled in this paste. Two small things to keep straight, both explained below: (1) the save/load call style here differs from your DeepFool notebook, and (2) the course mentions "batch norm statistics" in the checkpoint, which this particular model doesn't actually have.

---

## Part 1 - Why ElasticNet needs this much setup

### The idea

FGSM was one forward + one backward pass. ElasticNet is in a different league of cost:

|Attack|Compute per image|
|---|---|
|FGSM|1 forward + 1 backward|
|DeepFool|a few iterations|
|ElasticNet|**hundreds of FISTA iterations, nested inside 5-10 binary-search steps**|

So if one image takes hundreds of forward/backward passes and you run ~10 binary-search rounds, that's thousands of passes **per image**. That's why the section bothers with caching the model (don't waste minutes retraining) and detecting the GPU (don't run thousands of passes on CPU if you can help it).

The **HTB Evasion Library** (`htb_ai_library`) gives the same model, data loaders, training loop and plot theme across every attack, so you write attack logic, not boilerplate - exactly what you saw in DeepFool.

---

## Part 2 - Install and imports

### Block 1: install the library

```bash
# Install the AI Library (or update it)
pip install --upgrade git+https://github.com/PandaSt0rm/htb-ai-library
```

Same library as the DeepFool module. If you already ran DeepFool in this environment, it's installed; `--upgrade` just pulls any newer version.

### Block 2: imports and environment

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Import utilities from HTB Evasion Library
from htb_ai_library.utils import (
    set_reproducibility,
    save_model,
    load_model,
    HTB_GREEN,
    NODE_BLACK,
    HACKER_GREY,
    WHITE,
    AZURE,
    NUGGET_YELLOW,
    MALWARE_RED,
    VIVID_PURPLE,
    AQUAMARINE,
)
from htb_ai_library.data import get_mnist_loaders
from htb_ai_library.models import MNISTClassifierWithDropout
from htb_ai_library.training import train_model, evaluate_accuracy
from htb_ai_library.visualization import use_htb_style

# Apply HTB theme globally to all plots
use_htb_style()

# Set reproducibility
set_reproducibility(1337)

# Configure device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
if device.type == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")
```

|Code|What it does|
|---|---|
|`import torch / nn / F`|PyTorch core, layer types, and functions like softmax|
|`DataLoader`, `datasets`, `transforms`|feed data in batches (imported but the library's loader does the work)|
|`matplotlib.patches as patches`|lets later plots draw boxes/shapes on images|
|`from pathlib import Path`|tidy file paths (`Path("output")` instead of string juggling)|
|`warnings.filterwarnings('ignore')`|silences library deprecation chatter; cosmetic only|
|`set_reproducibility(1337)`|**fixes the seed for PyTorch, NumPy and CUDA** so runs match|
|the HTB colour names|the palette for the plots (green, black, red, etc.)|
|`get_mnist_loaders`, `MNISTClassifierWithDropout`, `train_model`, `evaluate_accuracy`|data, model, trainer, scorer|
|`use_htb_style()`|applies the HTB dark theme to every plot globally|
|`device = ... "cuda" ... else "cpu"`|**use the GPU if present, else CPU**|

Output:

```txt
Using device: cuda
GPU: NVIDIA GeForce RTX 5090
```

CPU is fine too - just slower for the thousands of passes Part 1 described.

---

## Part 3 - Build (or load) the target model

### Block 3: data, model, cache-or-train, evaluate

```python
# Get data loaders using library function
train_loader, test_loader = get_mnist_loaders(batch_size=128)
print(f"Training samples: {len(train_loader.dataset)}")
print(f"Test samples: {len(test_loader.dataset)}")

# Create output directory for saving models and results
output_dir = Path("output")
output_dir.mkdir(exist_ok=True)

# Define model checkpoint path in output directory
model_path = output_dir / "mnist_target.pth"

# Initialize model using MNISTClassifierWithDropout from library
model = MNISTClassifierWithDropout(num_classes=10).to(device)

# Check if trained model exists, otherwise train from scratch
if model_path.exists():
    print(f"\nLoading existing model from {model_path}")
    model = load_model(model, model_path, device)
else:
    print(f"\nNo existing model found. Training new model...")
    model = train_model(model, train_loader, test_loader, epochs=5, device=device)
    print(f"Saving trained model to {model_path}")
    save_model(model, model_path)

# Evaluate the trained model
accuracy = evaluate_accuracy(model, test_loader, device)
print(f"\nTest accuracy: {accuracy:.2f}%")
```

|Code|What it does|
|---|---|
|`get_mnist_loaders(batch_size=128)`|returns train and test loaders, 128 images per batch|
|`len(train_loader.dataset)`|number of images (60,000 train, 10,000 test)|
|`Path("output"); .mkdir(exist_ok=True)`|make the `output/` folder; don't error if it's already there|
|`model_path = output_dir / "mnist_target.pth"`|where the weights live (`/` joins paths with pathlib)|
|`MNISTClassifierWithDropout(num_classes=10).to(device)`|build the empty model, move it to GPU/CPU|
|`if model_path.exists(): load_model(model, model_path, device)`|**cache hit: pour saved weights into the model**|
|`else: train_model(...5 epochs...); save_model(...)`|**cache miss: train and save**|
|`evaluate_accuracy(model, test_loader, device)`|% of test images classified correctly|

Output (first run, training):

```txt
Training samples: 60000
Test samples: 10000

No existing model found. Training new model...
Epoch 1/5: Avg Loss = 0.2761, Test Accuracy = 98.07%
Epoch 2/5: Avg Loss = 0.0982, Test Accuracy = 98.64%
Epoch 3/5: Avg Loss = 0.0750, Test Accuracy = 98.75%
Epoch 4/5: Avg Loss = 0.0616, Test Accuracy = 98.90%
Epoch 5/5: Avg Loss = 0.0529, Test Accuracy = 98.89%
Saving trained model to output/mnist_target.pth

Test accuracy: 98.89%
```

Output (later runs, loading):

```txt
Training samples: 60000
Test samples: 10000

Loading existing model from output/mnist_target.pth

Test accuracy: 98.89%
```

> **If this errors:** a `weights_only` / unpickling error on load usually means a PyTorch version mismatch with the saved file - delete `output/mnist_target.pth` and let it retrain. A `size mismatch` means the model class changed since you saved; retrain the same way.

> **An honest note the course skips (save/load signatures):** here the course calls `load_model(model, model_path, device)` - you pass the **already-built model** and it fills in the weights - and `save_model(model, model_path)`. That's different from the caching in your DeepFool cheatsheet, which used a dictionary (`load_model(path)` returning `{'model': ...}`). Both are "the HTB library," but the call shape isn't the same. **Follow whatever the current notebook's library actually exposes, and don't copy the DeepFool caching cell into this one.** If `load_model(model, path, device)` errors about arguments, check the library's version.

---

## Part 4 - Memory and storage details

### The batching math (checked: correct)

|Quantity|Calculation|Result|
|---|---|---|
|One MNIST image|28 × 28 = 784 pixels × 4 bytes (float32)|3,136 bytes ≈ 3.06 KiB|
|A batch of 128|128 × 3,136|401,408 bytes ≈ 392 KiB (input data only)|
|Full per-batch (with gradients, activations, optimizer)| - |roughly 50-80 MB|
|Batch of 256|double the above|100-160 MB, steadier gradients, slower per step|

**128 is the balance**: big enough for stable gradient estimates, small enough to be quick. The 3.06 KiB comes from 3,136 / 1,024; the 392 KiB from 401,408 / 1,024.

### What's inside the `.pth` file

`save_model` writes a Python **pickle** holding an `OrderedDict` that maps parameter names (`conv1.weight`, `fc2.bias`, ...) to their tensor values. `load_model` deserializes it and restores **every weight down to the last floating-point bit**, so a loaded model is numerically identical to the one you trained - no retraining, perfect reproducibility.

> **An honest note the course skips (batch norm):** the course says the checkpoint stores "batch norm statistics." `MNISTClassifierWithDropout` uses **dropout, not batch normalization**, so its `state_dict` holds the conv filters and FC weights/biases but **no** batch-norm running mean/variance (and dropout itself has no saved parameters - it's just an on/off switch at train time). If you print the saved keys you'll see names like `conv1.weight`, not `bn1.running_mean`. Minor, but it's the kind of detail worth getting right for the report.

> **Pickle security tie-in (relevant):** because `.pth` is a pickle, **loading an untrusted checkpoint can run arbitrary code**. This is the same pickle risk from your _Pickles and Steganography_ / _AI Data Attacks_ material - only load model files you produced or fully trust. In practice, a planted `.pth` is a plausible foothold, so flag it if you see one.

---

## Copy-paste cells (mine, not the course's)

Run after the setup block. They use your real names: `model`, `test_loader`, `device`.

### Confirm the model is ready to attack

Puts it in eval mode (dropout OFF, mandatory before attacking) and counts parameters.

```python
model.eval() # dropout OFF for stable gradients
n_params = sum(p.numel() for p in model.parameters()) # total weights
print("eval mode (training flag should be False):", model.training)
print(f"parameter count: {n_params:,}")
```

Expected:

```txt
eval mode (training flag should be False): False
parameter count: 421,642
```

(The exact count depends on the library's layer sizes; the point is it prints a number and `training` is `False`.)

### Peek at the saved checkpoint's keys

Confirms what's actually stored - handy for the batch-norm note above.

```python
sd = torch.load(model_path, map_location=device) # raw saved dictionary
sd = sd.get("state_dict", sd) if isinstance(sd, dict) else sd # handle either save format
print("number of saved tensors:", len(sd))
print("first few keys:", list(sd.keys())[:6])
```

Expected (names will look like):

```txt
number of saved tensors: 8
first few keys: ['conv1.weight', 'conv1.bias', 'conv2.weight', 'conv2.bias', 'fc1.weight', 'fc1.bias']
```

No `running_mean` / `running_var` keys, which confirms there's no batch norm.

## Key facts to lock in

- **ElasticNet runs hundreds of FISTA iterations inside 5-10 binary-search steps per image**, thousands of forward/backward passes per example, which is why the setup caches the model and prefers a GPU - it's orders of magnitude costlier than FGSM's single step.
- **The victim is `MNISTClassifierWithDropout` trained to 98.89%** (9,889/10,000 correct), the same regularized CNN from the DeepFool module, giving a realistic, well-regularized target rather than a toy model.
- **Caching works by saving the model's weights to `output/mnist_target.pth` and reloading them instead of retraining**, so the first run is slow and every later run is instant and numerically identical.
- **A reloaded model is bit-for-bit identical because `load_model` restores every weight to the exact float value**, which is a stronger guarantee than `set_reproducibility` - the seed only makes _training_ deterministic, while reloading skips training entirely.
- **A batch of 128 MNIST images is ~392 KiB of input data (128 × 784 × 4 bytes)**, and ~50-80 MB once gradients, activations and optimizer state are added; doubling to 256 doubles memory for steadier but slower updates.
- **The `.pth` checkpoint is a Python pickle mapping parameter names to tensors**, and because pickles can execute code on load, you only ever load checkpoints you trust - the same pickle attack surface from the AI Data Attacks material.
- **This model uses dropout, not batch normalization**, so its saved `state_dict` contains conv and FC weights/biases but no batch-norm running statistics, contrary to the course's wording.
- **Method point:** **always call `model.eval()` before attacking** (dropout off -> stable, repeatable gradients), the same non-negotiable you learned for DeepFool; a model left in training mode gives random gradients and unreliable attacks.

