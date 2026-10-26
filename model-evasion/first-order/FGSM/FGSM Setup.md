# FGSM Setup


## The idea in one breath

Before attacking, you need a **victim**: a small image classifier (**SimpleCNN**) trained on MNIST digits to about 98% accuracy, set up so every run gives the same numbers (**reproducibility**). This section is plumbing: install the library, lock the randomness, load the data, build the model, train it once, and measure the clean accuracy. That clean accuracy (98.41%) is the "before" number the FGSM attack will knock down.

## Part 1: Install the HTB library

```bash
# Install the AI Library (or update it)
pip install --upgrade git+https://github.com/PandaSt0rm/htb-ai-library
```

- `pip install` installs a Python package.
- `--upgrade` replaces an older version if one is already installed.
- `git+https://github.com/...` installs straight from a GitHub repository instead of the usual package index.

The library bundles the boring parts (data loading, model, training) so the notebook can focus on attacks.

## Part 2: Imports and reproducibility

```python
import os
import random
import numpy as np
import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Import common utilities from HTB Evasion Library
from htb_ai_library import (
    set_reproducibility,
    SimpleCNN,
    get_mnist_loaders,
    mnist_denormalize,
    train_model,
    evaluate_accuracy
)

# Configure reproducibility
set_reproducibility(1337)
```

### What each import is for

|Import|What it is|Used for|
|---|---|---|
|`os`, `random`|Python built-ins|environment variables, Python's random numbers|
|`numpy as np`|number-array library|math on arrays|
|`torch`|PyTorch|tensors (multi-dimensional number arrays), gradients|
|`nn`|PyTorch's neural-network building blocks|layers, loss functions|
|`Tensor`|the tensor type|type hints in function definitions|
|`F` (`torch.nn.functional`)|functions like `cross_entropy`, `relu`|computing the loss in the attack|
|`DataLoader`|feeds data in batches|looping over images 128 at a time|
|`datasets`, `transforms`|torchvision helpers|downloading MNIST, converting images|
|`set_reproducibility`|HTB helper|locks the randomness|
|`SimpleCNN`|HTB helper|the victim model|
|`get_mnist_loaders`|HTB helper|ready-made train/test data|
|`mnist_denormalize`|HTB helper|undoes normalization (important in Section 5)|
|`train_model`, `evaluate_accuracy`|HTB helpers|training and scoring|

### What `set_reproducibility(1337)` locks down

A **seed** is a starting number for a random-number generator. The same seed gives the same "random" sequence every time, so `1337` makes every run identical. The section names three sources of randomness it controls:

|Source|What could vary without it|Fix|
|---|---|---|
|Python hashing|dictionary/set ordering|`PYTHONHASHSEED` environment variable|
|`random` and NumPy|shuffles, random picks|both seeded with 1337|
|cuDNN (GPU maths)|which algorithm runs, which causes tiny number differences|switched to **deterministic** mode (slightly slower)|

**Why it matters for attacks:** if gradients change a little between runs, you can't tell whether an attack improved or you just got lucky. **Reproducibility makes before/after comparisons meaningful.**

## Part 3: Device

```python
# Configure computation device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

This reads as: "use the NVIDIA GPU (`cuda`) if one is available, otherwise use the CPU." MNIST is small, so the CPU is fine.

**Common error:** `Expected all tensors to be on the same device`. It happens when the model is on one device and the images are on another. The fix is `.to(device)` on both. That's why you'll see `.to(device)` everywhere.

## Part 4: Data loading

```python
# Prepare data loaders using library function (normalized space)
train_loader, test_loader = get_mnist_loaders(batch_size=128, normalize=True)
```

### Toy first: what happens to one pixel

An MNIST pixel starts as a whole number from 0 (black) to 255 (white). `ToTensor()` divides by 255:

```
pixel 0 → 0/255 = 0.0
pixel 128 → 128/255 ≈ 0.502
pixel 255 → 255/255 = 1.0
```

So every pixel ends up between **0 and 1**. With `normalize=True`, a second step then shifts and rescales each pixel into a different range (the **normalized space**). Section 5 is all about this, and it's the trap I flagged in the module overview. For now, remember: **the model sees normalized pixels, not the plain 0-1 values**, which is why `mnist_denormalize` is imported.

### The rest of the line

|Piece|Meaning|
|---|---|
|`batch_size=128`|the model sees 128 images at a time|
|`train_loader`|60,000 training images, shuffled with seed 1337, so batch 0 is always the same 128 images|
|`test_loader`|10,000 test images|
|downloads to `./data`|only the first time you run it|
|`num_workers=0`|loads data in the main process (no parallel-loading surprises)|

Each batch gives you `images` with shape **[128, 1, 28, 28]** and `labels` with shape **[128]**. Reading the image shape: 128 images × 1 colour channel (greyscale) × 28 rows × 28 columns.

## Part 5: The model, SimpleCNN

```python
# Initialize model using library's SimpleCNN
model = SimpleCNN().to(device)
```

### Toy first: the three building blocks

**1. Convolution (a sliding 3×3 stencil).** A **filter** is a small 3×3 grid of weights. It slides across the image, and at each spot it does our old friend the **dot product** (multiply pair by pair, then add):

```
3×3 patch of image 3×3 filter result
[0 0 1] [0 1 0]
[0 1 1] · [0 1 0] → 0+0+0 + 0+1+0 + 0+1+0 = 2
[0 1 0] [0 1 0]
```

This filter scores high wherever there's a vertical stroke, like the stem of a "1" or "7". **A convolution layer is many of these stencils, each learning to spot one pattern.**

**Padding = 1** adds a border of zeros around the image, so the stencil can sit on edge pixels and the output stays the same size (28×28 in, 28×28 out).

**2. ReLU**: `ReLU(v) = max(0, v)`. Negative numbers become 0 and positive numbers pass through: ReLU(−3) = 0, ReLU(2) = 2. This is the "nonlinearity" that lets the network learn more than straight lines.

**3. Max-pooling (2×2, stride 2)** keeps the biggest number in each 2×2 block, which halves the width and height:

```
[1 3]
[2 0] → 3
```

### The real shape flow

|Stage|What happens|Shape (channels × height × width)|
|---|---|---|
|Input|one greyscale digit|1 × 28 × 28|
|conv1 + ReLU|32 filters of 3×3, padding 1|32 × 28 × 28|
|max-pool|halve|32 × 14 × 14|
|conv2 + ReLU|64 filters of 3×3|64 × 14 × 14|
|max-pool|halve|64 × 7 × 7|
|`view` (flatten)|lay everything out in one long line|64 × 7 × 7 = **3136**|
|fully connected layers|two layers, the last outputs 10 numbers|**10 logits**|

The final **10 logits** are raw scores, one per digit 0-9. The prediction is whichever score is highest (**argmax**):

```
logits = [0.1, −1.2, 0.3, 0.0, −0.5, 0.2, −0.8, 6.4, 0.9, 1.1]
argmax = 7 (index 7 has the biggest score, 6.4)
```

`.to(device)` moves all the model's weights onto the chosen device.

## Part 6: Training and evaluation

```python
# Train the model using library function
trained_model = train_model(model, train_loader, test_loader, epochs=1, device=device)

# Evaluate baseline accuracy using library function
baseline_acc = evaluate_accuracy(trained_model, test_loader, device)
print(f"Baseline test accuracy: {baseline_acc:.2f}%")
```

Output:

```txt
Epoch 1/1: Avg Loss = 0.1566, Test Accuracy = 98.41%
Baseline test accuracy: 98.41%
```

### What `train_model` does

- **One epoch** means one full pass over all 60,000 training images. That's enough for MNIST, which gets to about 98-99%.
- `model.train()` switches on training behaviour (dropout, batch-norm updates, if the model has them).
- **Adam, learning rate 0.001**, is the optimizer: it nudges the weights downhill on the loss (Section 1's "training changes weights").
- `zero_grad(set_to_none=True)` clears the previous batch's gradients before computing new ones. Otherwise PyTorch **adds** new gradients onto the old ones.
- **Loss weighted by batch size.** Toy check: 60,000 ÷ 128 = 468 full batches plus a last batch of only **96** images. Say a full batch has loss 0.2 and the small last batch has loss 0.5:

```
Plain average of the two: (0.2 + 0.5) ÷ 2 = 0.350 ← overweights the small batch
Weighted by size: (0.2×128 + 0.5×96) ÷ (128 + 96) = 73.6 ÷ 224 = 0.329
```

Weighting by size gives each **image** an equal vote, not each batch.

### What `evaluate_accuracy` does differently

|Setting|Training|Evaluation|**During an FGSM attack**|
|---|---|---|---|
|Mode|`model.train()`|`model.eval()`|`model.eval()`|
|Gradients|on|off: `torch.no_grad()`|****on**, because you need them for the input**|
|Weights change?|yes|no|**no** (frozen)|

**For attacks you use `model.eval()` but NOT `torch.no_grad()`.** You still need the gradient with respect to the pixels, you just never update the weights. Wrapping attack code in `no_grad()` is the classic bug: `x.grad` comes back `None`.

**Reading the result:** 98.41% of 10,000 test images is **9,841 correct and 159 wrong**. That's the "accuracy looks great" baseline that FGSM is about to break.

## Optional sanity-check cells (mine, not the course's)

These use only names you've already defined. Run them after the training cell.

### Check one batch's shape

Confirms the data looks the way Part 4 says.

```python
images, labels = next(iter(test_loader))
print(images.shape, labels.shape)
```

Expected: `torch.Size([128, 1, 28, 28]) torch.Size([128])`

### Check the normalized pixel range

Shows the model is not seeing plain 0-1 pixels. This sets up Section 5.

```python
print(images.min().item(), images.max().item())
```

Expected, if the library uses the standard MNIST mean and std: about `-0.42` and `2.82`. Anything outside 0-1 confirms you're in normalized space.

### Check the model outputs 10 logits per image

```python
trained_model.eval()
with torch.no_grad():
    logits = trained_model(images[:1].to(device))
print(logits.shape, logits.argmax(dim=1).item(), labels[0].item())
```

Expected: `torch.Size([1, 10])`, then the predicted digit and the true digit (they'll almost always match). `no_grad()` is fine here because we're only predicting, not attacking.

### Read the library's own code

Useful in labs and on the exam when a helper does something you need to know exactly.

```python
import inspect
print(inspect.getsource(set_reproducibility))
```

Expected: the function's source code printed out, so you can see every seed it sets.

## Quick self-check

Suppose SimpleCNN's two conv layers had **16** and **32** filters instead of 32 and 64, and everything else stayed the same. What would the flattened vector size be (the number that's 3136 in the real model)?

Reply with your answer and I'll check it.

## The facts worth locking in for the exam

- **`set_reproducibility(1337)` seeds Python's hashing, `random` and NumPy, and puts cuDNN in deterministic mode**, so gradients and results are identical across runs and attack comparisons are meaningful.
- ==`get_mnist_loaders(batch_size=128, normalize=True)` first scales pixels from 0-255 to 0-1 with `ToTensor()`, then normalizes them==. The model is trained on normalized inputs, so attacks must handle the conversion between pixel space and normalized space (Section 5).
- **SimpleCNN goes 1×28×28 → conv1 (32 filters) → pool → 14×14 → conv2 (64 filters) → pool → 64×7×7, which flattens to 3136 and ends in 10 logits**. The prediction is the argmax of those 10 raw scores.
- **`model.eval()` with `torch.no_grad()` is for clean evaluation only**. An attack keeps `model.eval()` but must leave gradient tracking on, because it needs the gradient with respect to the input pixels, while the weights stay frozen.
- **The model and the data must be on the same device (`.to(device)`)**, or PyTorch raises an "Expected all tensors to be on the same device" error.
- **One epoch gives a 98.41% clean baseline (9,841 of 10,000 test images correct)**, which shows that high clean accuracy says nothing about adversarial robustness.
- **Method point:** **always record the clean baseline accuracy before attacking**, because attack success is reported as the drop from that number.