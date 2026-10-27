# Setup Overview

## The idea in one breath

This section sets up the DP-SGD experiment: install the library, import the HTB helpers, and fix one shared config. Baseline and DP models get the **same** 20 epochs, batch size 256 and learning rate 0.1, so any accuracy difference comes from the privacy mechanism itself. Membership leakage is measured with a simple **confidence-threshold MIA** (`compute_mia_advantage`) on a **balanced 10,000 vs 10,000 pool**. Unlike Section 6, that makes 50% the honest baseline.

> **Heads-up:** no mangled formulas this time. One naming clash: the course's ==advantage = accuracy − 0.5==. Most papers define advantage = TPR − FPR, which is **2×** the course's number. Say which one you mean in a report.
> 
> **Checked against the library source (I read it, didn't run it):** everything below about what the helper functions do internally comes from the installed `htb_ai_library` code, not just the course prose.

---

## Part 1: Install and imports

### Step 1: Install

```bash
pip install --upgrade git+https://github.com/PandaSt0rm/htb-ai-library
```

> **If this errors:** if you get `ModuleNotFoundError: torchvision` (the library needs it for CIFAR-10) or later `No module named 'opacus'`, run `pip install torchvision opacus`. Neither is in the course's pip line.

### Step 2: Imports

```python
import os
import json
import torch
import torch.optim as optim
from safetensors.torch import save_file

from htb_ai_library import (
    set_reproducibility, use_htb_style,
    CIFAR10CNN,
    get_cifar10_loaders,
    train_baseline_sgd,
    train_dp_sgd,
    evaluate_accuracy,
    compute_mia_advantage,
    plot_accuracy_comparison,
    plot_privacy_utility_tradeoff,
)
```

|Code|What it does|
|---|---|
|`os`, `json`|create folders, save results|
|`torch.optim as optim`|the SGD optimizer that Opacus will wrap next section|
|`save_file` (safetensors)|saves weights in **safetensors** format. That's a hint: a typical **validation server takes a .safetensors model**|
|`set_reproducibility`, `use_htb_style`|seed everything; HTB plot colours|
|`CIFAR10CNN`|the DP-safe classifier (Part 2)|
|`get_cifar10_loaders`|data (Part 3)|
|`train_baseline_sgd`, `train_dp_sgd`|the two training loops (Part 4)|
|`evaluate_accuracy`, `compute_mia_advantage`|utility and privacy metrics (Part 5)|
|`plot_accuracy_comparison`, `plot_privacy_utility_tradeoff`|the final figures|

---

## Part 2: The model, `CIFAR10CNN`

From the source: three 3×3 conv layers (3->32->64->64), each followed by ReLU and 2×2 max-pool, then a 1024->64->10 fully connected head. There's **no BatchNorm and no dropout**.

|Name|Shape|Meaning|
|---|---|---|
|input|(N, 3, 32, 32)|RGB image|
|after block 1|(N, 32, 16, 16)|conv1 + pool halves 32->16|
|after block 2|(N, 64, 8, 8)||
|after block 3|(N, 64, 4, 4)||
|flattened|(N, 1024)|64 × 4 × 4|
|fc1|(N, 64)||
|output|(N, 10)|logits for 10 classes|

**Size: 122,570 parameters.** Plugging that into the Section 8 memory toy: 122,570 × 4 bytes ≈ 0.5 MB per gradient, × 256 per-sample copies ≈ **125 MB**. That's small, so it's fine on CPU or a modest GPU.

---

## Part 3: The data, `get_cifar10_loaders`

Returns **`(train_dataset, test_dataset, train_loader, test_loader)`**, in that order. The train loader shuffles, the test loader doesn't. The full signature also has `data_root="./data"`, which the course doesn't mention.

> **If this errors / is very slow:** it uses torchvision's CIFAR-10 download, the same slow Toronto server you hit in the DeepFool CIFAR sheet (~1 hour). If you still have the `cifar-10-batches-py` folder from then, pass `data_root=` pointing at the folder that _contains_ it, and it won't download again.

> **An honest note the course skips:** "sharing loaders between models would corrupt privacy budget calculations" isn't quite how it works. `make_private_with_epsilon` returns a **new** Poisson loader and leaves your plain loader untouched. What really must be fresh for each ε is the **model + optimizer + PrivacyEngine**, because each engine has its own accountant and a model can't be wrapped twice. Calling `get_cifar10_loaders()` per ε is still harmless and keeps things clean.

---

## Part 4: The two training loops

|Function|Signature (course)|What it really does (source)|Returns|
|---|---|---|---|
|`train_baseline_sgd`|`(model, train_loader, device, epochs, learning_rate, momentum)`|plain SGD (momentum defaults to 0.9), prints loss/acc each epoch|model|
|`train_dp_sgd`|`(model, train_loader, privacy_engine, optimizer, device, epochs, delta)`|the **same plain loop**, but on an already-wrapped model, optimizer and loader; prints `ε` each epoch via `privacy_engine.get_epsilon(delta)`|`(model, final_epsilon)`|

**`train_dp_sgd` doesn't create the privacy engine.** You call `make_private_with_epsilon` yourself first (next section), and then the ordinary loop becomes DP because the optimizer it calls is the wrapped one. The final ε it returns is the **spent** budget from Section 8's "report `get_epsilon`, not the target" rule.

---

## Part 5: The two metrics

### Step 1: `evaluate_accuracy`

Runs in eval mode with `no_grad`, and returns a percentage from 0 to 100 (not 0 to 1).

### Step 2: `compute_mia_advantage`, toy first

Three members with confidences [0.9, 0.8, 0.6] and three non-members with [0.7, 0.5, 0.4]. The rule is "member if conf ≥ t":

|Threshold t|Members right|Non-members right|Accuracy|
|---|---|---|---|
|0.8|2 (0.9, 0.8)|3|5/6 = 0.833|
|0.7|2|2 (0.7 wrongly called member)|4/6 = 0.667|
|0.6|3|2|5/6 = 0.833|

Best accuracy is 0.833, so the advantage is **0.333**.

### Step 3: What the real function does (source)

1. Gets the **max softmax confidence** for every train record (members) and every test record (non-members).
2. **Balances** by randomly picking n = min(50,000, 10,000) = **10,000** from each. Since 50/50 makes 0.5 the honest baseline, **this fixes the Section 6 base-rate trap.**
3. Tries 1,000 thresholds (percentiles of the pooled confidences) and keeps the **best accuracy**.
4. Returns `(best_accuracy, best_accuracy − 0.5)`.

> **An honest note the course skips:**
> 
> 1. The threshold is chosen on the same data it's scored on, so the result is slightly optimistic. With 20,000 points and only one threshold to pick, that's negligible.
> 2. It calls `np.random.seed(1337)`, which resets NumPy's global RNG as a side effect.
> 3. **Watch next section:** if they pass the **Poisson-wrapped** train loader into it, the "members" become the ~63% of records that loader happens to draw, some of them twice. You should pass the plain loader.
> 4. This is a **weak attack** (Section 7's lower bound). The shadow model from Sections 3-6, or LiRA, could find more.

---

## Part 6: Configuration

```python
RANDOM_SEED = 1337
BATCH_SIZE = 256
BASELINE_EPOCHS = 20
BASELINE_LR = 0.1
DP_EPOCHS = 20
DP_LR = 0.1
MAX_GRAD_NORM = 1.0
DELTA = 1e-5

set_reproducibility(RANDOM_SEED)
use_htb_style()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

os.makedirs("figs", exist_ok=True)
os.makedirs("output", exist_ok=True)
os.makedirs("models", exist_ok=True)
```

|Code|What it does|
|---|---|
|`BATCH_SIZE = 256`|gives q = 256/50,000 = 0.00512 (Section 8)|
|`BASELINE_* = DP_*`|**parallel conditions**, so the accuracy gap reflects the privacy mechanism|
|`MAX_GRAD_NORM = 1.0`|clipping norm C (Section 8's CIFAR default)|
|`DELTA = 1e-5`|below 1/50,000 = 2×10⁻⁵ (only by 2×, as noted before)|
|`device = ...`|GPU if available. DP-SGD is 2-5× slower, so CPU will take a while|
|`os.makedirs(...)`|creates the three output folders; `exist_ok=True` means no error if they already exist|

> **An honest note the course skips:** the prose names `OUTPUT_DIR`, `FIGS_DIR` and `MODELS_DIR`, but **the code never defines them**. It just uses the strings. If later code uses those names, you'll get a `NameError`. My first cell below fixes that.

---

## Reading the figure/output

There are no figures or output blocks yet. Training logs and plots come in the next sections.

---

## Copy-paste cells (mine, not the course's)

I didn't run these this time, but the expected values come from the library source I read.

### Define the missing folder names

Run this right after the config cell.

```python
FIGS_DIR, OUTPUT_DIR, MODELS_DIR = "figs", "output", "models"
```

Expected: no output, and later cells using these names won't crash.

### Check the model is DP-safe

Uses `CIFAR10CNN` and `torch` from the imports.

```python
from opacus.validators import ModuleValidator
m = CIFAR10CNN()
print("params:", sum(p.numel() for p in m.parameters()))
print("opacus errors:", ModuleValidator.validate(m, strict=False))
print("output shape:", m(torch.zeros(2, 3, 32, 32)).shape)
```

Expected: `params: 122570`, `opacus errors: []`, `output shape: torch.Size([2, 10])`.

### Check the data

Add `data_root="..."` if you're reusing an old download.

```python
train_dataset, test_dataset, train_loader, test_loader = get_cifar10_loaders(batch_size=BATCH_SIZE, download=True)
x, y = next(iter(train_loader))
print(len(train_dataset), len(test_dataset), x.shape, y[:10].tolist(), round(x.min().item(), 2), round(x.max().item(), 2))
```

Expected: `50000 10000 torch.Size([256, 3, 32, 32])`, ten class ids, and a **negative** min (meaning the images are normalized).

## Key facts to lock in

- **The baseline and DP models share the same 20 epochs, batch size 256 and learning rate 0.1** so that any accuracy difference reflects the privacy mechanism, not the training setup. The DP-specific settings are `MAX_GRAD_NORM = 1.0` and `DELTA = 1e-5`.
- **CIFAR10CNN is DP-safe because it uses only Conv2d, ReLU, MaxPool2d and Linear layers, with no BatchNorm**. It's small (122,570 parameters, 3 conv blocks then 1024->64->10), so per-sample gradients cost only ~125 MB at batch size 256.
- **`train_dp_sgd` is an ordinary training loop run on an already-wrapped model, optimizer and loader**. It doesn't make anything private by itself; `make_private_with_epsilon` has to be called first, and it returns the spent ε from `get_epsilon(delta)`.
- **`compute_mia_advantage` is a confidence-threshold MIA on a balanced 10,000 vs 10,000 pool**. It picks the best of 1,000 thresholds and returns (accuracy, accuracy − 0.5), so 0.5 is the honest no-leak baseline here.
- **The course's "advantage" (accuracy − 0.5) is half the paper definition (TPR − FPR)**. Always state which one a report number uses.
- **A confidence-threshold MIA is a weak, lower-bound measurement**. A low score doesn't prove privacy, and a shadow-model attack or LiRA could find more leakage.
- **Each ε setting needs a fresh model, optimizer and PrivacyEngine** because each engine owns one accountant. The course also reloads data per ε, which is harmless. Make sure the MIA gets the plain train loader, not the Poisson-wrapped one.
- **Method point:** **when you set up a privacy comparison, fix the seed, keep the training config identical across arms, balance the members and non-members, and save the model in the format the grader expects (here safetensors)**. Define every path variable your later code references.

---
