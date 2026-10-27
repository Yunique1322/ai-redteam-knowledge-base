# The Opacus Library

## The idea in one breath

**Opacus** is the PyTorch library that turns normal training into DP-SGD. You wrap your model, optimizer and data loader with a **PrivacyEngine**. **make_private_with_epsilon()** works out the noise multiplier for your target ε, and **get_epsilon()** tells you how much budget you've actually spent. The cost is real: DP-SGD needs **per-sample gradients** (memory grows with batch size, training is about 2-5× slower), **BatchNorm has to go**, and batches are drawn by **Poisson subsampling** (q = 256/50,000 = 0.00512).

> **Heads-up:** in `q=batch_size/dataset_size`, q is the **sampling rate**, the chance that each record lands in a given batch. `δ=10−5=1/100000` reads as **δ = 10⁻⁵ = 1/100,000**.

---

## Part 1: What Opacus does

### Step 1: The three wrapped pieces

|Piece|Normal PyTorch|After PrivacyEngine|
|---|---|---|
|model|computes one averaged gradient per parameter|hooks capture a **per-sample gradient** for every record|
|optimizer|steps with the averaged gradient|clips each sample to C, sums, adds noise N(0, (z·C)²), averages, then steps|
|data loader|shuffles, then takes fixed chunks|**Poisson subsampling** (random batch sizes)|

### Step 2: The two calls to remember

- `make_private_with_epsilon(...)`: you give it target ε, δ, epochs and `max_grad_norm`. The **privacy accountant** solves for the noise multiplier and returns the wrapped model, optimizer and loader.
- `get_epsilon(delta)`: the ε **actually spent so far**. If you train longer than the epochs you told it, this number goes past your target. **Report this value, not the target.**

---

## Part 2: Per-sample gradients and architecture limits

### Step 1: Why it's expensive (toy)

Normal backprop stores **1** gradient per parameter. Opacus stores **B** of them, one per sample, so each can be clipped on its own.

|Model size|One gradient|× 256 samples|
|---|---|---|
|1M parameters × 4 bytes|4 MB|**≈ 1 GB**|

So memory grows with batch size, and the course quotes **2-5× slower** training.

### Step 2: Why BatchNorm breaks DP (toy)

BatchNorm normalizes using the _batch_ mean. Take a batch of three values [1, 2, 6], with mean 3:

- Sample 1 becomes (1 − 3) = −2 (before scaling).
- Change sample 3 from 6 to 3. The mean drops to 2, so sample 1 becomes −1.

Sample 1's output, and therefore its gradient, **depends on the other samples**. You can't bound one record's influence by clipping it alone, and that breaks the per-sample independence DP-SGD needs.

### Step 3: The fixes

|Layer|Normalizes over|DP-safe?|
|---|---|---|
|BatchNorm|the whole batch|❌|
|GroupNorm|groups of channels **within one sample**|✅|
|LayerNorm|all features **within one sample**|✅|
|InstanceNorm|each channel **within one sample**|✅|

`ModuleValidator.validate(model)` lists the problems, and `ModuleValidator.fix(model)` swaps in safe layers automatically. Complex architectures may still need manual changes, for example custom layers that keep batch statistics, or attention that mixes samples.

> **An honest note the course skips:** GroupNorm with groups = channels normalizes each channel inside one sample. That's really InstanceNorm, not "similar to BatchNorm". Opacus's own `fix()` uses `min(32, channels)` groups. Either way, the swap can cost some accuracy on its own, separate from the noise.

---

## Part 3: Privacy amplification by subsampling

### Step 1: Toy, what Poisson batches look like

Each of the 50,000 records independently gets a 0.512% chance to join each batch:

- **Batch size:** 256 on average, with standard deviation √(256 × 0.995) ≈ **16**. The course's "240-270" is about ±1 std, and ~95% of batches fall within 224-288.
- **One epoch** ≈ 1/q ≈ 195 steps. How often a given record is used in that epoch:

|Times used in one epoch|Share of records|
|---|---|
|0 (skipped)|**≈ 36.8%**|
|1|≈ 36.8%|
|2|≈ 18.4%|
|3+|≈ 8%|

### Step 2: Why it helps privacy

The attacker doesn't know whether Alice was even _in_ a given batch. That uncertainty makes each step much cheaper in privacy, and it's the **privacy amplification** I flagged in Section 7. It's why the noise multiplier can stay around 0.5-1 instead of the ≈ 4.8 the closed-form formula gives.

### Step 3: The batch-size tug-of-war

|Batch size|Sampling rate q|Amplification|Steps per epoch|Net|
|---|---|---|---|---|
|Small (64)|low|strong|many (each costs budget)|noisy, unstable below 64|
|Large (1024)|high|weaker|few|the accountant decides|
|**128-512**|moderate|moderate|moderate|**usual sweet spot**|

---

## Part 4: Tuning the hyperparameters

### Step 1: The clipping threshold C (toy)

Take one sample's gradient with norm 1 and noise multiplier z = 1, so the noise std on the sum is z × C:

|C|Signal kept per sample|Noise std|Effect|
|---|---|---|---|
|10|1 (not clipped)|**10**|noise swamps signal|
|1|1|1|balanced|
|0.1|0.1 (clipped)|0.1|every gradient cut to the same length, so "how wrong was this sample" is lost and learning slows|

**Course rule:** train a few epochs without privacy, take the **75th percentile of the gradient norms**, and use that as C. CIFAR CNN norms are about 0.5-5, so **C = 1.0**.

> **An honest note the course skips:** the course says a too-big C "wastes privacy budget". It doesn't. **ε depends only on the noise multiplier z, the sampling rate q, the number of steps and δ. C isn't part of it.** A bad C wastes _utility_ (accuracy). Also, measuring gradient norms on the private data without DP technically leaks a little. In practice people tune C on public or similar data.

### Step 2: The other knobs

|Knob|Course guidance|Why|
|---|---|---|
|Learning rate|match or slightly undercut the baseline|the noise already weakens the signal; lower it only if training diverges|
|Momentum|works here (some practitioners
|Epochs|fewer epochs mean less noise per step at the same ε|if the model converges fast, stop early and keep the noise low|
|δ|δ < 1/n|10⁻⁵ vs 1/50,000 = 2×10⁻⁵ is only 2× below. "Comfortably" is generous; aim well below 1/n.|

---

## Part 5: What this adds to a report (mine)

- State the **architecture changes** (BatchNorm -> GroupNorm). Part of the accuracy drop comes from that, not from the privacy noise.
- Report **`get_epsilon(δ)` after training**, not the target ε. Also give C, z, q (or batch and n), epochs and the accountant used.
- If the client tuned C on the private data without DP, note it as a small extra leak.

---

## Reading the figure/output

There are no figures or code in this section. The Opacus code should show up next section.

---

## Copy-paste cells (mine, not the course's)

These are standalone. Per your request, I did **not** run them this time.

### Poisson batches and records skipped per epoch

```python
import numpy as np
rng = np.random.default_rng(0)
n, q = 50000, 256 / 50000
steps = int(1 / q) # ~one epoch = 195 steps
batch_sizes = rng.binomial(n, q, size=steps)
print("batch size mean/std/min/max:", batch_sizes.mean().round(1), batch_sizes.std().round(1), batch_sizes.min(), batch_sizes.max())
times_seen = rng.binomial(steps, q, size=n) # how often each record is used in one epoch
for k in range(4):
    print(f"seen {k} times this epoch: {np.mean(times_seen == k):.1%}")
```

Expected: mean ≈ 256, std ≈ 16; seen 0× ≈ 37%, 1× ≈ 37%, 2× ≈ 18%, 3× ≈ 6%.

### Catch and fix BatchNorm with Opacus

```python
import torch.nn as nn
from opacus.validators import ModuleValidator
tiny = nn.Sequential(nn.Conv2d(3, 16, 3), nn.BatchNorm2d(16), nn.ReLU())
print("errors before:", ModuleValidator.validate(tiny, strict=False))
tiny = ModuleValidator.fix(tiny)
print("errors after :", ModuleValidator.validate(tiny, strict=False))
print(tiny)
```

Expected: one BatchNorm error before, `[]` after, and the printout shows `GroupNorm(16, 16, ...)` where BatchNorm was.

## Key facts to lock in

- **Opacus implements DP-SGD by wrapping the model, optimizer and data loader in a PrivacyEngine**. `make_private_with_epsilon()` solves for the noise multiplier from target ε, δ, epochs and `max_grad_norm`, and `get_epsilon(δ)` reports the budget actually spent.
- **DP-SGD needs per-sample gradients**, so Opacus stores B gradients per parameter instead of 1. Memory grows with batch size (≈1 GB for a 1M-parameter model at B = 256) and training is roughly 2-5× slower.
- **BatchNorm is incompatible with DP-SGD because it couples samples through the batch statistics**. Replace it with GroupNorm, LayerNorm or InstanceNorm, or let `ModuleValidator.fix()` swap it.
- **Poisson subsampling includes each record independently with probability q (0.00512 for CIFAR-10)**, so batch sizes vary (256 ± 16). About 37% of records are skipped in any given epoch, and the attacker's uncertainty about batch membership amplifies privacy.
- **ε depends on the noise multiplier, sampling rate, number of steps and δ, not on the clipping norm C**. C only trades utility: too big and noise swamps the signal, too small and every gradient is cut to the same length. The course calibrates C to the 75th percentile of non-private gradient norms (≈1.0 for CIFAR CNNs).
- **More epochs or more steps force more noise per step at the same ε**, so if the model converges early, fewer epochs with less noise usually give better accuracy. Moderate batches (128-512) are the usual sweet spot, and below 64 training becomes unstable.
- **δ must be below 1/n, and ideally well below**. For CIFAR-10, 1/50,000 = 2×10⁻⁵, so δ = 10⁻⁵ passes, but only by 2×.
- **Method point:** **when reviewing or recommending DP-SGD, check the architecture (no BatchNorm), the δ vs 1/n check, and the spent `get_epsilon` rather than the target**, and give C, z, q and epochs. Attribute part of any accuracy drop to the normalization swap, not just the noise.

---
