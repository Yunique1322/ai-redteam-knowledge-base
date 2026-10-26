# Implementing FISTA Components

## The idea in one breath

Two small functions. The first, `compute_fista_momentum`, is the one-liner `k/(k+3)` - **momentum that starts cautious (0.25) and grows toward 1**. The second, `apply_shrinkage_thresholding`, is the soft-threshold from Section 4 turned into code: it **zeros every pixel change at or below the threshold and shrinks the rest by the threshold**. The demo on a 5×5 pattern: β = 0.1 turns 22 non-zero pixels into just 9 (64% sparse).

> **Heads-up: mangled formulas.** Read them as:
> 
> - `tk=k/(k+3)` → t_k = k / (k + 3).
> - `tk+1=(1+1+4tk2)/2` → t_{k+1} = ( 1 + √(1 + 4·t_k²) ) / 2 (the exact FISTA recursion, approximated by k/(k+3)).
> - `O(1/k2)` → O(1/k²).
> - `|p|≤0.1` → the dead-zone rule: changes with absolute value ≤ the threshold are zeroed.

---

## Part 1 - Nesterov momentum

### The idea

Momentum is "keep some of the direction you were already going." Section 4 showed FISTA uses a momentum coefficient that **grows with the iteration**: small early (don't over-commit when you know nothing), large late (accelerate once a direction has proven itself).

### The code

```python
def compute_fista_momentum(iteration):
    """
    Calculate FISTA momentum parameter for iteration k.

    Uses Nesterov acceleration: k/(k+3)
    Early iterations have small momentum, later ones accelerate.

    Parameters:
        iteration (int): Current FISTA iteration number

    Returns:
        float: Momentum coefficient in [0, 1)
    """
    return iteration / (iteration + 3.0)
```

|Code|What it does|
|---|---|
|`iteration / (iteration + 3.0)`|the momentum coefficient for step k|

### Why `+ 3.0` in the denominator

It **keeps the result below 1** (it only approaches 1 as k → ∞). Without it, momentum could grow without bound and overwhelm the gradient signal, destabilizing the whole optimization. The `3.0` also makes the function well-defined at k = 0 (gives 0, no momentum on the first move).

### The progression (verified)

```python
# Test momentum progression
print("FISTA Momentum Progression:")
print("=" * 40)
test_iterations = [1, 5, 10, 50, 100, 500, 1000]

for k in test_iterations:
    momentum = compute_fista_momentum(k)
    print(f"Iteration {k:4d}: momentum = {momentum:.4f}")
```

Output:

```txt
FISTA Momentum Progression:
========================================
Iteration 1: momentum = 0.2500
Iteration 5: momentum = 0.6250
Iteration 10: momentum = 0.7692
Iteration 50: momentum = 0.9434
Iteration 100: momentum = 0.9709
Iteration 500: momentum = 0.9940
Iteration 1000: momentum = 0.9970
```

|Iteration|Momentum|Phase|
|---|---|---|
|1|0.25|cautious - let the gradient pick the direction|
|10|0.77|confident - directions have proven productive|
|100|0.97|aggressive - final convergence, strong acceleration|
|1000|0.997|nearly full momentum|

The contrast the course draws: **standard momentum uses a fixed coefficient (often 0.9) that needs tuning**; Nesterov's schedule auto-adjusts with the iteration, removing that hyperparameter. (As the section admits, using `k/(k+3)` is a heuristic approximation of the exact O(1/k²) scheme from Section 4.)

---

## Part 2 - Soft thresholding in code

### The code

```python
def apply_shrinkage_thresholding(y, original_images, threshold, clip_min=0.0, clip_max=1.0):
    """
    Apply soft thresholding to create sparse perturbations.

    Values with magnitude at most threshold are zeroed (sparsity).
    Values exceeding the threshold are reduced by the threshold but remain non-zero.

    Parameters:
        y (torch.Tensor): Candidate adversarial images after gradient step
        original_images (torch.Tensor): Original unperturbed images
        threshold (float): Soft thresholding parameter (higher = sparser)
        clip_min (float): Minimum valid pixel value (default 0.0)
        clip_max (float): Maximum valid pixel value (default 1.0)

    Returns:
        torch.Tensor: Images after soft thresholding
    """
    # Compute the difference from original
    diff = y - original_images

    # Apply soft thresholding
    # Values within threshold are zeroed (sparsity!)
    # Values above threshold get reduced by threshold
    shrink_positive = torch.clamp(y - threshold, min=clip_min, max=clip_max)
    shrink_negative = torch.clamp(y + threshold, min=clip_min, max=clip_max)

    # Three-way decision: positive, zero, or negative
    cond_positive = (diff > threshold).float()
    cond_zero = (torch.abs(diff) <= threshold).float()
    cond_negative = (diff < -threshold).float()

    # Combine using conditions
    result = (
        cond_positive * shrink_positive
        + cond_zero * original_images
        + cond_negative * shrink_negative
    )

    return result
```

|Code|What it does|
|---|---|
|`diff = y - original_images`|the perturbation the gradient step produced, before thresholding|
|`shrink_positive = clamp(y - threshold, 0, 1)`|shrink a positive change by the threshold, kept a valid pixel|
|`shrink_negative = clamp(y + threshold, 0, 1)`|shrink a negative change (add threshold), kept valid|
|`cond_positive = (diff > threshold)`|mask: 1 where the change is a big positive|
|`cond_zero = (|diff|
|`cond_negative = (diff < -threshold)`|mask: 1 where the change is a big negative|
|`result = cond_positive*shrink_positive + cond_zero*original + cond_negative*shrink_negative`|pick the right value per pixel|

Shapes: every tensor here is the same image shape (e.g. 5×5 or (B, 1, 28, 28)); the masks are 0/1 tensors of that shape, so the final line is pure element-wise selection.

### How it maps to the three-region rule

|Pixel's change `diff`|Mask that fires|Value used|Net perturbation|
|---|---|---|---|
|`diff > threshold`|`cond_positive`|`y − threshold`|`diff − threshold` (shrunk)|
|`|diff|≤ threshold`|`cond_zero`|
|`diff < −threshold`|`cond_negative`|`y + threshold`|`diff + threshold` (shrunk)|

Same `sign(z)·max(|z|−λ, 0)` operation from Section 4, written as explicit masks. Note the **dead zone is inclusive (`≤`)** - a change exactly equal to the threshold is zeroed, not kept.

> **An honest note the course skips (the clamp can bite near 0 and 1):** because the clamp happens on the pixel value (`y ± threshold`) rather than on the perturbation, a shrunk value that would fall below 0 or above 1 gets pinned to the bound, so the actual perturbation there is `−original` or `1−original`, not `diff ∓ threshold`. With the demo's mid-gray 0.5 pixels this never triggers, but on real images with pixels near 0 or 1 it can slightly alter the shrinkage. Good to know for the report.

---

## Part 3 - Watching sparsity appear

### The code

```python
# Create synthetic perturbation for visualization
print("\nTesting shrinkage-thresholding operation...")

# Generate a 5x5 synthetic perturbation pattern
perturbation_pattern = torch.tensor([
    [-0.15, -0.08, -0.02, 0.03, 0.12],
    [-0.10, -0.05, 0.00, 0.06, 0.18],
    [-0.05, 0.00, 0.05, 0.10, 0.20],
    [ 0.00, 0.05, 0.08, 0.15, 0.25],
    [ 0.05, 0.08, 0.12, 0.20, 0.30]
], device=device)

# Assume original pixels are 0.5 (mid-gray)
original_values = torch.ones_like(perturbation_pattern) * 0.5
perturbed_values = original_values + perturbation_pattern

# Apply shrinkage with beta=0.1
beta_test = 0.1
thresholded = apply_shrinkage_thresholding(
    perturbed_values,
    original_values,
    beta_test,
    clip_min=0.0,
    clip_max=1.0
)

# Compute resulting perturbations
resulting_perturbation = thresholded - original_values

print(f"\nSoft Thresholding Results (beta={beta_test}):")
print("=" * 50)
print("\nOriginal perturbation pattern:")
print(perturbation_pattern.cpu().numpy())
print("\nAfter soft thresholding:")
print(resulting_perturbation.cpu().numpy())

# Count sparsity
original_nonzero = (perturbation_pattern.abs() > 1e-6).sum().item()
thresholded_nonzero = (resulting_perturbation.abs() > 1e-6).sum().item()
sparsity_gained = original_nonzero - thresholded_nonzero

print(f"\nSparsity Analysis:")
print(f" Original non-zero elements: {original_nonzero}/25")
print(f" After thresholding: {thresholded_nonzero}/25")
print(f" Elements zeroed out: {sparsity_gained}")
print(f" Sparsity achieved: {(25-thresholded_nonzero)/25*100:.1f}%")
```

Output:

```txt
Testing shrinkage-thresholding operation...

Soft Thresholding Results (beta=0.1):
==================================================

Original perturbation pattern:
[[-0.15 -0.08 -0.02 0.03 0.12]
 [-0.1 -0.05 0. 0.06 0.18]
 [-0.05 0. 0.05 0.1 0.2 ]
 [ 0. 0.05 0.08 0.15 0.25]
 [ 0.05 0.08 0.12 0.2 0.3 ]]

After soft thresholding:
[[-0.05 0. 0. 0. 0.02]
 [ 0. 0. 0. 0. 0.08]
 [ 0. 0. 0. 0. 0.1 ]
 [ 0. 0. 0. 0.05 0.15]
 [ 0. 0. 0.02 0.1 0.2 ]]

Sparsity Analysis:
  Original non-zero elements: 22/25
  After thresholding: 9/25
  Elements zeroed out: 13
  Sparsity achieved: 64.0%
```

### Reading the output

|What|Value|Meaning|
|---|---|---|
|Non-zero before|22/25|the pattern had two kinds of waste: tiny values and a few exact zeros|
|Non-zero after|9/25|only the big-magnitude changes survived|
|Zeroed out|13|killed by the dead zone|
|Sparsity|64.0%|(25 − 9)/25 - the fraction now exactly zero|

The spatial story: the pattern grows from small (upper-left) to large (bottom-right). β = 0.1 **bisects the magnitudes**: everything with |change| ≤ 0.1 (upper-left and the central diagonal) is zeroed; only the bottom-right corner survives, each survivor shrunk by exactly 0.1 (the 0.30 becomes 0.20, the 0.12 becomes 0.02). **The eliminated pixels become _exact_ zeros, not 0.001-style near-zeros that would still count toward L0.**

This is exactly what an attack wants: pixels with weak gradients produce small changes that **waste L0 budget without moving the decision boundary**, so thresholding discards them and concentrates the budget on high-gradient pixels.

> **An honest note the course skips (its β = 0.05 and β = 0.2 counts are wrong):** the section claims β = 0.2 would "eliminate 20 of 25 (80% sparse)" leaving 5, and β = 0.05 would "keep 14 (44% sparse)." I ran the actual function: ==β = 0.2 leaves only **2** (92% sparse), and β = 0.05 leaves **15** (40% sparse).== The β = 0.1 output shown is correct - only these two "what-if" asides are miscounted. The reason is the boundary values: the pattern contains exact 0.05, 0.10, and 0.20 changes, and the code's dead zone is inclusive (`|diff| ≤ threshold`), so a change _equal_ to the threshold is zeroed, not kept. Eyeballing without that rule gives the course's slightly-off numbers. **When a value sits exactly on the threshold, it dies.**

---

## Copy-paste cells (mine, not the course's)

### Check the course's β counts yourself

```python
import torch
def apply_shrinkage_thresholding(y, original_images, threshold, clip_min=0.0, clip_max=1.0):
    diff = y - original_images
    sp = torch.clamp(y - threshold, min=clip_min, max=clip_max)
    sn = torch.clamp(y + threshold, min=clip_min, max=clip_max)
    cp = (diff > threshold).float(); cz = (diff.abs() <= threshold).float(); cn = (diff < -threshold).float()
    return cp*sp + cz*original_images + cn*sn

pat = torch.tensor([[-0.15,-0.08,-0.02,0.03,0.12],[-0.10,-0.05,0.00,0.06,0.18],
                    [-0.05,0.00,0.05,0.10,0.20],[0.00,0.05,0.08,0.15,0.25],
                    [0.05,0.08,0.12,0.20,0.30]])
orig = torch.ones_like(pat) * 0.5
for beta in [0.05, 0.1, 0.2]:
    res = apply_shrinkage_thresholding(orig + pat, orig, beta) - orig
    nz = (res.abs() > 1e-6).sum().item()
    print(f"beta={beta}: {nz}/25 survive, {(25-nz)/25*100:.0f}% sparse")
```

Expected (the real numbers, not the course's asides):

```txt
beta=0.05: 15/25 survive, 40% sparse
beta=0.1: 9/25 survive, 64% sparse
beta=0.2: 2/25 survive, 92% sparse
```

---

## Quick self-check

Using the dead-zone rule (`|change| ≤ threshold` → zeroed; otherwise shrink by the threshold), apply soft thresholding with **threshold = 0.1** to this perturbation row: **[0.10, −0.25, 0.04, 0.30, −0.10]**. What does each value become, and what's the resulting L0?

---

## The facts worth locking in for the exam

- ==FISTA momentum is `k/(k+3)`, growing from 0.25 at k=1 toward 1==, which gives cautious early steps and aggressive late acceleration; the `+3` keeps it below 1 so momentum never overwhelms the gradient.
- **Nesterov's schedule removes the fixed-momentum hyperparameter** (vs the usual tuned 0.9), auto-matching acceleration to the optimization phase, and `k/(k+3)` is a stable heuristic approximation of the exact O(1/k²) recursion.
- **Soft thresholding zeros every change with |change| ≤ threshold and shrinks the rest by the threshold**, producing _exact_ zeros (not near-zeros), which is what actually reduces the L0 count.
- **The dead zone is inclusive**: a change exactly equal to the threshold is zeroed, which is why the course's β = 0.2 (really 2 survive, not 5) and β = 0.05 (really 15, not 14) asides are miscounted - only the β = 0.1 output is right.
- **Higher threshold (β) means sparser**: on the 5×5 demo, β = 0.05 → 40% sparse, β = 0.1 → 64% sparse, β = 0.2 → 92% sparse, since a larger threshold swallows more of the magnitude distribution.
- **Thresholding concentrates the L0 budget on high-gradient pixels** by discarding small changes that waste budget without shifting the decision boundary - the whole point of sparsity for an attack.
- **The clamp acts on the pixel value, not the perturbation**, so near pixel bounds 0 and 1 the realized shrinkage can differ from `diff ∓ threshold`; it doesn't show on mid-gray demo pixels but matters on real images.
- **Method point:** **verify counts against the actual code instead of trusting a narrative's eyeballed numbers**, especially where values sit exactly on a threshold and an inclusive-vs-strict boundary (`≤` vs `<`) flips the result.

Logged Section 7, both course miscounts, and the self-check answer key. Paste Section 8 (Loss Gradients and Optimization) when ready.