# Adversarial Loss

## The idea in one breath

The distance metrics measure the change; this section adds the part that actually **pushes the model to misclassify**. It's the **Carlini & Wagner margin loss**: compare the true class's logit to the best competitor, and once the competitor wins (by a margin κ), the loss **saturates to exactly 0** so the optimizer can stop pushing and focus on shrinking the change. Then `compute_total_loss` glues it to the squared-L2 term as `c·f + ‖δ‖₂²`.

> **Heads-up: mangled formulas.** Read them as:
> 
> - `f(x′,y)=max(Zy(x′)−maxj≠yZj(x′)+κ,0)` -> **f = max( Z_y − max_{j≠y} Z_j + κ , 0 )**.
> - `ℒtotal=c⋅f(x′,y)+||x′−x||22` -> **L_total = c·f(x′,y) + ‖x′−x‖₂²** (squared L2).
> - Zᵥ = the **logit** (raw pre-softmax score) of class y; κ (kappa) is the confidence margin.

---

## Part 1 - Why not a simpler loss

A naive "loss = 0 if misclassified, 1 if not" is a step function: **flat everywhere, so the gradient is zero everywhere**, and the optimizer gets no direction. The C&W loss is the smooth, differentiable stand-in: it gives a real slope to follow while it still matters, and goes quiet once the goal is met.

This is a real contrast with FGSM. **FGSM uses cross-entropy and only cares about the gradient's _direction_ (it takes `sign`), so the loss magnitude is irrelevant.** ElasticNet needs the actual loss _size_ to weigh against the distortion terms, which is why it needs a loss that saturates cleanly to 0.

---

## Part 2 - The margin loss (untargeted first)

### Toy walkthrough

True class is digit 7 with logit **Z₇ = 2.8**. Best competitor is class 4 with **Z₄ = 1.2**. With κ = 0:

|Stage|real (true logit)|other (best competitor)|margin = real − other + κ|loss = max(margin, 0)|
|---|---|---|---|---|
|Before attack|2.8|1.2|2.8 − 1.2 + 0 = **1.6**|**1.6** (true class still leads)|
|After successful flip|1.0|1.5 (class 2 now)|1.0 − 1.5 + 0 = **−0.5**|**0** (hinge clamps it)|

While the loss is positive, the gradient **pushes the true logit down and the competitor up**. The moment a competitor overtakes the true class (by κ), the margin goes negative, **the hinge clamps the loss to 0**, and optimization shifts entirely to minimizing distortion.

### The formula

```
f(x′, y) = max( Z_y(x′) − max_{j≠y} Z_j(x′) + κ , 0 )
```

|Symbol|Plain English|Toy|
|---|---|---|
|Z_y(x′)|logit of the **true** class y|2.8|
|max_{j≠y} Z_j(x′)|the **best competitor's** logit|1.2|
|Z_y − max_{j≠y} Z_j|how much the true class still leads|1.6|
|+ κ|demand the competitor win by at least κ, not just tie|+0|
|max( ... , 0 )|hinge: no "reward" for over-shooting past success|stays 1.6, or clamps to 0|

**Read aloud:** "Loss is how much the true class still beats its best rival (plus a safety margin κ), but never less than zero."

### What κ buys you

|κ|Attack succeeds when...|Result|
|---|---|---|
|0|any competitor exceeds the true class by any amount|minimal, but fragile|
|> 0|a competitor exceeds the true class by **at least κ**|**more robust** - survives small perturbations / preprocessing|

---

## Part 3 - The code: `compute_adversarial_loss`

```python
def compute_adversarial_loss(logits, labels_onehot, confidence, targeted=False):
    """
    Compute margin-based adversarial loss.

    Uses C&W formulation: encourage misclassification with
    confidence margin. Loss becomes zero once margin achieved.

    Parameters:
        logits (torch.Tensor): Model outputs before softmax (batch_size, num_classes)
        labels_onehot (torch.Tensor): One-hot encoded labels (batch_size, num_classes)
        confidence (float): Confidence margin kappa
        targeted (bool): Whether this is a targeted attack

    Returns:
        torch.Tensor: Adversarial loss per example (batch_size,)
    """
    # Extract scores
    real = torch.sum(labels_onehot * logits, dim=1)
    other = torch.max((1 - labels_onehot) * logits - labels_onehot * 10000, dim=1)[0]

    # Compute margin loss
    if targeted:
        # For targeted attacks: want target to exceed other classes by the margin
        loss = torch.clamp(other - real + confidence, min=0)
    else:
        # For untargeted attacks: want real class to be exceeded
        loss = torch.clamp(real - other + confidence, min=0)

    return loss
```

|Code|What it does|
|---|---|
|`labels_onehot * logits`|zero every class except the marked one (a selector mask)|
|`torch.sum(..., dim=1)`|collapse to one number per image -> `real`|
|`(1 - labels_onehot) * logits`|zero the marked class, keep the others at their logits|
|`- labels_onehot * 10000`|push the marked class to about −10000 so it **can never be the max**|
|`torch.max(..., dim=1)[0]`|the best remaining logit -> `other` (the `[0]` grabs the _values_, since `torch.max` returns values **and** indices)|
|`targeted: other - real + κ`|want the **marked** class (the target) to climb above the rest|
|`else: real - other + κ`|want the **marked** class (the true one) to be overtaken|
|`torch.clamp(..., min=0)`|the hinge|

### The one trap worth circling

**`labels_onehot` means different things depending on `targeted`.**

|Mode|`labels_onehot` marks|`real` is|`other` is|Goal|
|---|---|---|---|---|
|untargeted|the **true** class|true logit|best competitor|make `other` beat `real`|
|targeted|the **target** class|target logit|best non-target|make `real` (target) beat `other`|

### How the mask extracts `real` and `other`

For logits `[0.1, −0.3, 0.5, 2.8, 1.2, 0.7, −0.5, 0.3, 0.9, 0.4]` with the true class at the position holding 2.8:

- `real`: the one-hot keeps only that position -> **2.8**.
- `other`: `(1 − onehot)` zeroes that position, `− onehot·10000` drives it to −9997.2, so `max` of the rest = **1.2** (class 4).

Untargeted margin = 2.8 − 1.2 + 0 = **1.6**. ✓

> **An honest note the course skips (its worked example is inconsistent):** the course says "true label 7" and writes the one-hot as `[0,0,0,0,0,0,0,1,0,0]` - a 1 at **index 7**. But index 7 of its logit vector is **0.3**, not 2.8 (2.8 sits at index 3). I ran the code: with the one-hot truly at index 7 you get `real = 0.3, other = 2.8`, not the claimed `real = 2.8, other = 1.2`. Their own masked vector (`−9997.2` at index 3) shows they actually masked **index 3**, i.e. they treated the true class as the one holding 2.8. So either the one-hot should be at index 3 or "position 7" is a typo. **The mechanism they're teaching is correct; only the index in practiceple is wrong.** The lesson: the one-hot's _position_ must match where the true/target logit actually is.

### Targeted toy (from the course)

|Stage|target logit (real)|best competitor (other)|loss = max(other − real + κ, 0)|
|---|---|---|---|
|Not yet targeted|2.0|2.6|max(2.6 − 2.0, 0) = **0.6**|
|Target wins|2.7|2.6|max(2.6 − 2.7, 0) = **0**|

Mirror image of untargeted: now we want the **target** (marked) to climb above everyone else.

---

## Part 4 - Combining into the total loss

### The formula and a quick calc

```
L_total = c · f(x′, y) + ‖x′ − x‖₂²
```

|f (margin loss)|squared L2|c|L_total = c·f + L2²|
|---|---|---|---|
|0.6|2.25|0.1|0.1 × 0.6 + 2.25 = **2.31**|
|0.6|2.25|1.0|1.0 × 0.6 + 2.25 = **2.85**|

Bigger c puts more weight on fooling the model relative to staying small. **The L1 term is deliberately absent** - FISTA's soft-threshold step handles it, so it's not in the gradient.

### The code: `compute_total_loss`

```python
def compute_total_loss(
    adv_images,
    original_images,
    labels_onehot,
    const,
    model,
    beta,
    confidence,
    targeted=False,
):
    """
    Combine smooth components: c * adversarial + squared L2 (L1 via proximal operator).

    The constant c balances misclassification vs distortion.
    Beta controls L1 vs L2 trade-off (sparsity vs smoothness).

    Parameters:
        adv_images (torch.Tensor): Current adversarial images
        original_images (torch.Tensor): Original clean images
        labels_onehot (torch.Tensor): One-hot encoded labels
        const (torch.Tensor): Trade-off constants c per example
        model (nn.Module): Target model
        beta (float): L1 weight in elastic-net distance
        confidence (float): Margin for misclassification
        targeted (bool): Whether this is a targeted attack

    Returns:
        tuple: (total_loss, adversarial_loss, distances)
    """
    # Get model predictions
    logits = model(adv_images)

    # Compute adversarial loss
    adversarial_loss = compute_adversarial_loss(
        logits, labels_onehot, confidence, targeted
    )

    # Compute distances
    l1_dist, l2_dist, elastic_dist = compute_distances(
        adv_images, original_images, beta
    )

    # Combine: c * adversarial_loss + L2_distance
    # Note: L1 is handled by FISTA's proximal operator, not in this gradient
    total_loss = const * adversarial_loss + l2_dist

    return total_loss, adversarial_loss, (l1_dist, l2_dist, elastic_dist)
```

|Code|What it does|
|---|---|
|`logits = model(adv_images)`|forward pass - **the most expensive step, repeated hundreds of times in FISTA**|
|`compute_adversarial_loss(...)`|the margin loss from Part 3|
|`compute_distances(...)`|L1, squared-L2, elastic from Section 5|
|`total_loss = const * adversarial_loss + l2_dist`|the smooth objective; **no L1 here**|
|returns three things|total (for backprop), adv loss (to monitor success), distances (for analysis)|

Why return all three: the **total loss** feeds autograd, the **adversarial loss** tells you whether the attack has flipped yet, and the **distances** feed logging/analysis. `const` is a **per-example tensor**, so each image follows its own binary-searched c.

Autograd computes ∇[c·f + ‖x′−x‖₂²] for the gradient step; the **shrinkage step adds the L1 effect without ever needing its gradient**.

---

## Copy-paste cells (mine, not the course's)

### Reproduce both margin examples

```python
import torch

def compute_adversarial_loss(logits, labels_onehot, confidence, targeted=False):
    real = torch.sum(labels_onehot * logits, dim=1)
    other = torch.max((1 - labels_onehot) * logits - labels_onehot * 10000, dim=1)[0]
    if targeted:
        return torch.clamp(other - real + confidence, min=0)
    return torch.clamp(real - other + confidence, min=0)

# Untargeted: true class (index 3) leads 2.8 vs 1.2
logits = torch.tensor([[0.1, -0.3, 0.5, 2.8, 1.2, 0.7, -0.5, 0.3, 0.9, 0.4]])
onehot = torch.zeros(1, 10); onehot[0, 3] = 1 # mark the TRUE class (index 3!)
print("untargeted loss:", compute_adversarial_loss(logits, onehot, 0.0).item()) # expect 1.6
```

Expected:

```txt
untargeted loss: 1.5999...
```

Note the one-hot is at index **3** (where 2.8 lives), not index 7 - the fix for the course's slip.

## Key facts to lock in

- **The C&W margin loss is max(Z_y − max_{j≠y} Z_j + κ, 0) for untargeted**, pushing the true logit down and the best competitor up, and saturating to exactly 0 once a competitor wins by κ so the optimizer can switch to shrinking the perturbation.
- **A step-function "0 if wrong, 1 if right" loss gives no gradient**, which is why C&W uses a smooth margin; this also differs from FGSM, which uses cross-entropy and only reads the gradient's sign, ignoring the loss magnitude.
- **`labels_onehot` marks the true class in untargeted mode but the target class in targeted mode**, so `real` is the true logit (untargeted) or the target logit (targeted) - mixing these up silently inverts the attack.
- **`other` is found by masking the marked class with `−10000` and taking the max of the rest**, guaranteeing the marked class can never be selected as its own competitor; the `[0]` after `torch.max` grabs the values, not the indices.
- ==κ (the confidence margin) controls robustness: κ = 0 flips on any lead change, while κ > 0 forces the competitor to win by at least κ==, producing adversarial examples that survive small perturbations or preprocessing.
- ==The total smooth loss is L_total = c·f + ‖x′−x‖₂², and the L1 term is intentionally excluded==, because FISTA's soft-threshold (proximal) step applies the L1 effect without a gradient.
- **`const` (c) is a per-example tensor set by binary search**, so easy images use a small c with low distortion and resistant images use a large c accepting more distortion - the reason `compute_distances` kept per-image numbers.
- **Method point:** **design an adversarial loss that saturates to zero at success**, so once an example is fooled the optimizer stops adding distortion and spends its remaining effort shrinking the perturbation - the opposite of FGSM, which has no notion of "stop once fooled."

