# Single-Pixel Attack Fundamentals

## The idea in one breath

Three small helpers - **apply a one-pixel change**, **check if we've hit the target class**, **read the target's confidence** - and then one full iteration run by hand so you can watch Sections 15 and 16 feed into each other: Jacobian → α/β → mask → score → pick → perturb → update → check.

## Part 1 - Apply a single-pixel change

```python
def apply_single_pixel_perturbation(x, pixel_idx, theta, increase, clip_min=0.0, clip_max=1.0):
    original_shape = x.shape
    x_flat = x.view(-1).clone()

    perturbation = theta if increase else -theta
    x_flat[pixel_idx] = torch.clamp(
        x_flat[pixel_idx] + perturbation,
        clip_min,
        clip_max
    )

    x_modified = x_flat.view(original_shape)
    return x_modified
```

|Code|What it does|
|---|---|
|`original_shape = x.shape`|remember `(1,1,28,28)` so we can restore it|
|`x.view(-1).clone()`|flatten to 784 in C-H-W order **and clone** - `view` _shares memory_ with `x`, so without the clone you'd overwrite the original image and poison every later iteration|
|`theta if increase else -theta`|the direction flag from `select_best_direction` becomes a signed step|
|`torch.clamp(x_flat[pixel_idx] + perturbation, clip_min, clip_max)`|add the step, but **pin the result to [0,1]** - a pixel at 0.9 given +0.25 becomes 1.0, not 1.15|
|`x_flat.view(original_shape)`|fold the vector back to 4D, same order, so index 352 lands at the same spatial spot|

The `.clone()` and the matching C-H-W `view` both ways are the whole correctness story here - Section 15's "flatten/unflatten consistently" rule made concrete.

## Part 2 - Did we reach the target?

```python
def check_target_reached(x, target_class, model):
    with torch.no_grad():
        logits = model(x)
        prediction = int(logits.argmax(dim=1).item())

    return prediction == target_class
```

| Code | What it does |
| ----------------------------------- | -------------------------------------------------------------------------------------- |
| `with torch.no_grad()` | just a forward pass - no gradients, so skip building the graph (saves memory and time) |
| `logits.argmax(dim=1)` | the model's current top class |
| `.item()` → `int(...)` | pull it out as a plain Python int for a clean `==` compare |
| `return prediction == target_class` | `True` the moment the model predicts our target - the loop's stop signal |

## Part 3 - How confident is the target class?

```python
def compute_confidence(x, target_class, model):
    with torch.no_grad():
        logits = model(x)
        probs = F.softmax(logits, dim=1)
        confidence = float(probs[0, target_class].item())

    return confidence
```

|Code|What it does|
|---|---|
|`F.softmax(logits, dim=1)`|turn raw scores into probabilities that sum to 1|
|`probs[0, target_class]`|the probability the model assigns to **our target**|
|`float(...item())`|a plain float to track across iterations|

Softmax, in words: exponentiate every score and divide by the total, so you get a proper probability. This number starts near **0.0** for a class the model currently rejects, and you want to watch it climb toward 1.0 - misclassification happens as it crosses the other classes, typically once it clears **~0.5**.

## Part 4 - One full iteration, out loud

**Pick a correctly-classified sample and set up state:**

```python
# Get first correctly classified sample
for x_batch, y_batch in test_loader:
    x_batch, y_batch = x_batch.to(device), y_batch.to(device)
    with torch.no_grad():
        preds = model(x_batch).argmax(dim=1)

    for i in range(x_batch.size(0)):
        if preds[i].item() == y_batch[i].item():
            x = x_batch[i:i+1]
            original_class = int(y_batch[i].item())
            target_class = (original_class + 5) % 10
            break
    break

# Initialize attack state
x_adv = x.clone().detach()
theta = 0.25
clip_min, clip_max = 0.0, 1.0

# Initialize search space
num_features = int(np.prod(x.shape[1:]))
search_space = np.ones(num_features, dtype=bool)

print(f"Sample: digit {original_class}, target {target_class}")
print(f"Total features: {num_features}")
print(f"Initial target confidence: {compute_confidence(x_adv, target_class, model):.4f}")
```

Two things to note: `x = x_batch[i:i+1]` keeps the batch dimension (so shape stays `(1,1,28,28)` - slicing with `[i]` would drop it and break `compute_jacobian_matrix`). And `target_class = (original_class + 5) % 10` is just a quick way to pick a fixed, different target - digit 7 → target 2. Output: digit 7, target 2, confidence **0.0000** (the model gives "2" essentially no chance right now).

**Compute the Jacobian, split, mask:**

```python
jacobian = compute_jacobian_matrix(x_adv, model, num_classes=10, wrt='logits')
alpha = extract_target_gradient(jacobian, target_class)
beta = extract_other_gradients(jacobian, target_class)
alpha_masked = apply_search_mask(alpha, search_space)
beta_masked = apply_search_mask(beta, search_space)

print(f"Jacobian shape: {jacobian.shape}")
print(f"Target gradient range: [{alpha.min():.4f}, {alpha.max():.4f}]")
print(f"Other gradient range: [{beta.min():.4f}, {beta.max():.4f}]")
```

Shape `(10, 784)`; α and β both swing roughly −0.78 to +0.85. Big spread = some pixels matter a lot, most barely.

**Score and select:**

```python
inc_scores = score_increase_saliency(alpha_masked, beta_masked)
dec_scores = score_decrease_saliency(alpha_masked, beta_masked)
pixel_idx, saliency, increase = select_best_direction(inc_scores, dec_scores)

print(f"Increase: {(inc_scores > 0).sum()} valid pixels, max score {inc_scores.max():.6f}")
print(f"Decrease: {(dec_scores > 0).sum()} valid pixels, max score {dec_scores.max():.6f}")
print(f"Selected: pixel {pixel_idx}, saliency {saliency:.6f}, {'increase' if increase else 'decrease'}")
```

330 pixels pass the increase gate (best 0.5016), 349 pass the decrease gate (best 0.6412). Winner: **pixel 398, decrease, 0.6412**.

**Perturb:**

```python
pixel_before = x_adv.flatten()[pixel_idx].item()
x_adv = apply_single_pixel_perturbation(x_adv, pixel_idx, theta, increase, clip_min, clip_max)
pixel_after = x_adv.flatten()[pixel_idx].item()

print(f"Pixel {pixel_idx}: {pixel_before:.4f} → {pixel_after:.4f}")
```

`Pixel 398: 0.0000 → 0.0000`. Pixel 398 was already black (0.0); the decrease tried to subtract 0.25, but `clamp` held it at 0.0. The direction was "right" per the saliency, but there was nowhere to go.

**Update search space and check:**

```python
search_space = remove_saturated_pixels(search_space, x_adv, clip_min, clip_max)
success = check_target_reached(x_adv, target_class, model)
confidence_new = compute_confidence(x_adv, target_class, model)

print(f"Modifiable pixels: {search_space.sum()}")
print(f"Target reached: {success}")
print(f"Target confidence: 0.0001 → {confidence_new:.4f}")
```

`Modifiable pixels: 115`, `Target reached: False`, confidence still 0.0000. One iteration moved the needle not at all - the attack needs many more.

## Reading these outputs

- **784 → 115 in a single step** is the eye-catching number. `remove_saturated_pixels` stripped out _every_ pixel sitting at 0.0 or 1.0 - and on an MNIST "7" the whole black background is at 0.0, so ~669 pixels vanished at once.
- **Pixel 398 was a dud move:** selected for decrease but already at the floor. The saliency map ranked it top because its _gradients_ looked best, but the gradient doesn't know the pixel can't physically go lower. That mismatch is normal early on.

> **An honest note the course glosses over - and it matters.** Because `remove_saturated_pixels` drops pixels at _both_ boundaries, every black (0.0) background pixel is thrown out after iteration 1 - even though those pixels could still be **increased** (you just can't decrease them). So this course's JSMA ends up mostly nudging the existing grey pixels rather than drawing fresh white strokes onto black background. It still converges, but slower and with lower success than the classic **increase-only** JSMA, which keeps 0.0 pixels available and only removes pixels that hit 1.0. If your success rate looks disappointing later, this is a big part of why. (Section 16's note, now biting in practice.)

> **Cosmetic course bug:** the last `print` hardcodes `"Target confidence: 0.0001 → ..."` - that `0.0001` is a literal in the string, not the real starting value (which was 0.0000). Don't read meaning into it.

## Copy-paste cell (mine, not the course's)

The three utilities plus a tiny self-contained one-step demo on a stub model, so you can confirm the perturb/clamp/saturate mechanics without MNIST:

```python
# === JSMA single-pixel utilities (Section 17) ===
import torch, torch.nn as nn, torch.nn.functional as F, numpy as np

def apply_single_pixel_perturbation(x, pixel_idx, theta, increase, clip_min=0.0, clip_max=1.0):
    shape = x.shape
    xf = x.view(-1).clone()
    xf[pixel_idx] = torch.clamp(xf[pixel_idx] + (theta if increase else -theta), clip_min, clip_max)
    return xf.view(shape)

def check_target_reached(x, target_class, model):
    with torch.no_grad():
        return int(model(x).argmax(dim=1).item()) == target_class

def compute_confidence(x, target_class, model):
    with torch.no_grad():
        return float(F.softmax(model(x), dim=1)[0, target_class].item())

# --- mechanics check (no MNIST needed) ---
x = torch.tensor([[[[0.0, 0.1], [0.9, 1.0]]]]) # 4 pixels: at floor, low, high, at ceiling
print("decrease px1 (0.10, theta .25):", apply_single_pixel_perturbation(x, 1, 0.25, False).flatten()[1].item()) # -> 0.0 (clamped)
print("increase px2 (0.90, theta .25):", apply_single_pixel_perturbation(x, 2, 0.25, True ).flatten()[2].item()) # -> 1.0 (clamped)
```

## Quick self-check

A pixel currently at **0.10** is selected for the **decrease** direction, with `theta=0.25` and `clip_min=0.0`.

1. What is the pixel's value after `apply_single_pixel_perturbation`?
2. On the next `remove_saturated_pixels` call, is this pixel kept or dropped?

Work it, I'll confirm next message.

## The facts worth locking in for the exam

- **Flatten with `.view(-1).clone()`, perturb one index, then `.view(original_shape)` back** - the clone stops the flattened view from overwriting the real image, and the matching order keeps the pixel in the right place.
- **Every pixel change is clamped to [clip_min, clip_max]**, so a step that would overshoot just saturates the pixel at 0.0 or 1.0 rather than creating an illegal value.
- ==Success is `argmax(logits) == target_class`, checked under `torch.no_grad()`==; confidence is `softmax(logits)[target]`, which climbs from ~0 toward 1 and crosses the boundary near 0.5.
- ==One iteration = Jacobian → α/β → mask → score both directions → select → perturb → remove saturated → check==, exactly Sections 15 and 16 chained together.
- **The saliency map can pick a pixel that can't actually move** (top-ranked pixel 398 was already at 0.0), because gradients don't know about the clip boundaries - such a move does nothing and the pixel is dropped next update.
- **The search space can collapse dramatically in one step** (784 → 115 here) because all boundary pixels are removed at once, and on MNIST the black background is all at 0.0.
- Keep the batch dimension when selecting a sample (`x_batch[i:i+1]`, not `[i]`), or the Jacobian's batch-size-1 check fails.
- **Method point:** **this single hand-run iteration is the attack's whole body; the next section only wraps it in a loop** - repeating until `check_target_reached` is `True` or the L0 / iteration budget runs out, plus the bookkeeping to report what happened.

Paste Section 18 when ready - that's the full loop and results tracking, where these pieces finally run to a successful misclassification.