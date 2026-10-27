# Saliency And Search Space

## The idea in one breath

Turn the α and β from Section 15 into an actual **pixel ranking** - score every pixel for the increase direction _and_ the decrease direction, pick the single best, and keep a running **mask** of which pixels are still fair game so you never waste a step on a pixel that's already maxed out.

## Part 1 - Scoring the increase direction

```python
def score_increase_saliency(target_grad, other_grad):
    increase_mask = (target_grad > 0) & (other_grad < 0)
    scores = target_grad * np.abs(other_grad) * increase_mask
    return scores
```

|Code|What it does|
|---|---|
|`(target_grad > 0) & (other_grad < 0)`|the Section 14 sign gate: keep a pixel only if raising it **helps the target** (α > 0) **and** **hurts competitors** (β < 0). `True`/`False` per pixel|
|`target_grad * np.abs(other_grad)`|the magnitude product `α ×|
|`* increase_mask`|multiply by the `True/False` gate -> valid pixels keep their score, invalid pixels become **0**|

This is Section 14's formula verbatim, just vectorized over all 784 pixels at once.

## Part 2 - Scoring the decrease direction

```python
def score_decrease_saliency(target_grad, other_grad):
    decrease_mask = (target_grad < 0) & (other_grad > 0)
    scores = np.abs(target_grad) * other_grad * decrease_mask
    return scores
```

|Code|What it does|
|---|---|
|`(target_grad < 0) & (other_grad > 0)`|the **flipped** gate - see below|
|`np.abs(target_grad) * other_grad`|still `|
|`* decrease_mask`|zero out the pixels that fail the flipped gate|

Why the signs flip: if α is **negative**, that means "raising this pixel _lowers_ the target" - so **lowering** the pixel _raises_ the target, which is what we want. Same logic for β: if β is **positive**, raising the pixel helps competitors, so **lowering** it hurts them. **Both effects line up with our goal, so a pixel that looks useless for increasing can be the best pixel to decrease.** This is the Section 14 point - JSMA can erase pixels as readily as it adds them.

## Part 3 - The worked example

```python
# Gradients for 6 features
alpha = np.array([ 0.6, -0.3, 0.4, 0.1, -0.5, 0.2])
beta = np.array([-0.2, 0.4, -0.5, 0.3, 0.6, -0.1])

inc_scores = score_increase_saliency(alpha, beta)
dec_scores = score_decrease_saliency(alpha, beta)
```

|Feature|α|β|Inc gate (α>0 & β<0)|Inc score|Dec gate (α<0 & β>0)|Dec score|
|---|---|---|---|---|---|---|
|0|0.6|−0.2|✓|0.6×0.2 = **0.120**|✗|0|
|1|−0.3|0.4|✗|0|✓|0.3×0.4 = **0.120**|
|2|0.4|−0.5|✓|0.4×0.5 = **0.200**|✗|0|
|3|0.1|0.3|✗|0|✗|0|
|4|−0.5|0.6|✗|0|✓|0.5×0.6 = **0.300**|
|5|0.2|−0.1|✓|0.02|✗|0|

Feature 3 fails **both** gates (raising it helps everyone, lowering it helps no one) - dead. Feature 4 is the overall winner at 0.300, but only in the **decrease** direction.

> The `-0.000` you see in the course output is just a signed zero - a negative number times the `False` (0) gate. It's cosmetically negative but numerically zero; ignore the minus sign.

## Part 4 - Pick the winner

```python
def select_best_direction(inc_scores, dec_scores):
    max_inc_idx = int(np.argmax(inc_scores))
    max_dec_idx = int(np.argmax(dec_scores))

    max_inc_score = float(inc_scores[max_inc_idx])
    max_dec_score = float(dec_scores[max_dec_idx])

    if max_inc_score > max_dec_score:
        return max_inc_idx, max_inc_score, True
    else:
        return max_dec_idx, max_dec_score, False
```

|Code|What it does|
|---|---|
|`np.argmax(inc_scores)` / `np.argmax(dec_scores)`|the index of the best pixel in each direction|
|`max_inc_score`, `max_dec_score`|the actual scores at those indices|
|`if max_inc_score > max_dec_score`|a head-to-head: higher score wins|
|`return idx, score, True/False`|returns **(pixel index, its score, is-increase flag)** - `True` = increase, `False` = decrease|

Two edge cases worth noting. The comparison is **strict** (`>`), so an exact tie goes to **decrease**. And when every score is 0 (no valid pixel left), `np.argmax` returns index 0 and the score comes back `0.0` - the attack loop (next section) checks for that zero and **stops**, because there's nothing useful left to do.

On the example this returns `pixel 4, score 0.300, decrease`, matching the table.

## Part 5 - Start the search space

```python
def initialize_search_space(shape):
    num_features = int(np.prod(shape[1:]))
    return np.ones(num_features, dtype=bool)
```

|Code|What it does|
|---|---|
|`shape[1:]`|drop the batch dim: `(1,1,28,28)` -> `(1,28,28)`|
|`np.prod(...)`|multiply them: 1×28×28 = **784**|
|`np.ones(num_features, dtype=bool)`|a length-784 array of all `True` - every pixel starts available|

`dtype=bool` is deliberate: 1 byte per pixel instead of 4-8, and it signals "this is a yes/no mask, not a counter."

## Part 6 - Remove saturated pixels

```python
def remove_saturated_pixels(search_space, x, clip_min=0.0, clip_max=1.0, epsilon=1e-6):
    x_flat = x.detach().cpu().numpy().flatten()

    saturated_min = (x_flat <= clip_min + epsilon)
    saturated_max = (x_flat >= clip_max - epsilon)
    saturated = saturated_min | saturated_max

    updated_mask = search_space & ~saturated
    return updated_mask
```

|Code|What it does|
|---|---|
|`x_flat = ...flatten()`|the current image as a length-784 vector (same C-H-W order as the gradients)|
|`x_flat <= clip_min + epsilon`|pixels sitting at (or within 1e-6 of) **0.0** - can't go lower|
|`x_flat >= clip_max - epsilon`|pixels at (or within 1e-6 of) **1.0** - can't go higher|
|`saturated = ... \| ...`|either boundary counts as saturated|
|`search_space & ~saturated`|keep pixels that are **still available AND not saturated** (`~` flips the saturated mask)|

The `epsilon` is there because a pixel you think is exactly 0.0 might be stored as `-1e-8` after arithmetic; the tolerance catches "effectively at the boundary." On the toy image `[0.0, 0.3, 0.95, 1.0]` the mask becomes `[False, True, True, False]` - pixels 0 and 3 are pinned at the edges and drop out.

> **If this errors / behaves oddly:**
> 
> - Scores all zero on the very first iteration -> check you passed **β** (the others-sum) not a single competitor row, and that you're on **logits** (softmax forces β = −α, which can zero every gate - the Section 14 trap showing up here).
> - Mask length doesn't match the gradient length -> your image shape or flatten order drifted; both must be length `C×H×W` in the same order.
> - `argmax` keeps picking the same pixel forever -> you're not applying the search mask to the gradients (Section 15's `apply_search_mask`) before scoring, so saturated pixels never get excluded.

> **An honest note the course skips:** `remove_saturated_pixels` drops a pixel sitting at 0.0 from **all** future moves - but a pixel at 0.0 can still legitimately be **increased**; only its _decrease_ is impossible. Likewise a pixel at 1.0 could still be decreased. Removing it from both directions is a simplification that costs a little attack power, though in practice a freshly-saturated pixel is rarely the top saliency choice anyway. The textbook Papernot JSMA sidesteps this by moving pixels in a single fixed direction (and in pairs); this course scores both directions for teaching clarity, which is why the blanket removal is slightly lossy.

## Copy-paste cell (mine, not the course's)

All six functions plus a self-contained check that reproduces the course's pixel-4 result - no model needed:

```python
# === JSMA saliency + search space (Section 16) ===
import numpy as np, torch

def score_increase_saliency(target_grad, other_grad):
    m = (target_grad > 0) & (other_grad < 0)
    return target_grad * np.abs(other_grad) * m

def score_decrease_saliency(target_grad, other_grad):
    m = (target_grad < 0) & (other_grad > 0)
    return np.abs(target_grad) * other_grad * m

def select_best_direction(inc_scores, dec_scores):
    i, d = int(np.argmax(inc_scores)), int(np.argmax(dec_scores))
    si, sd = float(inc_scores[i]), float(dec_scores[d])
    return (i, si, True) if si > sd else (d, sd, False)

def initialize_search_space(shape):
    return np.ones(int(np.prod(shape[1:])), dtype=bool)

def remove_saturated_pixels(search_space, x, clip_min=0.0, clip_max=1.0, epsilon=1e-6):
    xf = x.detach().cpu().numpy().flatten()
    saturated = (xf <= clip_min + epsilon) | (xf >= clip_max - epsilon)
    return search_space & ~saturated

# --- reproduce the course example ---
alpha = np.array([ 0.6, -0.3, 0.4, 0.1, -0.5, 0.2])
beta = np.array([-0.2, 0.4, -0.5, 0.3, 0.6, -0.1])
inc, dec = score_increase_saliency(alpha, beta), score_decrease_saliency(alpha, beta)
idx, sc, up = select_best_direction(inc, dec)
print(f"pixel {idx}, score {sc:.3f}, {'increase' if up else 'decrease'}") # -> pixel 4, score 0.300, decrease
assert (idx, round(sc,3), up) == (4, 0.300, False)
print("OK")
```

## Key facts to lock in

- **A pixel's saliency is `|α| × |β|` only when the sign gate passes, else zero** - increase needs α>0 and β<0, decrease needs α<0 and β>0; the product rewards a pixel that strongly helps the target _and_ strongly hurts competitors.
- **Score both directions and take the single highest** across all pixels and both directions; this is why the best move is sometimes to _lower_ a pixel, not raise it.
- **When all scores are zero there's no legal move left**, so the attack must stop - `select_best_direction` returns a 0.0 score that the loop treats as a termination signal.
- **The search space is a boolean mask over flattened pixels, all `True` to start** (784 on MNIST), and pixels are removed as they're used or saturate.
- **A saturated pixel is one pinned at clip_min (0.0) or clip_max (1.0)**, detected with a small epsilon for float rounding, and removed so you don't waste iterations pushing a pixel that can't move.
- Removing saturated pixels uses **boolean AND with the inverted saturation mask** (`search & ~saturated`), preserving every pixel's index position.
- Ties between the two directions resolve to **decrease** because the comparison is strict `>`, a small implementation detail that won't matter for success but explains deterministic tie-breaks.
- **Method point:** **this section delivers "which pixel and which direction," but not the move itself** - applying the step θ, re-checking the prediction, and looping until success or the L0 budget is the full attack, which the next section assembles from Sections 15 and 16.

