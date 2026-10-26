# Pairwise Saliency

## The idea in one breath

The canonical JSMA moves **two pixels at once**. You add their gradients (`α_pq = α_p + α_q`, `β_pq = β_p + β_q`), apply the same sign-gated `|α|×|β|` score to the _pair_, and pick the best pair. Because the gate is on the **sums**, a pixel that's useless alone can team up with a complement - the synergy single-pixel JSMA couldn't see.

## Part 1 - The pairwise formula

$$S^{+}[p,q] = \alpha_{pq} \times |\beta_{pq}| \quad\text{if } \alpha_{pq} > 0 \text{ and } \beta_{pq} < 0$$

with `α_pq = α_p + α_q` and `β_pq = β_p + β_q`; decrease flips to `|α_pq| × β_pq` when `α_pq < 0` and `β_pq > 0`. It's Section 16's score with the pair sums in place of one pixel's gradients.

**The math moment - why pairs beat singles:**

||α|β|single-increase gate|single score|
|---|---|---|---|---|
|p|0.6|**+0.1**|✗ (β > 0)|0|
|q|0.1|−0.5|✓|0.05|
|**pair (p,q)**|α_pq = **0.7**|β_pq = **−0.4**|✓|**0.7 × 0.4 = 0.28**|

Pixel p fails on its own (raising it helps competitors), but paired with q the _sum_ still suppresses competitors and the pair scores 0.28 - far above q alone. **That rescue is the entire reason pairwise JSMA is stronger.**

**Read aloud:** "Add the two pixels' target slopes and the two competitors' slopes; if the sums still point the right way, score the pair by the product of their sizes."

## Part 2 - Prune candidates (keep it affordable)

```python
def prune_candidates(alpha, beta, search_space, top_k):
    alpha_masked = alpha * search_space
    beta_masked = beta * search_space
    valid = np.where(search_space)[0]

    if valid.size < 2 or top_k is None or valid.size <= top_k:
        return valid

    prelim_scores = np.abs(alpha_masked[valid]) * np.abs(beta_masked[valid])
    idx = np.argsort(-prelim_scores)[:top_k]
    return valid[idx]
```

|Code|What it does|
|---|---|
|`alpha * search_space`|zero out unavailable pixels before scoring|
|`np.where(search_space)[0]`|turn the boolean mask into integer pixel positions|
|early returns|can't pair < 2 pixels; `top_k=None` → no pruning; fewer than `top_k` → nothing to prune|
|`np.abs(...) * np.abs(...)`|preliminary per-pixel score `|
|`np.argsort(-prelim_scores)[:top_k]`|keep the top-k by that proxy (`-` for descending)|

## Part 3 - Evaluate all pairs

```python
def evaluate_pairs(alpha, beta, valid, direction):
    best_p, best_q, best_score = -1, -1, 0.0

    for i in range(valid.size):
        p = valid[i]
        for j in range(i + 1, valid.size):
            q = valid[j]
            a_pq = alpha[p] + alpha[q]
            b_pq = beta[p] + beta[q]

            if direction == 'increase':
                if a_pq <= 0 or b_pq >= 0:
                    continue
                score = a_pq * abs(b_pq)
            else:
                if a_pq >= 0 or b_pq <= 0:
                    continue
                score = abs(a_pq) * b_pq

            if score > best_score:
                best_score = float(score)
                best_p, best_q = int(p), int(q)

    return best_p, best_q, best_score
```

|Code|What it does|
|---|---|
|`for j in range(i + 1, ...)`|inner loop starts at **i+1** → no self-pairs (p=q), no duplicates ((p,q)=(q,p)); exactly C(n,2) pairs|
|`a_pq = alpha[p] + alpha[q]` / `b_pq = ...`|the pair sums|
|increase gate `a_pq<=0 or b_pq>=0: continue`|skip pairs that fail the sign constraint|
|`score = a_pq * abs(b_pq)` (or `abs(a_pq) * b_pq`)|`|
|`best_*` tracking, init `-1`|remember the top pair; `-1` is the "none found" sentinel|

## Part 4 - Assemble, and the complexity

```python
def compute_pairwise_saliency(alpha, beta, search_space, direction='increase', top_k=None):
    valid = prune_candidates(alpha, beta, search_space, top_k)

    if valid.size < 2:
        return -1, -1, 0.0

    best_p, best_q, best_score = evaluate_pairs(alpha, beta, valid, direction)

    if best_p == -1:
        return -1, -1, 0.0
    return best_p, best_q, best_score
```

The cost is **quadratic**: C(n,2) = n(n−1)/2 pairs. All 784 pixels → **306,936** pairs per iteration. Pruning to `top_k=128` → C(128,2) = **8,128** pairs, a ~97% cut. That pruning is what makes pairwise JSMA runnable at all.

## Part 5 - Apply the pair perturbation

```python
def apply_pair_perturbation(x, p, q, theta, increase, clip_min=0.0, clip_max=1.0):
    original_shape = x.shape
    x_flat = x.view(-1).clone()

    step = theta if increase else -theta
    x_flat[p] = torch.clamp(x_flat[p] + step, clip_min, clip_max)
    x_flat[q] = torch.clamp(x_flat[q] + step, clip_min, clip_max)

    return x_flat.view(original_shape)
```

Both pixels get the **same signed step**. This is not optional: the score was computed on `α_p + α_q` and `β_p + β_q`, which only holds if both pixels move the **same** way. Increasing p but decreasing q would break the math behind the pair score. Each pixel is clamped independently, so one can saturate while the other still has room.

## Part 6 - One pairwise iteration

After picking a correctly-classified sample (digit 7 → target 2) and computing the Jacobian → α, β as before, the demo prunes and searches:

```python
top_k = 128
valid = np.where(search_space)[0]
# ... prune to top-128 by |α|·|β| ...
```

```
Initial candidates: 784
Worst-case pairs: 306936
After pruning: 128 candidates
Pairs to evaluate: 8128
```

Then the search:

```python
p, q, score = compute_pairwise_saliency(alpha, beta, search_space, 'increase', top_k)
```

```
Best pair: pixels 327 and 538
Best score: 1.68123138
  Pixel 327: α=0.685326, β=-0.731848
  Pixel 538: α=0.498610, β=-0.688188
  α_pq = 1.183936
  β_pq = -1.420036
```

Score `1.184 × 1.420 = 1.681` - far above a typical single-pixel score (~0.6). Then apply with θ=1.0:

```
Pixel 327: 0.9922 → 1.0000
Pixel 538: 0.0000 → 1.0000
Target confidence after pair perturbation: 0.0000
```

Pixel 327 saturates from 0.99, pixel 538 jumps black→white in one step. Confidence is still 0.0000 - this is the hard 7→2 sample again; pairwise builds up over iterations.

## Reading the outputs - two course glitches and one real caveat

> **Course glitches:** the `Alpha range: [-0.001234, 0.001456]` / `Beta range: [...]` print shows tiny placeholder numbers that **don't match** the real gradients shown moments later (α=0.685, β=−0.732). Treat that range print as filler. And the confidence line again hardcodes `0.0001 →` in the string - not a real value.

> **The real caveat (honest note):** `prune_candidates` keeps pixels by their **individual** `|α|×|β|` - but the whole selling point of pairwise scoring is finding pairs where a pixel is _weak alone but synergistic_. Pruning by individual strength can throw those very pixels away before they're ever paired. The course admits it "may occasionally miss the globally best pair." So the pruning heuristic is in mild tension with the synergy it's meant to enable - a deliberate speed-for-optimality trade. In the demo both chosen pixels happen to be individually strong, so you don't see the tension bite, but it's there.

## Copy-paste cell (mine, not the course's)

The four functions plus a compact one-iteration demo that also prints how far the best _pair_ beats the best _single_ pixel:

```python
# === Pairwise JSMA saliency (Section 21) ===
import numpy as np, torch

def prune_candidates(alpha, beta, search_space, top_k):
    valid = np.where(search_space)[0]
    if valid.size < 2 or top_k is None or valid.size <= top_k:
        return valid
    prelim = np.abs((alpha*search_space)[valid]) * np.abs((beta*search_space)[valid])
    return valid[np.argsort(-prelim)[:top_k]]

def evaluate_pairs(alpha, beta, valid, direction):
    bp, bq, bs = -1, -1, 0.0
    for i in range(valid.size):
        p = valid[i]
        for j in range(i+1, valid.size):
            q = valid[j]
            apq, bpq = alpha[p]+alpha[q], beta[p]+beta[q]
            if direction == 'increase':
                if apq <= 0 or bpq >= 0: continue
                s = apq * abs(bpq)
            else:
                if apq >= 0 or bpq <= 0: continue
                s = abs(apq) * bpq
            if s > bs: bs, bp, bq = float(s), int(p), int(q)
    return bp, bq, bs

def compute_pairwise_saliency(alpha, beta, search_space, direction='increase', top_k=None):
    valid = prune_candidates(alpha, beta, search_space, top_k)
    if valid.size < 2: return -1, -1, 0.0
    bp, bq, bs = evaluate_pairs(alpha, beta, valid, direction)
    return (-1, -1, 0.0) if bp == -1 else (bp, bq, bs)

def apply_pair_perturbation(x, p, q, theta, increase, clip_min=0.0, clip_max=1.0):
    shape = x.shape; xf = x.view(-1).clone()
    step = theta if increase else -theta
    xf[p] = torch.clamp(xf[p] + step, clip_min, clip_max)
    xf[q] = torch.clamp(xf[q] + step, clip_min, clip_max)
    return xf.view(shape)

# demo (needs x_adv, model, target_class, Section 15 funcs):
# J = compute_jacobian_matrix(x_adv, model, 10, 'logits')
# a, b = extract_target_gradient(J, target_class), extract_other_gradients(J, target_class)
# ss = initialize_search_space(x_adv.shape)
# p, q, s = compute_pairwise_saliency(a, b, ss, 'increase', top_k=128)
# print("pair", p, q, "score", round(s,4), "| best single:", round(score_increase_saliency(a*ss, b*ss).max(),4))
```

## Quick self-check

Pixel **p** has α=0.3, β=+0.4 - it **fails** the single-pixel increase gate on its own (β positive). Pair it with **q** (α=0.5, β=−0.9):

1. Does the **pair** pass the increase gate?
2. If so, what's the pair's saliency score?

I'll confirm next message.

## The facts worth locking in for the exam

- **Pairwise JSMA sums the two pixels' gradients** (`α_pq = α_p + α_q`, `β_pq = β_p + β_q`) and scores the pair with the same sign-gated `|α_pq| × |β_pq|`.
- **The gate is on the sums, so a pixel that fails alone can pass in a complementary pair** - this synergy is why pairwise beats single-pixel.
- **Both pixels in a pair must move the same direction** (both +θ or both −θ), or the pair-sum scoring is invalid.
- **Cost is quadratic, C(n,2) pairs** - 784 pixels is ~307k pairs per iteration, so pruning to top-k (128 → ~8k pairs) is mandatory for feasibility.
- **The inner loop starts at i+1** to generate each unordered pair exactly once (no self-pairs, no duplicates).
- **Pruning keeps pixels by individual |α|×|β|**, a heuristic that's fast but can discard weak-yet-synergistic pixels and occasionally miss the globally best pair.
- A `-1` sentinel from `evaluate_pairs` means **no pair satisfied the sign constraints** - the loop's signal that the search space is exhausted.
- **Method point:** **pair scores run much higher than single-pixel scores** (1.68 vs ~0.6 here) because two coordinated pixels accumulate magnitude faster - the leverage that lifts JSMA's success rate in the full loop next section.

Paste Section 22 when ready - that's the full pairwise attack loop and batch evaluation, where you'll finally see the success rate jump above the single-pixel 70%.