# Pairwise Analysis

## The idea in one breath

Two investigations: **does a pair really synergize** (the demo says "can't tell from isolation"), and **is pairwise's O(n²) cost actually expensive** (no - the Jacobian dwarfs it, so pairwise is ~3× faster end-to-end because it needs far fewer iterations).

## Part 1 - The synergy experiment

```python
# Use first sample and first pair from attack
first_pair = (352, 389)
p, q = first_pair

# Reset to original
x_test = original_images[0].clone().detach()
baseline_conf = 0.0001 # Initial target confidence

# Impact of pixel p alone
x_p = apply_single_pixel_perturbation(x_test, p, 1.0, True, 0.0, 1.0)
conf_p = compute_confidence(x_p, target_labels[0], model)
impact_p = conf_p - baseline_conf
# ... same for q ...
```

```python
# Expected impact if purely additive
expected_impact = impact_p + impact_q

# Impact of pair together
x_pq = apply_pair_perturbation(x_test, p, q, 1.0, True, 0.0, 1.0)
impact_pq = compute_confidence(x_pq, target_labels[0], model) - baseline_conf

# Synergy
synergy = impact_pq - expected_impact
```

**synergy = pair impact − (impact of p alone + impact of q alone).** The interpretation:

- **positive** -> nonlinear boost (the pair does more together than apart - true interaction through the network's hidden layers)
- **negative** -> interference (the two cancel)
- **zero** -> purely additive (model treats them independently)

Output: every number is **0.000000**.

> **The demo is broken, but the lesson is real.** First, the glitches: the code hardcodes `first_pair = (352, 389)`, the printed output shows **327 and 538** (left over from Section 21), and the prose says "352 and 389." Classic copy-paste placeholder mess - and `baseline_conf = 0.0001` is the recurring fake literal. All zeros because these pixels have no isolated effect on this sample.
> 
> **But the conceptual point stands and it's worth remembering:** **measuring synergy in isolation is the wrong test for JSMA.** The attack is **cumulative** - a pair that matters at iteration 20 only matters _after_ 40 other pixels have already shifted the model's internal state. Drop that same pair onto the clean image and it does nothing. So "zero isolated synergy" doesn't mean the pair was useless in the attack; it means isolated testing can't capture context-dependent, sequential effects. A meaningful synergy study would test pairs _at their point in the attack sequence_, not against the untouched image.

## Part 2 - Timing: is O(n²) actually a problem?

```python
import time
# single-pixel saliency, averaged over 100 runs
start = time.time()
for _ in range(100):
    inc_scores = score_increase_saliency(alpha_test, beta_test)
    pixel_idx = int(np.argmax(inc_scores))
single_time = (time.time() - start) / 100
```

```python
# pairwise saliency with top_k=128
start = time.time()
for _ in range(100):
    p, q, score = compute_pairwise_saliency(alpha_test, beta_test, search_space_test, 'increase', top_k=128)
pair_time = (time.time() - start) / 100
```

```
Single-pixel: 0.00 ms (really <0.01 ms - below the measurement floor)
Pairwise: 2.13 ms
Ratio: 644.1x
```

> **Don't quote that 644×.** It's a measurement artifact - dividing a real 2.13 ms by a near-zero, unmeasurable single-pixel time inflates the ratio meaninglessly. The honest comparison is in **absolute** terms, which the next cell makes:

```python
jacobian_time = 15 # ms, empirically measured
```

```
Per-iteration breakdown:
  Jacobian: ~15 ms
  Single saliency: ~0.00 ms
  Pairwise saliency: ~2.13 ms

Total per iteration:
  Single-pixel: ~15.0 ms
  Pairwise: ~17.1 ms
```

The **Jacobian (15 ms, 10 backward passes) dominates everything**. Pairwise saliency adds 2.13 ms - just **+14%** per iteration. And here's the kicker: pairwise needs **22.4 iterations** on average vs single-pixel's **74.3**. So:

- single-pixel ≈ 74.3 × 15.0 ≈ **1,115 ms**
- pairwise ≈ 22.4 × 17.1 ≈ **383 ms**

Pairwise is ~**3× faster end-to-end** despite being slower _per iteration_. **The win comes from needing fewer iterations, not from a cheaper step.** The principle: optimize total_iterations × per_iteration × success, not the micro-cost of one component.

## Part 3 - top_k guidance

`top_k` caps how many candidate pixels get paired, so it sets the pair count C(top_k, 2):

|top_k|pairs/direction|~time|trade-off|
|---|---|---|---|
|**128** (default)|C(128,2) = **8,128**|~3 ms|rarely misses the best pair - recommended|
|64|C(64,2) = **2,016**|~1 ms|faster, slightly lower success (may prune a good pixel)|
|None|all valid pairs|10-20 ms|optimal, but the quality gain rarely justifies the cost|

θ and γ behave exactly as in single-pixel mode (θ=1.0 for fast saturation, γ=0.15 for the 117-pixel budget).

## Copy-paste cell (mine, not the course's)

A clean timing harness that reports **absolute** times and the honest end-to-end estimate (no misleading ratio), plus a top_k sweep:

```python
# === Section 24: honest cost model ===
import time

def time_fn(fn, runs=100):
    start = time.time()
    for _ in range(runs): fn()
    return (time.time() - start) / runs * 1000 # ms

ss = np.ones_like(alpha, dtype=bool)
t_single = time_fn(lambda: int(np.argmax(score_increase_saliency(alpha, beta))))
for k in (64, 128, None):
    t = time_fn(lambda: compute_pairwise_saliency(alpha, beta, ss, 'increase', top_k=k))
    pairs = "all" if k is None else k*(k-1)//2
    print(f"top_k={str(k):<4} pairs/dir={str(pairs):<6} saliency={t:.2f}ms")

JAC = 15.0
print(f"\nsingle-pixel saliency: {t_single:.3f} ms -> per-iter ~{JAC + t_single:.1f} ms")
print(f"end-to-end estimate: single {74.3*(JAC+t_single):.0f} ms vs pairwise {22.4*(JAC+2.13):.0f} ms")
```

## Key facts to lock in

- **Pairwise synergy can't be measured in isolation** - JSMA is cumulative, so a pair's real contribution appears only in sequence, after earlier pixels have shifted the model's state; an isolated test can read zero and still be wrong.
- ==Synergy = pair impact − sum of individual impacts==: positive is nonlinear boost, negative is interference, zero is additive.
- **The Jacobian (≈15 ms, 10 backward passes) dominates per-iteration cost**, so pairwise saliency's 2 ms is only ~14% extra.
- **Pairwise is faster end-to-end (~3×) despite being slower per iteration**, purely because it needs ~70% fewer iterations.
- **The reported 644× saliency ratio is a measurement artifact** (dividing by a sub-millisecond floor) - compare absolute times, not that ratio.
- **top_k sets the pair count C(top_k, 2)**: 128 -> 8,128 pairs (default, rarely misses), 64 -> 2,016 (faster, slightly worse), None -> full scan (optimal, usually not worth it).
- θ and γ carry over unchanged from single-pixel mode; `top_k` is the only pairwise-specific knob.
- **Method point:** ==overall attack efficiency = per-iteration cost × iterations × success rate==, not the speed of any one component - pairwise wins the product even though it loses the per-step micro-benchmark.

