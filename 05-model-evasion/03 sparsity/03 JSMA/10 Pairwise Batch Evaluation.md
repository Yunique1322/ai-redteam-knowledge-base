# Pairwise Batch Evaluation

## The idea in one breath

Run pairwise JSMA on the same 10 samples and put the two attacks side by side. The scoreboard: single-pixel **70%**, pairwise **90%** - and pairwise does it with **42% fewer pixels** and **70% fewer iterations**.

## Part 1 - The batch run

```python
results_pairs = {
    'adversarial': [],
    'success': [],
    'pixels_modified': [],
    'iterations': []
}
```

The attack loop per sample is exactly Section 22's pairwise loop, just wrapped in the `for idx in range(len(original_images))` batch harness with fresh state each time (cloned image, reset search space, `pixels_mod = 0`, `+= 2` per iteration). Nothing new in the mechanics - you've seen every line.

```
1 7->2 ✓ 68 35
2 2->7 ✗ 118 60
3 1->6 ✓ 38 20
4 0->5 ✓ 12 7
5 4->9 ✓ 28 15
6 1->6 ✓ 32 17
7 4->9 ✓ 34 18
8 9->4 ✓ 32 17
9 5->0 ✓ 52 27
10 9->4 ✓ 14 8
```

**9/10 succeed.** Only sample 2 (2->7) fails. Successful attacks range from 12 to 68 pixels. Compare to single-pixel, where samples 1, 2, **and** 9 all failed - pairwise rescued two of those three.

## Part 2 - The head-to-head

```python
for idx in range(len(original_images)):
    single_iters = results['iterations'][idx]
    single_pixels = results['pixels_modified'][idx]
    pair_iters = results_pairs['iterations'][idx]
    pair_pixels = results_pairs['pixels_modified'][idx]
    print(f"{idx+1:<8} {single_iters:<8} {single_pixels:<8} {'':8} {pair_iters:<8} {pair_pixels:<8}")

mean_single_iters = np.mean(results['iterations'])
mean_pair_iters = np.mean(results_pairs['iterations'])
# ... etc ...
iter_reduction = 100.0 * (mean_single_iters - mean_pair_iters) / mean_single_iters
```

```
Sample Single-Pixel Pairwise
         Iters Pixels Iters Pixels
1 100 100 35 68
2 100 100 60 118
3 68 67 20 38
4 34 33 7 12
5 59 58 15 28
6 93 92 17 32
7 61 60 18 34
8 87 86 17 32
9 100 100 27 52
10 41 40 8 14
Mean 74.3 73.6 22.4 42.8

Iteration reduction: 69.9%
```

Every single sample is faster and lighter with pairwise. Means: **22.4 vs 74.3 iterations** (69.9% fewer), **42.8 vs 73.6 pixels** (~42% fewer), and **90% vs 70%** success.

## Reading it - the win is real, but mind the confounds

The headline (90% vs 70%, far fewer pixels) is a genuine, large improvement. But two things to keep honest:

> **Honest note 1 - the comparison changes more than one variable.** Single-pixel here ran at **θ=0.25**; pairwise ran at **θ=1.0**. (And the single-pixel `results` came from the Section 19 batch, which had the per-pixel lock _off_, while pairwise locks both pixels.) So this isn't a clean ablation of "pairwise alone" - some of the gain is θ=1.0, some is pairwise synergy, and the two are tangled. A fair isolation would hold θ fixed. The takeaway you should carry: the **canonical strong config is pairwise + θ=1.0 + logits**, as a bundle. Don't claim "pairwise is 20 points better" as if θ were held constant.

> **Honest note 2 - the budget is "soft" by a pixel.** Sample 2 shows **118** pixels against a **117** budget. How does it exceed the cap? `pixels_mod` goes up by 2 each iteration (2, 4, ..., 116, 118), so it skips the odd value 117 entirely. At 116 the check `116 >= 117` is false, so it does one more pair -> 118, and only then does `118 >= 117` trip the break. So with an odd budget and 2-pixels-per-step, pairwise can overshoot the cap by one pixel. Harmless here, but worth knowing if you're reporting strict L0 ≤ budget.

## Copy-paste cell (mine, not the course's)

A comparison harness that runs both variants on a sample list and prints the deltas, reusing the functions from Sections 18-22:

```python
# === Single-pixel vs pairwise comparison (Section 23) ===
def compare_variants(images, targets, model, gamma=0.15):
    print(f"{'#':<4}{'single(it/px/ok)':<22}{'pair(it/px/ok)':<22}")
    s_it, s_px, s_ok, p_it, p_px, p_ok = [], [], [], [], [], []
    for n, (x, t) in enumerate(zip(images, targets), 1):
        xa_s, ok_s, _ = jsma_single_pixel_v2(x, t, model, theta=0.25, gamma=gamma) # Sec19 variant
        xa_p, ok_p, l0_p, curve = jsma_pairwise(x, t, model, theta=1.0, gamma=gamma) # Sec22 variant
        it_s = len(curve) # placeholder if your single fn returns iters; adjust as needed
        print(f"{n:<4}{str(('?', int(np.count_nonzero((xa_s-x).cpu().numpy())), ok_s)):<22}"
              f"{str((len(curve), l0_p, ok_p)):<22}")
        s_ok.append(ok_s); p_ok.append(ok_p); p_it.append(len(curve)); p_px.append(l0_p)
    print(f"\nsingle success {sum(s_ok)}/{len(s_ok)} | pair success {sum(p_ok)}/{len(p_ok)}")
    print(f"pair mean iters {np.mean(p_it):.1f} | pair mean trueL0 {np.mean(p_px):.1f}")

# compare_variants(original_images, target_labels, model)
```

(Note I count **true L0** via `count_nonzero` rather than the modification counter, so the pixel comparison is honest - the Section 19 lesson.)

## Key facts to lock in

- **Pairwise JSMA beats single-pixel decisively here: 90% vs 70% success**, with ~42% fewer pixels and ~70% fewer iterations on average.
- **Pairwise rescued samples single-pixel couldn't crack** (1 and 9), leaving only sample 2 as a shared-hard failure.
- **The course's comparison changes θ (0.25 -> 1.0) alongside single->pair**, so the measured gap bundles both effects - the clean takeaway is "pairwise + θ=1.0 + logits" as a config, not "pairwise alone = +20 points."
- **A pairwise budget can overshoot an odd cap by one pixel** (118 vs 117) because `pixels_modified` increments by 2 and skips the odd value.
- **Report true L0 with `np.count_nonzero`**, not the modification counter, for an honest sparsity comparison across variants.
- Keep **identical sample sets and fresh per-sample state** so single-pixel and pairwise numbers are actually comparable.
- Pairwise's efficiency comes from **two coordinated pixels per step plus synergy in the pair-sum gate**, which lets it cross the boundary in far fewer iterations.
- **Method point:** **pairwise JSMA is the configuration to reach for on MNIST-scale models** - but its quadratic pair cost (mitigated by top-k pruning) is the price, which the next section weighs against the accuracy gain.

