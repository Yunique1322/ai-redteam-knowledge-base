# Single-Pixel Batch Evaluation

## The idea in one breath

Run the single-pixel attack on 10 samples instead of 1, so you get a real success _rate_ instead of one anecdote. And buried in this section's loop is a **one-line change** that quietly fixes the Section 18 failure - the course glosses over it, so I won't.

## Part 1 - The batch harness (collect correctly-classified samples)

```python
print("Collecting samples...")
samples_found = 0
target_count = 10

original_images = []
original_labels = []
target_labels = []
```

```python
for x_batch, y_batch in test_loader:
    if samples_found >= target_count:
        break

    x_batch = x_batch.to(device)
    y_batch = y_batch.to(device)

    with torch.no_grad():
        preds = model(x_batch).argmax(dim=1)
```

```python
    for i in range(x_batch.size(0)):
        if samples_found >= target_count:
            break

        if preds[i].item() != y_batch[i].item():
            continue

        x = x_batch[i:i+1]
        original_class = int(y_batch[i].item())
        target_class = (original_class + 5) % 10

        original_images.append(x)
        original_labels.append(original_class)
        target_labels.append(target_class)

        samples_found += 1
        print(f" Sample {samples_found}: digit {original_class} → target {target_class}")

print(f"\nCollected {len(original_images)} samples")
```

|Code|What it does|
|---|---|
|three parallel lists|index _i_ ties image↔true-label↔target across all three|
|`if preds[i] != y_batch[i]: continue`|keep **only correctly-classified** samples - an already-wrong image is a meaningless baseline|
|`x_batch[i:i+1]`|keep the batch dim → shape `(1,1,28,28)`, which JSMA needs (`[i]` alone would drop it)|
|`(original_class + 5) % 10`|pick a fixed, "far" target (7→2, 2→7, ...) and wrap around at 10|

## Part 2 - Results storage

```python
print("\nRunning attacks...")
print(f"{'#':<4} {'Orig→Tgt':<10} {'Result':<10} {'Pixels':<8} {'Iters':<8}")
print("="*46)

results = {
    'adversarial': [],
    'success': [],
    'pixels_modified': [],
    'iterations': []
}
```

Same parallel-array pattern - one entry per sample. Storing the adversarial image lets you analyze perturbations afterward.

## Part 3 - The attack loop, with the one-line change that matters

```python
for idx in range(len(original_images)):
    x = original_images[idx]
    orig_class = original_labels[idx]
    tgt_class = target_labels[idx]

    # Initialize attack state
    x_adv = x.clone().detach()
    search_space = initialize_search_space(x.shape)
    pixels_mod = 0

    # Attack loop
    for iteration in range(config['max_iter']):
        if check_target_reached(x_adv, tgt_class, model):
            break
        if pixels_mod >= int(config['gamma'] * 784):
            break

        jacobian = compute_jacobian_matrix(x_adv, model, 10, config['wrt'])
        alpha = extract_target_gradient(jacobian, tgt_class)
        beta = extract_other_gradients(jacobian, tgt_class)
        alpha_masked = apply_search_mask(alpha, search_space)
        beta_masked = apply_search_mask(beta, search_space)

        inc_scores = score_increase_saliency(alpha_masked, beta_masked)
        dec_scores = score_decrease_saliency(alpha_masked, beta_masked)
        pixel_idx, saliency, increase = select_best_direction(inc_scores, dec_scores)

        if saliency <= 0:
            break

        x_adv = apply_single_pixel_perturbation(
            x_adv, pixel_idx, config['theta'], increase,
            config['clip_min'], config['clip_max']
        )

        search_space = remove_saturated_pixels(search_space, x_adv, 0.0, 1.0)
        pixels_mod += 1
```

Look at the state update at the bottom: ==the `search_space[pixel_idx] = False` line from Section 18 is **gone**.== Only `remove_saturated_pixels` remains.

The course says this "still prevents repeat selections." **That's backwards.** Removing the lock means a pixel is **no longer** retired after one use - if it still has the top saliency next iteration, it gets picked **again**, moving 0 → 0.25 → 0.50 → 0.75 → 1.0 over four selections, and only drops out when it finally saturates at 1.0. **This is the exact fix for Section 18's failure**: a pixel can now reach full white and actually swing the prediction. That's why Section 18's single run flopped and this batch version succeeds most of the time. (It's the thing I flagged two sections ago, now corrected in the code even though the prose misdescribes it.)

## Part 4 - Record and display

```python
    # Record results
    success = check_target_reached(x_adv, tgt_class, model)
    pred = model(x_adv).argmax(dim=1).item()

    results['adversarial'].append(x_adv)
    results['success'].append(success)
    results['pixels_modified'].append(pixels_mod)
    results['iterations'].append(iteration + 1)

    # Display progress
    status = "✓" if success else "✗"
    print(f"{idx+1:<4} {orig_class}→{tgt_class:<8} {status:<10} {pixels_mod:<8} {iteration+1:<8}")

print("="*46)
```

Output:

```
1 7→2 ✗ 100 100
2 2→7 ✗ 100 100
3 1→6 ✓ 67 68
4 0→5 ✓ 33 34
5 4→9 ✓ 58 59
6 1→6 ✓ 92 93
7 4→9 ✓ 60 61
8 9→4 ✓ 86 87
9 5→0 ✗ 100 100
10 9→4 ✓ 40 41
```

Small detail worth knowing: on a **success**, `iters = pixels_mod + 1` (the winning iteration checks the target at the top and breaks _before_ incrementing); on a **max-iter failure** they're equal (both 100). So `iteration + 1` corrects the zero-based count.

## Part 5 - Summary statistics

```python
success_count = sum(results['success'])
total_samples = len(results['success'])
success_rate = 100.0 * success_count / total_samples

pixels = results['pixels_modified']
mean_pixels = np.mean(pixels)
median_pixels = np.median(pixels)
std_pixels = np.std(pixels)
min_pixels = np.min(pixels)
max_pixels = np.max(pixels)

sparsity_pct = 100.0 * mean_pixels / 784
```

Output: **7/10 (70%)**, mean 73.6 ± 24.2 pixels, median 76.5, range [33, 100], "sparsity 9.39%". The three failures (1, 2, 9) all hit the 100-iteration wall.

## Part 6 - Perturbation analysis (the four norms)

```python
x_orig = original_images[0].cpu().numpy()
x_adv = results['adversarial'][0].cpu().numpy()

perturbation = x_adv - x_orig
pert_magnitude = np.abs(perturbation)

# Compute norms
l0_norm = np.count_nonzero(perturbation)
l1_norm = np.sum(pert_magnitude)
l2_norm = np.linalg.norm(perturbation)
linf_norm = np.max(pert_magnitude)
```

|Norm|Code|Sample 1|Means|
|---|---|---|---|
|L0|`np.count_nonzero`|51|51 **unique** pixels actually changed|
|L1|`np.sum(|Δ|)`|
|L2|`np.linalg.norm`|3.54|Euclidean size of the change|
|L∞|`np.max(|Δ|)`|

## The number that should bother you - and the honest note

Sample 1 shows `pixels_mod = 100` but `L0 = 51`. Those aren't the same quantity, and the gap is the whole story of Part 3:

> **An honest note the course skips:** once the per-pixel lock is removed, **`pixels_mod` counts _modifications_, not _unique pixels_.** It's incremented once per iteration, and iterations now reselect pixels. On sample 1, 100 modifications landed on only 51 distinct pixels - the other ~49 were repeat hits pushing pixels toward saturation (which is exactly why L∞ reached 0.9961). So the summary's "Sparsity 9.39%" (mean `pixels_mod` ÷ 784) **overstates the pixel count** - the true L0 is lower. If you care about real sparsity (and for an L0 attack you do), measure it with `np.count_nonzero(x_adv - x_orig)`, not the modification counter. The budget check `pixels_mod >= gamma*784` is likewise a cap on _modifications_, not on distinct pixels.

This also reframes "70% success": it's achieved by **repeatedly hammering pixels to saturation**, which is halfway toward what the canonical pairwise JSMA does more efficiently.

## Copy-paste cell (mine, not the course's)

A corrected single-pixel attack **without** the lock (matching Section 19, so repeats are allowed) plus a batch evaluator that reports **true L0** sparsity, not the modification count:

```python
# === Section 19: batch single-pixel JSMA with honest sparsity ===
def jsma_single_pixel_v2(x, target_class, model, theta=0.25, gamma=0.15, max_iter=100,
                         wrt='logits', clip_min=0.0, clip_max=1.0, num_classes=10):
    x_adv = x.clone().detach()
    search_space = initialize_search_space(x.shape)
    budget = int(gamma * int(np.prod(x.shape[1:])))
    mods = 0
    for iteration in range(max_iter):
        if check_target_reached(x_adv, target_class, model): break
        if mods >= budget: break
        J = compute_jacobian_matrix(x_adv, model, num_classes, wrt)
        a = apply_search_mask(extract_target_gradient(J, target_class), search_space)
        b = apply_search_mask(extract_other_gradients(J, target_class), search_space)
        idx, sal, inc = select_best_direction(score_increase_saliency(a, b),
                                              score_decrease_saliency(a, b))
        if sal <= 0: break
        x_adv = apply_single_pixel_perturbation(x_adv, idx, theta, inc, clip_min, clip_max)
        search_space = remove_saturated_pixels(search_space, x_adv, clip_min, clip_max) # NO per-pixel lock
        mods += 1
    success = check_target_reached(x_adv, target_class, model)
    return x_adv, success, mods

def evaluate_batch(images, targets, model, **cfg):
    rows = []
    for x, t in zip(images, targets):
        x_adv, ok, mods = jsma_single_pixel_v2(x, t, model, **cfg)
        true_l0 = int(np.count_nonzero((x_adv - x).detach().cpu().numpy()))
        rows.append((ok, mods, true_l0))
    succ = sum(r[0] for r in rows)
    l0s = [r[2] for r in rows if r[0]] # true L0 of successes
    print(f"Success: {succ}/{len(rows)} ({100*succ/len(rows):.0f}%)")
    if l0s:
        print(f"True L0 (successes): mean {np.mean(l0s):.1f}, range [{min(l0s)}, {max(l0s)}]")
    return rows

# rows = evaluate_batch(original_images, target_labels, model)
```

## Quick self-check

An attack reports `pixels_modified = 80`, but `np.count_nonzero(x_adv - x_orig) = 45`.

1. What is the true L0 (number of pixels actually changed)?
2. Why are the two numbers different?

I'll confirm next message.

## The facts worth locking in for the exam

- **Only attack correctly-classified samples**, because a targeted attack needs a known-good starting prediction to flip; already-wrong images give no meaningful baseline.
- **Removing the per-pixel lock lets a pixel be reselected until it saturates**, which is what makes the batch attack succeed where the single-run baseline failed - a pixel can finally reach full intensity.
- **`pixels_modified` counts modifications, not unique pixels**, once repeats are allowed, so it overstates sparsity; measure true L0 with `np.count_nonzero(x_adv - x_orig)`.
- **Single-pixel JSMA scores ~70% here, not 100%**, and the failures exhaust the iteration cap - a reminder that one-pixel-at-a-time is a weak baseline.
- **Report L0, L1, L2, and L∞ together** to characterize a perturbation: L0 for sparsity, L∞ near 1.0 signals a saturated pixel, and L1 ÷ L0 gives the average change per touched pixel.
- On success `iterations = pixels_modified + 1` (the winning check doesn't increment), while a max-iter failure leaves them equal - useful when reading result tables.
- Give each sample **fresh state (cloned image, reset search space, zeroed counter)** so attacks don't contaminate each other.
- **Method point:** **the 70% rate and the saturation-hammering behavior expose single-pixel JSMA's inefficiency**, motivating the canonical **pairwise** variant and the parameter-tuning guidance in the sections ahead.

Paste Section 20 when ready - that should be the configuration guidelines, where θ and γ get swept against success rate so you can see how the knobs actually trade off.