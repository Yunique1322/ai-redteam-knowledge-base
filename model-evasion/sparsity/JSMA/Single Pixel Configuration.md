# Single Pixel Configuration

## The idea in one breath

Sweep the two knobs - step size **θ** and feature budget **γ** - to see how they trade off. The headline: on a hard sample, only **θ = 1.0** succeeds. But there's a catch the course doesn't call out - this section quietly **puts the per-pixel lock back**, which is _why_ small θ fails here.

> **Heads-up - the lock flip-flops, and it changes everything.** Section 18 had the lock (`search_space[pixel_idx] = False`). Section 19 **removed** it and got 70% at θ=0.25. This section **adds it back**. With the lock on, each pixel is used exactly once, so a pixel can only ever move by θ - which is the whole reason the θ sweep below behaves as it does. Watch for the line in the loop.

## Part 1 - The θ sweep

```python
theta_values = [0.10, 0.25, 0.50, 1.00]

print("\nStep Size Analysis:")
print(f"{'theta':<8} {'Iterations':<12} {'Pixels':<8} {'Result':<8}")
print("="*42)
```

```python
for theta_test in theta_values:
    # Initialize fresh attack
    x_test = original_images[0].clone().detach()
    search_space_test = initialize_search_space(x_test.shape)
    config_test = {**config, 'theta': theta_test}
    pixels_mod = 0

    # Run attack loop
    for iteration in range(100):
        if check_target_reached(x_test, target_labels[0], model):
            break
        if pixels_mod >= int(config_test['gamma'] * 784):
            break

        jacobian = compute_jacobian_matrix(x_test, model, 10, config_test['wrt'])
        alpha = extract_target_gradient(jacobian, target_labels[0])
        beta = extract_other_gradients(jacobian, target_labels[0])
        alpha_masked = apply_search_mask(alpha, search_space_test)
        beta_masked = apply_search_mask(beta, search_space_test)

        inc_scores = score_increase_saliency(alpha_masked, beta_masked)
        dec_scores = score_decrease_saliency(alpha_masked, beta_masked)
        pixel_idx, saliency, increase = select_best_direction(inc_scores, dec_scores)

        if saliency <= 0:
            break

        x_test = apply_single_pixel_perturbation(
            x_test, pixel_idx, config_test['theta'], increase, 0.0, 1.0
        )

        search_space_test[pixel_idx] = False
        search_space_test = remove_saturated_pixels(search_space_test, x_test, 0.0, 1.0)
        pixels_mod += 1
```

| Code | What it does |
| ----------------------------------------------- | --------------------------------------------------------------------------------------- |
| `config_test = {**config, 'theta': theta_test}` | copy the base config, override **only** θ (dict unpacking keeps γ, max_iter, etc.) |
| `x_test = original_images[0].clone().detach()` | fresh image each run so experiments don't bleed into each other |
| `search_space_test[pixel_idx] = False` | **the lock is back** - each pixel used once, so θ is the _total_ move a pixel ever gets |

## Part 2 - Results

```python
    success = check_target_reached(x_test, target_labels[0], model)
    result = "SUCCESS" if success else "FAILED"
    print(f"{theta_test:<8.2f} {iteration+1:<12} {pixels_mod:<8} {result:<8}")
```

```
theta Iterations Pixels Result
0.10 100 100 FAILED
0.25 100 100 FAILED
0.50 100 100 FAILED
1.00 64 63 SUCCESS
```

## Part 3 - Reading the θ trade-off

A **threshold effect**: 0.10, 0.25, 0.50 all fail at the 100-iteration wall; only **1.0 succeeds** (64 iterations, 63 pixels). Why, mechanically:

- With the **lock on**, a pixel is touched once and retired. So at θ=0.25 a background pixel reaches only 0.25 and is locked - it never becomes a strong white stroke. Spreading faint grey dots doesn't build enough change to cross the boundary.
- At **θ=1.0**, each selected pixel jumps 0.0 → 1.0 (or 1.0 → 0.0) in that single hit, so each pixel contributes maximum magnitude. Enough accumulates to flip the prediction.

Also note **pixels ≈ iterations (63 vs 64)** - a **one-to-one** correspondence. That's the fingerprint of the lock being on: every iteration consumes a brand-new pixel, no repeats.

> **An honest note - reconcile this with Section 19.** Section 19 got 70% at θ=0.25 _with the lock off_, by hammering pixels to saturation over repeated selections. This section fails at θ=0.25 _with the lock on_, because a locked pixel can't be hammered. So there are two working recipes: **lock off + small θ + repeats**, or **lock on + θ=1.0 + one pixel per step**. The latter is the **canonical** MNIST setup and what the course recommends. Neither is wrong, but the course swaps between them section to section without saying so - if your own results don't match, check which lock behavior your loop has.

## Part 4 - The γ budget table

```python
gamma_values = [0.10, 0.15, 0.20, 0.30]

print("\nFeature Budget Analysis:")
print(f"{'gamma':<8} {'Max Pixels':<12} {'Percentage':<12}")
print("="*38)

for gamma_test in gamma_values:
    max_pixels = int(gamma_test * 784)
    print(f"{gamma_test:<8.2f} {max_pixels:<12} {gamma_test*100:<12.0f}%")

print("="*38)
```

```
gamma Max Pixels Percentage
0.10 78 10%
0.15 117 15%
0.20 157 20%
0.30 235 30%
```

This is just arithmetic (`γ × 784`), not an attack run. The reasoning that follows:

- The 10-sample batch averaged 73.6 pixels, ~63% of the γ=0.15 budget (117).
- **γ=0.10 (78 px)** would choke the attacks that needed 86-92 pixels → lower success.
- **γ=0.30 (235 px)** adds headroom most attacks never use → wasted allowance, no real gain.
- **γ=0.15** is the balance: enough room for most attacks while still enforcing sparsity.

## Copy-paste cell (mine, not the course's)

A reusable sweep helper so you can try θ (and γ) on any sample and print the table - with a `lock` flag so you can _see_ the Section 19-vs-20 difference yourself:

```python
# === Section 20: parameter sweep helper ===
def sweep_theta(x, target_class, model, thetas=(0.10, 0.25, 0.50, 1.00),
                gamma=0.15, max_iter=100, lock=True, clip=(0.0, 1.0)):
    print(f"{'theta':<8} {'Iters':<8} {'Pixels':<8} {'trueL0':<8} {'Result':<8}")
    for th in thetas:
        xa = x.clone().detach()
        ss = initialize_search_space(xa.shape)
        budget, mods = int(gamma * int(np.prod(xa.shape[1:]))), 0
        it = 0
        for it in range(max_iter):
            if check_target_reached(xa, target_class, model): break
            if mods >= budget: break
            J = compute_jacobian_matrix(xa, model, 10, 'logits')
            a = apply_search_mask(extract_target_gradient(J, target_class), ss)
            b = apply_search_mask(extract_other_gradients(J, target_class), ss)
            idx, sal, inc = select_best_direction(score_increase_saliency(a, b),
                                                  score_decrease_saliency(a, b))
            if sal <= 0: break
            xa = apply_single_pixel_perturbation(xa, idx, th, inc, clip[0], clip[1])
            if lock: ss[idx] = False # flip this to compare Sec19 vs Sec20
            ss = remove_saturated_pixels(ss, xa, clip[0], clip[1])
            mods += 1
        ok = check_target_reached(xa, target_class, model)
        l0 = int(np.count_nonzero((xa - x).detach().cpu().numpy()))
        print(f"{th:<8.2f} {it+1:<8} {mods:<8} {l0:<8} {'SUCCESS' if ok else 'FAILED':<8}")

# sweep_theta(original_images[0], target_labels[0], model, lock=True) # Section 20 behavior
# sweep_theta(original_images[0], target_labels[0], model, lock=False) # Section 19 behavior
```

## Quick self-check

Lock **on**, `theta = 0.5`, clip `[0, 1]`, a pixel starting at 0.0.

1. After it's selected once, what's its value?
2. Can it be selected again?
3. Over 30 iterations with no wasted moves, how many unique pixels change, and what's the L∞ of the perturbation?

I'll confirm next message.

## The facts worth locking in for the exam

- ==θ is a hard threshold knob here: on a difficult sample only θ=1.0 succeeds== because, with the lock on, each pixel moves once and needs to saturate in that single hit.
- **With the per-pixel lock on, pixels modified ≈ iterations (one-to-one)**, the signature of one new pixel per step with no repeats.
- ==θ=1.0 is the recommended / canonical MNIST setting== (maximum magnitude per pixel, fewest pixels), trading visual subtlety for reliable success.
- **`{**config, 'theta': v}` overrides one hyperparameter while preserving the rest** - the clean way to sweep.
- ==max_pixels = int(γ × num_features)==, so γ=0.10/0.15/0.20/0.30 → 78/117/157/235 pixels on MNIST; γ sets the hard sparsity ceiling.
- ==γ=0.15 balances room against sparsity== - tighter starves the pixel-hungry successes, looser just wastes allowance most attacks don't use.
- The same θ can succeed or fail depending on whether the **per-pixel lock is on (one move each) or off (hammer to saturation)** - know which your loop uses.
- **Method point:** ==single-pixel JSMA's need for θ=1.0 and its steadily-declining saliency are symptoms of greedy, one-pixel-at-a-time selection== - the canonical **pairwise** algorithm in the next sections captures two-pixel interactions to succeed more often with less.

Paste Section 21 when ready - that should be the canonical **pairwise** JSMA, where saliency is scored over _pairs_ of pixels and the attack finally gets its real teeth.