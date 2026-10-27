# Pairwise Attack Loop

## The idea in one breath

Wrap Section 21's pairwise saliency in the full loop - and watch the **exact sample that failed single-pixel JSMA (7->2) now succeed**, in 35 iterations with 68 pixels. This is the payoff for all the pairwise machinery.

## Part 1 - Configuration

```python
# Reset
x_adv = x.clone().detach()
search_space = initialize_search_space(x.shape)

# Configuration
config = {
    'theta': 1.0,
    'gamma': 0.15,
    'max_iter': 90,
    'wrt': 'logits',
    'clip_min': 0.0,
    'clip_max': 1.0,
    'top_k': 128
}
```

`theta=1.0` (saturate each pixel in one hit - pairwise is efficient enough that it doesn't need to re-hit pixels), `gamma=0.15` (same 117-pixel budget), and the new `top_k=128` for pair pruning.

## Part 2 - Budget

```python
num_features = int(np.prod(x.shape[1:]))
max_pixels = int(config['gamma'] * num_features)
```

117 pixels. But now each iteration spends **two** pixels, so the attack can run at most ⌊117/2⌋ = **58 iterations** before the budget is gone (it usually succeeds well before).

## Part 3 - The loop

**Termination checks, then gradients:**

```python
for iteration in range(config['max_iter']):
    # Check termination conditions
    if check_target_reached(x_adv, target_class, model):
        print(f"\n✓ Target reached at iteration {iteration}!")
        break

    if pixels_modified >= max_pixels:
        print(f"\n✗ Budget exhausted")
        break

    # Compute Jacobian and extract gradients
    jacobian = compute_jacobian_matrix(x_adv, model, 10, config['wrt'])
    alpha = extract_target_gradient(jacobian, target_class)
    beta = extract_other_gradients(jacobian, target_class)
```

Identical to single-pixel up to here - same Jacobian, same α/β. Only the _use_ changes.

**Score both directions:**

```python
    # Score both directions
    p_inc, q_inc, score_inc = compute_pairwise_saliency(
        alpha, beta, search_space, 'increase', config['top_k']
    )
    p_dec, q_dec, score_dec = compute_pairwise_saliency(
        alpha, beta, search_space, 'decrease', config['top_k']
    )

    # Check for exhausted search space
    if max(score_inc, score_dec) <= 0.0:
        print(f"\n✗ No valid pairs")
        break
```

Two full pair searches per iteration - one for increase, one for decrease - then the zero-score stop.

**Select and validate:**

```python
    # Select best direction
    if score_inc >= score_dec:
        p, q, score, increase = p_inc, q_inc, score_inc, True
    else:
        p, q, score, increase = p_dec, q_dec, score_dec, False

    # Validate selected pair
    if p < 0 or q < 0:
        print(f"\n✗ Invalid pair")
        break
```

Higher score wins (ties -> increase via `>=`); the `p < 0` check catches the `-1` sentinel defensively.

**Perturb and update:**

```python
    # Apply pair perturbation
    x_adv = apply_pair_perturbation(
        x_adv, p, q, config['theta'], increase,
        config['clip_min'], config['clip_max']
    )

    # Update search space
    search_space[p] = False
    search_space[q] = False
    search_space = remove_saturated_pixels(search_space, x_adv, clip_min, clip_max)

    # Track progress
    pixels_modified += 2
    confidence = compute_confidence(x_adv, target_class, model)

    # Print progress
    if iteration % 3 == 0 or iteration < 3:
        print(f"{iteration:<6} {pixels_modified:<8} {confidence:<12.4f} {score:<12.8f}")
```

Both pixels get locked (`search_space[p] = search_space[q] = False`), then saturated pixels are swept out. The counter goes up by **2**.

```
Iter Pixels Confidence Score
0 2 0.0000 2.44320202
...
24 50 0.0185 0.37239930
27 56 0.2927 0.55573213
30 62 0.3491 0.46899793
33 68 0.7262 0.41797119

✓ Target reached at iteration 34!
```

## Part 4 - Final evaluation

```python
final_success = check_target_reached(x_adv, target_class, model)
final_pred = model(x_adv).argmax(dim=1).item()

print("\n✓ Target reached at iteration 35!")
print("\n" + "="*46)
print(f"Final result: {'SUCCESS' if final_success else 'FAILED'}")
...
```

```
Final result: SUCCESS
Final prediction: 2
Pixels modified: 68
Iterations: 35
```

## Reading the result - the win, and the confidence shape

**The win:** this is the same digit 7 -> target 2 sample that single-pixel JSMA couldn't crack (failed at 100 iterations / 100 pixels). Pairwise nails it with **68 pixels (8.7%)** - 32% fewer pixels - and stops at 68 of the 117 budget, so it didn't even need the full allowance.

**The confidence curve is the thing to notice:** it sits at ~0.0000 through iteration ~24, then detonates - 0.0185 -> 0.2927 -> 0.3491 -> 0.7262 -> flip. **Confidence isn't linear; the early pairs lay foundation that does nothing visible, then a threshold is crossed and it rockets across the boundary.** Don't panic when an attack shows flat confidence early; that's normal.

> **Course glitches to flag:**
> 
> - The final cell prints a **hardcoded** `"✓ Target reached at iteration 35!"` - a literal string. The real break happened at iteration **34** (printed inside the loop); `iteration + 1 = 35` is just the 1-based count.
> - The course explains the 68 pixels as "35 iterations × 2 = 70, minus 2 due to saturated pixels being skipped." **That's wrong.** `pixels_modified` always goes up by exactly 2 whenever the loop body runs. The real arithmetic: the success was detected at the **top** of iteration 34, so only iterations 0-33 modified pairs - **34 × 2 = 68**. The 35th pass did a success check and broke without touching a pixel. No saturated-skip involved.

## Copy-paste cell (mine, not the course's)

The whole pairwise attack as one reusable function returning the adversarial image, success, true pixel count, and the confidence curve for plotting:

```python
# === Full pairwise JSMA (Section 22) ===
def jsma_pairwise(x, target_class, model, theta=1.0, gamma=0.15, max_iter=90, top_k=128,
                  wrt='logits', clip_min=0.0, clip_max=1.0, num_classes=10):
    x_adv = x.clone().detach()
    search_space = initialize_search_space(x.shape)
    max_pixels = int(gamma * int(np.prod(x.shape[1:])))
    pixels_modified, conf_curve = 0, []

    for iteration in range(max_iter):
        if check_target_reached(x_adv, target_class, model): break
        if pixels_modified >= max_pixels: break

        J = compute_jacobian_matrix(x_adv, model, num_classes, wrt)
        a = extract_target_gradient(J, target_class)
        b = extract_other_gradients(J, target_class)
        pi, qi, si = compute_pairwise_saliency(a, b, search_space, 'increase', top_k)
        pd, qd, sd = compute_pairwise_saliency(a, b, search_space, 'decrease', top_k)
        if max(si, sd) <= 0.0: break

        if si >= sd: p, q, inc = pi, qi, True
        else: p, q, inc = pd, qd, False
        if p < 0 or q < 0: break

        x_adv = apply_pair_perturbation(x_adv, p, q, theta, inc, clip_min, clip_max)
        search_space[p] = False; search_space[q] = False
        search_space = remove_saturated_pixels(search_space, x_adv, clip_min, clip_max)
        pixels_modified += 2
        conf_curve.append(compute_confidence(x_adv, target_class, model))

    ok = check_target_reached(x_adv, target_class, model)
    true_l0 = int(np.count_nonzero((x_adv - x).detach().cpu().numpy()))
    return x_adv, ok, true_l0, conf_curve

# x_adv, ok, l0, curve = jsma_pairwise(x, target_class, model)
# print("SUCCESS" if ok else "FAILED", "| true L0:", l0, "| iters:", len(curve))
```

(I return `true_l0 = count_nonzero` as well, since - same lesson as Section 19 - `pixels_modified` counts modifications, and locking both pixels each step means it's usually close to but not guaranteed equal to the unique-pixel count once saturation removal interacts.)

## Key facts to lock in

- **Pairwise JSMA spends two pixels per iteration**, so with a 117-pixel budget it runs at most ⌊117/2⌋ = 58 iterations, and `pixels_modified` increments by 2.
- **The loop scores both directions every iteration** (`compute_pairwise_saliency` for increase and decrease) and takes the higher-scoring pair.
- **The same sample that failed single-pixel JSMA succeeds with pairwise**, using fewer pixels (68 vs 100) - the concrete proof that pair synergy matters.
- **Target confidence is highly nonlinear**: flat near zero for many iterations, then a sharp climb across the boundary once enough foundation pairs are placed.
- **Both pixels of the chosen pair are locked after use** (`search_space[p]=search_space[q]=False`), consistent with the canonical lock-on + θ=1.0 setup.
- Pixel count equals **2 × the number of iterations that actually modified** - a top-of-loop success break means the final pass modifies nothing (so 34 mods -> 68 pixels, not 70).
- Keep the `top_k` pruning in the loop or each iteration pays the full ~307k-pair cost - pruning to 128 keeps it near 8k pairs.
- **Method point:** ==pairwise + logits + θ=1.0 is the canonical, effective JSMA configuration== - pair sums capture feature interactions the single-pixel greedy selection misses, lifting success while keeping the attack sparse.

