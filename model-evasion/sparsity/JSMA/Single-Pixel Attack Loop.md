# Single-Pixel Attack Loop

## The idea in one breath

Wrap Section 17's one iteration in a `for` loop that repeats - check-success, check-budget, Jacobian, score, pick, perturb, lock the pixel - until the target is reached or the budget runs out. The twist: **on this sample it fails**, and understanding _why_ is the whole point, because it's what motivates the pairwise version later.

## Part 1 - Configuration

```python
# Reset for full attack
x_adv = x.clone().detach()
search_space = initialize_search_space(x.shape)

# Attack configuration
config = {
    'theta': 0.25,
    'gamma': 0.15,
    'max_iter': 100,
    'wrt': 'logits',
    'clip_min': 0.0,
    'clip_max': 1.0
}
```

|Key|Meaning|
|---|---|
|`theta`|step size - how far each chosen pixel moves per hit (±0.25)|
|`gamma`|sparsity budget as a **fraction** of pixels|
|`max_iter`|hard cap on iterations, so a stalled attack still stops|
|`wrt`|`'logits'` - the Section 14 rule (never softmax)|
|`clip_min/max`|legal pixel range|

## Part 2 - The pixel budget

```python
num_features = int(np.prod(x.shape[1:]))
max_pixels = int(config['gamma'] * num_features)
```

`gamma × num_features = 0.15 × 784 = 117`. So the attack may touch at most **117 pixels (14.9% of the image)** before it gives up. The abstract fraction γ becomes a concrete pixel cap.

## Part 3 - The theta/gamma trade-off (and a catch)

The course explains:

- **θ small (0.25):** gentle, subtle changes; a black pixel needs several hits to reach white. Slower.
- **θ large (1.0):** a pixel saturates 0→1 in one hit; fast but visually obvious.
- **γ:** the hard sparsity cap - a smaller γ forces the attack to be more selective.

> **Important catch - read this before the trade-off confuses you.** That "a pixel needs four selections to saturate" story describes the **canonical** JSMA. But look at the state update in Part 6: this code does `search_space[pixel_idx] = False` the instant a pixel is used, so in **this** implementation a pixel is selected **exactly once** and then locked out forever. With θ=0.25, every modified background pixel moves 0 → 0.25 and stays there - it never reaches white. The "select it again" paragraph is describing a different algorithm than the one on the page. Hold that thought; it's the reason the attack fails.

## Part 4 - Progress tracking

```python
stats = {
    'iterations': [],
    'pixels_modified': [],
    'target_confidence': [],
    'saliency_scores': []
}

pixels_modified = 0
```

Four parallel lists - element _i_ of each is iteration _i_. It's a lightweight time-series: `stats['target_confidence']` gives you the whole confidence curve at once, cheaper than a list-of-dicts.

## Part 5 - The loop: termination checks first, then gradients

```python
for iteration in range(config['max_iter']):
    # Check success condition
    if check_target_reached(x_adv, target_class, model):
        print(f"\n✓ Target reached at iteration {iteration}!")
        break

    # Check budget exhaustion
    if pixels_modified >= max_pixels:
        print(f"\n✗ Budget exhausted")
        break

    # Compute Jacobian and extract gradients
    jacobian = compute_jacobian_matrix(x_adv, model, num_classes=10, wrt=config['wrt'])
    alpha = extract_target_gradient(jacobian, target_class)
    beta = extract_other_gradients(jacobian, target_class)
    alpha_masked = apply_search_mask(alpha, search_space)
    beta_masked = apply_search_mask(beta, search_space)
```

The ordering matters: the two cheap checks (a forward pass; an integer compare) run **before** the expensive Jacobian (10 backward passes, ~10-15 ms). No point paying for gradients if you've already won or run out of budget.

## Part 6 - Score, select, and the early-out

```python
    # Score both directions and select best
    inc_scores = score_increase_saliency(alpha_masked, beta_masked)
    dec_scores = score_decrease_saliency(alpha_masked, beta_masked)
    pixel_idx, saliency, increase = select_best_direction(inc_scores, dec_scores)

    # Check for exhausted search space
    if saliency <= 0:
        print(f"\n✗ No valid pixels remaining")
        break
```

A **third** stop condition: if the best saliency is 0, no pixel satisfies the sign gate anymore (all saturated or sign-violating), so continuing would change nothing. Break.

## Part 7 - Perturb and update state

```python
    # Apply perturbation
    x_adv = apply_single_pixel_perturbation(
        x_adv, pixel_idx, config['theta'], increase,
        config['clip_min'], config['clip_max']
    )

    # Update search space
    search_space[pixel_idx] = False
    search_space = remove_saturated_pixels(search_space, x_adv, clip_min, clip_max)
```

|Code|What it does|
|---|---|
|`apply_single_pixel_perturbation(...)`|move the chosen pixel by ±θ, clamped - **permanent**, JSMA never undoes a change|
|`search_space[pixel_idx] = False`|lock the just-used pixel so it's never reselected (single-pixel JSMA = one pixel per step)|
|`remove_saturated_pixels(...)`|also drop any pixel now sitting at 0.0 or 1.0|

That `search_space[pixel_idx] = False` is the line behind the Part 3 catch: used once, gone for good.

## Part 8 - Track and print

```python
    # Track metrics
    pixels_modified += 1
    confidence = compute_confidence(x_adv, target_class, model)

    stats['iterations'].append(iteration)
    stats['pixels_modified'].append(pixels_modified)
    stats['target_confidence'].append(confidence)
    stats['saliency_scores'].append(saliency)

    # Print progress
    if iteration % 5 == 0 or iteration < 3:
        print(f"{iteration:<6} {pixels_modified:<8} {confidence:<12.4f} {saliency:<12.6f}")
```

Prints the first 3 iterations plus every 5th, so the log stays short.

## Part 9 - Final evaluation

```python
final_success = check_target_reached(x_adv, target_class, model)
final_pred = model(x_adv).argmax(dim=1).item()

print("\n" + "="*46)
print(f"Final result: {'SUCCESS' if final_success else 'FAILED'}")
print(f"Final prediction: {final_pred}")
print(f"Pixels modified: {pixels_modified}/{max_pixels}")
print(f"Iterations: {len(stats['iterations'])}")
```

```
Final result: FAILED
Final prediction: 7
Pixels modified: 100/117
Iterations: 100
```

## Reading the result (this is the lesson)

The attack **failed**: still predicts 7, used 100 pixels, and hit the 100-iteration cap _before_ the 117-pixel budget. Two signals in the trace tell the story:

- **Saliency decays** 0.641 → 0.031: the best pixels get spent first, and each later pixel contributes less. A greedy attack eating its own high-value moves.
- **Confidence barely moves** (0.0000 → 0.0002): it never gets within a mile of the 0.5 boundary.

Why it's so weak, concretely:

1. A **decrease** on a black (0.0) background pixel does nothing - `clamp` holds it at 0.0 - and then it's removed as saturated. Wasted move.
2. Every pixel that _does_ move is **locked after one 0.25 step**, so it lands at grey 0.25 and can never reach white. You end up scattering faint grey dots that don't add up to a "2".

> **An honest note the course half-admits:** it calls this a "baseline" and says the "canonical algorithm uses pairwise selection." The deeper reason it fails is the combination above - single-use-per-pixel + both-direction saturation removal - which leaves each pixel at θ. Don't take this failure as "JSMA doesn't work"; take it as "this deliberately hobbled single-pixel variant doesn't, and the next sections fix it." The real JSMA either keeps picking the same pixel until it saturates, or modifies **pixel pairs**, both of which deliver far more punch per step.

## Copy-paste cell (mine, not the course's)

The whole loop wrapped into one reusable function, returning the adversarial image, whether it worked, and the stats for plotting - so you can run it on any sample, not just the demo one. Assumes Sections 15-17 functions are already defined:

```python
# === Full single-pixel JSMA (Section 18) ===
def jsma_single_pixel(x, target_class, model, theta=0.25, gamma=0.15, max_iter=100,
                      wrt='logits', clip_min=0.0, clip_max=1.0, num_classes=10, verbose=False):
    x_adv = x.clone().detach()
    search_space = initialize_search_space(x.shape)
    max_pixels = int(gamma * int(np.prod(x.shape[1:])))
    stats = {'iterations': [], 'pixels_modified': [], 'target_confidence': [], 'saliency_scores': []}
    pixels_modified = 0

    for iteration in range(max_iter):
        if check_target_reached(x_adv, target_class, model):
            break
        if pixels_modified >= max_pixels:
            break

        J = compute_jacobian_matrix(x_adv, model, num_classes=num_classes, wrt=wrt)
        a = apply_search_mask(extract_target_gradient(J, target_class), search_space)
        b = apply_search_mask(extract_other_gradients(J, target_class), search_space)
        inc, dec = score_increase_saliency(a, b), score_decrease_saliency(a, b)
        pixel_idx, saliency, increase = select_best_direction(inc, dec)
        if saliency <= 0:
            break

        x_adv = apply_single_pixel_perturbation(x_adv, pixel_idx, theta, increase, clip_min, clip_max)
        search_space[pixel_idx] = False
        search_space = remove_saturated_pixels(search_space, x_adv, clip_min, clip_max)

        pixels_modified += 1
        conf = compute_confidence(x_adv, target_class, model)
        stats['iterations'].append(iteration)
        stats['pixels_modified'].append(pixels_modified)
        stats['target_confidence'].append(conf)
        stats['saliency_scores'].append(saliency)
        if verbose and (iteration % 5 == 0 or iteration < 3):
            print(f"{iteration:<6} {pixels_modified:<8} {conf:<12.4f} {saliency:<12.6f}")

    success = check_target_reached(x_adv, target_class, model)
    return x_adv, success, stats

# x_adv, ok, stats = jsma_single_pixel(x, target_class, model, verbose=True)
# print("SUCCESS" if ok else "FAILED", "| pixels:", len(stats['iterations']))
```

## Quick self-check

You run this **exact** implementation on a CIFAR image (3 channels, 32×32) with `gamma=0.2` and `theta=0.5`.

1. What is `max_pixels`?
2. How many times can one particular pixel be selected before it's locked out?

I'll confirm next message.

## The facts worth locking in for the exam

- ==The pixel budget is `max_pixels = int(gamma × num_features)`== (0.15 × 784 = 117 here), a hard cap that ends the attack even if it hasn't succeeded.
- **Check success and budget before computing the Jacobian**, because the Jacobian's 10 backward passes are the expensive step and both checks are nearly free.
- **The loop has three exits: target reached, budget exhausted, and zero saliency** (no valid pixel left) - a well-formed greedy attack needs all three.
- **In this single-pixel implementation each pixel is used exactly once** (`search_space[pixel_idx] = False` right after modifying), so with θ=0.25 a pixel only ever moves 0.25 and never fully saturates - unlike canonical JSMA.
- **Saliency scores decay over the run** as the best pixels are consumed first, and a flat near-zero confidence curve is the signature of an attack that isn't crossing the boundary.
- **This baseline fails on the sample** (confidence peaks at 0.0002, prediction stays 7) because decreases on black pixels do nothing and each used pixel is frozen at grey 0.25.
- Track metrics in parallel lists for a cheap time-series, and take the final iteration count from `len(stats[...])` so early termination is counted correctly.
- **Method point:** **single-pixel JSMA is a teaching baseline whose weakness motivates the canonical pairwise version** - modifying two complementary pixels per step (or re-selecting until saturation) is what gives real JSMA the leverage to cross the boundary within budget.

Paste Section 19 when ready - that's the batch evaluation, where you'll see this single-pixel variant's success rate across many samples (and it won't be pretty, which sets up the pairwise upgrade).