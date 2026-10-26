# Complete FISTA Iteration And Binary Search

## The idea in one breath

The pieces from Sections 4-8 finally snap together. **One FISTA step** = evaluate the loss at the look-ahead point → gradient step → soft-threshold → extrapolate the next look-ahead point. A **success check** gives a yes/no verdict. The **binary search** uses that verdict to tune c per image. Two nested loops: inner FISTA finds the perturbation, outer binary search tunes the pressure, **both adapting to each image's difficulty**.

> **Heads-up:** no mangled formulas this time - the section is mostly code. Two reminders: `y(k)` is the momentum/look-ahead point, `x(k)` is the actual current solution, and the soft-threshold level here is `learning_rate * beta` (that's η·β from Section 4).

---

## Part 1 - One FISTA step

```python
def fista_step(
    adv_images,
    y_momentum,
    original_images,
    labels_onehot,
    const,
    model,
    beta,
    learning_rate,
    confidence,
    iteration,
    targeted=False,
    clip_min=0.0,
    clip_max=1.0,
):
    """
    Perform one complete FISTA iteration.

    Combines gradient computation, shrinkage-thresholding,
    and momentum update into a single optimization step.

    Parameters:
        adv_images (torch.Tensor): Current adversarial images
        y_momentum (torch.Tensor): Momentum point for gradient evaluation
        original_images (torch.Tensor): Original clean images
        labels_onehot (torch.Tensor): One-hot encoded labels
        const (torch.Tensor): Trade-off constants per example
        model (nn.Module): Target model
        beta (float): L1 weight parameter
        learning_rate (float): FISTA step size
        confidence (float): Margin for misclassification
        iteration (int): Current FISTA iteration number for momentum calculation
        targeted (bool): Whether this is a targeted attack
        clip_min (float): Minimum valid pixel value
        clip_max (float): Maximum valid pixel value

    Returns:
        tuple: (new_adv_images, new_y_momentum, loss_value, distances)
    """
    # Ensure y_momentum requires gradients for backprop
    y_momentum = y_momentum.detach().requires_grad_(True)

    # Compute loss at momentum point
    total_loss, adversarial_loss, distances = compute_total_loss(
        y_momentum,
        original_images,
        labels_onehot,
        const,
        model,
        beta,
        confidence,
        targeted,
    )

    # Compute gradient of total loss w.r.t. momentum point
    total_loss_summed = total_loss.sum()
    total_loss_summed.backward()
    grad = y_momentum.grad

    # Gradient step: move in negative gradient direction
    y_new = y_momentum - learning_rate * grad

    # Apply shrinkage-thresholding (proximal operator for L1)
    adv_new = apply_shrinkage_thresholding(
        y_new, original_images, learning_rate * beta, clip_min, clip_max
    )

    # Compute momentum coefficient
    momentum_coef = compute_fista_momentum(iteration)

    # Update momentum point for next iteration
    y_new_momentum = adv_new + momentum_coef * (adv_new - adv_images)

    return adv_new, y_new_momentum, total_loss_summed.item(), distances
```

|Code|What it does|Section it came from|
|---|---|---|
|`y_momentum.detach().requires_grad_(True)`|break last step's graph, start a fresh one| - |
|`compute_total_loss(y_momentum, ...)`|loss at the **look-ahead point**, not the current solution|6|
|`total_loss.sum().backward()`|one backward pass; sum lets each image's gradient reach its own pixels|8|
|`y_new = y_momentum - learning_rate * grad`|gradient step on the smooth part|4|
|`apply_shrinkage_thresholding(y_new, ..., learning_rate * beta, ...)`|soft-threshold at η·β → the L1 part|4, 7|
|`momentum_coef = compute_fista_momentum(iteration)`|the k/(k+3) coefficient|7|
|`y_new_momentum = adv_new + momentum_coef * (adv_new - adv_images)`|extrapolate the next look-ahead point|4|

This is exactly the FISTA recipe from Section 4: `x_{k+1} = adv_new`, and `y_{k+1} = x_{k+1} + coef·(x_{k+1} − x_k)` where `x_k = adv_images`.

### The two subtle lines

- **Why `detach().requires_grad_(True)`?** The `detach()` **cuts the computational graph from the previous iteration**. Without it, PyTorch would chain all iterations together - a memory leak and wrong gradients. `requires_grad_(True)` then starts tracking fresh for this step only.
- **Why evaluate at `y_momentum`, not `adv_images`?** This is the **FISTA look-ahead**. Measuring the gradient at the extrapolated point (where momentum says you're heading) is what beats plain proximal gradient and gives the O(1/k²) speed.

---

## Part 2 - The success check

```python
def check_attack_success(adv_images, labels, model, targeted=False):
    """
    Verify whether adversarial examples achieve misclassification.

    Compares model predictions on adversarial images to target labels.

    Parameters:
        adv_images (torch.Tensor): Adversarial images to evaluate
        labels (torch.Tensor): True labels (or target labels if targeted)
        model (nn.Module): Target model
        targeted (bool): Whether this is a targeted attack

    Returns:
        torch.Tensor: Boolean mask indicating successful attacks (batch_size,)
    """
    with torch.no_grad():
        outputs = model(adv_images)
        predictions = outputs.argmax(dim=1)

        if targeted:
            # Success = prediction matches target
            success = predictions.eq(labels)
        else:
            # Success = prediction differs from true label
            success = predictions.ne(labels)

    return success
```

|Code|What it does|
|---|---|
|`with torch.no_grad()`|no gradients - this is just an evaluation, not optimization|
|`outputs.argmax(dim=1)`|the predicted class per image|
|`targeted: predictions.eq(labels)`|==success = prediction _equals_ the target==|
|`else: predictions.ne(labels)`|==success = prediction _differs_ from the true class==|

`labels` means the **true** labels in untargeted mode, the **target** labels in targeted mode - the same convention as the one-hot in Section 6. This is a **hard, binary verdict**: fooled or not, no "almost." It contrasts with the soft margin loss, which gives smooth gradients _during_ optimization; the success check gives the definitive yes/no _after_ it, for the binary search to act on.

---

## Part 3 - Updating the binary search bounds

```python
def update_binary_search_bounds(lower_bound, upper_bound, const, success_mask):
    """
    Update binary search bounds based on attack success.

    Successful attacks lower upper bound (c was sufficient).
    Failed attacks raise lower bound (c was insufficient).

    Parameters:
        lower_bound (torch.Tensor): Lower bounds on c per example
        upper_bound (torch.Tensor): Upper bounds on c per example
        const (torch.Tensor): Current c values per example
        success_mask (torch.Tensor): Boolean mask of successful attacks

    Returns:
        tuple: (new_lower_bound, new_upper_bound, new_const)
    """
    # Process each example individually
    for i in range(len(success_mask)):
        if success_mask[i]:
            # Success: try smaller c
            upper_bound[i] = min(upper_bound[i], const[i])
            if upper_bound[i] < 1e10:
                const[i] = (lower_bound[i] + upper_bound[i]) / 2
        else:
            # Failure: need larger c
            lower_bound[i] = max(lower_bound[i], const[i])
            if upper_bound[i] < 1e10:
                const[i] = (lower_bound[i] + upper_bound[i]) / 2
            else:
                const[i] *= 10 # Exponential increase for persistent failures

    return lower_bound, upper_bound, const
```

|Code|What it does|
|---|---|
|`for i in range(len(success_mask))`|each image gets its own c, lower, upper|
|success → `upper_bound[i] = min(upper_bound[i], const[i])`|c worked → it may be too big → pull the ceiling down|
|failure → `lower_bound[i] = max(lower_bound[i], const[i])`|c failed → raise the floor|
|`if upper_bound[i] < 1e10: const = (lower+upper)/2`|**both bounds finite → bisect (midpoint)**|
|`else: const[i] *= 10`|**no success yet → multiply c by 10**|

### Two phases (verified)

**Phase 1 - no success yet (upper still ∞), so grow ×10:**

```
c = 0.001 → 0.01 → 0.1 → 1 → 10 → ...
```

This sweeps the scale space fast instead of crawling up in tiny steps.

**Phase 2 - first success sets a finite ceiling, so bisect:**

|Event|lower|upper|new c|
|---|---|---|---|
|start|0|1e10|0.001|
|success|0|**0.001**|(0 + 0.001)/2 = **0.0005**|
|failure|**0.0005**|0.001|(0.0005 + 0.001)/2 = **0.00075**|

Each bisection halves the interval, so ==5 steps narrow it 2⁵ = 32×==. That's the logarithmic convergence from Section 2.

---

## Part 4 - How the loops nest

```
OUTER: binary search over c (tune adversarial pressure)
  └─ INNER: FISTA to convergence or max-iter (find the minimal perturbation for this c)
       └─ then: check_attack_success → update_binary_search_bounds
```

|Loop|Job|Adapts via|
|---|---|---|
|Inner (FISTA)|minimize distortion for a fixed c|iteration count|
|Outer (binary search)|find the smallest c that still fools|success feedback|

Crucially, **each binary-search round restarts FISTA from the original image** (zero perturbation). That fresh start stops perturbations from ballooning across rounds. The result is automatic adaptation: **easy images** need few FISTA iterations and settle on a small c fast; **hard images** need many iterations and a bigger c.

### Why this is unique to ElasticNet

|Attack|Adaptation|
|---|---|
|FGSM|none - one fixed ε for every image|
|DeepFool|iteration count only, same algorithm for all|
|ElasticNet|**both FISTA iterations (perturbation) and binary search (pressure)**|

---

## Copy-paste cell (mine, not the course's)

### Trace the binary search by hand

Confirms the bisect-then-grow logic with a few simulated verdicts.

```python
def update_binary_search_bounds(lower_bound, upper_bound, const, success_mask):
    for i in range(len(success_mask)):
        if success_mask[i]:
            upper_bound[i] = min(upper_bound[i], const[i])
            if upper_bound[i] < 1e10:
                const[i] = (lower_bound[i] + upper_bound[i]) / 2
        else:
            lower_bound[i] = max(lower_bound[i], const[i])
            if upper_bound[i] < 1e10:
                const[i] = (lower_bound[i] + upper_bound[i]) / 2
            else:
                const[i] *= 10
    return lower_bound, upper_bound, const

lo, up, c = [0.0], [1e10], [0.001]
for verdict in [False, False, True, False, True]: # F,F grow x10; then bisect
    lo, up, c = update_binary_search_bounds(lo, up, c, [verdict])
    print(f"{'success' if verdict else 'failure':8} -> lower {lo[0]:.5f} upper {up[0]:.5g} c {c[0]:.5f}")
```

Expected:

```txt
failure -> lower 0.00100 upper 1e+10 c 0.01000
failure -> lower 0.01000 upper 1e+10 c 0.10000
success -> lower 0.01000 upper 0.1 c 0.05500
failure -> lower 0.05500 upper 0.1 c 0.07750
success -> lower 0.05500 upper 0.0775 c 0.06625
```

See it grow ×10 while failing, then bisect once a success caps the ceiling.

---

## Quick self-check

Binary search state for one image: `lower = 0.0005`, `upper = 0.001`, `const = 0.00075`. This FISTA round **fails** (no misclassification). What are the new `lower`, `upper`, and `const`?

---

## The facts worth locking in for the exam

- **One `fista_step` evaluates the loss at the look-ahead point `y_momentum` (not the current solution), takes a gradient step, soft-thresholds at η·β, then extrapolates the next momentum point** - the full gradient-descent-plus-proximal-plus-momentum recipe in one function.
- **`detach().requires_grad_(True)` breaks the previous iteration's graph and starts a fresh one**, preventing cross-iteration gradient accumulation and memory leaks.
- **The success check is a hard binary verdict under `torch.no_grad()`: `argmax` equals the target (targeted) or differs from the true class (untargeted)**, unlike the soft margin loss that drives optimization.
- **Binary search keeps per-example lower/upper bounds on c: success lowers the upper bound, failure raises the lower bound, and when both are finite it bisects to the midpoint**, halving the interval each step (5 steps → 32× narrower).
- **Before any success the upper bound is infinite, so c is multiplied by 10 each failure** (0.001 → 0.01 → 0.1 → ...) to sweep scales fast; the first success switches it to bisection.
- **FISTA (inner loop) minimizes distortion for a fixed c; binary search (outer loop) tunes c from success feedback**, and each binary-search round restarts FISTA from the original image so perturbations never grow unboundedly.
- **ElasticNet adapts on both axes - FISTA iterations and binary-search pressure - unlike FGSM (fixed ε, no adaptation) and DeepFool (iteration count only)**, giving fine-grained per-example control.
- **Method point:** **separate the two subproblems - "find the perturbation" (FISTA) and "tune the pressure" (binary search) - into nested loops with a clean success signal between them**, rather than entangling distortion minimization and strength tuning into one loop.

Logged Section 9 and the self-check answer key. Paste Section 10 (Attack Execution and Performance) when ready - that's where these three functions get driven by the full nested loop on real MNIST images.