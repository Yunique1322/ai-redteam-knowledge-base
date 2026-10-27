# Section 10: Attack Execution and Performance

## The idea in one breath

The driver that runs the whole attack. Set hyperparameters, grab 20 **correctly classified** MNIST images, then run the nested loops: **5 binary-search steps, each running up to 1000 FISTA iterations**, keeping the smallest successful perturbation per image. For 20 images that's ~5,000 forward + 5,000 backward passes - a few minutes on GPU, up to an hour on CPU.

> **Heads-up:** no mangled formulas. One naming reminder carried from Section 5: the results print "Average Squared L2 Distortion" - that's ‖δ‖₂² with no square root, while L∞ is computed separately as the true max-per-pixel.

---

## Part 1 - Attack configuration

```python
# Attack hyperparameters
config = {
    "beta": 0.01, # L1 vs L2 trade-off (higher = sparser)
    "confidence": 0, # Margin for misclassification
    "learning_rate": 0.01, # FISTA step size
    "max_iterations": 1000, # FISTA iterations per binary search
    "binary_search_steps": 5, # Number of binary search iterations
    "initial_const": 0.001, # Starting trade-off constant
    "clip_min": 0.0, # Minimum pixel value
    "clip_max": 1.0, # Maximum pixel value
}

print("\nAttack Configuration:")
for key, value in config.items():
    print(f" {key}: {value}")
```

|Parameter|Value|What it controls|Course's guidance|
|---|---|---|---|
|`beta`|0.01|L1 vs L2 (sparsity)|0.05-0.1 much sparser; 0.001-0.005 nearly dense (≈ pure L2)|
|`confidence`|0|the κ margin|0 = flip on any lead change; 5 = demand a 5-logit lead (robust, bigger)|
|`learning_rate`|0.01|FISTA step size|>0.05 -> diverge/oscillate; <0.001 -> needs 2000-5000 iters|
|`max_iterations`|1000|FISTA steps per binary step| - |
|`binary_search_steps`|5|outer loop count|3-5 fast, 7-9 precise|
|`initial_const`|0.001|starting c|grows ×10 until first success|

Each parameter maps to a concept you've already met: `beta` is Section 2's sparsity knob, `confidence` is Section 6's κ, `learning_rate` is the η in Section 9's gradient step, `initial_const` seeds Section 9's binary search.

---

## Part 2 - Sample selection

```python
print("\nSelecting correctly classified samples for attack...")
model.eval()
num_samples = 20

for data, targets in test_loader:
    data, targets = data.to(device), targets.to(device)
    outputs = model(data)
    predictions = outputs.argmax(dim=1)

    # Select correctly classified samples
    correct_mask = predictions.eq(targets)
    attack_data = data[correct_mask][:num_samples]
    attack_targets = targets[correct_mask][:num_samples]

    if len(attack_data) >= num_samples:
        print(f"Selected {len(attack_data)} samples")
        break
```

|Code|What it does|
|---|---|
|`correct_mask = predictions.eq(targets)`|True where the model got it right|
|`data[correct_mask][:num_samples]`|keep only correctly classified images, up to 20|
|`if len(attack_data) >= num_samples: break`|stop once we have enough|

**Only attack images the model already classifies correctly** - fooling an image it already gets wrong proves nothing. This is the same "count success only on originally-correct images" discipline you learned in DeepFool.

---

## Part 3 - Initialization

```python
print("\nPerforming ElasticNet attack...")
print("This may take several minutes due to iterative optimization...\n")

batch_size = len(attack_data)
original_images = attack_data.clone()

# Convert labels to one-hot encoding
labels_onehot = torch.zeros(batch_size, 10).to(device)
labels_onehot.scatter_(1, attack_targets.unsqueeze(1), 1)

# Initialize binary search bounds
lower_bound = torch.zeros(batch_size).to(device)
upper_bound = torch.ones(batch_size).to(device) * 1e10
const = torch.ones(batch_size).to(device) * config["initial_const"]

# Track best adversarial examples
best_adv = original_images.clone()
best_l2 = torch.ones(batch_size).to(device) * 1e10
```

|Code|What it does|
|---|---|
|`original_images = attack_data.clone()`|keep a pristine copy to measure change against|
|`labels_onehot.scatter_(1, attack_targets.unsqueeze(1), 1)`|**build one-hot labels** (1 at the true class)|
|`lower_bound = zeros`|no pressure floor|
|`upper_bound = ones * 1e10`|**effectively infinite ceiling** (the "unbounded" flag)|
|`const = ones * 0.001`|every image starts at c = 0.001|
|`best_adv = original.clone()`|best-so-far starts as the untouched image|
|`best_l2 = ones * 1e10`|best distortion starts "infinitely bad" so anything beats it|

`scatter_(1, idx, 1)` writes a 1 into each row at the column given by the label (I verified it produces the correct one-hot). **One-hot is used because Section 6's loss multiplies it against logits as a mask** - cheaper than index lookups inside a loop that runs thousands of times. Wide initial bounds (0 to 1e10) let the search cover any c the attack might need.

---

## Part 4 - The nested loops

```python
# Binary search over trade-off constant c
for binary_step in range(config["binary_search_steps"]):
    print(f"Binary search step {binary_step + 1}/{config['binary_search_steps']}")

    # Initialize FISTA from original images
    adv_images = original_images.clone().detach()
    y_momentum = adv_images.clone()

    # FISTA optimization loop
    for iteration in range(config["max_iterations"]):
        # Perform one FISTA step
        adv_images, y_momentum, loss, distances = fista_step(
            adv_images,
            y_momentum,
            original_images,
            labels_onehot,
            const,
            model,
            config["beta"],
            config["learning_rate"],
            config["confidence"],
            iteration,
            targeted=False,
            clip_min=config["clip_min"],
            clip_max=config["clip_max"],
        )

    # Check which examples successfully fooled the model
    success_mask = check_attack_success(
        adv_images, attack_targets, model, targeted=False
    )

    # Update best adversarial examples
    l1_dist, l2_dist, elastic_dist = compute_distances(
        adv_images, original_images, config["beta"]
    )

    for i in range(batch_size):
        if success_mask[i] and l2_dist[i] < best_l2[i]:
            best_adv[i] = adv_images[i]
            best_l2[i] = l2_dist[i]

    # Update binary search bounds
    lower_bound, upper_bound, const = update_binary_search_bounds(
        lower_bound, upper_bound, const, success_mask
    )

    # Progress reporting
    num_success = success_mask.sum().item()
    print(f" Successfully generated {num_success}/{batch_size} adversarial examples\n")
```

The structure:

|Loop level|Code|Job|
|---|---|---|
|Outer (5×)|`for binary_step in range(...)`|tune c|
| - reset|`adv_images = original.clone().detach()`|**restart from the clean image each step**|
|Inner (1000×)|`for iteration in range(...)` -> `fista_step(...)`|find the perturbation for this c|
|After inner|`check_attack_success(...)`|who got fooled?|
|Save best|`if success_mask[i] and l2_dist[i] < best_l2[i]`|**keep only successes with smaller squared-L2**|
|Update|`update_binary_search_bounds(...)`|move c for next round|

Two things worth circling:

- **Restart from original each binary step.** Without `adv_images = original.clone().detach()`, a perturbation tuned for c = 0.001 would **carry over and contaminate** the c = 0.01 round. Fresh start = fair evaluation per constant (this is the Section 9 point in action).
- **Best-tracking keeps the minimum.** A perturbation is saved only if it **both fools the model and has smaller squared-L2 than any previous success** for that image. So `best_adv` always holds the smallest successful change found across all 5 steps.

> **An honest note the course skips:** Section 8 described a "hybrid" stop that ends a binary step early when the whole batch is fooled. **This particular code doesn't do that** - the inner loop always runs the full 1000 iterations regardless of early success. So it's correct but not speed-optimized the way Section 8 implied; if you wanted the early-exit, you'd add a success check with a `break` inside the inner loop.

---

## Part 5 - Results

```python
# Evaluate final adversarial examples
with torch.no_grad():
    adv_outputs = model(best_adv)
    adv_predictions = adv_outputs.argmax(dim=1)

    # Calculate success rate
    final_success = adv_predictions.ne(attack_targets)
    success_rate = final_success.float().mean().item() * 100

    # Calculate final distortions
    l1_dist, l2_dist, elastic_dist = compute_distances(
        best_adv, original_images, config["beta"]
    )
    linf_dist = torch.max(
        torch.abs(best_adv - original_images).view(batch_size, -1), dim=1
    )[0]

# Display results
print("=" * 60)
print("ElasticNet Attack Results:")
print("=" * 60)
print(f"Success Rate: {success_rate:.2f}% ({final_success.sum()}/{batch_size})")
print(f"Average L1 Distortion: {l1_dist.mean().item():.4f}")
print(f"Average Squared L2 Distortion: {l2_dist.mean().item():.4f}")
print(f"Average L∞ Distortion: {linf_dist.mean().item():.4f}")
print(f"Average Elastic Distortion: {elastic_dist.mean().item():.4f}")
print("=" * 60)
```

|Code|What it does|
|---|---|
|`adv_predictions.ne(attack_targets)`|which `best_adv` images now misclassify|
|`success_rate = final_success.float().mean() * 100`|% fooled|
|`compute_distances(best_adv, ...)`|L1, squared-L2, elastic|
|`torch.max(torch.abs(...).view(batch_size, -1), dim=1)[0]`|**L∞: biggest single-pixel change per image**|

Four rulers, four views of the same change: L1 (total movement), squared-L2 (energy), L∞ (worst single pixel), elastic (the blend). A subtle correctness point: images that ==never succeeded keep `best_adv = original`==, so they predict the true label, correctly counting as failures in the success rate.

---

## Part 6 - Performance

### The cost (verified)

5 binary steps × 1000 FISTA iterations = **5,000 forward + 5,000 backward passes** on the batch, ≈ 100,000 per-image evaluations at batch size 20. On an RTX 3090 or better: 2-3 minutes. On CPU: 30-60 minutes.

### Tuning trade-offs

|Change|Effect|
|---|---|
|Binary steps 9 -> 5|−45% compute, +5-15% distortion (less precise c)|
|Binary steps -> 9|−3-8% distortion, +80% compute|
|learning_rate > 0.05|**diverges** (oscillates)|
|learning_rate < 0.001|needs 2000-5000 iters (4× runtime)|
|beta 0.01 -> 0.02|+10-15 pp sparsity, −5-10% success on hard examples|
|batch 20 -> 50|+15-25% throughput, 1 GB memory|
|batch 20 -> 10|−10-15% throughput, 200 MB memory|

The use-case guidance: **adversarial training wants speed (3-5 binary steps); robustness evaluation wants precision (7-9)**.

---

## Copy-paste cell (mine, not the course's)

### Confirm the one-hot and the compute budget

```python
import torch
t = torch.tensor([3, 7, 0])
oh = torch.zeros(3, 10)
oh.scatter_(1, t.unsqueeze(1), 1) # build one-hot like the attack does
print("one-hot argmax:", oh.argmax(1).tolist()) # -> [3, 7, 0]

steps, fista, n = 5, 1000, 20
print(f"batch passes: {steps*fista} fwd + {steps*fista} bwd | per-image evals: {steps*fista*n}")
```

Expected:

```txt
one-hot argmax: [3, 7, 0]
batch passes: 5000 fwd + 5000 bwd | per-image evals: 100000
```

## Key facts to lock in

- **The attack only targets images the model already classifies correctly** (`predictions.eq(targets)`), since fooling an already-misclassified image proves nothing about robustness.
- **The outer binary-search loop (5 steps) restarts FISTA from the original image each time** (`adv_images = original.clone().detach()`), so a perturbation tuned for one c never contaminates the next.
- **`best_adv` saves a perturbation only when it both fools the model and has smaller squared-L2 than any prior success** for that image, so the final result is the minimal successful change found across all binary steps.
- **Labels are one-hot encoded with `scatter_(1, targets.unsqueeze(1), 1)`** because the C&W loss uses the one-hot as a mask to extract logits cheaply inside the thousands-of-iterations inner loop.
- ==Binary bounds start wide (lower = 0, upper = 1e10) and c starts at 0.001==, letting the search grow c ×10 until a first success then bisect, covering any difficulty.
- **The run costs ~5,000 forward + 5,000 backward passes for 20 images (5 × 1000)**, a few minutes on GPU and up to an hour on CPU - orders of magnitude more than FGSM's single pass.
- **Learning rate is the most sensitive knob: >0.05 diverges, <0.001 quadruples runtime**, while binary-search steps trade precision for speed (3-5 for adversarial training, 7-9 for robustness evaluation).
- **Method point:** **track the best successful result separately from the live optimization state**, comparing each round's distortion against the stored best, so that exploring more aggressive constants can never make your saved answer worse.

