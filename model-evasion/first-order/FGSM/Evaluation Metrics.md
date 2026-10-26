# Evaluation Metrics

## The idea in one breath

"How many predictions flipped" is only part of the story. This section builds `evaluate_attack`, which reports four kinds of numbers for one batch:

- **Accuracy** before and after the attack
- **Attack success rate**: flips among only the images that were originally correct
- **Confidence**: how sure the model is of the _true_ class, before and after
- **Perturbation size**: average L2 per image, and the largest single-pixel change (L∞) to prove the ε budget was respected

Together they answer how often it worked, how badly it confused the model, and how big the change was.

## Part 1: The complete function

The course splits **one** function, `evaluate_attack`, across four consecutive blocks. None of the pieces runs on its own. I've reassembled them into the single real function below, with every line unchanged and in order. Run it as one cell.

```python
from typing import Dict

def evaluate_attack(model: nn.Module,
                   clean_images: Tensor,
                   adversarial_images: Tensor,
                   true_labels: Tensor) -> Dict[str, float]:
    """Compute accuracy, success rate, confidence shift, and norms.

    Args:
        model: Evaluated classifier in evaluation mode
        clean_images: Clean inputs in the model's expected domain (e.g., normalized MNIST)
        adversarial_images: Adversarial counterparts in the same domain as `clean_images`
        true_labels: Ground-truth labels

    Returns:
        Dict[str, float]: Aggregated metrics summarizing attack impact
    """
    model.eval()
    with torch.no_grad():
        clean_logits = model(clean_images)
        adv_logits = model(adversarial_images)

        clean_probs = F.softmax(clean_logits, dim=1)
        adv_probs = F.softmax(adv_logits, dim=1)

        clean_pred = clean_logits.argmax(dim=1)
        adv_pred = adv_logits.argmax(dim=1)

        clean_correct = (clean_pred == true_labels)
        adv_correct = (adv_pred == true_labels)

        originally_correct = clean_correct
        flipped = (~adv_correct) & originally_correct

        conf_clean = clean_probs.gather(1, true_labels.view(-1, 1)).squeeze(1)
        conf_adv = adv_probs.gather(1, true_labels.view(-1, 1)).squeeze(1)

        l2 = (adversarial_images - clean_images).view(clean_images.size(0), -1).norm(p=2, dim=1)
        linf = (adversarial_images - clean_images).abs().amax()

        return {
            "clean_accuracy": clean_correct.float().mean().item(),
            "adversarial_accuracy": adv_correct.float().mean().item(),
            # Success rate among originally correct samples only
            "attack_success_rate": (
                flipped.float().sum() / originally_correct.float().sum().clamp_min(1.0)
            ).item(),
            "avg_clean_confidence": conf_clean.mean().item(),
            "avg_adv_confidence": conf_adv.mean().item(),
            "avg_confidence_drop": (conf_clean - conf_adv).mean().item(),
            "avg_l2_perturbation": l2.mean().item(),
            "max_linf_perturbation": linf.item(),
        }
```

Everything after `with torch.no_grad():` is indented inside it, including the `return`. **`no_grad()` is correct here because this function only _measures_; it never attacks.** That's the opposite of the attack function (Section 6).

The rest of this section walks through the function in the course's four parts.

## Part 2: Predictions and probabilities (the first course block)

### Toy first: softmax turns scores into probabilities

Take 3 classes with logits `[2, 1, 0]`. Softmax raises _e_ (≈ 2.718) to each score, then divides by the total:

```
e² = 7.389 e¹ = 2.718 e⁰ = 1.000 total = 11.107

probs = [7.389/11.107, 2.718/11.107, 1.000/11.107]
      = [0.665, 0.245, 0.090] → adds up to 1.0
```

The biggest logit gets the biggest probability, and every probability is positive. **Softmax turns raw scores into "how sure am I of each class," and the row adds up to 1.**

`argmax` then picks the position of the biggest value: index 0 here.

### The real lines

| Code | What it does | Shape for 128 images |
| ---------------------------------------- | -------------------------------------------------------- | ----------------------------- |
| `model.eval()` | evaluation mode (no dropout / batch-norm updates) | |
| `clean_logits = model(clean_images)` | raw scores for clean images | [128, 10] |
| `adv_logits = model(adversarial_images)` | raw scores for attacked images | [128, 10] |
| `F.softmax(..., dim=1)` | softmax across the 10 classes (`dim=1` = along each row) | [128, 10], each row sums to 1 |
| `.argmax(dim=1)` | predicted class for each image | [128] |

## Part 3: Accuracy and success masks (the second course block)

|Code|Meaning|
|---|---|
|`clean_correct = (clean_pred == true_labels)`|True where the model was right on the clean image|
|`adv_correct = (adv_pred == true_labels)`|True where it's still right after the attack|
|`originally_correct = clean_correct`|just a clearer name for the same thing|
|`~adv_correct`|`~` means **NOT**: True where the model is now **wrong**|
|`flipped = (~adv_correct) & originally_correct`|was right before **and** is wrong now|

The course's example: 125 of 128 originally correct and 80 flipped gives 80 ÷ 125 = **0.64**.

**The `.float().mean()` trick** (used in the `return`): True counts as 1 and False as 0, so the average of a True/False list is the **fraction** that are True.

```
[T, T, F, T] → [1, 1, 0, 1] → mean = 3/4 = 0.75
```

## Part 4: Confidence with `gather` (the third course block)

### Toy first: pick each image's true-class probability

Two images, 3 classes:

```
clean_probs = [[0.10, 0.70, 0.20], ← image 0
               [0.05, 0.15, 0.80]] ← image 1
true_labels = [1, 2]
```

|Step|Code|Result|
|---|---|---|
|1. reshape labels into a column|`true_labels.view(-1, 1)`|`[[1], [2]]`|
|2. pick column 1 from row 0, column 2 from row 1|`clean_probs.gather(1, ...)`|`[[0.70], [0.80]]`|
|3. remove the extra brackets|`.squeeze(1)`|`[0.70, 0.80]`|

**`gather` answers "how much probability did the model give the _correct_ answer, for each image?"** The `1` in `gather(1, ...)` means "pick along the columns (the classes)."

The course's example: true-class confidence going from 0.92 to 0.31 is a drop of 0.61 (61 percentage points), and that counts as damage even if the prediction didn't flip.

> **Note:** `avg_adv_confidence` is the confidence in the **true** class after the attack, not in whatever the model now predicts. A low number means "the model no longer believes the right answer." It doesn't tell you how sure the model is of its wrong answer. One of the cells further down measures that.

## Part 5: Perturbation norms and the results dictionary (the fourth course block)

### Toy first: per-image L2 vs batch-wide L∞

Two tiny 2×2 perturbations (adversarial minus clean):

```
image A: [[0.8, 0.0], [−0.8, 0.0]] → flatten → [0.8, 0, −0.8, 0]
         L2 = √(0.64 + 0 + 0.64 + 0) = √1.28 = 1.131

image B: [[0.8, −0.8], [0.8, −0.8]] → flatten → [0.8, −0.8, 0.8, −0.8]
         L2 = √(4 × 0.64) = √2.56 = 1.600

avg L2 = (1.131 + 1.600) / 2 = 1.366
L∞ = biggest |change| anywhere in the batch = 0.8
```

### The real lines

|Code|What it does|
|---|---|
|`(adversarial_images - clean_images)`|the perturbation δ for every image|
|`.view(clean_images.size(0), -1)`|flattens each image: [128, 1, 28, 28] becomes [128, 784]|
|`.norm(p=2, dim=1)`|L2 of each row, i.e. one number per image, shape [128]|
|`.abs().amax()`|the largest absolute change in the **whole batch**, one number|

So **L2 is computed **per image, then averaged**, while L∞ is **one maximum over the entire batch****, which is the right check for "did anything exceed ε?"

### The returned dictionary

|Key|Formula|Meaning|
|---|---|---|
|`clean_accuracy`|`clean_correct.float().mean()`|fraction right before the attack|
|`adversarial_accuracy`|`adv_correct.float().mean()`|fraction right after|
|`attack_success_rate`|flipped ÷ originally correct|flip rate among originally-correct images only; `.clamp_min(1.0)` prevents dividing by zero|
|`avg_clean_confidence`|mean of `conf_clean`|average true-class probability before|
|`avg_adv_confidence`|mean of `conf_adv`|average true-class probability after|
|`avg_confidence_drop`|mean of (before − after)|how much belief in the right answer was lost|
|`avg_l2_perturbation`|mean of per-image L2|average overall distortion|
|`max_linf_perturbation`|batch-wide max|should equal ε exactly|

## Part 6: Running it

```python
# Assume images, labels, x_adv from the Core Implementation section
metrics = evaluate_attack(model, images, x_adv, labels)
for k, v in metrics.items():
    print(f"{k}: {v:.4f}")
```

Output:

```txt
clean_accuracy: 0.9766
adversarial_accuracy: 0.3203
attack_success_rate: 0.6797
avg_clean_confidence: 0.9824
avg_adv_confidence: 0.3891
avg_confidence_drop: 0.5933
avg_l2_perturbation: 10.8451
max_linf_perturbation: 0.8000
```

`metrics.items()` gives each (name, value) pair, and `:.4f` prints 4 decimals.

### Reading each number

|Metric|Value|In plain words|
|---|---|---|
|Clean accuracy|0.9766|**125 of 128** right before the attack|
|Adversarial accuracy|0.3203|**41 of 128** still right after|
|Success rate|0.6797|about **68%** of correct predictions flipped|
|Confidence: before → after|0.9824 → 0.3891|belief in the true class fell from 98% to 39%|
|Confidence drop|0.5933|0.9824 − 0.3891, about 59 points lost on average|
|Avg L2|10.85|overall distortion in **normalized** units, ≈ 10.85 × 0.3081 ≈ **3.34** in pixel units|
|Max L∞|0.8000|exactly ε, so **the budget was respected** (≈ 0.25 pixel ≈ 63/255)|

### Why L2 is 10.85 and not higher

If every one of the 784 pixels moved the full 0.8, L2 would be 0.8 × √784 = 0.8 × 28 = **22.4**, the maximum from Section 2's norm chain. The real value is about half that, largely because of the **clamp**. Black background pixels already sit at the floor (−0.424), so any push "darker" gets clamped back and their change becomes 0. **The clamp makes the real L2 smaller than the theoretical ε·√n.**

### Two honest notes on the numbers

- **They don't perfectly line up.** 0.6797 is exactly 87 ÷ **128**, but the code divides by the **125** originally-correct images. 87 flips out of 125 would be 0.696. The printed output most likely came from a slightly different run than the other lines.
- **Section 6 reported 71.09% on "the same" batch.** Same reason: different training runs give different weights. **Your own numbers will differ a little; judge whether the pattern matches, not the exact digits.**

## Copy-paste cells (mine, not the course's)

These use names already in your notebook (`model`, `images`, `labels`, `x_adv`, `flipped`, `metrics`, `test_loader`, `device`, `fgsm_attack`).

### Convert the norms into pixel units

Use this whenever you report sizes to a human.

```python
SIGMA = 0.3081
print("avg L2 (pixel):", round(metrics["avg_l2_perturbation"] * SIGMA, 3))
print("max Linf (pixel):", round(metrics["max_linf_perturbation"] * SIGMA, 4),
      "=", round(metrics["max_linf_perturbation"] * SIGMA * 255), "/255")
```

Expected: about `3.34`, then `0.2465 = 63 /255`.

### How confident is the model in its wrong answers?

This measures the top probability on the images that flipped. `flipped` comes from the Section 6 test cell.

```python
with torch.no_grad():
    adv_probs = F.softmax(model(x_adv), dim=1)
conf_wrong = adv_probs.max(dim=1).values[flipped]
print(round(conf_wrong.mean().item(), 4))
```

Expected: a probability between 0 and 1. Above about 0.5 means the model is fairly sure of its _wrong_ answer, a stronger finding than "barely flipped."

### Metrics over the whole test set (not just one batch)

This counts across all 10,000 test images. It totals the raw counts instead of averaging per-batch percentages, so every image gets an equal vote (the same idea as the weighted loss in Section 4).

```python
model.eval()
n_total = n_clean_ok = n_adv_ok = n_flipped = 0
for xb, yb in test_loader:
    xb, yb = xb.to(device), yb.to(device)
    xa = fgsm_attack(model, xb, yb, 0.8)
    with torch.no_grad():
        pc = model(xb).argmax(dim=1)
        pa = model(xa).argmax(dim=1)
    ok = (pc == yb)
    n_total += yb.size(0)
    n_clean_ok += ok.sum().item()
    n_adv_ok += (pa == yb).sum().item()
    n_flipped += ((pa != yb) & ok).sum().item()

print(f"clean acc: {n_clean_ok / n_total:.4f}")
print(f"adv acc: {n_adv_ok / n_total:.4f}")
print(f"success: {n_flipped / max(n_clean_ok, 1):.4f}")
```

Expected: clean accuracy about `0.984` (your Section 4 baseline), with adversarial accuracy and success rate in the same ballpark as the single batch. It takes roughly 20-60 seconds on CPU.

## Quick self-check

A batch of **200** images: the model gets **190** right on clean images and **57** right on adversarial images, and all 57 were among the originally correct ones. What are the **clean accuracy**, **adversarial accuracy** and **attack success rate**?

Reply with your three numbers and I'll check them.

## The facts worth locking in for the exam

- ==Attack success rate counts flips only among originally correct samples: flipped ÷ originally_correct, where flipped = (~adv_correct) & originally_correct==. That separates the attack's effectiveness from the model's own mistakes, unlike adversarial accuracy, which is over all samples.
- **Clean accuracy, adversarial accuracy and success rate are different numbers**. In HTB's batch, 0.9766 clean (125/128) dropped to 0.3203 adversarial (41/128), with a success rate of about 0.68.
- **Confidence is measured as the softmax probability of the true class, picked out per image with `gather(1, true_labels.view(-1, 1))`**. An average drop from 0.9824 to 0.3891 shows the attack destroys the model's belief in the right answer even where the label doesn't flip.
- **`avg_adv_confidence` is confidence in the true class, not in the new wrong prediction**. Measuring how sure the model is of its wrong answer needs the max probability on the flipped images.
- **L2 is computed per image after flattening to [N, 784] and then averaged, while L∞ is a single maximum over the whole batch**. `max_linf_perturbation` equal to ε (0.8000) proves the budget was respected.
- ==FGSM's real L2 (10.85 normalized ≈ 3.34 pixel) sits well below the theoretical ε·√n = 0.8 × 28 = 22.4== because clamping zeroes the change on pixels already at the valid-range limit, such as black background pushed darker.
- ==Metric norms computed on normalized tensors are in normalized units; multiply by σ = 0.3081 (and by 255) to report them in pixel terms.==
- **Method point:** **evaluate on the whole test set by totalling raw counts across batches, not by averaging per-batch percentages**, and report clean accuracy, adversarial accuracy, success rate, confidence drop and the ε/L∞ check together, so a reader sees both effectiveness and cost.