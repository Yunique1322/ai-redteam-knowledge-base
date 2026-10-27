# I-FGSM Implementation

## The idea in one breath

This turns Section 10's formula into one function, `iterative_fgsm`. It loops T times: re-compute the gradient at the current image, take a small sign step of size α, then project back inside the ε-box around the **original** image and the valid pixel range. An optional **random start** adds noise before the loop. On the first test batch, at the same ε = 0.8, **I-FGSM flipped **100%** of correct predictions** (single-step FGSM got about 71%), and the model's belief in the true class collapsed from 98.5% to 1.2%.

## Part 1: The function

```python
def iterative_fgsm(model: nn.Module,
                   images: Tensor,
                   labels: Tensor,
                   epsilon: float,
                   num_iter: int,
                   alpha: float | None = None,
                   targeted: bool = False,
                   random_start: bool = False) -> Tensor:
    """Iterative FGSM (Basic Iterative Method) with projection.

    Args:
        model: Target classifier in evaluation mode
        images: Clean images (normalized)
        labels: Ground-truth or target labels
        epsilon: L_infinity budget (in normalized space)
        num_iter: Number of iterations
        alpha: Step size per iteration (defaults to epsilon/T)
        targeted: If True, targeted attack
        random_start: If True, initialize within the epsilon ball

    Returns:
        Tensor: Adversarial images (normalized)
    """
    # Valid normalized range for MNIST
    MNIST_NORM_MIN = (0.0 - 0.1307) / 0.3081
    MNIST_NORM_MAX = (1.0 - 0.1307) / 0.3081

    if alpha is None:
        alpha = epsilon / max(num_iter, 1)
    if random_start:
        torch.manual_seed(1337)
        delta = torch.empty_like(images).uniform_(-epsilon, epsilon)
        x_adv = torch.clamp(images + delta, MNIST_NORM_MIN, MNIST_NORM_MAX)
    else:
        x_adv = images.clone()

    for _ in range(num_iter):
        x_adv = x_adv.detach().requires_grad_(True)
        logits = model(x_adv)
        loss = F.cross_entropy(logits, labels)
        model.zero_grad(set_to_none=True)
        loss.backward()
        step_dir = -1.0 if targeted else 1.0
        x_adv = x_adv + step_dir * alpha * x_adv.grad.sign()
        x_adv = torch.clamp(images + (x_adv - images).clamp(-epsilon, epsilon), MNIST_NORM_MIN, MNIST_NORM_MAX)

    return x_adv.detach()
```

> **If this errors:** `alpha: float | None` needs Python 3.10+. On older Python, use `from typing import Optional` and write `alpha: Optional[float] = None`.

### Toy first: one pixel through the loop

One normalized pixel, original **0.5**, ε = **0.25**, T = **4**, so the default α = 0.25 / 4 = **0.0625**.

**Without random start**, suppose the gradient signs are +, +, +, +:

```
start 0.5 -> 0.5625 -> 0.625 -> 0.6875 -> 0.75 drift = +0.25 = exactly ε, never more
```

With α = ε/T, four same-direction steps add up to **exactly** ε, so the ε-clip never has to act (the Section 10 honest note).

**With random start**, suppose the noise draw is **+0.2**, so the pixel starts at **0.7**. Now the same first + step:

```
0.7 + 0.0625 = 0.7625 drift from original = +0.2625 > ε = 0.25
projection: 0.5 + clamp(0.2625, −0.25, 0.25) = 0.5 + 0.25 = 0.75
```

==With a random start, the projection really does bind even when α = ε/T==, because the noise already spent part of the budget before the first step. So the course's "total movement is capped at 10 × 0.08 = 0.8" is only true **without** random start. The projection line is what guarantees the budget in both cases.

### Line by line: setup

|Code|What it does|
|---|---|
|`MNIST_NORM_MIN / MAX`|the valid normalized range [−0.424, 2.821] (Section 5)|
|`if alpha is None:`|if you didn't choose a step size...|
|`alpha = epsilon / max(num_iter, 1)`|...use ε/T. `max(num_iter, 1)` avoids dividing by zero if `num_iter=0`. With ε = 0.8 and T = 10, **α = 0.08**|
|`torch.manual_seed(1337)`|fixes the random-number generator so the "random" start is reproducible|
|`torch.empty_like(images)`|a new tensor with the same shape as the images (contents not set yet)|
|`.uniform_(-epsilon, epsilon)`|fills it with random values spread **evenly** between −ε and +ε (`_` = in place)|
|`torch.clamp(images + delta, MIN, MAX)`|adds the noise, then keeps pixels valid|
|`else: x_adv = images.clone()`|no random start: begin at the clean image (a copy, so the original is never modified)|

**Why a random start?** If the gradient at the clean image happens to point "nowhere useful" (flat, or a misleading spot), starting from a random nearby point gives the attack a fresh view. **I-FGSM plus a random start is essentially PGD** (the Section 10 vocabulary note).

### Line by line: the loop

|Code|What it does|Why it matters|
|---|---|---|
|`for _ in range(num_iter):`|repeat T times; `_` = "I don't need the counter"||
|`x_adv.detach().requires_grad_(True)`|cut the link to the previous iteration, then track slopes on the **current** image|**without `detach()`, PyTorch would chain all iterations into one huge calculation (memory grows each step)**; without `requires_grad_`, `x_adv.grad` is `None`|
|`logits = model(x_adv)` / `F.cross_entropy(...)`|forward pass and loss **at the current point**|this is the "fresh gradient" that makes iteration work|
|`model.zero_grad(set_to_none=True)`|clear old weight gradients||
|`loss.backward()`|fills `x_adv.grad`||
|`step_dir = -1.0 if targeted else 1.0`|+1 untargeted, −1 targeted (as in Sections 6 and 9)|targeted: pass **target** labels|
|`x_adv + step_dir * alpha * x_adv.grad.sign()`|a small sign step of size α|not ε, which is the difference from FGSM|
|`(x_adv - images)`|drift from the **original** images||
|`.clamp(-epsilon, epsilon)`|**cap the drift at ±ε (the projection)**||
|`images + ...`|rebuild the image from the original plus the capped drift||
|`torch.clamp(..., MIN, MAX)`|keep every pixel in the valid range||
|`return x_adv.detach()`|hand back a plain tensor||

The course's projection example: a pixel that has drifted **+0.9** with ε = 0.8 is clamped to **+0.8**. **Because the drift is always measured from `images` (the original), ε means "at most ε from the start," never "ε per step."**

> **An honest note the course skips (two of them):**
> 
> 1. `torch.manual_seed(1337)` **inside** the function resets the random generator every call. Every random start therefore uses the **same** noise pattern, which is reproducible, but "try several random restarts" would just repeat the same one. It also silently resets the seed for any random code you run afterwards.
> 2. Unlike `_forward_and_loss` (Section 6), this function has **no `model.eval()` check**. Call `model.eval()` yourself before using it.

## Part 2: Testing it on a batch

```python
# Assume model, test_loader, device from FGSM Setup
images, labels = next(iter(test_loader))
images, labels = images.to(device), labels.to(device)

epsilon = 0.8
num_iter = 10
alpha = epsilon / num_iter # alpha = 0.08

with torch.no_grad():
    clean_pred = model(images).argmax(dim=1)

x_adv_ifgsm = iterative_fgsm(
    model, images, labels,
    epsilon=epsilon,
    num_iter=num_iter,
    alpha=alpha,
    targeted=False,
    random_start=True
)

with torch.no_grad():
    adv_pred_ifgsm = model(x_adv_ifgsm).argmax(dim=1)

originally_correct = clean_pred == labels
flipped_ifgsm = (adv_pred_ifgsm != labels) & originally_correct
print(
    f"I-FGSM flips (first batch): "
    f"{(flipped_ifgsm.float().sum() / originally_correct.float().sum().clamp_min(1.0)).item():.2%}"
)
```

Output:

```txt
I-FGSM flips (first batch): 100.00%
```

|Code|What it does|
|---|---|
|`images, labels = next(iter(test_loader))`|the **first test batch again** (the test loader isn't shuffled, so it's the same 128 images as Section 6)|
|`epsilon, num_iter, alpha`|ε = 0.8, T = 10, α = 0.08|
|`clean_pred` in `no_grad()`|predictions before the attack|
|`iterative_fgsm(..., random_start=True)`|the attack, **outside** `no_grad()` (it needs gradients)|
|`flipped_ifgsm`|wrong now **and** right before (the Section 6/7 success rule)|
|`.clamp_min(1.0)`|avoids dividing by zero|

**Notebook note:** this cell **re-assigns** `images`, `labels`, `epsilon`, `clean_pred` and `originally_correct`. They get the same values as before (same batch, same ε), so nothing breaks, but it also means the batch-level `x_adv` that Section 9 overwrote is still a single image. Re-run `x_adv = fgsm_attack(model, images, labels, 0.8)` if you need the FGSM batch.

## Part 3: The full metrics

```python
# Reuse evaluate_attack function from the Evaluation Metrics section
metrics_ifgsm = evaluate_attack(model, images, x_adv_ifgsm, labels)
for k, v in metrics_ifgsm.items():
    print(f"{k}: {v:.4f}")
```

Output:

```txt
clean_accuracy: 1.0000
adversarial_accuracy: 0.0000
attack_success_rate: 1.0000
avg_clean_confidence: 0.9853
avg_adv_confidence: 0.0115
avg_confidence_drop: 0.9739
avg_l2_perturbation: 14.0519
max_linf_perturbation: 0.8000
```

### Reading it next to FGSM (Section 7)

|Metric|FGSM, 1 step (Sec 7)|I-FGSM, 10 steps (Sec 11)|What changed|
|---|---|---|---|
|Clean accuracy|0.9766|1.0000|different training run (see note below)|
|Adversarial accuracy|0.3203|**0.0000**|no image survived|
|Success rate|≈ 0.68|**1.0000**|every correct prediction flipped|
|True-class confidence after|0.3891|**0.0115**|**the model is now confidently wrong**: ~1% on the right answer|
|Confidence drop|0.5933|**0.9739**|about 97 points lost on average|
|Avg L2 (normalized)|10.85|14.05|≈ 3.34 -> ≈ **4.33** in pixel units|
|Max L∞|0.8000|0.8000|**same budget, proven**|

**Same ε, far stronger attack.** Section 8's "barely flipped, 51% vs 43%" weakness is gone: I-FGSM leaves the true class with about 1% probability.

> **An honest note the course skips:**
> 
> - **The two runs used different models.** Clean accuracy is 125/128 in Section 7 but 128/128 here, on the same batch. So FGSM's "71.09%" and I-FGSM's "100%" come from different trained weights. The direction of the result is real, but for a fair comparison run both attacks on **your** model. The first cell below does that.
> - **The course contradicts itself on L2.** It says the higher L2 reflects the budget "used broadly," and then that "the L2 norm may drop slightly." Here L2 clearly **rose** (10.85 -> 14.05). A likely reason: FGSM left many background pixels at exactly zero change (pushes below black got clamped away, Section 7), while the random start and the back-and-forth steps leave fewer pixels untouched. Trust your measured numbers over the general claim.

**Why the L∞ is still exactly 0.8000:** the projection caps every pixel's drift at ε, and at least one pixel reached the cap. More steps can raise success **without** raising max‖δ‖∞, which is the course's closing point.

## Copy-paste cells (mine, not the course's)

These use names in your notebook: `model`, `images`, `labels`, `fgsm_attack`, `iterative_fgsm`, `evaluate_attack`, `x_adv_ifgsm`, `candidate`, `target_label`.

### Fair head-to-head: FGSM vs I-FGSM on your model

Runs both attacks on the same batch with the same ε and prints the key metrics side by side.

```python
model.eval()
x_adv_fgsm = fgsm_attack(model, images, labels, 0.8)
m_f = evaluate_attack(model, images, x_adv_fgsm, labels)
m_i = evaluate_attack(model, images, x_adv_ifgsm, labels)
for k in ["attack_success_rate", "avg_adv_confidence", "avg_l2_perturbation", "max_linf_perturbation"]:
    print(f"{k:24s} FGSM {m_f[k]:.4f} | I-FGSM {m_i[k]:.4f}")
```

Expected: I-FGSM shows higher success and much lower adversarial confidence, and both show `max_linf_perturbation 0.8000`. Exact values depend on your weights.

### How many iterations are enough?

Sweeps T with α = ε/T and no random start. T = 1 is mathematically identical to FGSM, which makes a nice sanity check.

```python
ok = (clean_pred == labels)
for T in [1, 3, 5, 10, 20]:
    xa = iterative_fgsm(model, images, labels, epsilon=0.8, num_iter=T)
    with torch.no_grad():
        pa = model(xa).argmax(dim=1)
    rate = ((pa != labels) & ok).float().sum() / ok.float().sum().clamp_min(1.0)
    print(f"T={T:2d} alpha={0.8/T:.3f} success={rate.item():.2%}")
```

Expected: T = 1 matches your FGSM success rate. Success then climbs with T and levels off near 100%, the diminishing returns from Section 10.

### Targeted I-FGSM on the Section 9 "1"

Section 9's single-step FGSM **failed** at ε = 0.5. This tries 10 iterative steps at the same 0.5.

```python
xa = iterative_fgsm(model, candidate.unsqueeze(0), target_label,
                    epsilon=0.5, num_iter=10, targeted=True)
with torch.no_grad():
    probs = F.softmax(model(xa), dim=1)[0]
print("pred:", probs.argmax().item(), "| p(7) =", round(probs[7].item(), 4))
```

Expected: most likely `pred: 7`, with p(7) well above Section 9's 43.54%. Iteration lets a targeted attack succeed at a budget where one step couldn't.

### Check the budget and where the pixels ended up

Confirms nothing exceeds ε, and shows how many pixels sit right at the ±ε edge (FGSM's "box corners") versus inside.

```python
for name, xa in [("FGSM", x_adv_fgsm), ("I-FGSM", x_adv_ifgsm)]:
    d = (xa - images).abs()
    print(f"{name:7s} max={d.max().item():.4f} at ±eps: {(d > 0.8 - 1e-4).float().mean().item():.2%}")
```

Expected: both `max=0.8000`. FGSM's "at ±eps" share is high (every unclamped pixel sits at the edge); I-FGSM's is usually lower, because its pixels can stop anywhere inside [−ε, +ε].

## Key facts to lock in

- **`iterative_fgsm` loops T times: `detach().requires_grad_(True)` on the current image, forward pass, cross-entropy loss, `backward()`, a step of α·sign(grad), then projection**, so every step uses a freshly computed gradient at the current point.
- **The projection line `images + (x_adv - images).clamp(-ε, ε)` measures drift from the ORIGINAL images and caps it at ±ε, and the outer clamp keeps pixels inside MNIST's normalized range [−0.424, 2.821]**, so ε always means "at most ε from the start," never "ε per step."
- ==The default step size is α = ε / max(T, 1), which is 0.08 for ε = 0.8 and T = 10==. Without a random start, T same-direction steps total exactly ε, but with a random start the noise already uses part of the budget, so the projection actively clips.
- **A random start adds uniform noise in [−ε, ε] before iterating (I-FGSM plus a random start is essentially PGD)**, which helps escape spots where the gradient is uninformative. This implementation reseeds with 1337 inside the function, so every call uses the same noise.
- ==At the same ε = 0.8, I-FGSM reached a 100% attack success rate with adversarial accuracy 0.0000, and true-class confidence collapsed from 0.9853 to 0.0115==, versus about 68-71% success and 0.389 confidence for single-step FGSM.
- ==Both attacks report max_linf_perturbation = 0.8000, proving the gain comes from better directions, not a bigger budget==. I-FGSM's average L2 was higher (14.05 vs 10.85 normalized, ≈ 4.33 vs 3.34 pixel), because fewer pixels are left at zero change.
- ==Targeted I-FGSM uses the same function with target labels and `targeted=True` (step_dir = −1)==. Iterating lets targeted attacks succeed at budgets where single-step targeted FGSM fails.
- **Method point:** **compare attacks fairly by running them on the same model, the same batch and the same ε, and report success rate, true-class confidence and the L∞ check together**, because numbers copied from different training runs (125/128 vs 128/128 clean) aren't directly comparable.

