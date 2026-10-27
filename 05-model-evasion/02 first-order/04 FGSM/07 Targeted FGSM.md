# Targeted FGSM

## The idea in one breath

Untargeted FGSM only says "be wrong." **Targeted FGSM** says "say _this_ class." It makes two changes: the gradient is computed with the **target label** yₜ instead of the true label, and the step is **subtracted** instead of added. The ε budget and the clamping stay the same. Steering toward one specific class is harder than just leaving the current one. In HTB's run, ε = 0.5 failed to turn a "1" into a "7," and ε = 0.8 succeeded.

## Part 1: The toy example, built up step by step

### Step 1: A 3-class toy model

Two pixels, `x = [0.5, 0.2]`, and three classes ("one," "two," "seven"). Each class gets its own score recipe:

```
z_one = 3·p1 + 0·p2 = 3(0.5) = 1.5 <- highest -> predicts "one" ✅
z_two = 2·p1 + 1·p2 = 1.0 + 0.2 = 1.2 <- close rival
z_seven = 0·p1 + 2·p2 = 2(0.2) = 0.4 <- far rival
```

The slopes of each score with respect to the pixels are just the recipe numbers:

```
∇z_one = [3, 0] ∇z_two = [2, 1] ∇z_seven = [0, 2]
```

Softmax turns the three scores into probabilities (Section 7): **p = [0.482, 0.357, 0.161]**.

### Step 2: The untargeted gradient pushes away from "one"

Using the Section 3 formula, each class gets the coefficient `pᵢ − 𝟙[i = true]`, with true = "one":

```
coefficients = [0.482 − 1, 0.357, 0.161] = [−0.518, 0.357, 0.161]

∇ℒ = −0.518·[3,0] + 0.357·[2,1] + 0.161·[0,2]
   = [−1.554 + 0.714 + 0, 0 + 0.357 + 0.321]
   = [−0.839, +0.678]
sign = [−1, +1]
```

Untargeted **adds** ε·sign. With ε = 0.2: `x = [0.5 − 0.2, 0.2 + 0.2] = [0.3, 0.4]`.

### Step 3: The targeted gradient pulls toward "seven"

Now the label in the loss is the **target**, "seven," so the "−1" moves to seven's coefficient:

```
coefficients = [0.482, 0.357, 0.161 − 1] = [0.482, 0.357, −0.839]

∇ℒ_t = 0.482·[3,0] + 0.357·[2,1] − 0.839·[0,2]
     = [1.446 + 0.714, 0.357 − 1.679]
     = [+2.161, −1.322]
sign = [+1, −1]
```

Targeted **subtracts** ε·sign. With ε = 0.2: `x = [0.5 − 0.2, 0.2 + 0.2] = [0.3, 0.4]`.

**What the minus does:** subtracting this gradient raises "seven" (it carried the big −0.839, which flips to a push upward) and lowers every class the model currently believes in, weighted by their probabilities. **Lowering the loss for yₜ means making the model more confident in yₜ.**

### Step 4: Check where each attack lands

At ε = 0.2, the new point `[0.3, 0.4]`:

```
z_one = 3(0.3) = 0.9
z_two = 2(0.3) + 0.4 = 1.0 <- wins
z_seven = 2(0.4) = 0.8
```

The model now says "two." For the **untargeted** attack that's a success, since any wrong class counts. For the **targeted** attack it's a **targeted miss**: wrong, but not the class you asked for. **Small budgets tend to land on the nearest rival, not your chosen target.**

Try ε = 0.3, giving `[0.2, 0.5]`:

```
z_one = 3(0.2) = 0.6
z_two = 2(0.2) + 0.5 = 0.9
z_seven = 2(0.5) = 1.0 <- wins, barely
```

"Seven" wins by 1.0 vs 0.9, so **the targeted attack needed a bigger ε to get past the closer rival.** This is exactly HTB's real result: 0.5 fails, 0.8 works. (In a 2-pixel toy both attacks happen to step to the same corner. With 784 pixels they generally point in different directions, which is why targeting gets a different outcome.)

### Step 5: The real formulas

```
Untargeted: x_adv = x + ε · sign( ∇x ℒ(θ, x, y) ) ascend the loss on the TRUE label
Targeted: x_adv = x − ε · sign( ∇x ℒ(θ, x, yₜ) ) descend the loss on the TARGET label
```

|Symbol|Plain English|Toy|
|---|---|---|
|`y`|the true label|"one"|
|`yₜ` (y-t)|the class you want the model to say|"seven"|
|`ℒ(θ, x, yₜ)`|how wrong the model is _if the right answer were yₜ_|high at the start (p_seven = 0.16)|
|`−` before ε|step **downhill** on that loss (toward yₜ)||
|"ascending −ℒ"|the same thing said differently: making −ℒ bigger = making ℒ smaller||

Read aloud: **"Targeted FGSM computes the gradient as if the target were the correct answer, then steps the opposite way, so the model grows more convinced of the target."**

In `fgsm_attack` from Section 6, this is just `step_dir = −1.0` plus passing the target labels.

## Part 2: Setup and configuration

```python
eps_candidates = [0.5, 0.8, 1.0]
success_image, success_label, success_eps = None, None, None
```

|Code|What it does|
|---|---|
|`eps_candidates`|budgets to try, smallest first, in **normalized** units (0.5 ≈ 0.154 pixel ≈ 39/255; 0.8 ≈ 63/255; 1.0 ≈ 79/255)|
|`a, b, c = None, None, None`|sets three variables at once; `None` means "nothing found yet"|

If they're still `None` at the end, the search failed. That's what the last block checks.

## Part 3: Finding a correctly classified "1"

```python
model.eval()
candidate, candidate_label = None, None

for xb, yb in test_loader:
    xb, yb = xb.to(device), yb.to(device)
    match_indices = (yb == 1).nonzero(as_tuple=True)[0]
    if len(match_indices) == 0:
        continue

    # Check predictions for all digit 1s in this batch
    with torch.no_grad():
        preds = model(xb[match_indices]).argmax(dim=1)
        correct_mask = (preds == 1)
        if correct_mask.any():
            # Take first correctly classified digit 1
            local_idx = correct_mask.nonzero(as_tuple=True)[0][0].item()
            idx = match_indices[local_idx].item()
            candidate = xb[idx]
            candidate_label = yb[idx]
            break

if candidate is None:
    raise RuntimeError("Could not find a correctly classified digit 1 in test set")
```

### Toy first: the double indexing

A tiny batch with labels `[3, 1, 7, 1, 2]`:

|Step|Code|Result|
|---|---|---|
|1. which positions are "1"?|`(yb == 1)`|`[F, T, F, T, F]`|
|2. their positions|`.nonzero(as_tuple=True)[0]`|`match_indices = [1, 3]`|
|3. predict only those two|`model(xb[match_indices]).argmax(dim=1)`|say `preds = [7, 1]` (the first "1" is misread)|
|4. which were right?|`correct_mask = (preds == 1)`|`[F, T]`|
|5. first correct one, in the **short** list|`correct_mask.nonzero(...)[0][0]`|`local_idx = 1`|
|6. map back to the **full** batch|`match_indices[local_idx]`|`idx = match_indices[1] = 3`|

**Step 6 is the key: `local_idx` counts within the filtered list of "1"s, so you translate it back to the real batch position.**

### The other lines

|Code|What it does|
|---|---|
|`model.eval()`|evaluation mode before predicting|
|`for xb, yb in test_loader:`|goes through the test set batch by batch|
|`continue`|"skip to the next batch" if there's no "1" here|
|`with torch.no_grad():`|only predicting here, so no gradients needed|
|`correct_mask.any()`|True if at least one "1" was predicted correctly|
|`candidate = xb[idx]`|the chosen image, shape **[1, 28, 28]**|
|`break`|stop searching once we have one|
|`raise RuntimeError(...)`|stop with a clear message if no "1" was ever classified correctly|

**Why insist on a correctly classified "1"?** If the model already misreads it, a "successful" attack proves nothing. It's the same rule as Section 6's `originally_correct`.

## Part 4: Trying each ε

```python
target_label = torch.tensor([7], device=device)

for eps_try in eps_candidates:
    x_adv = fgsm_attack(
        model,
        candidate.unsqueeze(0),
        target_label,
        epsilon=eps_try,
        targeted=True,
    )
    with torch.no_grad():
        pred = model(x_adv).argmax(dim=1).item()
    print(f"epsilon={eps_try:.2f} -> predicted {pred}")

    if pred == 7:
        success_image = candidate
        success_label = candidate_label
        success_eps = eps_try
        break
```

|Code|What it does|
|---|---|
|`torch.tensor([7], device=device)`|the **target** label as a 1-element tensor, shape [1], on the right device|
|`candidate.unsqueeze(0)`|[1, 28, 28] -> [1, 1, 28, 28], a batch of one|
|`fgsm_attack(model, ..., target_label, ..., targeted=True)`|==passes the **target** label in the labels slot and sets `targeted=True`==, which makes `step_dir = −1`|
|`fgsm_attack` call outside `no_grad()`|the attack needs gradients (as always)|
|`pred = ...argmax(dim=1).item()`|the new prediction as a plain number|
|`if pred == 7:` ... `break`|stop at the **first** ε that works (**early stopping**)|

> **An honest note the course skips:** this cell **overwrites `x_adv`**. Before this, `x_adv` held the whole Section 6 batch [128, 1, 28, 28]; now it holds one image [1, 1, 28, 28]. Any earlier cell that uses the batch `x_adv` (like `evaluate_attack(model, images, x_adv, labels)`) will now break with a size-mismatch error. Re-run the Section 6 test cell to restore it.

> **A second honest note:** "minimal ε" here only means smallest **in this list**. 0.5 failed and 0.8 worked, so the true minimum is somewhere in between. The first cookbook cell below searches more finely.

Output:

![Part 4: Trying each ε](/assets/Pasted%20image%2020261016165620.png)

At 0.5 the model still says "1" (not even a wrong answer). At 0.8 it says "7," a full targeted success. The loop then stops, so ε = 1.0 is never tried.

## Part 5: Handling failure and visualizing

```python
if success_image is None:
    raise RuntimeError("Targeted FGSM did not achieve 1 -> 7 within the tested epsilons.")

_ = visualize_fgsm_attack(
    model,
    success_image.detach().cpu(),
    success_label.detach().cpu(),
    success_eps,
    targeted=True,
    target_class=7,
)
```

| Code | What it does |
| ----------------------------------------------------------- | -------------------------------------------------------------------------------- |
| `if success_image is None:` | no ε in the list worked, so stop with a helpful message |
| `visualize_fgsm_attack(..., targeted=True, target_class=7)` | the Section 8 wrapper; it builds the target labels itself with `torch.full_like` |
| `success_label` | the **true** label (1), used for the red/green title check |
| `.detach().cpu()` | a clean CPU copy for plotting |

## Part 6: Reading the figure ("FGSM Targeted")
![Targeted FGSM](/assets/9%20Targeted%20FGSM-20261015221750224.png)
### Top row

- **Original | class=1 | p=99.92%:** a clean "1," near-certain.
- **Adversarial | class=7 | p=43.54% | target=7:** red title (no longer the true label), and ==`class=7` equals `target=7`, so this is a real targeted success==, not just a miss into some other class (the Section 8 check). To a human it's still obviously a "1." The background is speckled grey (brightened by about 63/255).
- **Perturbation (x10):** same reading as Section 8. **White** = brightened, **orange** = no change (pushes below black were clamped away), **black** = the stroke of the "1" got darker.

### Bottom row (class probabilities)

|Class|Clean (blue)|Adversarial (red)|
|---|---|---|
|1 (true)|≈ 1.00|≈ 0.27|
|2|≈ 0|≈ 0.24|
|3, 4, 5|≈ 0|≈ 0.01-0.02 each|
|**7 (target)**|≈ 0|**≈ 0.44**|

**The win is narrow: 7 at 44%, with the true "1" at 27% and "2" at 24%.** The model is confused three ways, just like the toy where "two" was the close rival that the target had to beat. A single targeted step reached the target, but not confidently. Iterating (I-FGSM, next section) is how you push it further.

## Copy-paste cells (mine, not the course's)

These use names already in your notebook (`model`, `candidate`, `target_label`, `fgsm_attack`, `images`, `labels`, `clean_pred`, `device`).

### Find the minimum ε more precisely

Scans ε from 0.50 to 0.80 in steps of 0.02 for the same "1" -> "7."

```python
for eps_try in np.arange(0.50, 0.81, 0.02):
    xa = fgsm_attack(model, candidate.unsqueeze(0), target_label,
                     epsilon=float(eps_try), targeted=True)
    with torch.no_grad():
        p = model(xa).argmax(dim=1).item()
    if p == 7:
        print(f"first success at epsilon={eps_try:.2f} "
              f"(pixel {eps_try * 0.3081:.3f} = {round(eps_try * 0.3081 * 255)}/255)")
        break
else:
    print("no success in 0.50-0.80")
```

Expected: one line like `first success at epsilon=0.6x ...`, somewhere between 0.50 and 0.80. The `else` on a `for` loop runs only if the loop finished **without** a `break`.

> Single-step FGSM isn't guaranteed to work at every ε above the first success, because each ε is a separate one-step guess. That's why this scans upward instead of using a binary search.

### Targeted vs untargeted on a whole batch

Gives every image in your Section 6 batch the target "(true label + 1) mod 10," then counts hits, misses and failures.

```python
targets = (labels + 1) % 10
xa = fgsm_attack(model, images, targets, 0.8, targeted=True)
with torch.no_grad():
    pa = model(xa).argmax(dim=1)
ok = (clean_pred == labels)
hit = ((pa == targets) & ok).sum().item()
miss = ((pa != targets) & (pa != labels) & ok).sum().item()
fail = ((pa == labels) & ok).sum().item()
n = ok.sum().item()
print(f"targeted hit: {hit/n:.2%} | wrong-but-not-target: {miss/n:.2%} | still correct: {fail/n:.2%}")
```

Expected: three percentages that add to 100%. The **hit** rate will most likely be well below the ~70% untargeted success from Section 6. That gap is the cost of choosing the destination.

### Restore the batch `x_adv` after this section

Re-creates the Section 6 untargeted batch so batch-level cells work again.

```python
x_adv = fgsm_attack(model, images, labels, 0.8)
print(x_adv.shape)
```

Expected: `torch.Size([128, 1, 28, 28])`

## Key facts to lock in

- ==Targeted FGSM is x_adv = x − ε·sign(∇xℒ(θ, x, yₜ)), which descends the loss on the target label==, while untargeted FGSM is x + ε·sign(∇xℒ(θ, x, y)), which ascends the loss on the true label. Descending ℒ(yₜ) is the same as ascending −ℒ(yₜ).
- **Exactly two things change for targeted FGSM: the label used in the gradient (target yₜ instead of true y) and the sign of the step (minus instead of plus)**. The L∞ budget ε and the clamping to the valid range are unchanged.
- ==In `fgsm_attack`, a targeted attack means passing the target labels in the labels argument and setting `targeted=True`, which sets `step_dir = −1`==. Passing the true labels with `targeted=True` would push the model _toward_ its correct answer.
- **Targeted attacks usually need a larger ε than untargeted ones because they must reach one specific class, not just the nearest rival**. In HTB's run, ε = 0.5 (≈ 39/255) left the "1" unchanged and ε = 0.8 (≈ 63/255) forced a "7."
- ==A targeted attempt has three outcomes: a hit (prediction = target), a miss (wrong but not the target) and a failure (still correct)==. Only a hit counts as targeted success, so check `class=` against `target=`.
- **HTB's targeted success was narrow: the target "7" won at 43.54%, with the true "1" at about 27% and "2" at about 24%**. That shows single-step targeted FGSM often reaches the target only weakly, which motivates iterative attacks.
- **Early stopping, testing ε values in ascending order and breaking at the first success, finds the smallest working budget in the tested list**. The true minimum lies between the last failure and the first success, so a finer scan narrows it.
- **Method point:** **only attack inputs the model originally classifies correctly, and report a targeted result as the smallest ε that achieves the target class**, converted to pixel units (for example 0.8 normalized ≈ 0.246 ≈ 63/255).

---
