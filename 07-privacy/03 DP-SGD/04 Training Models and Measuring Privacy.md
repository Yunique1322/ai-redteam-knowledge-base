# Training Models And Measuring Privacy

## The idea in one breath

This is the experiment itself. First we train a normal baseline, then measure how much it leaks. Then we retrain the same CNN with DP-SGD at ε = 10 and at ε = 3 and measure again. The course's numbers:

|Model|Test acc|Gap|MIA adv|
|---|---|---|---|
|Baseline|67%|10 pts|**0.019**|
|ε = 10|58%|3 pts|**0.008**|
|ε = 3|53%|1 pt|**0.004**|

**Stronger privacy costs accuracy and shrinks memorization.** The ε = 10 model is the one saved as `models/dp_model.safetensors` for the validation server.

> **Heads-up:** the ε symbols in your paste broke across lines (`ε\n=\n10`). Every one of them just means **ε = 10** or **ε = 3**. The `# Output:` comments in the code are the course's summaries, not the exact print format. The real line looks like `Epoch [1/20] Loss: 2.3100 | Acc: 25.40% | ε: 1.24`.

---

## Part 1: Setting up the script

```python
print("=" * 80)
print(" DP-SGD PRIVACY MITIGATION DEMONSTRATION")
print("=" * 80)
print(f"\nDevice: {device}")
print(f"Random seed: {RANDOM_SEED}")
```

|Code|What it does|
|---|---|
|`"=" * 80`|prints a line of 80 `=` signs as a banner|
|`f"...{device}"`|shows `cuda` or `cpu`, and the seed 1337|

```python
print("\nLoading CIFAR-10 dataset...")
train_dataset, test_dataset, train_loader, test_loader = get_cifar10_loaders(batch_size=BATCH_SIZE, download=True)

print(f"Training samples: {len(train_dataset):,}")
print(f"Test samples: {len(test_dataset):,}")
print(f"Batch size: {BATCH_SIZE}")
```

|Code|What it does|
|---|---|
|`get_cifar10_loaders(...)`|returns the 4-tuple from Section 9. `train_loader` here is the **plain (non-Poisson) loader**, so keep it for fair measurement later|
|`:,`|prints numbers with commas: 50,000 and 10,000|

---

## Part 2: The baseline model

```python
print("\n" + "=" * 80)
print(" TRAINING: BASELINE MODEL (No Privacy Protection)")
print("=" * 80)

baseline_model = CIFAR10CNN().to(device)
baseline_model = train_baseline_sgd(baseline_model, train_loader, device, epochs=BASELINE_EPOCHS, learning_rate=BASELINE_LR)
# Output: Epoch 1/20 - Loss: 2.31, Acc: 25.4%
# Output: Epoch 10/20 - Loss: 1.12, Acc: 61.2%
# Output: Epoch 20/20 - Loss: 0.68, Acc: 77.3%
```

|Code|What it does|
|---|---|
|`CIFAR10CNN().to(device)`|a fresh, untrained 122,570-parameter CNN|
|`train_baseline_sgd(...)`|20 epochs of plain SGD, lr 0.1, momentum 0.9 (the library default). **No clipping, no noise.**|

**Why the loss starts at 2.3:** a fresh model guesses about evenly, so p(true class) ≈ 1/10, and the cross-entropy is −ln(0.1) = **ln 10 = 2.303**.

> **An honest note the course skips:** "about 25% (random guessing across 10 classes)" is wrong. Random guessing on 10 classes is **10%**. The 25% is the _average over epoch 1_, and the model is already learning during that epoch.

```python
train_acc_baseline = evaluate_accuracy(baseline_model, train_loader, device)
test_acc_baseline = evaluate_accuracy(baseline_model, test_loader, device)

print("\nBaseline Model Performance:")
print(f" Training accuracy: {train_acc_baseline:.2f}%")
print(f" Test accuracy: {test_acc_baseline:.2f}%")
print(f" Overfitting gap: {train_acc_baseline - test_acc_baseline:.2f}%")
```

|Code|What it does|
|---|---|
|`evaluate_accuracy(...)`|eval mode + `no_grad`, returns a percentage from 0 to 100|
|`train − test`|the **overfitting gap**, the same memorization signal as the Adult target's 11.1 pts in Section 6|

```python
torch.save(baseline_model.state_dict(), "output/baseline_model.pth")
print("\nSaved baseline model to output/baseline_model.pth")
```

Saves the baseline weights. It isn't Opacus-wrapped, so no `._module` is needed here.

```python
print("\n" + "=" * 80)
print(" MEMBERSHIP INFERENCE MEASUREMENT: Baseline Model")
print("=" * 80)

mia_acc_baseline, mia_adv_baseline = compute_mia_advantage(
    baseline_model, train_loader, test_loader, device
)

print("\nMIA Results (Baseline):")
print(f" Attack accuracy: {mia_acc_baseline:.4f}")
print(f" Attack advantage: {mia_adv_baseline:.4f}")
print(" Random baseline: 0.5000")
```

|Code|What it does|
|---|---|
|`compute_mia_advantage(...)`|max-confidence threshold attack on a balanced 10k/10k pool. Returns (best accuracy, accuracy − 0.5)|
|`"Random baseline: 0.5000"`|honest this time, because the pool is balanced (the Section 6 lesson)|

**Course result:** about 77% train, 67% test, gap ≈ 10 pts, **MIA advantage ≈ 0.019**. 0.019 × 10,000 = **190** extra correct guesses per 10,000 balanced queries.

> **An honest note the course skips:** the advantage is "modest" for two reasons. One is the course's reason: 5,000 images per class is a lot of data. The other is that this attack is weak. It uses only max confidence and ignores the true label, unlike Section 2's loss attack. A stronger attack would likely show more leakage.

---

## Part 3: DP-SGD at ε = 10

```python
from opacus import PrivacyEngine
from opacus.validators import ModuleValidator
```

```python
print("\n" + "=" * 80)
print(" TRAINING: DP-SGD MODEL (Target ε=10)")
print("=" * 80)

TARGET_EPSILON_10 = 10.0

_, _, train_loader_dp, test_loader_dp = get_cifar10_loaders(batch_size=BATCH_SIZE, download=False)

dp_model_10 = CIFAR10CNN().to(device)
dp_model_10 = ModuleValidator.fix(dp_model_10)
optimizer_dp = optim.SGD(dp_model_10.parameters(), lr=DP_LR, momentum=0.9)
```

|Code|What it does|
|---|---|
|`_, _, train_loader_dp, test_loader_dp = ...`|fresh loaders. `_` throws away the two dataset objects; `download=False` uses the cached files|
|`ModuleValidator.fix(...)`|swaps out BatchNorm and similar layers. CIFAR10CNN has none, so the model comes back unchanged|
|`optim.SGD(..., lr=DP_LR, momentum=0.9)`|the same optimizer settings as the baseline (parallel conditions)|

```python
privacy_engine = PrivacyEngine(accountant="rdp")
dp_model_10, optimizer_dp, train_loader_dp = privacy_engine.make_private_with_epsilon(
    module=dp_model_10,
    optimizer=optimizer_dp,
    data_loader=train_loader_dp,
    target_epsilon=TARGET_EPSILON_10,
    target_delta=DELTA,
    epochs=DP_EPOCHS,
    max_grad_norm=MAX_GRAD_NORM,
)

print(f"\nConfiguration:")
print(f" Target epsilon: {TARGET_EPSILON_10}")
print(f" Delta: {DELTA}")
print(f" Max gradient norm: {MAX_GRAD_NORM}")
```

|Code|What it does|
|---|---|
|`PrivacyEngine(accountant="rdp")`|one engine with the **RDP accountant** (Section 7)|
|`make_private_with_epsilon(...)`|solves for the noise multiplier that hits ε = 10 at δ = 10⁻⁵ after 20 epochs, then returns **3 wrapped objects**|
|`dp_model_10` (returned)|now a `GradSampleModule`, which records per-sample gradients|
|`optimizer_dp` (returned)|now a `DPOptimizer`, which clips to 1.0, sums, adds noise and averages|
|`train_loader_dp` (returned)|**now the Poisson loader** (q = 0.00512). It **overwrites** the plain loader you had under that name|

```python
dp_model_10, final_epsilon_10 = train_dp_sgd(
    dp_model_10, train_loader_dp, privacy_engine, optimizer_dp, device
)
# Output: Epoch 1/20 - Loss: 2.35, Acc: 22.1%, ε: 1.24
# Output: Epoch 10/20 - Loss: 1.52, Acc: 48.3%, ε: 5.82
# Output: Epoch 20/20 - Loss: 1.21, Acc: 58.4%, ε: 10.00

print(f"\nFinal privacy guarantee: (ε={final_epsilon_10:.2f}, δ={DELTA})")
# Output: Final privacy guarantee: (ε=10.00, δ=1e-05)

train_acc_dp10 = evaluate_accuracy(dp_model_10, train_loader_dp, device)
test_acc_dp10 = evaluate_accuracy(dp_model_10, test_loader_dp, device)

print(f"\nDP Model (ε=10) Performance:")
print(f" Training accuracy: {train_acc_dp10:.2f}%")
print(f" Test accuracy: {test_acc_dp10:.2f}%")
print(f" Overfitting gap: {train_acc_dp10 - test_acc_dp10:.2f}%")
# Output: Training accuracy: 61.24%, Test accuracy: 58.15%, Overfitting gap: 3.09%
```

|Code|What it does|
|---|---|
|`train_dp_sgd(...)`|the ordinary training loop, made private because the optimizer is the wrapped one. Prints the **spent** ε each epoch|
|`ε: 1.24 -> 5.82 -> 10.00`|the budget being spent as training goes on (Section 7's composition). It lands on the target because the noise was solved for exactly these 20 epochs|
|`evaluate_accuracy(dp_model_10, train_loader_dp, ...)`|"train accuracy" measured on **Poisson draws**, not the full training set|

> **If this errors / silently misbehaves:** `train_dp_sgd` is called **without** `epochs=` and `delta=`, so it falls back to its defaults of 20 and 1e-5. Those match the config by luck. **If you lower `DP_EPOCHS` to save time, the noise is sized for fewer epochs but it still trains 20, and ε overshoots your target.** Safer: `train_dp_sgd(dp_model_10, train_loader_dp, privacy_engine, optimizer_dp, device, epochs=DP_EPOCHS, delta=DELTA)`.

> **An honest note the course skips:** "about 60-70% of per-sample gradients get clipped" isn't measured anywhere in this code. Treat it as a claim, not a result.

```python
torch.save(dp_model_10._module.state_dict(), "output/dp_model_eps10.pth")

# Save in safetensors format for validator API
save_file(dp_model_10._module.state_dict(), "models/dp_model.safetensors")
print("Saved DP model (ε=10) to models/dp_model.safetensors for validation")
```

|Code|What it does|
|---|---|
|`._module`|**unwraps the Opacus wrapper**. Without it, every key gets a `_module.` prefix (e.g. `_module.conv1.weight`), and a plain `CIFAR10CNN` or the validator can't load it|
|`save_file(..., "models/dp_model.safetensors")`|**the saved model file**. A good target is test accuracy **above 50%** with MIA advantage **below 5%**|

```python
print("\n" + "=" * 80)
print(" MEMBERSHIP INFERENCE MEASUREMENT: DP Model (ε=10)")
print("=" * 80)

mia_acc_dp10, mia_adv_dp10 = compute_mia_advantage(
    dp_model_10, train_loader_dp, test_loader_dp, device
)

print(f"\nMIA Results (DP ε=10):")
print(f" Attack accuracy: {mia_acc_dp10:.4f}")
print(f" Attack advantage: {mia_adv_dp10:.4f}")

improvement_10 = mia_adv_baseline - mia_adv_dp10
print(f"\nPrivacy Improvement vs Baseline:")
print(f" MIA advantage reduction: {improvement_10:+.4f}")
print(f" Accuracy cost: {test_acc_baseline - test_acc_dp10:.2f}%")
```

|Code|What it does|
|---|---|
|`compute_mia_advantage(dp_model_10, train_loader_dp, ...)`|the same attack, but "members" come from the **Poisson** loader. This is the case I flagged in Section 9|
|`improvement_10`|baseline advantage − DP advantage: 0.019 − 0.008 = **+0.011**|
|`Accuracy cost`|67.13 − 58.15 = **8.98**. These are **percentage points**, despite the `%` sign|

---

## Part 4: DP-SGD at ε = 3

```python
print("\n" + "=" * 80)
print(" TRAINING: DP-SGD MODEL (Target ε=3)")
print("=" * 80)

TARGET_EPSILON_3 = 3.0

_, _, train_loader_dp3, test_loader_dp3 = get_cifar10_loaders(batch_size=BATCH_SIZE, download=False)

dp_model_3 = CIFAR10CNN().to(device)
dp_model_3 = ModuleValidator.fix(dp_model_3)
optimizer_dp3 = optim.SGD(dp_model_3.parameters(), lr=DP_LR, momentum=0.9)

privacy_engine_3 = PrivacyEngine(accountant="rdp")
dp_model_3, optimizer_dp3, train_loader_dp3 = privacy_engine_3.make_private_with_epsilon(
    module=dp_model_3,
    optimizer=optimizer_dp3,
    data_loader=train_loader_dp3,
    target_epsilon=TARGET_EPSILON_3,
    target_delta=DELTA,
    epochs=DP_EPOCHS,
    max_grad_norm=MAX_GRAD_NORM,
)

dp_model_3, final_epsilon_3 = train_dp_sgd(
    dp_model_3, train_loader_dp3, privacy_engine_3, optimizer_dp3, device
)
# Output: Epoch 20/20 - Loss: 1.45, Acc: 53.2%, ε: 3.00

print(f"\nFinal privacy guarantee: (ε={final_epsilon_3:.2f}, δ={DELTA})")
# Output: Final privacy guarantee: (ε=3.00, δ=1e-05)

torch.save(dp_model_3._module.state_dict(), "output/dp_model_eps3.pth")
```

This is the ε = 10 recipe repeated, with **its own fresh model, optimizer and PrivacyEngine** (one accountant per budget). The only real change is `TARGET_EPSILON_3 = 3.0`, which forces a bigger noise multiplier. That shows up as a slower loss drop (1.45 vs 1.21 at epoch 20).

> **An honest note the course skips:** the course again says the noise multiplier is "around 3.8 vs 1.2, roughly 3× more noise". My Opacus accountant check from Section 7, with this exact setup (RDP, q = 0.00512, 20 epochs, δ = 10⁻⁵), gave **≈ 0.83 vs ≈ 0.57**. That's about **1.5×**, not 3×. My first copy-paste cell prints the real values.

```python
train_acc_dp3 = evaluate_accuracy(dp_model_3, train_loader_dp3, device)
test_acc_dp3 = evaluate_accuracy(dp_model_3, test_loader_dp3, device)

print(f"\nDP Model (ε=3) Performance:")
print(f" Training accuracy: {train_acc_dp3:.2f}%")
print(f" Test accuracy: {test_acc_dp3:.2f}%")
print(f" Overfitting gap: {train_acc_dp3 - test_acc_dp3:.2f}%")
# Output: Training accuracy: 54.12%, Test accuracy: 53.01%, Overfitting gap: 1.11%

mia_acc_dp3, mia_adv_dp3 = compute_mia_advantage(
    dp_model_3, train_loader_dp3, test_loader_dp3, device
)

print(f"\nMIA Results (DP ε=3):")
print(f" Attack accuracy: {mia_acc_dp3:.4f}")
print(f" Attack advantage: {mia_adv_dp3:.4f}")
# Output: Attack accuracy: 0.5040, Attack advantage: 0.0040

improvement_3 = mia_adv_baseline - mia_adv_dp3
print(f"\nPrivacy Improvement vs Baseline:")
print(f" MIA advantage reduction: {improvement_3:+.4f}")
print(f" Accuracy cost: {test_acc_baseline - test_acc_dp3:.2f}%")
# Output: MIA advantage reduction: +0.0150, Accuracy cost: 14.12%
```

**Arithmetic check:** gap 54.12 − 53.01 = 1.11 ✓. Reduction 0.019 − 0.004 = 0.015 ✓. Cost 67.13 − 53.01 = 14.12, so the baseline test accuracy is 67.13% ✓.

---

## Part 5: How small is "small"? The MIA noise floor (the key lesson)

### Step 1: Toy, luck with 4 samples

Two members [0.7, 0.4] and two non-members [0.6, 0.3], with confidences picked **at random** so there's zero real leakage. Try threshold t = 0.7: 0.7 is called member ✓, 0.4 non-member ✗, 0.6 and 0.3 non-member ✓✓. That's **3/4 = 0.75**, an "advantage" of 0.25 from pure luck. Picking the best of many thresholds always finds some lucky cut.

### Step 2: Luck shrinks with more samples, but never to zero

The luck shrinks roughly like 1/√n. For this exact setup (10,000 vs 10,000, best of 1,000 thresholds), a **model that leaks nothing scores a course-style advantage of about 0.004 on average, and about 0.008 in an unlucky run** (one-sided Kolmogorov - Smirnov math: 0.627 / √5000 / 2 ≈ 0.0044).

### Step 3: What that does to the table

|Model|Measured adv|vs noise floor (≈ 0.004, up to ~0.008)|Honest reading|
|---|---|---|---|
|Baseline|0.019|~4× the floor|**real leakage**|
|ε = 10|0.008|at the top of the luck range|can't tell it apart from zero|
|ε = 3|0.004|exactly the average luck level|**same as zero leakage**|

> **An honest note the course skips:** "reduces MIA advantage to 0.4%, barely above random" is really "**indistinguishable from random with this attack**". For a report, say the DP models' residual advantage is within the measurement's noise floor, and back that with a null check (my cell 3) or a confidence interval. Don't quote 0.004 as a real leak.

### Step 4: The Poisson-loader issue

`train_loader_dp` and `train_loader_dp3` were **overwritten** by `make_private_with_epsilon`. So the DP "train accuracy" and the MIA "members" come from Poisson draws: about 63% of distinct records, some of them twice. They're still real members, so the numbers stay roughly right, but the fair measurement uses your plain `train_loader`, which still exists from Part 1. My cell 2 does that.

---

## Reading the output

|Model|Train|Test|Gap|MIA adv|Accuracy cost|What it means|
|---|---|---|---|---|---|---|
|Baseline|~77%|67.13%|~10 pts|0.019| - |the accuracy ceiling and the "privacy floor" (most leakage)|
|ε = 10|61.24%|58.15%|3.09|0.008|8.98 pts|gap cut by ~70%, MIA near the noise floor. **This is the one submitted (>50% acc, <5% adv)**|
|ε = 3|54.12%|53.01%|1.11|0.004|14.12 pts|almost no memorization; costs 5 more points than ε = 10|

The training trajectory also shows the cost of noise. At epoch 20 the baseline loss is 0.68 vs 1.21 (ε = 10) and 1.45 (ε = 3). **The gap shrinking from 10 to 3 to 1 is the clearest evidence DP-SGD works**, and it's a sturdier signal than these tiny MIA numbers.

---

## Copy-paste cells (mine, not the course's)

I didn't run these. They use your existing names.

### See the real noise multipliers

Uses `optimizer_dp` and `optimizer_dp3` (the `DPOptimizer`s).

```python
print(f"noise multiplier eps=10: {optimizer_dp.noise_multiplier:.3f} eps=3: {optimizer_dp3.noise_multiplier:.3f}")
```

Expected: about **0.57** and **0.83** (not the course's 1.2 / 3.8).

### Fair MIA with the plain loader

Uses `train_loader` and `test_loader` from Part 1.

```python
mia_acc_dp10_plain, mia_adv_dp10_plain = compute_mia_advantage(dp_model_10, train_loader, test_loader, device)
mia_acc_dp3_plain, mia_adv_dp3_plain = compute_mia_advantage(dp_model_3, train_loader, test_loader, device)
print(f"eps=10 adv (plain loader): {mia_adv_dp10_plain:.4f} eps=3 adv (plain loader): {mia_adv_dp3_plain:.4f}")
```

Expected: close to the Poisson versions, within a few thousandths.

### Measure the noise floor (what a zero-leak model scores)

This is pure NumPy and copies `compute_mia_advantage`'s threshold search exactly.

```python
import numpy as np
rng = np.random.default_rng(0)
null_adv = []
for trial in range(20):
    conf = rng.beta(5, 2, size=20000) # members and non-members from the SAME distribution
    labels = np.r_[np.ones(10000), np.zeros(10000)]
    ths = np.percentile(conf, np.linspace(0, 100, 1000))
    best = max(np.mean((conf >= t) == labels) for t in ths)
    null_adv.append(best - 0.5)
print(f"zero-leak model, course-style advantage: mean {np.mean(null_adv):.4f}, max {np.max(null_adv):.4f}")
```

Expected: mean ≈ **0.004**, max ≈ **0.008**. Takes a few seconds.

## Key facts to lock in

- **The baseline (no clipping, no noise) reached ~77% train / 67% test, a ~10-point overfitting gap, with a confidence-MIA advantage of 0.019**, about 190 extra correct guesses per 10,000 balanced queries. It's the accuracy ceiling and the most-leaky reference point.
- ==DP-SGD at ε = 10 cut the gap to 3.09 points and test accuracy to 58.15% (a ~9-point cost); ε = 3 cut the gap to 1.11 points at 53.01% (a 14.12-point cost)==. Stronger privacy means less memorization and lower utility.
- **`make_private_with_epsilon` returns three wrapped objects (GradSampleModule, DPOptimizer, Poisson DataLoader) that overwrite the originals**. Each ε needs its own fresh model, optimizer and PrivacyEngine, and you should keep a plain train loader for fair measurement.
- **Always save `model._module.state_dict()` from an Opacus model**, because the wrapper adds a `_module.` prefix to every key. The validator expects the ε = 10 model as `models/dp_model.safetensors`, targeting test accuracy above 50% and MIA advantage below 5%.
- ==Pass `epochs=DP_EPOCHS` and `delta=DELTA` to `train_dp_sgd` explicitly==, because its defaults (20, 1e-5) only match the config by coincidence. Training longer than the epochs the noise was sized for pushes the spent ε past the target.
- **On a balanced 10k/10k pool with the best of 1,000 thresholds, a model that leaks nothing still scores a course-style advantage of about 0.004 (up to ~0.008)**. So the ε = 3 (0.004) and ε = 10 (0.008) results can't be told apart from zero with this attack, while the baseline's 0.019 is real.
- ==The course's noise-multiplier claims (1.2 for ε = 10, 3.8 for ε = 3) don't match Opacus's RDP accountant for this setup (≈ 0.57 and ≈ 0.83)==. Read `optimizer.noise_multiplier` instead of trusting the prose. Initial loss 2.3 = ln 10 is correct, but random accuracy on 10 classes is 10%, not 25%.
- **Method point:** **in a report, show the before/after table (test accuracy, overfitting gap, MIA advantage, spent ε with δ), quote accuracy cost in percentage points, and compare residual MIA advantage against a null noise floor**. Remember a weak max-confidence attack only gives a lower bound on leakage.

---
