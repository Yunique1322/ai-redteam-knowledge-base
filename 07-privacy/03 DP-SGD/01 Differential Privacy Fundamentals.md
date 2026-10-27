# Differential Privacy Fundamentals

## The idea in one breath

The shadow-model attack worked because the model was **overfit**: it was more confident on records it had trained on. **DP-SGD** (Differentially Private Stochastic Gradient Descent) fixes this during training. It **clips** each training example's gradient to a fixed size, then **adds Gaussian noise** before every update, so no single record can shape the final weights much. How strong the protection is gets set by the **privacy budget ε**, where smaller means more private. The course trains CIFAR-10 at ε = 10 (modest privacy) and ε = 3 (stronger privacy, lower accuracy).

> **Heads-up:** your paste lost the superscripts and square roots. Here are the correct readings:
> 
> - `P[M(D)∈S]≤eε⋅P[M(D′)∈S]+δ` -> **P[M(D) ∈ S] ≤ e^ε · P[M(D′) ∈ S] + δ**
> - `eε` -> **e^ε** ("e to the power ε")
> - `σ=Δf⋅2ln(1.25/δ)/ε` -> **σ = Δf · √(2 ln(1.25/δ)) / ε** (the square root covers the whole 2 ln(...))
> - `δ=10−5` -> **δ = 10⁻⁵ = 0.00001**
> 
> **Symbol clashes. These are the ones to watch:**
> 
> - **ε** here is the **privacy budget**. It is NOT the FGSM ε (perturbation size). In FGSM, a bigger ε meant a stronger attack. Here, a bigger ε means _weaker privacy_.
> - **δ** here is a tiny failure probability. It is NOT the perturbation δ from the evasion modules.
> - **σ** here is the noise standard deviation. It is NOT the StandardScaler σ from the Adult data.

---

## Part 1: Defend the root cause, not the output

### Step 1: Why training-time

In Sections 1-6 the leak came from the gap between how the model treats members and non-members. You can hide outputs (return only labels, round the probabilities), but label-only MIAs still exist. That's the Section 2 note: hiding outputs reduces the leak but doesn't remove it. DP-SGD goes after the cause instead. If no single record can change the weights much, there is very little membership signal left for any attack to find.

### Step 2: "Controlled amnesia"

Here's an analogy. A teacher grades a class, but each student's complaint can move the curve by at most 1 point (that's clipping). The teacher also adds a random jitter to the curve (that's noise). Looking at the final curve, you can't tell whether Alice was in the class.

### Step 3: The running experiment

- **Dataset:** **CIFAR-10**, with 50,000 training images, 10,000 test images and 10 classes.
- **Plan:** train a normal CNN, then measure how vulnerable it is to a **confidence-threshold MIA**. Then retrain the _same architecture_ with DP-SGD at ε = 10 and ε = 3, and compare.

---

## Part 2: The formal definition of differential privacy

### Step 1: Toy first, a coin-flip survey (randomized response)

You ask someone "were you in the training set?" They secretly flip a coin:

- **Heads:** they answer truthfully.
- **Tails:** they flip again and say "yes" on heads, "no" on tails, whatever the truth is.

|Truth|P(answer "yes")|Arithmetic|
|---|---|---|
|Really yes|**0.75**|0.5 (truthful) + 0.5 × 0.5 (random yes)|
|Really no|**0.25**|0 + 0.5 × 0.5 (random yes)|

So a "yes" is at most **0.75 / 0.25 = 3 times** more likely if the person really was in. That ratio is the whole idea of DP: **e^ε = 3, so ε = ln 3 ≈ 1.10.**

### Step 2: What the attacker learns

Say the attacker starts at 50/50 and hears "yes". Their belief becomes 0.75 / (0.75 + 0.25) = **75%**. That's the same as e^ε / (1 + e^ε) = 3/4. **ε caps how far any observation can move the attacker's belief.**

### Step 3: The real formula

**P[M(D) ∈ S] ≤ e^ε · P[M(D′) ∈ S] + δ**

|Symbol|Plain English|Toy|Real (course)|
|---|---|---|---|
|M|the randomized algorithm|the coin-flip survey|DP-SGD training|
|D, D′|two datasets that differ in exactly ONE record|Alice in / Alice out|CIFAR-10 with / without one image|
|S|any set of outcomes you might observe|the answer "yes"|any set of trained weights|
|P[M(D) ∈ S]|chance of seeing that outcome|0.75|(not computable by hand)|
|e^ε|the max multiplier between the two worlds|3|e¹⁰ ≈ 22,026 or e³ ≈ 20.1|
|δ|small slack where the bound may not hold|0|10⁻⁵|

**Read aloud:** "Whatever you look at, it's at most e^ε times more likely with Alice's record than without it, plus a tiny slack δ."

### Step 4: What ε means for an MIA (the number for your report)

On a balanced 50/50 pool, the best possible attacker's accuracy is capped at (e^ε + δ) / (1 + e^ε):

|ε|e^ε|Worst-case max MIA accuracy|What it means|
|---|---|---|---|
|1|2.72|**73.1%**|real limit|
|2|7.39|**88.1%**|weaker limit|
|3|20.1|**95.3%**|loose|
|8|2,981|**99.97%**|**no worst-case limit at all**|
|10|22,026|**99.995%**|**no worst-case limit at all**|

==ε = 10 is "provably private" on paper, but its worst-case bound puts no real limit on an MIA.== In practice it still cuts real attacks a lot. That gap is Part 7.

> **An honest note the course skips:** the course calls ε = 1 "near-complete indifference". Odds that can still move 2.7× (up to 73% attacker accuracy) are strong privacy, but not near-complete. The course also says δ = 10⁻⁵ means "holds with probability 0.99999". That's a loose reading. The rule practitioners use is **δ much smaller than 1/n**: here 1/50,000 = 2×10⁻⁵, so 10⁻⁵ just passes. If δ ≈ 1/n, a mechanism could fully expose about one record and still "satisfy" DP.

---

## Part 3: Gradient clipping and sensitivity

### Step 1: Toy, three per-sample gradients, C = 1

Here C is `max_grad_norm`. Sample B reuses your norms toy (L2 = 0.5).

|Sample|Gradient|L2 norm|Scale = min(1, C/norm)|Clipped|
|---|---|---|---|---|
|A (outlier)|[3, 4]|5.0|1/5 = 0.2|**[0.6, 0.8]**|
|B|[0.3, −0.4]|0.5|min(1, 2) = 1|[0.3, −0.4] (unchanged)|
|C|[−0.6, 0.8]|1.0|1|[−0.6, 0.8] (unchanged)|

### Step 2: Why this bounds sensitivity

- **Raw sum** = [2.7, 4.4]. Removing A changes it by [3, 4], which has norm **5**. A single outlier dominates the update.
- **Clipped sum** = [0.3, 1.2]. Removing A changes it by [0.6, 0.8], which has norm **1 = C**.

**Sensitivity** is the most one record can change the result. After clipping, ==sensitivity = C==, whoever is added or removed. Without a known sensitivity you can't decide how much noise to add.

### Step 3: The real rule and the course's numbers

**g <- g · min(1, C / ‖g‖₂)**. The course's example: norm 5.2 with C = 1.0 gets scale 1/5.2 ≈ 0.192, so the new norm is exactly 1.0.

**The tradeoff:** a small C gives stronger privacy for the same noise, but it throws away signal from large-gradient (often the most informative) samples, so training is slower and accuracy is lower.

> **An honest note the course skips:** "sensitivity is _exactly_ max_grad_norm" should be "_at most_". That's for add/remove-one neighbours, which is what Opacus uses. For replace-one neighbours it's 2C.

---

## Part 4: Noise and the Gaussian mechanism

### Step 1: Toy noise, continuing Part 3

The noise multiplier is z = 1.2 and C = 1, so std = z × C = **1.2** on each coordinate of the **sum**. Suppose the random draw is [0.9, −1.5]:

- Noisy sum = [0.3 + 0.9, 1.2 − 1.5] = [1.2, −0.3]
- Divide by batch size B = 3 -> **[0.40, −0.10]**
- True clipped average = [0.10, 0.40]. With 3 samples the noise even flipped the sign of the second coordinate.

### Step 2: Why a big batch rescues it

The noise is added _once per batch_, to the sum. With B = 256, the noise on the averaged gradient is 1.2 / 256 ≈ **0.0047**. The true signal still averages to a normal-sized step. **Big batches dilute the noise**, which is why DP-SGD configs use large batches.

### Step 3: The real formulas

- **DP-SGD noise:** std = `noise_multiplier × max_grad_norm`.
- **Closed-form Gaussian mechanism:** σ = Δf · √(2 ln(1.25/δ)) / ε. Toy check with Δf = 1, δ = 10⁻⁵, ε = 1: ln(125,000) = 11.74, so √23.47 = 4.84, giving **σ ≈ 4.84**.

|Symbol|Plain English|Toy|Real|
|---|---|---|---|
|Δf|sensitivity (max change from one record)|1 (= C)|max_grad_norm = 1.0|
|δ|failure slack|10⁻⁵|10⁻⁵|
|ε|privacy budget|1|10 or 3|
|σ|noise standard deviation|4.84|noise_multiplier × C|

**Read aloud:** "Noise std = how much one record can move things, times a δ factor, divided by ε. More privacy (smaller ε) means more noise."

> **An honest note the course skips:** (1) That closed form only holds for **ε < 1**, so it can't be used at ε = 3 or 10. The course does say the accountant sets the real values. (2) **Subsampling amplification:** each step only touches about 256/50,000 ≈ 0.5% of the data, which makes each step much more private. That's why the needed noise multiplier is far below 4.84. (3) **I checked the course's numbers with Opacus's RDP accountant** (B = 256, n = 50,000, δ = 10⁻⁵). ε = 10 needs only about **0.53 / 0.57 / 0.59** at 10 / 20 / 30 epochs, and ε = 3 needs about **0.76 / 0.83 / 0.90**. The course's "1.2 for ε = 10, 3.8 for ε = 3" doesn't match a standard setup (1.2 is what ε ≈ 1 needs at 10 epochs). Trust whatever their code prints in the next section.

---

## Part 5: Privacy adds up across training (composition)

### Step 1: Toy

**Composition** means each step leaks a little and the leaks add up. Naive composition: 1,000 steps × ε 0.01 = **ε 10**. Real CIFAR runs take many more steps: 50,000 / 256 ≈ 195 steps per epoch, so 20 epochs ≈ **3,900 steps**.

### Step 2: Tighter accounting

Naive addition overestimates the total. **Rényi Differential Privacy (RDP)** accounting tracks the total much more tightly. Opacus's **PrivacyEngine** has a **privacy accountant** built in. You give it a target ε, δ and number of epochs, and it solves for the noise multiplier.

### Step 3: The key consequence

**More epochs means more noise per step for the same final ε**, because each step must leak less. You can see it in my accountant numbers in Part 4: 0.53 at 10 epochs rises to 0.59 at 30. The budget is something you _spend_ over training.

---

## Part 6: Choosing budgets (ε = 10 vs ε = 3)

|Budget|Privacy|Utility|Where you see it|
|---|---|---|---|
|ε = 10|modest|keeps most accuracy|"reasonable privacy" research baseline (ε = 8-10)|
|ε = 3|strong|noticeable accuracy drop|"strong privacy" papers (ε = 1-3)|
|ε = 1|very strong|hard on CIFAR-10 without public pretraining, careful architecture and long training|skipped by the course|

> **An honest note the course skips:** Apple's "ε ≈ 2-8" values are **per-day, local DP** budgets (the noise is added on each phone). That's a different setting from central DP-SGD, so the numbers aren't directly comparable. Don't cite them in a report as a benchmark for model training.

---

## Part 7: Theoretical guarantee vs empirical MIA

|Aspect|DP guarantee (ε, δ)|Empirical MIA (what you ran in Secs 3-6)|
|---|---|---|
|Kind of bound|**upper bound** (worst case)|**lower bound** (what one known attack achieves)|
|Holds against|any attack, including future ones|only the attack you tried|
|Assumes|worst-case sample, unlimited adversary|average sample, a simple threshold or shadow attack|
|Typical result|loose (ε = 10 gives no worst-case limit)|much lower (e.g. AUC near 0.5)|

**Why they differ:** the bound assumes the single most influential record and an optimal attacker. Most records have modest influence, and a confidence threshold is a weak attack. Outliers, as in your Section 2 experiment (AUC 0.991 vs 0.905), sit closer to the bound.

**Use both:** low empirical MIA _without_ DP can be broken by a better attack tomorrow. DP holds no matter how attacks improve. This ties straight back to Section 6: measure the empirical side with **AUC on a balanced pool**, never raw accuracy.

---

## Part 8: Writing DP-SGD as a fix in a report (mine, not the course's)

This is the priority piece for an engagement report:

- **Finding:** Membership inference against [model]. CWE-359 (Exposure of Private Personal Information) or CWE-200.
- **Root cause:** the model is overfit (train/test gap of X points), so it is more confident on training records. A shadow-model or threshold attack reached AUC X on a balanced pool.
- **Remediation, short term:** return only labels or top-1, or rounded probabilities. Add early stopping and weight decay. These reduce the leak but don't eliminate it.
- **Remediation, medium term:** retrain with **DP-SGD** (Opacus `PrivacyEngine`, TensorFlow Privacy). State **ε AND δ** (δ much smaller than 1/n), plus the clipping norm C, noise multiplier, epochs, sampling rate and accountant (RDP/PRV). State the accuracy cost (the privacy-utility tradeoff). ε ≤ 8 is the common target.
- **Remediation, long term:** pretrain on public data (makes low ε usable). Consider PATE (coming next). Re-run the MIA audit in CI and report AUC / TPR at low FPR before and after.

**Never report ε alone. Give ε with δ, and back it with the empirical before/after MIA AUC.**

---

## Reading the figure/output

There are no figures or output blocks in this section. The training and MIA comparison plots will come with the code in the next sections.

---

## Copy-paste cells (mine, not the course's)

These are standalone; they don't use any notebook names yet, because this section has no course code.

### Clip and noise toy

Reproduces Parts 3-4 with NumPy.

```python
import numpy as np
C = 1.0
grads = np.array([[3.0, 4.0], [0.3, -0.4], [-0.6, 0.8]]) # per-sample gradients A, B, C
norms = np.linalg.norm(grads, axis=1)
clipped = grads * np.minimum(1.0, C / norms)[:, None]
print("norms :", norms)
print("clipped :", clipped.round(3).tolist())
print("sum raw :", grads.sum(0), "| sum clipped:", clipped.sum(0).round(3))
rng = np.random.default_rng(0)
noise_multiplier = 1.2
noisy_avg = (clipped.sum(0) + rng.normal(0, noise_multiplier * C, size=2)) / len(grads)
print("noisy average:", noisy_avg.round(3), "| true clipped average:", (clipped.sum(0) / len(grads)).round(3))
```

Expected: norms `[5. 0.5 1.]`, clipped `[[0.6, 0.8], [0.3, -0.4], [-0.6, 0.8]]`, sums `[2.7 4.4]` vs `[0.3 1.2]`. The noisy average lands far from the true `[0.1 0.4]` because B = 3 is tiny.

### ε -> worst-case MIA accuracy

Prints the Part 2 table for any budget a client gives you.

```python
import numpy as np
delta = 1e-5
for eps in [0.5, 1, 2, 3, 8, 10]:
    acc = (np.exp(eps) + delta) / (1 + np.exp(eps))
    print(f"eps={eps:>4}: worst-case MIA accuracy <= {acc:.4f} (advantage <= {acc - 0.5:+.4f})")
```

Expected: 0.6225, 0.7311, 0.8808, 0.9526, 0.9997, 1.0000.

### Check the course's noise multipliers with Opacus

Needs `pip install opacus`.

```python
from opacus.accountants.utils import get_noise_multiplier
q = 256 / 50000
for E in [10, 20, 30]:
    print(E, [round(get_noise_multiplier(target_epsilon=e, target_delta=1e-5,
                                         sample_rate=q, epochs=E, accountant='rdp'), 2) for e in (10, 3, 1)])
```

Expected (tested): `10 [0.53, 0.76, 1.21]`, `20 [0.57, 0.83, 1.5]`, `30 [0.59, 0.9, 1.76]`.

## Key facts to lock in

- **DP-SGD defends against MIA at training time** by clipping every per-sample gradient to L2 norm C (`max_grad_norm`) and adding Gaussian noise with std = noise_multiplier × C to the batch sum. It attacks the root cause (overfitting) instead of hiding outputs.
- **(ε, δ)-DP means any outcome is at most e^ε times more likely with one record than without it, plus slack δ**. Smaller ε means stronger privacy, and δ should be much smaller than 1/n (10⁻⁵ for CIFAR-10's 50,000 records).
- **Clipping is what makes the noise calibratable**: after clipping, one record can change the gradient sum by at most C (the sensitivity). Without clipping, one outlier (norm 5 in the toy) can dominate the update.
- **Privacy composes across steps**, so more epochs need more noise per step for the same final ε. The RDP accountant in Opacus's PrivacyEngine solves for the noise multiplier from target ε, δ, epochs and sampling rate.
- **The worst-case MIA accuracy cap is e^ε / (1 + e^ε)**: 73% at ε = 1, 95% at ε = 3, and essentially 100% at ε = 8-10. High ε puts no worst-case limit on an attacker, even though it still cuts real attacks a lot.
- **The DP guarantee is an upper bound (any attack, worst case); empirical MIA is a lower bound (one known attack, average case)**. A model with low empirical MIA but no DP could be broken by a better attack later.
- **Privacy costs utility**: smaller ε, smaller C and more noise all lower accuracy. Large batches dilute the noise (1.2/256 ≈ 0.005 on the average), which is why DP-SGD uses them.
- **Method point:** **in a report, name the fix (DP-SGD via Opacus), give ε AND δ plus C, noise multiplier, epochs and accountant, state the accuracy cost, and prove it with before/after MIA AUC on a balanced pool**. Never quote ε alone or raw MIA accuracy on an imbalanced pool.

---
