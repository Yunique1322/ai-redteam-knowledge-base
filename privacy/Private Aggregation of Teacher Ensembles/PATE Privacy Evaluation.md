# PATE Privacy Evaluation

## The idea in one breath

We test PATE with the same kind of **confidence-threshold MIA** used before. First, a deliberately overfit **baseline** trained directly on 10,000 private images leaks: advantage **≈ 0.064**. Then the PATE **student** is attacked with the private data as "members", and scores **≈ 0.002**, which is random. The student _does_ leak about its own public training queries (**≈ 0.075**).

The big caveat: **the "private" test is a built-in pass**. The student never trained on the private data _or_ the holdout, so any such model scores ~50% there, even one with no noise at all. It proves the separation works; it says nothing about ε.

> **Heads-up:** no mangled formulas. You pasted each figure twice; both copies are identical.
> 
> **I merged three code units:** the course split `compute_mia_advantage`, `plot_confidence_distributions` and `plot_mia_comparison` each across **two** blocks (the second block is the indented rest of the same function). Each is given below as **one cell**, with every line unchanged.
> 
> **Name clash:** this `compute_mia_advantage` takes arrays and returns **3** values. The DP-SGD library function of the same name takes loaders and returns 2. Likewise, `baseline_model` reuses a DP-SGD name. Keep PATE in its own notebook.

---

## Part 1: The vulnerable baseline

```python
print("\n" + "=" * 60)
print("PHASE 8: Privacy Evaluation")
print("=" * 60)

BASELINE_CONFIG = {
    "hidden_layers": [256, 128],
    "dropout": 0.0,
    "epochs": 50,
    "batch_size": 64,
    "learning_rate": 0.001,
    "train_size": 10000,
}
```

|Key|Value|Why|
|---|---|---|
|`hidden_layers`|[256, 128]|bigger than the teachers' [128, 64], so more capacity to memorize|
|`dropout`|0.0|**no regularization**, to maximize overfitting (Section 2's factors)|
|`epochs`|50|long training, which widens the train/test gap|
|`train_size`|10,000|trained **directly on private data**: this is the "no PATE" reference|

```python
# Select subset of private data for baseline training
np.random.seed(RANDOM_SEED)
baseline_indices = np.random.choice(
    len(X_private_norm), BASELINE_CONFIG['train_size'], replace=False
)
X_baseline_members = X_private_norm[baseline_indices]
y_baseline_members = y_private[baseline_indices]

# Track which samples are NOT in the baseline training set
baseline_nonmember_mask = np.ones(len(X_private_norm), dtype=bool)
baseline_nonmember_mask[baseline_indices] = False
X_baseline_nonmembers = X_private_norm[baseline_nonmember_mask]
y_baseline_nonmembers = y_private[baseline_nonmember_mask]

print(f"Baseline members: {len(X_baseline_members)} samples")
print(f"Baseline non-members: {len(X_baseline_nonmembers)} samples")
```

|Code|What it does|
|---|---|
|`np.random.choice(48000, 10000, replace=False)`|10,000 distinct private images become the **members**|
|`np.ones(..., dtype=bool)` then `[baseline_indices] = False`|a mask that is True everywhere except the members|
|`X_private_norm[baseline_nonmember_mask]`|the other **38,000** are non-members from the **same distribution**, which makes this a fair MIA test|

```python
baseline_member_loader = create_dataloader(
    X_baseline_members, y_baseline_members,
    BASELINE_CONFIG['batch_size']
)

baseline_model = MLP(
    input_size=num_features,
    hidden_layers=BASELINE_CONFIG['hidden_layers'],
    num_classes=DATASET_CONFIG['num_classes'],
    dropout=BASELINE_CONFIG['dropout']
)

train_model(
    baseline_model, baseline_member_loader, holdout_loader,
    device=DEVICE,
    epochs=BASELINE_CONFIG['epochs'],
    learning_rate=BASELINE_CONFIG['learning_rate']
)
```

Builds the loader and the [256, 128] MLP, then trains it with Adam for 50 epochs. The holdout is only monitored, and it prints 50 lines.

---

## Part 2: The confidence-threshold attack (merged)

```python
def compute_mia_advantage(model, X_members, y_members, X_nonmembers, y_nonmembers, device):
    """
    Compute membership inference attack advantage using confidence threshold.
    """
    member_probs = get_model_predictions(model, X_members, device)
    member_confidence = np.max(member_probs, axis=1)

    nonmember_probs = get_model_predictions(model, X_nonmembers, device)
    nonmember_confidence = np.max(nonmember_probs, axis=1)
    n_samples = min(len(member_confidence), len(nonmember_confidence))
    member_conf_balanced = member_confidence[:n_samples]
    nonmember_conf_balanced = nonmember_confidence[:n_samples]

    all_confidence = np.concatenate([member_conf_balanced, nonmember_conf_balanced])
    all_labels = np.concatenate([np.ones(n_samples), np.zeros(n_samples)])

    thresholds = np.percentile(all_confidence, np.linspace(0, 100, 1000))
    best_accuracy = 0.0

    for threshold in thresholds:
        predictions = (all_confidence >= threshold).astype(int)
        accuracy = np.mean(predictions == all_labels)
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_threshold = threshold

    attack_advantage = best_accuracy - 0.5
    return best_accuracy, attack_advantage, best_threshold
```

**Toy, using Section 9's numbers:** members [0.9, 0.8, 0.6] and non-members [0.7, 0.5, 0.4]. At threshold 0.8: 2 members right + 3 non-members right = 5/6 = **0.833**, so the advantage is **0.333**.

|Code|What it does|
|---|---|
|`get_model_predictions(...)`|eval mode + `no_grad`, softmax probabilities|
|`np.max(..., axis=1)`|**confidence** = the top probability ([0.95, 0.05, ...] gives 0.95)|
|`n_samples = min(...)` and `[:n_samples]`|**balance** by taking the **first** n of each, so 50% is the honest baseline (the Section 6 lesson)|
|`np.percentile(..., linspace(0, 100, 1000))`|1,000 candidate thresholds spread across the observed confidences|
|`(all_confidence >= threshold)`|"member" if confident enough|
|the `if accuracy > best_accuracy` block|keeps the best threshold, which **gives the attacker every advantage**|
|returns|(best accuracy, accuracy − 0.5, the threshold that achieved it)|

> **An honest note the course skips:** `y_members` and `y_nonmembers` are passed in but **never used**. This attack ignores the true label (weaker than Section 2's loss attack). And as in Section 10, best-of-1,000 thresholds gives a **noise floor**: a leak-free model still scores about **0.004** at n = 10,000 and about **0.006** at n = 5,000.

---

## Part 3: The three attacks

```python
print("\n--- Attacking Baseline Model ---")
print(f" Members: {len(X_baseline_members)} samples (baseline training data)")
print(f" Non-members: {len(X_baseline_nonmembers)} samples")

baseline_mia_acc, baseline_mia_adv, baseline_threshold = compute_mia_advantage(
    baseline_model,
    X_baseline_members, y_baseline_members,
    X_baseline_nonmembers, y_baseline_nonmembers,
    DEVICE
)

print(f" Attack accuracy: {baseline_mia_acc:.4f}")
print(f" Attack advantage: {baseline_mia_adv:.4f}")
```

Members (10,000) vs the first 10,000 non-members. Figure: **accuracy ≈ 0.564, advantage ≈ 0.064**. That's well above the noise floor, so it's real leakage.

```python
print("\n--- Attacking PATE Student Model ---")

student_mia_acc_private, student_mia_adv_private, _ = compute_mia_advantage(
    student_model,
    X_private_norm, y_private,
    X_holdout_norm, y_holdout,
    DEVICE
)

print(f" Attack accuracy: {student_mia_acc_private:.4f}")
print(f" Attack advantage: {student_mia_adv_private:.4f}")
```

"Members" = private images (the first 10,000), "non-members" = holdout. Figure: **≈ 0.502 / 0.002**, below the noise floor, so nothing is detectable.

```python
student_mia_acc_query, student_mia_adv_query, _ = compute_mia_advantage(
    student_model,
    X_query, student_labels,
    X_holdout_norm, y_holdout,
    DEVICE
)

print(f" Attack on query data accuracy: {student_mia_acc_query:.4f}")
print(f" Attack on query data advantage: {student_mia_adv_query:.4f}")
```

Members = the 5,000 public images the student **did** train on (n = 5,000 each). Figure: **≈ 0.575 / 0.075**, which is real leakage.

### What the three tests actually measure

|Test|Did the model train on the "members"?|Role|Result|
|---|---|---|---|
|Baseline vs its training data|yes|the real attack|0.064, leaks|
|Student vs private data|**no, and not on the holdout either**|**negative control**|0.002, random|
|Student vs query data|yes|**positive control** (the attack works)|0.075, leaks|

> **An honest note the course skips:**
> 
> 1. **The "private" test can't fail.** Both groups are non-members of the student, so any model not trained on them scores ~50%. That includes a student trained on **clean** votes (ε = ∞). It shows the **separation** works, not the **noise**. The course's claim that this "validates PATE's theoretical guarantees" is an overclaim; its own figure caption even says the student "never trained on either". My cell 3 demonstrates it.
> 2. **The student leaks its own training data _more_ than the overfit baseline** (0.075 vs 0.064): it's trained on only 5,000 samples, with noisy labels. PATE gives the public data **no** protection, so that data must really be non-sensitive. "Public by assumption" is something to verify in an audit, not assume.
> 3. Where the private data really is exposed: the **teachers**. Each one overfits its 192 private images (my cell 2). That's why they must stay offline.

```python
print("\n" + "-" * 50)
print("SUMMARY: Attack Advantage Comparison")
print("-" * 50)
print(f" Baseline (no protection): {baseline_mia_adv:.4f}")
print(f" PATE Student (private data): {student_mia_adv_private:.4f}")
print(f" PATE Student (query data): {student_mia_adv_query:.4f}")

if baseline_mia_adv > 0:
    reduction = (baseline_mia_adv - student_mia_adv_private) / baseline_mia_adv * 100
    print(f"\n Protection improvement: {reduction:.1f}%")
```

|Code|What it does|
|---|---|
|`if baseline_mia_adv > 0`|avoids dividing by zero|
|`reduction`|relative drop: (0.0645 − 0.002) / 0.0645 ≈ **96.9%** with the figure's numbers|

---

## Part 4: The plotting functions (merged)

```python
def plot_confidence_distributions(baseline_member, baseline_nonmember,
                                   student_private, student_holdout, save_path=None):
    """Plot density difference between member and non-member confidence."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Baseline model
    bins = np.linspace(0.5, 1.0, 50)
    member_hist, _ = np.histogram(baseline_member, bins=bins, density=True)
    nonmember_hist, _ = np.histogram(baseline_nonmember, bins=bins, density=True)
    diff = member_hist - nonmember_hist
    bin_centers = (bins[:-1] + bins[1:]) / 2

    axes[0].bar(bin_centers, diff, width=0.01, color=MALWARE_RED, alpha=0.8)
    axes[0].axhline(y=0, color=WHITE, linestyle='--', alpha=0.5)
    axes[0].set_xlabel('Prediction Confidence')
    axes[0].set_ylabel('Density Difference (Member - Non-member)')
    axes[0].set_title('Baseline Model', color=MALWARE_RED, fontweight='bold')
    # PATE student
    student_member_hist, _ = np.histogram(student_private, bins=bins, density=True)
    student_nonmember_hist, _ = np.histogram(student_holdout, bins=bins, density=True)
    student_diff = student_member_hist - student_nonmember_hist

    axes[1].bar(bin_centers, student_diff, width=0.01, color=HTB_GREEN, alpha=0.8)
    axes[1].axhline(y=0, color=WHITE, linestyle='--', alpha=0.5)
    axes[1].set_xlabel('Prediction Confidence')
    axes[1].set_ylabel('Density Difference (Private - Holdout)')
    axes[1].set_title('PATE Student', color=HTB_GREEN, fontweight='bold')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=PLOT_CONFIG['dpi'], bbox_inches='tight', facecolor=NODE_BLACK)
    plt.close()
```

|Code|What it does|
|---|---|
|`bins = np.linspace(0.5, 1.0, 50)`|50 edges, so 49 bins about 0.0102 wide (as in Section 4)|
|`np.histogram(..., density=True)`|turns counts into a density (area = 1) **within 0.5-1.0 only**. Confidences below 0.5 are dropped from the plot|
|`diff = member − nonmember`|**above zero: members are more common at this confidence; below zero: non-members are**|
|`bin_centers`|the middle of each bin, where each bar is drawn|
|`axhline(y=0)`|dashed line at zero, meaning "no difference"|

```python
def plot_mia_comparison(baseline_adv, student_adv_private, student_adv_query,
                        baseline_acc, student_acc_private, student_acc_query, save_path=None):
    """Plot bar chart comparing MIA results across models."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    labels = ['Baseline', 'PATE\n(Private)', 'PATE\n(Query)']
    advantages = [baseline_adv, student_adv_private, student_adv_query]
    accuracies = [baseline_acc, student_acc_private, student_acc_query]
    colors = [MALWARE_RED, HTB_GREEN, AQUAMARINE]

    axes[0].bar(labels, advantages, color=colors, edgecolor=HACKER_GREY, linewidth=2)
    axes[0].axhline(y=0.05, color=NUGGET_YELLOW, linestyle='--', linewidth=2, label='5% threshold')
    axes[0].set_ylabel('Attack Advantage')
    axes[0].set_title('MIA Attack Advantage', color=HTB_GREEN, fontweight='bold')
    axes[0].legend()
    axes[1].bar(labels, accuracies, color=colors, edgecolor=HACKER_GREY, linewidth=2)
    axes[1].axhline(y=0.5, color=NUGGET_YELLOW, linestyle='--', linewidth=2, label='Random (50%)')
    axes[1].set_ylabel('Attack Accuracy')
    axes[1].set_title('MIA Attack Accuracy', color=HTB_GREEN, fontweight='bold')
    axes[1].legend()

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=PLOT_CONFIG['dpi'], bbox_inches='tight', facecolor=NODE_BLACK)
    plt.close()
```

|Code|What it does|
|---|---|
|`'PATE\n(Private)'`|`\n` puts the label on two lines|
|`axhline(y=0.05)`|the **5% advantage line**. It's this course's validator bar (DP-SGD: "advantage below 5%"), not an industry standard|
|`axhline(y=0.5)`|the random-guess line for accuracy|

```python
baseline_member_conf = np.max(get_model_predictions(baseline_model, X_baseline_members, DEVICE), axis=1)
baseline_nonmember_conf = np.max(get_model_predictions(baseline_model, X_baseline_nonmembers, DEVICE), axis=1)
student_private_conf = np.max(get_model_predictions(student_model, X_private_norm, DEVICE), axis=1)
student_holdout_conf = np.max(get_model_predictions(student_model, X_holdout_norm, DEVICE), axis=1)

plot_confidence_distributions(
    baseline_member_conf, baseline_nonmember_conf,
    student_private_conf, student_holdout_conf,
    save_path="figs/pate_confidence_distributions.png"
)
```

Computes the four confidence arrays (all 38,000 non-members and all 48,000 private images this time, unbalanced, which is fine for densities) and saves figure 1.

```python
plot_mia_comparison(
    baseline_mia_adv, student_mia_adv_private, student_mia_adv_query,
    baseline_mia_acc, student_mia_acc_private, student_mia_acc_query,
    save_path="figs/pate_mia_comparison.png"
)
```

Saves figure 2.

---

## Part 5: Interpreting the results

|Course claim|My verdict|
|---|---|
|Baseline vulnerability scales with overfitting|✓ the same lesson as Sections 2-6|
|PATE protection is architectural; the student answers queries for free|✓ (post-processing)|
|"The student achieves DP through the information bottleneck"|✗ the DP comes from the **Laplace noise** on the votes. The bottleneck is intuition (Section 13)|
|Query leakage "confirms methodology"|✓ as a positive control, but it also shows the public data is exposed|
|"Validates PATE's theoretical guarantees"|✗ the private test is a negative control that passes even at ε = ∞|

---

## Reading the figure/output

### Figure 1: Confidence density differences
![PATE Privacy Evaluation](/assets/19%20PATE%20Privacy%20Evaluation-20261023191009992.png)

|Panel|What you see|Meaning|
|---|---|---|
|**Baseline** (red, y-axis −1 to 6)|a huge spike of **+6.05** in the top bin (0.99-1.0); bars growing more **negative** from ~0.83 to 0.98 (deepest about −0.88 at ~0.985); ≈ 0 below 0.8|**members pile into near-perfect confidence; non-members sit just below it**. That gap is the attack surface|
|**PATE student** (green, y-axis only −0.1 to 0.3)|random ± bars around 0, a top bin of **+0.32**, and +0.14 near 0.975|about **20× smaller** than the baseline's spike. It's mostly noise, plus a tiny tilt from MNIST-train vs MNIST-test differences (and the scaler being fitted on private data)|

**A caution on reading it:** the attack's best threshold is probably _inside_ the top bin (e.g. 0.9999), which a 0.01-wide bar can't show. And `density=True` hides every confidence below 0.5.

### Figure 2: Attack advantage and accuracy
![PATE Privacy Evaluation](/assets/19%20PATE%20Privacy%20Evaluation-20261023191014979.png)

|Model|Advantage (left)|Accuracy (right)|vs 5% line / noise floor|
|---|---|---|---|
|Baseline|**≈ 0.0645**|≈ 0.564|above 5%, real leak|
|PATE (Private)|**≈ 0.002**|≈ 0.502|below the noise floor (negative control)|
|PATE (Query)|**≈ 0.075**|≈ 0.575|**the highest of the three**: the student's own training data leaks|

Protection improvement: (0.0645 − 0.002) / 0.0645 ≈ **96.9%**.

---

## Copy-paste cells (mine, not the course's)

I didn't run these. They use your names from Sections 14-19.

### Null control: the noise floor for this attack

```python
perm = np.random.default_rng(0).permutation(len(X_holdout_norm))
a, b = perm[:5000], perm[5000:]
null_acc, null_adv, _ = compute_mia_advantage(student_model, X_holdout_norm[a], y_holdout[a],
                                              X_holdout_norm[b], y_holdout[b], DEVICE)
print(f"null test (random holdout half vs half): advantage {null_adv:.4f}")
print(f"PATE private {student_mia_adv_private:.4f} | baseline {baseline_mia_adv:.4f} | query {student_mia_adv_query:.4f}")
```

Expected: null ≈ 0.004-0.009. The private result sits at or below it; the baseline and query results are far above it.

### Where the private data really leaks: one teacher

```python
t0_members = X_private_norm[teacher_partitions[0]]
t0_acc, t0_adv, _ = compute_mia_advantage(teachers[0], t0_members, y_private[teacher_partitions[0]],
                                          X_holdout_norm[:len(t0_members)], y_holdout[:len(t0_members)], DEVICE)
print(f"single teacher MIA on its own 192 private samples: acc {t0_acc:.3f}, advantage {t0_adv:.3f}")
```

Expected: a **much larger** advantage than the baseline's 0.064 (it's 192 samples trained for 30 epochs). The noise floor at n = 192 is about 0.03, so read it against that. **This is why the teachers must never be deployed or queryable.**

### Proof the "private" test can't see ε

Needs `clean_student` from my Section 16 cell.

```python
if 'clean_student' in globals():
    cs_acc, cs_adv, _ = compute_mia_advantage(clean_student, X_private_norm, y_private, X_holdout_norm, y_holdout, DEVICE)
    print(f"student trained on CLEAN votes (eps = infinity) - private-data MIA advantage: {cs_adv:.4f}")
else:
    print("run the Section 16 clean-label student cell first")
```

Expected: also ≈ 0.00x, the same as the noisy PATE student, so this test can't tell ε = 86.5 from ε = ∞.

To view the figures: `display(Image("figs/pate_mia_comparison.png"))` and `display(Image("figs/pate_confidence_distributions.png"))`.

---

## Quick self-check

Using the figure's numbers (advantage: baseline 0.0645, PATE private 0.002, PATE query 0.075):  
(a) What does the summary cell print as "Protection improvement"?  
(b) Suppose the teachers' votes had been released with **no noise at all** (ε = ∞). What would the PATE (Private) attack score, and why? What does that tell you about what this test measures?  
(c) Which model leaks most about its **own** training data, and what does that mean if the "public" data were actually sensitive?

---

## The facts worth locking in for the exam

- **A deliberately overfit baseline trained directly on 10,000 private MNIST images ([256, 128], no dropout, 50 epochs) leaks membership with a confidence-threshold advantage of about 0.064 (accuracy ~56.4%)**, well above the ~0.004 noise floor of a balanced 10k/10k best-threshold attack.
- **The PATE student scores about 0.002 advantage (50.2%) when attacked with private images as "members" and the holdout as non-members**, which is indistinguishable from random guessing: about a 97% reduction relative to the baseline.
- **That private-data test is a negative control, because the student trained on neither group**. Any student, even one trained on clean, un-noised votes (ε = ∞), would pass it. It demonstrates the architectural separation, not the strength of the noise or the value of ε.
- **The student leaks its own public training queries more than the baseline leaks its data (advantage ~0.075 vs ~0.064)**, because it trains on only 5,000 samples with noisy labels. PATE gives the public data no protection, so that data must really be non-sensitive.
- **PATE's formal guarantee comes from the Laplace noise on teacher votes, not from the information bottleneck**. And the most exposed component is the teacher ensemble (each teacher overfits 192 private images), which must never be deployed or queryable.
- **The course's confidence MIA balances by taking the first n of each group, searches 1,000 percentile thresholds, ignores the true labels, and returns (accuracy, accuracy − 0.5, threshold)**. The 5% advantage line is this course's validator convention, not an industry standard.
- **A confidence density-difference plot shows the attack surface: the baseline's members spike in the 0.99-1.0 bin (+6.05) while non-members sit just below it**. The student's differences stay within about ±0.1, but such plots hide confidences below 0.5 and any separation inside the top bin.
- **Method point:** **when empirically evaluating a privacy defense, run a positive control (data the model did train on), a negative control (two groups it never saw), and a null baseline for the noise floor, and attack the components that touched the private data (here the teachers)**. Report the result as a lower bound alongside the formal ε and δ.

---

Logged to COAE-progress.md, with the self-check answer key. Next is likely the PATE challenge, which we'll do hands-on.