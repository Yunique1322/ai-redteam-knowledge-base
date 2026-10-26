# Executing And Evaluating The Attack

## The idea in one breath

You finally point the trained attack model at the real target. It gets **69.15% accuracy, 97.58% recall, AUC 0.5675**. The course sells this as a "19% advantage over random". It isn't. The evaluation pool is **2:1 members to non-members**, so just saying "member" to everyone already scores **66.67%**. The attack's real edge over that is **+2.5 points**, and the honest, balance-proof number is **AUC 0.5675**. This section is the single most important one for your report writing: it's where "accuracy" lies and "AUC" tells the truth.

> **Heads-up:** No mangled formulas. The key traps, all verified by recomputing the course's own marginals:
> 
> - **The 69% is base-rate inflation.** "Always member" = 24,421 / 36,632 = **0.6667**. Honest advantage = 0.6915 − 0.6667 = **+0.0248**, not +0.1915.
> - **Specificity is ~0.123.** The attack correctly identifies only **12%** of non-members. It catches members (recall 0.976) by labeling almost everyone a member.
> - **Balanced accuracy ≈ 0.549**, which matches AUC 0.5675. That's the attack's true discriminative power.
> - **Non-members = 12,211, not 12,210.** The prose says 12,210 but the total 36,632 requires 12,211.
> - **The target numbers here (93.60 / 82.47 / gap 11.13) finally match Section 3's figure**, not the 90/83/7 prose. Trust these.

---

## Part 1: Executing the Attack (collect target predictions)

```python
print("\n" + "=" * 60)
print("Executing Membership Inference Attack")
print("=" * 60)

preds_members = get_model_predictions(target_model, X_target_norm, DEVICE)
preds_non_members = get_model_predictions(target_model, X_attack_eval_norm, DEVICE)

print(f"\nTarget model predictions collected:")
print(f" Members: {len(preds_members)} samples")
print(f" Non-members: {len(preds_non_members)} samples")
```

|Code|What it does|
|---|---|
|`get_model_predictions(target_model, X_target_norm, DEVICE)`|Target's softmax on its own **training** rows → 24,421 members|
|`get_model_predictions(target_model, X_attack_eval_norm, DEVICE)`|Target's softmax on data it never saw → 12,211 non-members|

This is the **genuine black-box scenario**: query the target, read its probability vectors, decide membership. Unlike the shadow sections, the labels here are real (we know who was in the target's training set because we built it).

---

## Part 2: Preparing Attack Input (the empty-array trick)

```python
attack_X_members, attack_y_members = prepare_attack_data(
    preds_members, np.zeros((0, preds_members.shape[1])),
    y_target, np.array([], dtype=np.int64)
)

attack_X_non_members, attack_y_non_members = prepare_attack_data(
    np.zeros((0, preds_non_members.shape[1])), preds_non_members,
    np.array([], dtype=np.int64), y_attack_eval
)

print(f"\nAttack input prepared:")
print(f" Member features: {attack_X_members.shape}")
print(f" Non-member features: {attack_X_non_members.shape}")
```

### Why this looks weird

Recall from Section 4: `prepare_attack_data(preds_in, preds_out, labels_in, labels_out)` labels the **first** argument as member (1) and the **second** as non-member (0), **by position**. If you concatenated members and non-members first, the function couldn't tell which was which. So you call it twice, each time passing an **empty placeholder** for the role you're not filling.

|Code|What it does|
|---|---|
|`np.zeros((0, preds_members.shape[1]))`|An array with **0 rows** but **2 columns**. It satisfies the function's shape needs without adding any samples|
|`np.array([], dtype=np.int64)`|The matching empty label array|
|first call|All 24,421 target rows → labeled member (1)|
|second call|All 12,211 eval rows → labeled non-member (0)|

Output:

```txt
Attack input prepared:
  Member features: (24421, 4)
  Non-member features: (12211, 4)
```

> **If this errors:** `ValueError` about concatenation dims inside `prepare_attack_data` means the empty array's **column count** didn't match (e.g. you wrote `np.zeros((0,))` instead of `np.zeros((0, 2))`). The function does `np.eye(2)[empty_labels]` and concatenates, so it needs the 2 columns even with 0 rows.

---

## Part 3: Combining Attack Data

```python
attack_X_eval = np.concatenate([attack_X_members, attack_X_non_members], axis=0)
attack_y_eval = np.concatenate([attack_y_members, attack_y_non_members], axis=0)

print(f"\nTotal attack evaluation samples: {len(attack_X_eval)}")
print(f" Members: {np.sum(attack_y_eval == 1)}")
print(f" Non-members: {np.sum(attack_y_eval == 0)}")
```

Output:

```txt
Total attack evaluation samples: 36632
  Members: 24421
  Non-members: 12211
```

|Code|What it does|
|---|---|
|`np.concatenate([...], axis=0)`|Stacks members on top of non-members → (36632, 4)|

**Note the 2:1 imbalance: 24,421 members vs 12,211 non-members.** The course flags it here and then half-forgets it when interpreting the numbers. 36,632 = 24,421 + 12,211, which confirms non-members is **12,211**, not the 12,210 the prose says.

---

## Part 4: Running the Attack

```python
attack_eval_loader = create_dataloader(attack_X_eval, attack_y_eval, ATTACK_MODEL_CONFIG['batch_size'], shuffle=False)

_, attack_predictions, attack_probs = evaluate_model(attack_model, attack_eval_loader, DEVICE)

membership_probs = attack_probs[:, 1]

print(f"\nAttack predictions generated")
print(f" Mean membership probability: {membership_probs.mean():.4f}")
```

|Code|What it does|
|---|---|
|`create_dataloader(..., shuffle=False)`|**Must** be unshuffled so predictions line up with `attack_y_eval`|
|`evaluate_model(attack_model, ...)`|Returns `(_, predictions, probs)`; the `_` is the accuracy we recompute below|
|`attack_probs[:, 1]`|Column 1 = **P(member)**, from 0.0 (non-member) to 1.0 (member)|
|`membership_probs`|These continuous scores feed ROC/PR (not the hard 0/1 predictions)|

The hard `attack_predictions` use the default 0.5 cutoff. The continuous `membership_probs` are what you sweep thresholds over.

---

## Part 5: Computing Attack Metrics

```python
attack_accuracy = accuracy_score(attack_y_eval, attack_predictions)
attack_precision = precision_score(attack_y_eval, attack_predictions)
attack_recall = recall_score(attack_y_eval, attack_predictions)
attack_f1 = f1_score(attack_y_eval, attack_predictions)

print(f"\nMembership Inference Attack Results:")
print(f" Attack Accuracy: {attack_accuracy:.4f}")
print(f" Attack Precision: {attack_precision:.4f}")
print(f" Attack Recall: {attack_recall:.4f}")
print(f" Attack F1 Score: {attack_f1:.4f}")
```

Output:

```txt
  Attack Accuracy: 0.6915
  Attack Precision: 0.6899
  Attack Recall: 0.9758
  Attack F1 Score: 0.8083
```

### Step 1: Decode the four numbers honestly

From Section 5 you know the formulas. Let me reconstruct the full confusion matrix from these numbers (I verified the arithmetic):

|Quantity|Value|How|
|---|---|---|
|Members (M)|24,421|positives|
|Non-members (N)|12,211|negatives|
|TP (members caught)|≈ 23,830|recall 0.9758 × 24,421|
|FN (members missed)|≈ 591|24,421 − TP|
|FP (non-members wrongly flagged)|≈ 10,711|TP/precision − TP|
|TN (non-members cleared)|≈ 1,500|12,211 − FP|

### Step 2: The number the course never prints: specificity

```
specificity = TN / (TN + FP) = 1,500 / 12,211 ≈ 0.123
```

**The attack correctly identifies only 12% of non-members.** It gets its high recall (97.6%) by calling almost everyone a member. Look at FP = 10,711: of 12,211 real non-members, it wrongly flags 10,711 as members.

|Metric|Value|Plain English|
|---|---|---|
|Recall (members caught)|0.976|Finds nearly all members|
|**Specificity (non-members caught)**|**0.123**|Misses nearly all non-members|
|Precision|0.690|Of flagged "members", 69% are real... but 66.7% would be by the base rate alone|

---

## Part 6: Attack Advantage (the trap, spelled out)

The course says:

> advantage = accuracy − 0.5 = 0.6915 − 0.5 = **0.1915**

### Why 0.5 is the wrong baseline here

0.5 is the coin-flip baseline for a **balanced** pool. This pool is 2:1. On a 2:1 pool, the trivial strategy "always say member" scores:

```
always-member accuracy = M / (M + N) = 24,421 / 36,632 = 0.6667
```

So the **honest** advantage is:

```
honest advantage = 0.6915 − 0.6667 = +0.0248 (2.5 points)
```

|Baseline used|Baseline value|"Advantage"|
|---|---|---|
|Random coin flip (course)|0.500|+0.1915 (looks huge)|
|**Always-member (honest, for 2:1)**|**0.6667**|**+0.0248 (modest)**|

|Symbol|Plain English|This pool|
|---|---|---|
|M / (M+N)|Fraction that are members = "always member" accuracy|0.6667|
|accuracy − that|Real skill above the trivial strategy|+0.0248|

**Read aloud:** "On a 2-to-1 pool, I start at 66.7% for free, so 69% is only 2.5 points of actual skill."

The course's own thresholds for judgment ("advantage > 0.15 = high vulnerability") are built on the 0.5 baseline, so applied here they overstate the risk. **Always compare to the base rate, not to 0.5, on an imbalanced pool.**

---

## Part 7: Storing Results

```python
results = {
    'attack_accuracy': attack_accuracy,
    'attack_precision': attack_precision,
    'attack_recall': attack_recall,
    'attack_f1': attack_f1,
    'attack_y_true': attack_y_eval,
    'attack_y_pred': attack_predictions,
    'attack_probs': membership_probs,
    'confidence_members': np.max(preds_members, axis=1),
    'confidence_non_members': np.max(preds_non_members, axis=1),
}

print("\nResults stored for visualization")
```

|Code|What it does|
|---|---|
|scalar keys|The four metrics|
|`attack_y_true`, `attack_y_pred`, `attack_probs`|Ground truth, hard predictions, and P(member) scores for the plots|
|`confidence_members` / `confidence_non_members`|`max` probability per sample, i.e. the raw confidence signal, for the histogram|

---

## Part 8: ROC Curve

```python
print("\n" + "=" * 60)
print("Generating Visualizations")
print("=" * 60)

auc_score = plot_attack_roc_curve(
    results['attack_y_true'],
    results['attack_probs'],
    save_path=os.path.join(FIGS_DIR, f"{FIG_PREFIX}attack_roc.png")
)
results['attack_auc'] = auc_score

print(f"Attack AUC: {auc_score:.4f}")
```

|Code|What it does|
|---|---|
|`plot_attack_roc_curve(y_true, probs, ...)`|Sweeps every threshold, plots TPR vs FPR, returns the AUC|
|`results['attack_auc'] = auc_score`|Stores it (0.5675)|

**AUC is the one metric immune to class balance**, because it's built from rates within each class (TPR uses only members, FPR uses only non-members). That's why it's the honest summary: 0.5675 means a random member scores higher than a random non-member about 57% of the time.

---

## Part 9: PR Curve

```python
plot_precision_recall_curve(
    results['attack_y_true'],
    results['attack_probs'],
    save_path=os.path.join(FIGS_DIR, f"{FIG_PREFIX}attack_pr.png")
)
```

|Code|What it does|
|---|---|
|`plot_precision_recall_curve(...)`|Precision vs recall across thresholds; baseline = fraction of positives = **0.667**|

**The PR baseline being 0.667 is itself the proof of the 2:1 imbalance.** Average precision 0.710 above a 0.667 baseline is a real but small edge, consistent with AUC 0.568.

---

## Part 10: Confidence Distribution

```python
plot_confidence_distributions(
    results['confidence_members'],
    results['confidence_non_members'],
    save_path=os.path.join(FIGS_DIR, f"{FIG_PREFIX}confidence_distributions.png")
)

print(f"\nMean confidence - Members: {np.mean(results['confidence_members']):.4f}")
print(f"Mean confidence - Non-Members: {np.mean(results['confidence_non_members']):.4f}")
```

Output:

```txt
Mean confidence - Members: 0.9301
Mean confidence - Non-Members: 0.9226
```

|Code|What it does|
|---|---|
|`plot_confidence_distributions(...)`|Overlaid histograms of max confidence, members vs non-members|

The gap is ==0.9301 − 0.9226 = 0.0075==, a 0.75% difference. Here's the key paradox: the target has a big **accuracy** gap (11 points) but a tiny **confidence** gap (0.75%). **Both members and non-members saturate near 1.0**, so there's little room for confidence to separate them. That's precisely why the AUC is only 0.568: the signal the attack reads (confidence) is almost identical for the two groups even though the model clearly overfits.

---

## Part 11: Attack Metrics Bar

```python
plot_attack_accuracy_comparison(
    results,
    save_path=os.path.join(FIGS_DIR, f"{FIG_PREFIX}attack_metrics.png")
)
```

|Code|What it does|
|---|---|
|`plot_attack_accuracy_comparison(results, ...)`|Bar chart of accuracy / AUC / precision / recall vs the 0.5 line|

The course's own gloss finally admits the issue: "AUC at 0.5675 appears lower because it measures discrimination across all thresholds, while the other metrics use a fixed 0.5 threshold that happens to work well for this class-imbalanced dataset."

> **An honest note the course skips (1):** That framing is backwards. The 0.5 threshold doesn't "happen to work well". It **rides the base rate**. Because the pool is 2:1 and the attack predicts "member" for almost everyone (recall 0.976), it automatically scoops up the 2/3 of the pool that really are members. **AUC and balanced accuracy strip that freebie away, and they're both ~0.55-0.57. That's the real leakage.**

---

## Part 12: Saving Results

```python
output = {
    'target_model': {
        'train_accuracy': float(train_acc),
        'test_accuracy': float(test_acc),
        'overfitting_gap': float(train_acc - test_acc),
    },
    'attack_results': {
        'accuracy': float(results['attack_accuracy']),
        'precision': float(results['attack_precision']),
        'recall': float(results['attack_recall']),
        'f1_score': float(results['attack_f1']),
        'auc': float(results['attack_auc']),
        'advantage': float(results['attack_accuracy'] - 0.5),
    },
    'configuration': {
        'random_seed': RANDOM_SEED,
        'num_shadow_models': SHADOW_MODEL_CONFIG['num_shadow_models'],
        'target_architecture': TARGET_MODEL_CONFIG['hidden_layers'],
        'attack_architecture': ATTACK_MODEL_CONFIG['hidden_layers'],
    }
}
```

```python
results_path = os.path.join(FIGS_DIR, f"{FIG_PREFIX}attack_results.json")
with open(results_path, 'w') as f:
    json.dump(output, f, indent=2)

print(f"\nResults saved to {results_path}")
```

|Code|What it does|
|---|---|
|`float(...)`|NumPy floats aren't JSON-serializable; this converts them|
|`'advantage': ... - 0.5`|The library hard-codes the **0.5** baseline. On this 2:1 pool, treat it as a label, not the real advantage|
|`json.dump(output, f, indent=2)`|Writes `figs/Introduction_attack_results.json`|

Note the stored `advantage` uses 0.5, so the JSON will say 0.1915. When you cite it, correct it to the base-rate version (+0.025) or cite AUC instead.

---

## Reading the figure/output

### Figure 1: ROC Curve (AUC = 0.568)
![Executing and Evaluating the Attack](/assets/6%20Executing%20and%20Evaluating%20the%20Attack-20261023005025306.png)

|What to read|Value|Meaning|
|---|---|---|
|Curve vs diagonal|Above it throughout|Better than random at every threshold|
|AUC|0.568|A random member outranks a random non-member ~57% of the time|
|Biggest bulge|around FPR 0.4-0.6|Where the attack separates best|
|Shape|gently bowed, close to the line|Weak but real signal; this is the **honest** headline|

### Figure 2: PR Curve (Avg Precision = 0.710, baseline 0.667)
![Executing and Evaluating the Attack](/assets/6%20Executing%20and%20Evaluating%20the%20Attack-20261023005029807.png)

|What to read|Value|Meaning|
|---|---|---|
|Baseline dashed line|0.667|= member fraction = confirms the 2:1 pool|
|Precision across recall|hovers ~0.70|Only slightly above the always-member 0.667|
|Early spike then dip to ~0.67|at very low recall|Noise from the most-confident handful of samples|

### Figure 3: Confidence Distribution on the Target
![Executing and Evaluating the Attack](/assets/6%20Executing%20and%20Evaluating%20the%20Attack-20261023005036959.png)

|What to read|Value|Meaning|
|---|---|---|
|Both histograms|pile up in the last bin (density ~61)|~60% of all samples sit at 99%+ confidence|
|Member mean|0.930|teal dashed line, slightly right|
|Non-member mean|0.923|red dashed line, slightly left|
|Gap between lines|0.0075|tiny; the lines nearly touch|
|Overlap|almost total|why the attack can't cleanly separate, only exploit a statistical lean|

Contrast this with Section 4's shadow histogram (means 0.857/0.857, gap 0.000). The target does have a gap, but it's a razor-thin 0.0075. **A huge accuracy gap can coexist with a tiny confidence gap.**

### Figure 4: Metrics Bar
![Executing and Evaluating the Attack](/assets/6%20Executing%20and%20Evaluating%20the%20Attack-20261023005041183.png)

|Bar|Value|Reading|
|---|---|---|
|Accuracy|0.691|Inflated by the 2:1 base rate (0.667 for free)|
|**AUC**|**0.568**|The honest number|
|Precision|0.690|≈ the 0.667 base rate|
|**Recall**|**0.976**|Towers over everything: the attack calls almost everyone a member|

The visual story: three bars near 0.69 and one (recall) near 1.0. That recall spike **is** the base-rate-riding behavior made visible.

---

## Copy-paste cells (mine, not the course's)

These use `attack_y_eval`, `attack_predictions`, `membership_probs`, `results`, `RANDOM_SEED`, `FIGS_DIR`, `FIG_PREFIX`. Run the course's cells first. I tested the logic on a mock with these exact marginals.

### Cell 1: The honest decomposition (what the report should actually say)

```python
import numpy as np
from sklearn.metrics import accuracy_score, recall_score, roc_auc_score, confusion_matrix

tn, fp, fn, tp = confusion_matrix(attack_y_eval, attack_predictions).ravel()
spec = tn / (tn + fp)
always_member = (attack_y_eval == 1).mean()
acc = accuracy_score(attack_y_eval, attack_predictions)
rec = recall_score(attack_y_eval, attack_predictions)

print(f"Reported accuracy : {acc:.4f}")
print(f"'Always member' baseline (NOT 0.5): {always_member:.4f}")
print(f"Honest advantage over that baseline: {acc - always_member:+.4f}")
print(f"Recall (members caught) : {rec:.3f}")
print(f"Specificity (non-members caught) : {spec:.3f} <- attack misses most non-members")
print(f"Balanced accuracy (recall+spec)/2 : {(rec + spec) / 2:.4f}")
print(f"AUC (threshold-free, HONEST) : {roc_auc_score(attack_y_eval, membership_probs):.4f}")
```

Expected (matching the course's numbers): always-member baseline 0.6667, honest advantage ≈ **+0.025**, specificity ≈ **0.12**, balanced accuracy ≈ **0.549**, AUC ≈ **0.568**.

### Cell 2: Re-score on a BALANCED pool (watch accuracy collapse toward the AUC)

This subsamples the majority (members) down to the non-member count, so 0.500 is the real baseline again. AUC should barely move; accuracy should drop from ~0.69 to ~0.55.

```python
rng = np.random.default_rng(RANDOM_SEED)
idx_m = np.where(attack_y_eval == 1)[0]
idx_n = np.where(attack_y_eval == 0)[0]
k = min(len(idx_m), len(idx_n))
keep = np.r_[rng.choice(idx_m, k, replace=False), idx_n] # members cut down to k; all non-members kept

yb = attack_y_eval[keep]
pb = membership_probs[keep]
predb = (pb > 0.5).astype(int)

print(f"Balanced pool: {int((yb == 1).sum())} members + {int((yb == 0).sum())} non-members (baseline 0.500)")
print(f"Balanced accuracy @0.5 threshold : {accuracy_score(yb, predb):.4f} <- the honest 'accuracy'")
print(f"AUC (unchanged by balancing) : {roc_auc_score(yb, pb):.4f}")
```

Expected: `12211 members + 12211 non-members`, accuracy around **0.55**, AUC around **0.568** (same as the imbalanced pool, proving AUC doesn't care about balance). This one cell is your proof that the 69% was mostly the base rate.

### Cell 3: See the four saved figures

```python
from IPython.display import Image, display
for name in ["attack_roc", "attack_pr", "confidence_distributions", "attack_metrics"]:
    path = os.path.join(FIGS_DIR, f"{FIG_PREFIX}{name}.png")
    display(Image(filename=path)) if os.path.exists(path) else print("Missing:", path)
```

Expected: the ROC, PR, confidence-distribution, and metrics-bar figures inline.

---

## Quick self-check

An attacker audits a model on a pool of **900 members and 100 non-members**, and their attack predicts "member" for **all 1,000**.

(a) What's the accuracy?  
(b) What are recall, precision, specificity, and AUC?  
(c) In one sentence: does the high accuracy mean the attack works, and what's the honest number?

---

## The facts worth locking in for the exam

- **The attack is evaluated on the target's real members (24,421) vs genuine non-members (12,211), a 2:1 imbalanced pool of 36,632**. The predictions must come from a `shuffle=False` loader so they stay aligned with the labels.
- **`prepare_attack_data` labels by position, so members and non-members are prepared in separate calls with an empty `np.zeros((0, 2))` placeholder for the missing side**, then concatenated. The empty array must carry the right column count.
- **The headline 69.15% accuracy is base-rate inflation: on a 2:1 pool, "always member" already scores 66.67%, so the honest advantage is only +2.5 points, not the +19 the course quotes against a 0.5 baseline**.
- **Recall 0.976 with specificity ~0.123 means the attack catches nearly all members by flagging almost everyone, while correctly clearing only ~12% of non-members**. That's why precision (0.690) sits right on the base rate.
- **AUC 0.5675 is the honest metric because it's immune to class balance**, built from TPR (members only) and FPR (non-members only). Balanced accuracy ≈ 0.549 agrees with it. On a re-balanced pool the accuracy falls from ~0.69 to ~0.55 while AUC is unchanged.
- **The target has an 11-point accuracy gap but only a 0.0075 confidence gap (members 0.9301 vs non-members 0.9226), because both saturate near 1.0**. A large accuracy gap does not guarantee a large confidence signal, which is why this heavily-overfit model still only yields AUC 0.57.
- **The PR baseline of 0.667 equals the member fraction and is itself the signature of the 2:1 imbalance**; average precision 0.710 just above it is a real but modest edge.
- **Method point:** **In a report, never quote raw accuracy for a membership attack on an imbalanced pool. Quote AUC (and balanced accuracy, or TPR at a fixed low FPR), and state the base rate explicitly**. Here the defensible claim is "AUC 0.57, a modest but real membership leak driven by the overfitting gap", not "69% accurate". The fix for the finding is the Section 3 remediation list: dropout, early stopping, more data, and differential privacy (DP-SGD / PATE), which the next sections cover.

---

I logged this section in COAE-progress.md: the full honest decomposition (specificity 0.12, balanced accuracy 0.549, base-rate advantage +2.5 points), the AUC-as-truth lesson, all four figure readings, and the self-check answer key.

That completes the Shadow Model attack (Sections 2-6). You now have the full pipeline: shadow models → attack model → execute on target → **read the metrics honestly**. Next up is either the Shadow Model skills assessment or the start of DP-SGD (the first defense). Paste whatever's next when you're ready.