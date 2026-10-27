# Section 5: Building the Attack Classifier

## The idea in one breath

You train the **attack model** (a tiny 4-input network) on the 61,050 shadow rows to tell members from non-members. On shadow data it scores a dismal **50-51%**, basically a coin flip, because the shadows barely overfit. That is **expected, not failure**: the same attack model, pointed at the heavily-overfit target next section, is predicted to jump to **65-66%**. This section builds, trains, saves, and dissects that attack model, and introduces **precision / recall / F1** and **threshold tuning**.

> **Heads-up:** No mangled formulas. Four things to flag up front, all verified against the library source and the split math:
> 
> - **Parameter count:** the course says "only 2,600 parameters", then lists 320 + 2,080 + 66, which sums to **2,466**. The breakdown is right; the total is wrong.
> - **The attack test set is balanced 6,105 / 6,105 (50/50).** But the threshold-tuning prose talks about "2:1 member to non-member ratio in our evaluation set". That 2:1 is the **target** evaluation (24,421 members vs 12,211 non-members) in a later section, not this attack test set. The course mixes the two.
> - **The decision-boundary thresholds** the generic prose quotes ("around 0.80-0.85") do **not** match the actual figure. I'll read the real figure below.
> - **The target numbers** here (90% / 83%, gap 7%) disagree with Section 3's figure (93.6% / 82.5%, gap 11.1). Same recurring inconsistency; trust your own run.

---

## Part 1: Why this is a transfer problem

The attack model faces a **generalization challenge**: it learns from shadow models but must work on the **target**, which has different weights and different overfitting. If it **overfits to shadow-specific artifacts** (quirks of individual shadow runs), it fails on the target. Three defenses, which map to the Section 4 config:

|Mechanism|How|Where it came from|
|---|---|---|
|**Diverse data**|61,050 rows from 5 different shadows|Section 4's "combine all shadows"|
|**Small architecture** ([64, 32])|Too few parameters to memorize shadow quirks|`ATTACK_MODEL_CONFIG`|
|**Held-out validation**|Early stopping watches a shadow validation set|The three-way split below|

This is the surrogate-transfer idea again: train on copies, deploy on the real thing, and keep the learner simple so it generalizes.

---

## Part 2: Splitting Attack Data (three splits, not two)

The course uses **three splits**: train (learn patterns), validation (guide early stopping), test (unbiased estimate). With only train/test you'd either skip early stopping (overfit to shadows) or burn training data on validation.

### Step 1: Hold out 20% as the test set

```python
print("\n" + "=" * 60)
print("Training Attack Model")
print("=" * 60)

X_attack_train, X_attack_test, y_attack_train, y_attack_test = train_test_split(
    attack_X, attack_y, test_size=0.2, random_state=RANDOM_SEED, stratify=attack_y
)

print(f"\nAttack data split:")
print(f" Training + Validation: {len(X_attack_train)} samples")
print(f" Test: {len(X_attack_test)} samples")
```

Output (I verified the arithmetic):
![Building the Attack Classifier](/assets/5%20Building%20the%20Attack%20Classifier-20261023155500485.png)

|Code|What it does|
|---|---|
|`train_test_split(attack_X, attack_y, test_size=0.2, ...)`|80% (48,840) for train+val, 20% (12,210) held out as test|
|`stratify=attack_y`|Keeps the 50/50 member split in **both** parts|
|`random_state=RANDOM_SEED`|Reproducible split|

The test set is **6,105 members + 6,105 non-members**. Remember that for the threshold discussion later.

### Step 2: Carve a validation set out of the training part

```python
X_attack_tr, X_attack_val, y_attack_tr, y_attack_val = train_test_split(
    X_attack_train, y_attack_train, test_size=0.2, random_state=RANDOM_SEED, stratify=y_attack_train
)

print(f" Training: {len(X_attack_tr)} samples")
print(f" Validation: {len(X_attack_val)} samples")
```

Output:

```txt
  Training: 39072 samples
  Validation: 9768 samples
```

|Code|What it does|
|---|---|
|`train_test_split(X_attack_train, ..., test_size=0.2, ...)`|Splits the 48,840 into 39,072 train and 9,768 validation|

So the full three-way split is **39,072 train / 9,768 validation / 12,210 test**, exactly as the course prose says.

### Step 3: DataLoaders

```python
attack_train_loader = create_dataloader(X_attack_tr, y_attack_tr, ATTACK_MODEL_CONFIG['batch_size'])
attack_val_loader = create_dataloader(X_attack_val, y_attack_val, ATTACK_MODEL_CONFIG['batch_size'], shuffle=False)
attack_test_loader = create_dataloader(X_attack_test, y_attack_test, ATTACK_MODEL_CONFIG['batch_size'], shuffle=False)

print(f"\nDataLoaders created with batch size {ATTACK_MODEL_CONFIG['batch_size']}")
```

|Code|What it does|
|---|---|
|`attack_train_loader`|Batches of 128, **shuffled** (training)|
|`attack_val_loader`, `attack_test_loader`|Batches of 128, **not shuffled** (consistent evaluation)|

Batch size 128 (vs 64 elsewhere) because the input is only 4 numbers and the net is tiny, so bigger batches give smoother gradients at no memory cost.

|Name|Shape|Meaning|
|---|---|---|
|`X_attack_train` / `X_attack_test`|(48840, 4) / (12210, 4)|Train+val pool / untouched test|
|`X_attack_tr` / `X_attack_val`|(39072, 4) / (9768, 4)|Actual training / early-stopping validation|

---

## Part 3: Attack Model Architecture

```python
attack_input_size = attack_X.shape[1]
attack_model = AttackModel(
    input_size=attack_input_size,
    hidden_layers=ATTACK_MODEL_CONFIG['hidden_layers'],
    dropout=ATTACK_MODEL_CONFIG['dropout']
)

print(f"\nAttack model architecture: {attack_input_size} -> {ATTACK_MODEL_CONFIG['hidden_layers']} -> 2")
print(f"Dropout: {ATTACK_MODEL_CONFIG['dropout']}")
```

Output:

```txt
Attack model architecture: 4 -> [64, 32] -> 2
Dropout: 0.2
```

|Code|What it does|
|---|---|
|`attack_input_size = attack_X.shape[1]`|4 (the `[p0, p1, label0, label1]` width)|
|`AttackModel(input_size=4, hidden_layers=[64,32], dropout=0.2)`|4 -> 64 -> 32 -> 2 logits `[non_member, member]`, ReLU + dropout between layers|

|Layer|Shape|Weights + biases|
|---|---|---|
|Linear 1|4 -> 64|4×64 + 64 = 320|
|Linear 2|64 -> 32|64×32 + 32 = 2,080|
|Output|32 -> 2|32×2 + 2 = 66|
|**Total**||**2,466**|

The course calls this "about 2,600". The real total is **2,466** (its own three numbers add to 2,466, not 2,600). Compare to the target's ~37,000. **Small on purpose**: a bigger net would memorize shadow noise that doesn't transfer. Dropout is 0.2, lower than the shadows' 0.3, because this task is simpler and you want to keep what little signal exists.

> **An honest note the course skips (1):** The course says "the primary signal is simply confidence level". But your Section 4 signal report showed the confidence gap in the shadow data is ~0.000. So the attack model is being asked to learn a confidence rule from data where confidence barely separates the classes. That's exactly why the next figure looks like a coin flip.

---

## Part 4: Training the Attack Model

```python
print("\nTraining attack model...")

history_attack = train_with_early_stopping(
    attack_model, attack_train_loader, attack_val_loader,
    device=DEVICE,
    epochs=ATTACK_MODEL_CONFIG['epochs'],
    learning_rate=ATTACK_MODEL_CONFIG['learning_rate'],
    patience=ATTACK_MODEL_CONFIG['early_stopping_patience']
)

plot_training_history(
    history_attack,
    "Attack Model Training",
    save_path=os.path.join(FIGS_DIR, f"{FIG_PREFIX}attack_training.png")
)
```

|Code|What it does|
|---|---|
|`train_with_early_stopping(..., patience=15)`|Trains up to 100 epochs, stops after 15 with no val-loss improvement. Patience 15 > shadows' 10 so the subtle pattern has time to appear|
|`history_attack`|The 4 metric lists, used by the plot|
|`plot_training_history(...)`|Saves `figs/Introduction_attack_training.png` (doesn't show inline)|

The course warns: loss drops only from **0.694 to 0.693**, accuracy hovers **50-51%**, and **validation accuracy on shadow data does not predict attack performance on the target**. That last sentence is the whole point of the section. Note the Section 3 early-stopping bug still applies: the restored weights are from the stop epoch, not the best one. Here it barely matters because every epoch is near-identical.

---

## Part 5: Evaluating on Shadow Data + precision/recall/F1

```python
attack_test_acc, attack_test_predictions, attack_test_probs = evaluate_model(attack_model, attack_test_loader, DEVICE)

print(f"\nAttack Model Test Performance:")
print(f" Accuracy: {attack_test_acc:.4f}")
print(f" Samples: {len(attack_test_predictions)}")

print("\nDetailed Classification Report:")
print(classification_report(
    y_attack_test,
    attack_test_predictions,
    target_names=['Non-Member', 'Member'],
    digits=4
))

# Save the attack model
attack_model_path = os.path.join(MODEL_DIR, "attack_model.pt")
torch.save(attack_model.state_dict(), attack_model_path)
print(f"\nAttack model saved to {attack_model_path}")
```

|Code|What it does|
|---|---|
|`evaluate_model(attack_model, attack_test_loader, DEVICE)`|Accuracy (~0.50-0.51), the 0/1 predictions, and the probabilities on the 12,210 test rows|
|`classification_report(..., target_names=['Non-Member','Member'], digits=4)`|Per-class precision, recall, F1 (label 0 = Non-Member, 1 = Member)|
|`torch.save(attack_model.state_dict(), ...)`|Saves ~20 KB of weights. Just the `state_dict`, no scaler needed|

### Step 1: Toy for precision, recall, F1

Imagine the attack is run on **10 people: 5 real members, 5 non-members**. It flags 6 as "member". Of those 6, 4 are truly members and 2 are not. So it caught 4 of the 5 real members.

Confusion matrix:

|Outcome|Count|Meaning|
|---|---|---|
|True Positive (TP)|4|members correctly flagged|
|False Positive (FP)|2|non-members wrongly flagged|
|False Negative (FN)|1|members missed|
|True Negative (TN)|3|non-members correctly cleared|

### Step 2: The three formulas

```
precision = TP / (TP + FP)
recall = TP / (TP + FN)
F1 = 2 · precision · recall / (precision + recall)
```

|Symbol|Plain English|Toy value|
|---|---|---|
|precision|Of those I **called** member, how many really were?|4 / (4+2) = **0.667**|
|recall|Of all **real** members, how many did I catch?|4 / (4+1) = **0.800**|
|F1|Harmonic mean (punishes a big gap between the two)|2·0.667·0.800 / (0.667+0.800) = **0.727**|

**Read aloud:** "Precision = how trustworthy my 'member' flags are. Recall = how many members I don't miss. F1 = one number that's only high when both are."

### Step 3: Which to prioritize

|Context|Prioritize|Why|
|---|---|---|
|**Privacy audit**|Recall|Find every exposed member, even with some false alarms. Use a **lower** threshold (e.g. 0.3)|
|**Legal / accusation**|Precision|Never wrongly label someone a training member. Use a **higher** threshold (e.g. 0.7)|

**Higher recall with lower precision means the attack aggressively calls things "member".** On balanced shadow test data you should see all three metrics near 0.50.

---

## Part 6: Understanding What the Attack Learned

```python
boundary_analysis = analyze_attack_decision_boundary(attack_model, DEVICE)

print("\nDecision Boundary Analysis:")
for cls, data in boundary_analysis.items():
    threshold_idx = np.argmin(np.abs(data['membership_probs'] - 0.5))
    threshold_conf = data['confidences'][threshold_idx]
    print(f" Class {cls}: Membership threshold at confidence ~{threshold_conf:.3f}")
```

|Code|What it does|
|---|---|
|`analyze_attack_decision_boundary(attack_model, DEVICE)`|Sweeps confidence from 0.5 to 1.0, **on the true class**, and records the attack's membership probability for each class|
|`np.argmin(np.abs(data['membership_probs'] - 0.5))`|Finds the confidence where the membership probability is **closest to 0.5** (the crossover)|
|`threshold_conf`|That crossover confidence|

I read the function source. For each true class it builds a fake prediction: confidence `conf` on the true class, the rest on the other class, plus the one-hot label, then asks the attack model for P(member). This is a clean probe of "what confidence does this attack treat as the member/non-member line?"

The course's generic prose says "typical thresholds around 0.80-0.85". **Your actual figure doesn't show that** (read below). Trust the figure and the printed numbers from your run, not the boilerplate.

---

## Part 7: Threshold Selection and Attack Tuning

The attack model outputs P(member) in [0, 1]. Default decision: **member if P > 0.5**. But 0.5 isn't always best:

|Goal|Threshold|Effect|
|---|---|---|
|Catch all members (audit)|lower, e.g. 0.3|↑ recall, ↓ precision|
|Avoid false accusations (legal)|higher, e.g. 0.7|↑ precision, ↓ recall|

Two ways to pick:

1. **Validation sweep:** try thresholds 0.1->0.9 in 0.05 steps, compute F1 on the validation set, keep the best.
2. **Class balance:** the attack learns a boundary for the 50/50 shadow data via its **final-layer bias**. When the real evaluation pool is imbalanced, that boundary is off, which is why precision and recall diverge.

> **An honest note the course skips (2):** This section says "with 2:1 member to non-member ratio in our evaluation set" and "most samples are members". That is **not true of this attack test set**, which is balanced 6,105/6,105. The 2:1 ratio describes the **target** evaluation later (24,421 members vs 12,211 non-members). My Cell 3 below shows the trap: on a raw 2:1 pool, "always say member" already scores **66.7%**, so a reported 65-66% there would be _worse than guessing one label_. The meaningful 65-66% the course promises is on a **balanced** target pool, where 50% is the baseline. Always balance before quoting attack accuracy.

---

## Part 8: Visualizing the Decision Boundary

```python
plot_decision_boundary(
    boundary_analysis,
    save_path=os.path.join(FIGS_DIR, f"{FIG_PREFIX}decision_boundary.png")
)
```

|Code|What it does|
|---|---|
|`plot_decision_boundary(boundary_analysis, ...)`|Plots P(member) vs confidence, one line per class, with a dashed 0.5 line and a dotted 0.8 marker. Saves the PNG|

---

## Part 9: Performance Expectations

|Pool|Overfitting|Expected attack accuracy|
|---|---|---|
|**Shadow test** (this section)|~1-2% gap (dropout + early stopping)|50-51%, looks like nothing|
|**Target** (next section)|7% gap per this text (11.1% per Section 3 figure)|**65-66%, a 15-16% advantage**|

The course's rough reading guide for the final target attack:

|Target attack accuracy|Verdict|
|---|---|
|**> 60%**|meaningful membership leakage|
|55-60%|moderate vulnerability|
|< 55%|largely ineffective|

**Shadow evaluation deliberately underestimates target performance**, because the shadows overfit less than the vulnerable target. This is the one place where a bad-looking metric is good news.

---

## Reading the figure/output

### Figure 1: "Attack Model Training"
![Building the Attack Classifier](/assets/5%20Building%20the%20Attack%20Classifier-20261023004447113.png)

|Panel|What to read|Value|Meaning|
|---|---|---|---|
|Left (loss)|Train + Val loss|hover **0.6925-0.694**|0.693 = ln(2) = the loss of pure guessing on 2 classes. The model is barely below a coin flip|
|Left|Val loss drifts slightly below train loss|~0.6925 vs ~0.6928|Not overfitting; there's just nothing to fit|
|Right (accuracy)|Train + Val accuracy|jitter **0.497-0.513**|Pure noise around 50%. Train and Val are tangled, no clear winner|
|Right|No upward trend|flat|The attack learned essentially nothing **on shadow data**|

The y-axis on the loss panel spans only **0.6922 to 0.6940**, a range of 0.0018. The curve looks dramatic but it's flat as a pancake. **A loss stuck at ln(2) and accuracy at 50% is the signature of "no signal in this data".**

### Figure 2: "Attack Model Decision Boundary"
![Building the Attack Classifier](/assets/5%20Building%20the%20Attack%20Classifier-20261023004454587.png)

|Line|What it does|Reading|
|---|---|---|
|**True Class 1 (green)**|P(member) vs confidence|Flat at **~0.51**, just above the 0.5 line, **never crosses it**. The attack can't separate members from non-members for class 1|
|**True Class 0 (blue)**|P(member) vs confidence|**Decreases** from ~0.50 at low confidence to ~**0.37** at high confidence|
|Grey dashed|Decision threshold|0.5|
|Yellow dotted|Confidence marker|0.8|

The counterintuitive part: for **class 0, higher confidence lowers** the membership probability. The attack learned that **a very confident class-0 prediction is more likely a non-member**, the opposite of the usual "confident = member" intuition. This happens because in the shadow data (Section 4 figure), the non-members were actually slightly **more** common in the top confidence bin.

> **An honest note the course skips (3):** The printed "threshold ~0.80-0.85" is generic boilerplate and doesn't match this figure. Class 1 never crosses 0.5, so `argmin|p − 0.5|` just lands wherever the flat line is closest (near the left edge, ~0.5), not a real threshold. Class 0 crosses 0.5 near confidence ~0.63 going **downward**. So what your printout actually reports is "the attack has no usable threshold on shadow data", which agrees with the 50% accuracy. Don't quote 0.80-0.85 in a writeup; read your own numbers.

---

## Copy-paste cells (mine, not the course's)

These use names that now exist: `attack_X`, `attack_y`, `attack_model`, `attack_test_loader`, `y_attack_test`, `boundary_analysis`, `evaluate_model`, `get_model_predictions`, `target_model`, `X_target_norm`, `y_target`, `X_attack_eval_norm`, `y_attack_eval`, `DEVICE`, `RANDOM_SEED`. Run the course's cells first. I tested the NumPy parts on stand-in arrays; PyTorch parts are checked against the library source.

### Cell 1: Confirm the split sizes and that the test set is balanced

```python
import numpy as np
print("train+val:", len(X_attack_train), "| test:", len(X_attack_test))
print("train:", len(X_attack_tr), "| val:", len(X_attack_val))
print("Test members:", int((y_attack_test == 1).sum()),
      "| non-members:", int((y_attack_test == 0).sum()),
      "-> balanced, so 0.500 is the right baseline here")
```

Expected: `48840 | 12210`, then `39072 | 9768`, then `Test members: 6105 | non-members: 6105`.

### Cell 2: Read the REAL decision thresholds from your run

This prints what the generic course line should have printed: where each class's membership probability actually crosses 0.5, and whether it crosses at all.

```python
print("Decision boundary, read honestly:")
for cls, data in sorted(boundary_analysis.items()):
    mp, conf = data['membership_probs'], data['confidences']
    crosses = np.where(np.diff(np.sign(mp - 0.5)) != 0)[0]
    where = f"crosses 0.5 at confidence ~{conf[crosses[0]]:.3f}" if len(crosses) else "NEVER crosses 0.5"
    print(f" Class {cls}: P(member) range [{mp.min():.3f}, {mp.max():.3f}] -> {where}")
```

Expected (matching Figure 2): Class 0 range roughly [0.37, 0.50], crosses near ~0.63; Class 1 range roughly [0.51, 0.51], **never crosses 0.5**. If a class never crosses, it has no usable threshold, which is why shadow accuracy is ~50%.

### Cell 3: The payoff preview, apply the attack to the TARGET (balanced vs 2:1)

This is the relevant move: the attack that looked dead on shadow data comes alive on the overfit target. It builds attack features from the target's own predictions and scores the attack model two ways: a **balanced** pool (50% baseline) and the raw **2:1** pool (where "always member" = 66.7%). It uses a private RNG so it won't disturb the global seed, and does no training.

```python
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def attack_feats(model, Xn, y):
    probs = get_model_predictions(model, Xn, DEVICE) # target's softmax on these rows
    return np.concatenate([probs, np.eye(2)[y]], axis=1) # [p0, p1, label0, label1]

Fm = attack_feats(target_model, X_target_norm, y_target) # members
Fn = attack_feats(target_model, X_attack_eval_norm, y_attack_eval) # non-members

def attack_member_prob(F):
    probs = attack_model.predict_proba(torch.tensor(F, dtype=torch.float32).to(DEVICE))
    return probs[:, 1].detach().cpu().numpy() # P(member)

pm, pn = attack_member_prob(Fm), attack_member_prob(Fn)

# Balanced pool: equal members and non-members -> 0.500 is the real baseline
rng = np.random.default_rng(RANDOM_SEED)
n = min(len(pm), len(pn))
pool = np.r_[pm[rng.choice(len(pm), n, replace=False)], pn[rng.choice(len(pn), n, replace=False)]]
truth = np.r_[np.ones(n), np.zeros(n)].astype(int)
pred = (pool > 0.5).astype(int)
print("BALANCED target pool (baseline 0.500):")
print(f" accuracy {accuracy_score(truth, pred):.3f} | precision {precision_score(truth, pred):.3f} "
      f"| recall {recall_score(truth, pred):.3f} | F1 {f1_score(truth, pred):.3f}")

# Raw 2:1 pool: always-member already scores 0.667 -> the trap
truth_i = np.r_[np.ones(len(pm)), np.zeros(len(pn))].astype(int)
pred_i = (np.r_[pm, pn] > 0.5).astype(int)
base = len(pm) / (len(pm) + len(pn))
print(f"RAW 2:1 pool: attack accuracy {accuracy_score(truth_i, pred_i):.3f} "
      f"| 'always member' baseline {base:.3f}")
```

Expected: on the **balanced** pool, accuracy clearly above 0.50 (the course promises ~0.65), so the attack is real. On the **2:1** pool, accuracy near ~0.50-0.55 while "always member" sits at **0.667**, proving why you must balance before claiming success. If your balanced number is below 0.55, the matched shadows from Section 4 Cell 3 (train the attack on `attack_X_m` instead) usually push it higher.

## Key facts to lock in

- **The attack model uses a three-way split, 39,072 train / 9,768 validation / 12,210 test (80/20 then 80/20 of the 61,050 rows)**. Validation drives early stopping so the attack doesn't memorize shadow-specific quirks; the test set gives an unbiased shadow-data estimate.
- **The attack model is deliberately tiny: 4 -> [64, 32] -> 2, which is 2,466 parameters (the course miscounts it as 2,600), versus ~37,000 in the target**. Small capacity plus diverse multi-shadow data is what makes it transfer rather than overfit.
- **50-51% accuracy on shadow test data is expected, not failure**. Loss stuck at ~0.693 (ln 2) and accuracy jittering around 0.50 mean the regularized shadows leaked almost nothing. Shadow-data accuracy does not predict target performance.
- ==Precision = TP/(TP+FP) is how trustworthy "member" flags are; recall = TP/(TP+FN) is how many real members you catch; F1 is their harmonic mean==. Privacy audits favor recall (lower threshold ~0.3), legal use favors precision (higher threshold ~0.7).
- **The default decision is member if P(member) > 0.5, but the best threshold depends on the cost of false positives vs false negatives and on class balance**. The attack bakes a 50/50 boundary into its final-layer bias, so an imbalanced evaluation pool skews precision and recall apart.
- **In the decision-boundary figure, class 1 stays flat at ~0.51 (never crosses 0.5) and class 0 actually decreases to ~0.37 at high confidence**, meaning a very confident class-0 prediction reads as a non-member. The generic "0.80-0.85 threshold" text does not match the figure.
- **The attack is expected to jump to 65-66% (a 15-16% advantage) on the overfit target**: above 60% = meaningful leakage, 55-60% = moderate, below 55% = ineffective. Shadow evaluation underestimates the target on purpose.
- **Method point:** **Always balance the member/non-member pool before quoting attack accuracy**. On the raw 2:1 target pool, "always say member" already scores 66.7%, so a 65-66% figure there is meaningless; the real 65-66% is on a balanced pool where 50% is the baseline. Report the balanced accuracy, the AUC, and precision/recall together, never accuracy alone.

---

