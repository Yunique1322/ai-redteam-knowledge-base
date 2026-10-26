# Training Shadow Models

## The idea in one breath

You train **5 shadow models**, each on a different random half of the attacker's shadow data. For each one you record its probabilities on rows it trained on (**members**) and rows it never saw (**non-members**), then glue those into **61,050 labelled attack rows** shaped `[p0, p1, label0, label1]`. The surprise is in HTB's own figure: **members and non-members both average 0.857 confidence**, so these shadows show almost **no membership signal**. The attack model will be learning from very little, and this section explains why.

> **Heads-up:** No mangled formulas. Four things to know before reading:
> 
> - **Merged code:** the course splits the main shadow-training `for` loop across **four** blocks. It's one loop, so I give it as **one cell** in Part 2, with every line unchanged.
> - **The numbers contradict each other inside the section.** The prose says the shadows have a "1-2% gap similar to the target" (the target's gap was 11.1 points) and a confidence gap "typically 0.03-0.05". The figure shows **0.857 vs 0.857, a gap of 0.000**. Trust the figure.
> - **A labelling mistake in the course code** (Part 3): 20% of the rows labelled "member" were never trained on.
> - **Plots don't show inline.** Every HTB privacy `plot_*` function saves a PNG and then calls `plt.close()`, so nothing appears in your notebook. My Cell 2 displays the saved file.

---

## Part 1: Creating Shadow Model Data Splits

```python
print("\n" + "=" * 60)
print("Training Shadow Models")
print("=" * 60)

shadow_splits = []
for i in range(SHADOW_MODEL_CONFIG['num_shadow_models']):
    seed = RANDOM_SEED + i
    X_train_s, X_out_s, y_train_s, y_out_s = train_test_split(
        X_shadow, y_shadow, train_size=SHADOW_MODEL_CONFIG['shadow_data_size'],
        random_state=seed, stratify=y_shadow
    )
    shadow_splits.append((X_train_s, X_out_s, y_train_s, y_out_s))

print(f"\nCreated {len(shadow_splits)} shadow model data splits")
print(f"Samples per shadow model: ~{len(shadow_splits[0][0])} in, ~{len(shadow_splits[0][1])} out")
```

Output (I checked the split arithmetic):

![Training Shadow Models](/assets/4%20Training%20Shadow%20Models-20261023154900835.png)

|Code|What it does|
|---|---|
|`for i in range(5)`|One split per shadow model|
|`seed = RANDOM_SEED + i`|Seeds 1337, 1338, ..., 1341, so each shadow gets a **different** random half|
|`train_test_split(X_shadow, y_shadow, train_size=0.5, ...)`|Cuts the 12,210 shadow rows into 6,105 "in" (will train) and 6,105 "out" (held back)|
|`stratify=y_shadow`|Both halves keep the same ~24% share of >50K earners|
|`shadow_splits.append((...))`|Stores the four arrays as one tuple per shadow|

This is the Section 2 toy table for real. A given person can be "in" for Shadow 1 and "out" for Shadow 3. Because every row appears in both roles across the shadows, the attack learns how _behaviour changes with membership_, not quirks of particular people.

|Name|Shape|Meaning|
|---|---|---|
|`shadow_splits`|list of 5 tuples|One `(X_train_s, X_out_s, y_train_s, y_out_s)` per shadow|
|`X_train_s`, `y_train_s`|(6105, 14), (6105,)|This shadow's "in" rows|
|`X_out_s`, `y_out_s`|(6105, 14), (6105,)|This shadow's "out" rows|

The course says five models is "the right balance": fewer gives too little diversity, and more gives diminishing returns at linear extra cost. That's a reasonable default, not a law. As noted in Section 2, later research found even **one** shadow can be enough.

---

## Part 2: Training and Collecting Predictions (the merged loop)

**Merged:** the course split this one `for` loop across four code blocks (setup + validation split / train / collect predictions / overfitting check). None of those pieces can run on its own, so here it is as **one cell**. Every line is unchanged from the course, in the original order.

```python
all_attack_X = []
all_attack_y = []
all_preds_in = []
all_preds_out = []

for i, (X_train_s, X_out_s, y_train_s, y_out_s) in enumerate(shadow_splits):
    print(f"\nTraining Shadow Model {i+1}/{SHADOW_MODEL_CONFIG['num_shadow_models']}")

    # Normalize using target scaler for transferability
    X_train_s_norm = scaler.transform(X_train_s)
    X_out_s_norm = scaler.transform(X_out_s)

    # Create validation split for early stopping
    X_tr_s, X_val_s, y_tr_s, y_val_s = train_test_split(
        X_train_s_norm, y_train_s, test_size=0.2,
        random_state=RANDOM_SEED + i, stratify=y_train_s
    )
    train_loader_s = create_dataloader(X_tr_s, y_tr_s, SHADOW_MODEL_CONFIG['batch_size'])
    val_loader_s = create_dataloader(X_val_s, y_val_s, SHADOW_MODEL_CONFIG['batch_size'], shuffle=False)

    # Initialize and train shadow model
    shadow_model = MLP(
        input_size=num_features,
        hidden_layers=SHADOW_MODEL_CONFIG['hidden_layers'],
        num_classes=DATASET_CONFIG['num_classes'],
        dropout=SHADOW_MODEL_CONFIG['dropout']
    )
    train_with_early_stopping(
        shadow_model, train_loader_s, val_loader_s,
        device=DEVICE,
        epochs=SHADOW_MODEL_CONFIG['epochs'],
        learning_rate=SHADOW_MODEL_CONFIG['learning_rate'],
        patience=SHADOW_MODEL_CONFIG['early_stopping_patience'],
        verbose=False
    )

    # Collect predictions on members and non-members
    preds_in = get_model_predictions(shadow_model, X_train_s_norm, DEVICE)
    preds_out = get_model_predictions(shadow_model, X_out_s_norm, DEVICE)

    # Transform to attack features and accumulate
    attack_X_s, attack_y_s = prepare_attack_data(preds_in, preds_out, y_train_s, y_out_s)
    all_attack_X.append(attack_X_s)
    all_attack_y.append(attack_y_s)
    all_preds_in.append(preds_in)
    all_preds_out.append(preds_out)

    # Verify overfitting gap exists
    full_train_loader_s = create_dataloader(X_train_s_norm, y_train_s,
                                            SHADOW_MODEL_CONFIG['batch_size'], shuffle=False)
    full_out_loader_s = create_dataloader(X_out_s_norm, y_out_s,
                                          SHADOW_MODEL_CONFIG['batch_size'], shuffle=False)
    train_acc_s, _, _ = evaluate_model(shadow_model, full_train_loader_s, DEVICE)
    out_acc_s, _, _ = evaluate_model(shadow_model, full_out_loader_s, DEVICE)
    print(f" Shadow {i+1} - Train Acc: {train_acc_s:.4f}, Out Acc: {out_acc_s:.4f}")
```

Output, one line per shadow (course example):

![Training Shadow Models](/assets/4%20Training%20Shadow%20Models-20261023155034688.png)
### Step 1: Four empty lists

|Code|What it does|
|---|---|
|`all_attack_X = []`|Collects each shadow's 4-number attack rows|
|`all_attack_y = []`|Collects each shadow's 1/0 member labels|
|`all_preds_in = []`, `all_preds_out = []`|Raw `(n, 2)` probabilities, kept only for the confidence histogram|

The course says it uses lists because "the counts vary slightly". They don't: every shadow gives exactly 6,105 + 6,105. Lists are simply the easy way to collect things in a loop.

### Step 2: Normalize and carve out a validation set

|Code|What it does|
|---|---|
|`enumerate(shadow_splits)`|Loops over the 5 tuples; `i` = 0..4|
|`scaler.transform(X_train_s)` / `(X_out_s)`|Scales with the **target's** μ and σ (fitted in Section 3), with `transform`, never `fit_transform`|
|`train_test_split(X_train_s_norm, ..., test_size=0.2, ...)`|Splits the 6,105 "in" rows into **4,884 for training** and **1,221 for validation**|
|`create_dataloader(X_tr_s, y_tr_s, 64)`|Training batches of 64, shuffled|
|`create_dataloader(X_val_s, y_val_s, 64, shuffle=False)`|Validation batches, used only to decide when to stop|

The course's reason for reusing the target's scaler: identical raw values should give identical scaled values in every model, so prediction differences reflect **membership**, not preprocessing. That logic is sound.

> **An honest note the course skips (1):** A real black-box attacker never sees the target's scaler. The API takes raw records and scales them internally. In practice you'd fit your own scaler on your shadow data. Because it's the same population, your μ and σ would be almost identical, so this shortcut barely changes anything here. Just don't describe it as part of the threat model in a report.

### Step 3: Build and train the shadow

|Code|What it does|
|---|---|
|`MLP(..., hidden_layers=[128, 64], ..., dropout=0.3)`|A **smaller, regularized** network than the target ([256, 128], dropout 0.0)|
|`train_with_early_stopping(..., patience=10, verbose=False)`|Trains until validation loss hasn't improved for 10 epochs. `verbose=False` hides the progress bar and the "Early stopping at epoch N" message|

The return value (the history dict) is thrown away. That's fine, because training changes `shadow_model` **in place**. Also remember the Section 3 bug: the "restore best weights" step doesn't really work, so each shadow keeps its weights from the **stop** epoch (best + 10).

### Step 4: Collect probabilities and build attack rows

|Code|What it does|
|---|---|
|`get_model_predictions(shadow_model, X_train_s_norm, DEVICE)`|Probabilities `(6105, 2)` for the "in" rows. Uses `eval()` + `no_grad()`, so dropout is switched off|
|`get_model_predictions(shadow_model, X_out_s_norm, DEVICE)`|Probabilities `(6105, 2)` for the "out" rows|
|`prepare_attack_data(preds_in, preds_out, y_train_s, y_out_s)`|Glues probs + one-hot labels → `attack_X_s` `(12210, 4)`; labels → `attack_y_s` (6,105 ones then 6,105 zeros)|
|the four `.append(...)`|Adds this shadow's results to the running lists|

### Step 5: Check that the shadow overfits at all

|Code|What it does|
|---|---|
|`full_train_loader_s` / `full_out_loader_s`|Loaders over **all** 6,105 "in" rows and all 6,105 "out" rows|
|`evaluate_model(...)`|Accuracy on each|
|`print(...)`|e.g. `Train Acc: 0.8612, Out Acc: 0.8423`, a gap of **1.9 points**|

|Name|Shape|Meaning|
|---|---|---|
|`X_train_s_norm`|(6105, 14)|Scaled "in" rows (**includes** the 1,221 validation rows)|
|`X_tr_s` / `X_val_s`|(4884, 14) / (1221, 14)|Rows actually trained on / validation-only rows|
|`preds_in`, `preds_out`|(6105, 2) each|This shadow's probabilities|
|`attack_X_s`, `attack_y_s`|(12210, 4), (12210,)|This shadow's attack rows and labels|

After the loop ends, `shadow_model`, `preds_in` and `preds_out` hold **Shadow 5 only**. The other four models are gone; only their predictions were kept in the lists.

> **If this errors:** `ValueError: Found input variables with inconsistent numbers of samples` inside `prepare_attack_data` or `train_test_split` usually means a cell was re-run out of order and arrays from different shadows got mixed. Fix: re-run Part 1, then this whole cell. That's another reason it's one cell: half of a loop re-run on its own is the classic way to get mismatched arrays.

---

## Part 3: The Labelling Problem the Course Doesn't Mention

### Step 1: Toy

Take a shadow with 10 records. `train_size=0.5` gives 5 "in" and 5 "out". The 20% validation split then takes 1 of the 5 "in" records, leaving **4 actually trained on**.

|Group|Count|Average confidence|Labelled as|
|---|---|---|---|
|Trained "in" records|4|0.95 (memorized)|member ✓|
|Validation record (never trained on)|1|0.80 (behaves like a stranger)|**member ✗**|
|"Out" records|5|0.80|non-member ✓|

- **True** gap (trained members vs non-members) = 0.95 − 0.80 = **0.15**
- Mean of everything **labelled** member = (4 × 0.95 + 1 × 0.80) / 5 = 4.60 / 5 = **0.92**
- **Measured** gap = 0.92 − 0.80 = **0.12**, which is 0.8 × 0.15

### Step 2: The rule

```
measured gap = (fraction of "member" rows really trained on) × true gap
```

|Symbol|Plain English|Toy|Real|
|---|---|---|---|
|fraction really trained on|Share of member-labelled rows that the optimizer actually fitted|4/5 = 0.8|4,884 / 6,105 = 0.8|
|true gap|Confidence gap between real members and non-members|0.15|unknown|
|measured gap|What the stats block prints|0.12|0.000 in the figure|

**Read aloud:** "Every validation row I call a member drags the member average back toward the non-member average."

> **An honest note the course skips (2):** `preds_in` is computed on `X_train_s_norm`, all 6,105 "in" rows. But the shadow only trained on `X_tr_s` (4,884). The 1,221 validation rows were used only to decide when to stop, and the model never fitted them, so they behave like non-members. **20% of the attack model's "member" examples are mislabelled**. That makes the signal weaker and teaches the attack model wrong answers. The printed shadow `Train Acc` is diluted the same way. The fix is to use only the rows actually trained on (`X_tr_s`) as members. My Cell 3 does that.

---

## Part 4: Combining Attack Training Data

```python
attack_X = np.concatenate(all_attack_X, axis=0)
attack_y = np.concatenate(all_attack_y, axis=0)

print(f"\nTotal attack training samples: {len(attack_X)}")
print(f" Members: {np.sum(attack_y == 1)}")
print(f" Non-members: {np.sum(attack_y == 0)}")
```

Output (exact, from the split sizes):

![Training Shadow Models](/assets/4%20Training%20Shadow%20Models-20261023155047416.png)

|Code|What it does|
|---|---|
|`np.concatenate(all_attack_X, axis=0)`|Stacks the 5 blocks of shape (12210, 4) on top of each other → (61050, 4)|
|`np.concatenate(all_attack_y, axis=0)`|→ (61050,)|
|`np.sum(attack_y == 1)`|Counts members: 5 × 6,105 = 30,525|

The row order is shadow 1 in, shadow 1 out, shadow 2 in, shadow 2 out, and so on, each block 6,105 rows.

**The attack training set is balanced (30,525 vs 30,525)**. As the course says, an imbalanced set would push the classifier toward the majority label. This is the Section 1 base-rate lesson applied to **training** data. The evaluation pool will need the same care.

---

## Part 5: Understanding Attack Features

```python
print(f"\nAttack feature dimensions: {attack_X.shape[1]}")
print(f"Example member feature: {attack_X[0].round(3)}")
print(f"Example non-member feature: {attack_X[len(attack_X)//2].round(3)}")
```

|Code|What it does|
|---|---|
|`attack_X.shape[1]`|4 (`[prob_0, prob_1, label_0, label_1]`)|
|`attack_X[0]`|First row = Shadow 1's first "in" row → a member|
|`attack_X[len(attack_X)//2]`|Row 30,525 = **Shadow 3's first "out" row** → a non-member|
|`.round(3)`|Display to 3 decimals|

The course's illustration: member `[0.12, 0.88, 0.0, 1.0]` vs non-member `[0.25, 0.75, 0.0, 1.0]`, the same true label (class 1) but different confidence. As in Section 1, the attack model reads two signals from each row: **confidence** (the bigger of the first two numbers) and **correctness** (does the bigger probability sit on the slot the one-hot marks?).

> **An honest note the course skips (3):** `len(attack_X)//2` lands on a non-member **by luck**. With 5 shadows, the halfway row is exactly where Shadow 3's "out" block starts. With 4 shadows, the halfway row (24,420) would be Shadow 3's first **member**, and the print would mislabel it. The safe way is `attack_X[attack_y == 0][0]` (my Cell 1).

---

## Part 6: Visualizing Shadow Model Behavior

```python
plot_shadow_confidence_distributions(
    all_preds_in, all_preds_out,
    save_path=os.path.join(FIGS_DIR, f"{FIG_PREFIX}shadow_confidence.png")
)
```

![Training Shadow Models](/assets/4%20Training%20Shadow%20Models-20261023155342787.png)

| Code | What it does |
| ------------------------------- | ------------------------------------------------------------------------------------------------------------------------------- |
| `all_preds_in`, `all_preds_out` | The 5 per-shadow probability arrays; the function takes the max of each row (confidence) and pools all 5 shadows |
| `save_path=...` | Writes `figs/Introduction_shadow_confidence.png`. Nothing displays inline (it calls `plt.close()`), so use my Cell 2 to view it |

The course's own reading (and it's right): **both distributions pile up above 0.85 with nearly identical means**. Because the shadows use **dropout and early stopping**, they barely overfit, so the "expected confidence gap essentially disappears". What remains is small differences in the **tails**, which the attack model has to find across 61,050 rows.

---

## Part 7: Attack Data Statistics

```python
member_confidences = attack_X[attack_y == 1, :2].max(axis=1)
non_member_confidences = attack_X[attack_y == 0, :2].max(axis=1)

print(f"\nAttack Data Statistics:")
print(f" Member confidence - Mean: {member_confidences.mean():.4f}, Std: {member_confidences.std():.4f}")
print(f" Non-member confidence - Mean: {non_member_confidences.mean():.4f}, Std: {non_member_confidences.std():.4f}")
print(f" Confidence gap: {member_confidences.mean() - non_member_confidences.mean():.4f}")
```

| Code | What it does |
| ----------------------------- | ------------------------------------------------------------------------ |
| `attack_X[attack_y == 1, :2]` | Member rows, first two columns only (the probabilities, not the one-hot) |
| `.max(axis=1)` | Confidence = the larger probability in each row |
| `.mean()`, `.std()` | Average and spread |
| last print | **Confidence gap** = member mean − non-member mean |

The course says the gap is "typically 0.03-0.05", but its own figure shows **0.857 − 0.857 = 0.000** (to 3 decimals). Expect your printed gap to be **near zero**, not 0.03.

> **An honest note the course skips (4):** This is the shadow-target mismatch from Section 3 playing out. The target overfits hard (11.1-point accuracy gap, validation loss doubling). The shadows, being smaller, using dropout 0.3, early stopping, and with 20% mislabelled members, show essentially no gap. **The attack model is trained to find a signal its training data barely contains, then pointed at a target where the signal is large.** The course also says each shadow has a "1-2% gap **similar to the target**", but 1-2 points is nowhere near 11.1. Section 2's rule ("match the target's training setup") exists to prevent exactly this. My Cell 3 builds **matched** shadows so you can see the difference.

---

## Reading the figure/output
![Training Shadow Models](/assets/4%20Training%20Shadow%20Models-20261023002809265.png)
### Figure: "Shadow Model Confidence: Members vs Non-Members"

|What to read|Value|What it means|
|---|---|---|
|Legend counts|Members n = 30,525, Non-members n = 30,525|All 5 shadows pooled; balanced|
|Member mean / Non-member mean|**0.857 / 0.857**|Identical. Only the **red** dashed line is visible because it's drawn on top of the teal one|
|x-axis range|0.5 to 1.0|With 2 classes, the bigger probability can never be below 0.5 ([0.5, 0.5] is the worst case)|
|Bins|49 bins, each ~0.0102 wide|A density of 1 ≈ 1% of rows in that bin|
|Flat region 0.5 → 0.9|density ≈ 1 per bin|~40% of rows are spread evenly across "unsure to fairly sure"|
|Rise 0.9 → 0.99|density 1.3 → 7|Increasingly confident|
|Last bin (0.99-1.0)|density ≈ 29|≈ 29 × 0.0102 ≈ **30% of all rows** are at 99%+ confidence, members and non-members alike|
|Colour|Almost everything is the overlap colour (brownish)|The two histograms sit on top of each other|
|Tiny teal slivers|~0.59, ~0.95, ~0.975|Spots where members are slightly more common|
|Tiny red sliver|last bin|Non-members are slightly **more** common at 99%+, the opposite of the expected direction|

**Bottom line:** if you put this histogram next to the target's training curves from Section 3, the contrast is the lesson. **The target leaks a lot; these shadows barely leak at all**, so an attack model trained on them has almost nothing to learn.

---

## Copy-paste cells (mine, not the course's)

These use names that now exist: `attack_X`, `attack_y`, `all_attack_X`, `all_preds_in`, `shadow_splits`, `scaler`, `MLP`, `TARGET_MODEL_CONFIG`, `DATASET_CONFIG`, `num_features`, `train_fixed_epochs`, `create_dataloader`, `get_model_predictions`, `prepare_attack_data`, `DEVICE`, `FIGS_DIR`, `FIG_PREFIX`.

**Run order:** finish the course's cells **first**. My Cell 3 trains extra models, which uses up PyTorch's random numbers. If you run it _before_ the next section's course cells, your later numbers will drift a little from HTB's. I tested the NumPy parts on stand-in arrays. The PyTorch parts are checked against the library source.

### Cell 1: Inspect the attack data and count the mislabelled "members"

```python
n_in = len(all_preds_in[0])
n_val = int(np.ceil(0.2 * n_in))
print("attack_X:", attack_X.shape, "| attack_y:", attack_y.shape)
print("Rows per shadow:", [len(a) for a in all_attack_X])
print("Label of row 0:", attack_y[0], "| label of row len//2:", attack_y[len(attack_X) // 2])
print("Safe member example: ", attack_X[attack_y == 1][0].round(3))
print("Safe non-member example:", attack_X[attack_y == 0][0].round(3))
print(f"Per shadow: {n_in} rows labelled member, but only {n_in - n_val} were trained on "
      f"-> {n_val} ({n_val / n_in:.0%}) are validation rows mislabelled as members")
```

Expected:

```txt
attack_X: (61050, 4) | attack_y: (61050,)
Rows per shadow: [12210, 12210, 12210, 12210, 12210]
Label of row 0: 1 | label of row len//2: 0
Safe member example: [... your numbers ...]
Safe non-member example: [... your numbers ...]
Per shadow: 6105 rows labelled member, but only 4884 were trained on -> 1221 (20%) are validation rows mislabelled as members
```

### Cell 2: Actually see the saved figures

The HTB plot functions only save to disk, so this cell displays the saved PNGs.

```python
from IPython.display import Image, display
for name in ["overfitting_gap", "shadow_confidence"]:
    path = os.path.join(FIGS_DIR, f"{FIG_PREFIX}{name}.png")
    if os.path.exists(path):
        display(Image(filename=path))
    else:
        print("Not found (run the plotting cell first):", path)
```

Expected: the Section 3 bar chart and this section's histogram, shown inline.

### Cell 3: Matched shadows (configured like the target, with clean member labels)

This retrains 5 shadows with the **target's** settings ([256, 128], dropout 0.0, fixed epochs, no early stopping), using the same `shadow_splits`. There's no validation split, so **every row labelled "member" was really trained on**. It's one loop, so it's one cell. All new names end in `_m`, so nothing from the course is overwritten.

Fill in: `SHADOW_EPOCHS_MATCHED`. Use 100 to match the target exactly, or 30-50 if your CPU is slow (it's 5 training runs).

```python
SHADOW_EPOCHS_MATCHED = 100 # fill-in: 100 = same as target; lower = faster

all_attack_X_m, all_attack_y_m = [], []
all_preds_in_m, all_preds_out_m = [], []

for i, (X_train_s, X_out_s, y_train_s, y_out_s) in enumerate(shadow_splits):
    print(f"\nMatched Shadow {i+1}/{len(shadow_splits)} "
          f"({TARGET_MODEL_CONFIG['hidden_layers']}, dropout {TARGET_MODEL_CONFIG['dropout']}, "
          f"{SHADOW_EPOCHS_MATCHED} fixed epochs)")
    X_in_norm = scaler.transform(X_train_s)
    X_out_norm = scaler.transform(X_out_s)

    m = MLP(input_size=num_features,
            hidden_layers=TARGET_MODEL_CONFIG['hidden_layers'],
            num_classes=DATASET_CONFIG['num_classes'],
            dropout=TARGET_MODEL_CONFIG['dropout'])
    train_fixed_epochs(m,
                       create_dataloader(X_in_norm, y_train_s, TARGET_MODEL_CONFIG['batch_size']),
                       create_dataloader(X_out_norm, y_out_s, TARGET_MODEL_CONFIG['batch_size'], shuffle=False),
                       device=DEVICE,
                       epochs=SHADOW_EPOCHS_MATCHED,
                       learning_rate=TARGET_MODEL_CONFIG['learning_rate'])

    p_in = get_model_predictions(m, X_in_norm, DEVICE) # every one of these rows WAS trained on
    p_out = get_model_predictions(m, X_out_norm, DEVICE)
    X_a, y_a = prepare_attack_data(p_in, p_out, y_train_s, y_out_s)
    all_attack_X_m.append(X_a); all_attack_y_m.append(y_a)
    all_preds_in_m.append(p_in); all_preds_out_m.append(p_out)
    print(f" Train Acc: {(p_in.argmax(1) == y_train_s).mean():.4f}, "
          f"Out Acc: {(p_out.argmax(1) == y_out_s).mean():.4f}")

attack_X_m = np.concatenate(all_attack_X_m, axis=0)
attack_y_m = np.concatenate(all_attack_y_m, axis=0)
print(f"\nMatched attack data: {attack_X_m.shape}, members {np.sum(attack_y_m == 1)}, "
      f"non-members {np.sum(attack_y_m == 0)}")
```

Expected: 5 progress bars, then per-shadow lines where Train Acc is clearly **above** Out Acc (a gap of several points, like the target, not the course shadows' ~2). At the end: `Matched attack data: (61050, 4), members 30525, non-members 30525`. These shadows train on 6,105 rows vs the target's 24,421, and less data means more memorization (Section 2), so their gap may even be **bigger** than the target's.

### Cell 4: How much signal does each attack dataset contain?

`signal_report()` measures the confidence gap, the correctness gap, and the threshold-free **AUC** of "probability on the true label" (the Section 2 loss-attack score). AUC ≈ 0.5 means nothing to learn.

```python
from sklearn.metrics import roc_auc_score

def signal_report(X_att, y_att, name):
    probs, onehot = X_att[:, :2], X_att[:, 2:]
    conf = probs.max(axis=1) # signal 1: confidence
    correct = probs.argmax(axis=1) == onehot.argmax(axis=1) # signal 2: correctness
    p_true = (probs * onehot).sum(axis=1) # prob on the true label (loss-attack score)
    mem = y_att == 1
    print(f"{name}")
    print(f" confidence gap : {conf[mem].mean() - conf[~mem].mean():+.4f}")
    print(f" accuracy gap : {correct[mem].mean() - correct[~mem].mean():+.4f}")
    print(f" AUC of p_true : {roc_auc_score(mem, p_true):.3f} (0.500 = no signal to learn)")

signal_report(attack_X, attack_y, "Course shadows (dropout 0.3, early stopping, 20% mislabelled)")
if 'attack_X_m' in globals():
    signal_report(attack_X_m, attack_y_m, "Matched shadows (target config, clean labels)")
```

Expected: the course shadows show a confidence gap near **+0.000**, an accuracy gap of about **+0.01 to +0.02**, and an AUC only slightly above **0.500**. The matched shadows should show clearly bigger numbers on all three. That difference is what the attack model gets to learn from. In the next section, compare how well each attack set does against the target.

---

## Quick self-check

Suppose the course had used a **30%** validation split instead of 20% (`test_size=0.3` on each shadow's 6,105 "in" rows).

(a) How many rows end up in validation, and how many are actually trained on? (sklearn rounds the validation count **up**.)  
(b) What percentage of the member-labelled rows are then mislabelled?  
(c) If truly-trained members average **0.90** confidence and non-members (including the validation rows) average **0.86**, what confidence gap would the stats block print, and what's the true gap?

---

## The facts worth locking in for the exam

- **Each of the 5 shadows trains on a different random half of the 12,210 shadow rows (6,105 in / 6,105 out, seeds 1337-1341)**. Over the 5 shadows the same person is sometimes in and sometimes out, so the attack learns how behaviour changes with membership rather than quirks of one split.
- **The attack training set is built by labelling each shadow's predictions on its own training rows as 1 (member) and on its held-out rows as 0 (non-member)**, giving 61,050 balanced rows of `[p0, p1, label0, label1]`. Balance matters because an imbalanced set would bias the attack toward the majority label.
- **In HTB's own figure, shadow members and non-members both average 0.857 confidence**. With dropout 0.3 and early stopping, the shadows barely overfit (~2-point accuracy gap vs the target's 11.1), so the confidence gap essentially disappears. The course's "typically 0.03-0.05" and "similar to the target" contradict its own figure.
- **A "member" must be a row the optimizer actually trained on**. The course labels all 6,105 "in" rows as members, but 1,221 (20%) were only validation rows, so the measured gap shrinks to 0.8 × the true gap and the attack learns some wrong answers. In general, measured gap = (fraction really trained on) × true gap.
- **Shadow-target mismatch is the main reason a shadow attack underperforms**. Shadows that are smaller, regularized and early-stopped can't teach the attack model a signal the overfit target has in abundance. Matching architecture, dropout, epochs and stopping rules is what Section 2 demanded, and this section is the counter-example.
- **Scaling shadow data with the same μ and σ as the target keeps prediction differences about membership, not preprocessing**. But a real black-box attacker would fit their own scaler on their own same-population data, which gives almost the same numbers.
- **With 2 classes, maximum confidence is always at least 0.5**, and in the figure ~30% of all rows sit at 99%+ confidence for members and non-members alike. High confidence alone does not mean member; the attack has to use the subtle differences in the tails plus the correctness signal.
- **Method point:** **Before training an attack model, measure how much signal your attack dataset actually contains: confidence gap, correctness gap, and above all a threshold-free AUC**. Check that the member labels are truly training rows and that the pool is balanced. If the AUC is near 0.5, fix the shadows (match the target, clean the labels) before blaming the attack model.

---

I logged this section in COAE-progress.md: the merged loop, the new names, the 20% mislabelling, the figure reading and the self-check answer key. Paste Section 5 when you're ready.