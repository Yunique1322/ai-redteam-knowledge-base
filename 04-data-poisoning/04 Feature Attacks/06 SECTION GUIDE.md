# Data Poisoning: Clean-Label (Feature) Attack Playbook for a New Dataset

This is the drag-and-drop version of what you did by hand in Sections 11 to 13. You plug your dataset into the config cell, run top to bottom, and it picks a target, crafts the perturbation, poisons the data, retrains, and tells you whether the target flipped, all while keeping labels untouched.

How it maps to what you learned: ==Cell 6 = Section 11 (identify a target), Cells 7 to 9 = Section 12 (craft and apply the clean-label perturbation), Cells 10 to 12 = Section 13 (retrain and evaluate).==

Two honest notes before you start, so nothing surprises you:

- **This attack is built for multiclass data (3+ classes), which is exactly the HTB scenario.** On a strictly 2-class dataset the single linear boundary is very rigid and often will not flip without an implausibly large push. The code still runs on 2-class data and will tell you if it could not flip.
- **Clean-label is data-dependent.** A target sitting near the boundary flips cheaply and stealthily. A target deep inside its class may need a big, less plausible push, or may not flip at all. Cell 6 prints the target's margin so you know which situation you are in, and Cell 11 auto-escalates if the default push is too weak.

## Cell 0: Config (the only cell you normally edit)

```python
# ============================================================
# CONFIG - plug your values in here. This is the only cell you edit.
# ============================================================
CSV_PATH     = "path/to/your_dataset.csv"   # <-- CHANGE: path to your dataset
LABEL_COLUMN = "label"                       # <-- CHANGE: exact name of the label column

# The attack goal: make a point whose TRUE class is SOURCE_CLASS get predicted as DEST_CLASS.
# Write the class values EXACTLY as they appear in the label column (text like "acceptable" or a number like 1).
SOURCE_CLASS = 1     # <-- CHANGE: true class of the point you want to hide
DEST_CLASS   = 0     # <-- CHANGE: the class you want it misread AS

# Attack strength. Start with these; only raise them if the target does not flip (see Cell 11).
N_NEIGHBORS_TO_PERTURB = 5      # how many DEST-class points to move
EPSILON_CROSS          = 0.25   # how far to push them, in standardized units (std devs)

SEED = 1337   # HTB requires 1337. Keep it for reproducibility.
```

## Cell 1: Imports

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, classification_report

np.random.seed(SEED)
```

## Cell 2: Load and look at the data

Always eyeball the data first so you know the label column name and the class values are what you think.

```python
df = pd.read_csv(CSV_PATH)

print("Shape:", df.shape)
print("\nFirst rows:")
print(df.head())
print("\nColumns:", list(df.columns))
print(f"\nClass counts in '{LABEL_COLUMN}':")
print(df[LABEL_COLUMN].value_counts())
```

## Cell 3: Prepare the data

Same prep you know from the label-flipping playbook: split features from label, turn text features into numbers, encode the label, split, and **standardize (this is what makes `EPSILON_CROSS` mean "std devs," so 0.25 is comparable across any dataset)**.

```python
# Split features from the label
X_df  = df.drop(columns=[LABEL_COLUMN])
y_raw = df[LABEL_COLUMN]

# Turn any text/categorical FEATURE columns into numbers (one-hot).
# If your features are already all numeric this changes nothing, so it is always safe to keep.
X_df = pd.get_dummies(X_df).astype(float)

# Encode the LABEL into integers 0,1,2,...
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y_raw)

# Map your chosen classes to those integer codes. Works whether your classes are text or numbers.
SOURCE_CLASS_ENC = label_encoder.transform([SOURCE_CLASS])[0]
DEST_CLASS_ENC   = label_encoder.transform([DEST_CLASS])[0]
print("Label mapping (original -> encoded):",
      dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_))))
print(f"SOURCE {SOURCE_CLASS!r} -> {SOURCE_CLASS_ENC}   DEST {DEST_CLASS!r} -> {DEST_CLASS_ENC}")

# Train/test split (stratify keeps class balance in both halves)
X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    X_df.values, y, test_size=0.30, random_state=SEED, stratify=y
)

# Standardize features (mean 0, std 1). EPSILON_CROSS is measured in these units.
scaler  = StandardScaler()
X_train = scaler.fit_transform(X_train_raw)
X_test  = scaler.transform(X_test_raw)

print("\nTrain features:", X_train.shape, "| Test features:", X_test.shape)
```

## Cell 4: Baseline model

The clean model you will attack. **Keep these exact settings; the poisoned model in Cell 10 must use the same ones so the comparison is fair.**

```python
baseline_model = OneVsRestClassifier(
    LogisticRegression(random_state=SEED, C=1.0, solver="liblinear")
)
baseline_model.fit(X_train, y_train)

baseline_accuracy = accuracy_score(y_test, baseline_model.predict(X_test))
print(f"Baseline accuracy on clean test set: {baseline_accuracy:.4f}")
print("Classes the model learned (encoded):", baseline_model.classes_)
```

## Cell 5: Build the DEST-vs-SOURCE boundary

This is the `w_diff_01_base` idea from Section 11, generalized. **You do not edit this cell.** The helper subtracts the source class hyperplane from the dest class hyperplane, and it also quietly handles the awkward 2-class case (where One-vs-Rest builds only a single hyperplane).

```python
# You do NOT need to edit this cell. It builds the boundary between DEST and SOURCE.
def dest_minus_source(ovr_model, dest_class, source_class):
    """(w_diff, b_diff) = dest hyperplane minus source hyperplane.
    Handles 3+ classes (one estimator per class) and the 2-class case (one estimator total)."""
    classes = list(ovr_model.classes_)
    if len(ovr_model.estimators_) == 1:                      # 2-class problem
        w = ovr_model.estimators_[0].coef_[0].copy()
        b = ovr_model.estimators_[0].intercept_[0]
        pos = classes[1]                                     # the single estimator scores this class
        sign_dest   = 1.0 if dest_class   == pos else -1.0
        sign_source = 1.0 if source_class == pos else -1.0
        return (sign_dest - sign_source) * w, (sign_dest - sign_source) * b
    i_dest   = classes.index(dest_class)                     # 3+ classes
    i_source = classes.index(source_class)
    w_dest,   b_dest   = ovr_model.estimators_[i_dest].coef_[0].copy(),   ovr_model.estimators_[i_dest].intercept_[0]
    w_source, b_source = ovr_model.estimators_[i_source].coef_[0].copy(), ovr_model.estimators_[i_source].intercept_[0]
    return w_dest - w_source, b_dest - b_source

w_diff, b_diff = dest_minus_source(baseline_model, DEST_CLASS_ENC, SOURCE_CLASS_ENC)

# f_ds(x) > 0 means DEST wins over SOURCE, < 0 means SOURCE wins. This is the "which side of the line" score.
def f_ds(points):
    return points @ w_diff + b_diff

print("Boundary built. ||w_diff|| =", round(float(np.linalg.norm(w_diff)), 4))
```

## Cell 6: Pick the target automatically

The target is the SOURCE point sitting closest to the DEST boundary, which is the cheapest one to flip. ==Read the printed margin: near 0 = easy and stealthy, very negative = deep in its class and hard.==

```python
margins = f_ds(X_train)                              # each train point's distance from the DEST/SOURCE line
baseline_pred_train = baseline_model.predict(X_train)

# candidates: truly SOURCE and currently predicted SOURCE (so a flip actually means something)
candidate_mask = (y_train == SOURCE_CLASS_ENC) & (baseline_pred_train == SOURCE_CLASS_ENC)
candidate_indices = np.where(candidate_mask)[0]
if len(candidate_indices) == 0:
    raise ValueError("No correctly-classified SOURCE points found. Check SOURCE_CLASS / DEST_CLASS.")

# closest to the boundary = the LARGEST (least negative) margin among SOURCE points
target_index = candidate_indices[np.argmax(margins[candidate_indices])]
X_target = X_train[target_index]

print(f"Target training index: {target_index}")
print(f"  true class (encoded):    {y_train[target_index]}   (SOURCE = {SOURCE_CLASS_ENC})")
print(f"  baseline prediction:     {baseline_model.predict(X_target.reshape(1, -1))[0]}")
print(f"  margin f_ds (want < 0):  {margins[target_index]:.4f}")
print("  Near 0 = easy to flip. Very negative = far from the boundary = hard.")

# TO TARGET A SPECIFIC ROW INSTEAD, comment out the block above and set:
# target_index = 373
# X_target = X_train[target_index]
```

## Cell 7: Find the DEST-class neighbours

The scikit-learn `NearestNeighbors` search, exactly like Section 12, but on the DEST class (the class you are allowed to move while keeping its label plausible). **Remember the gotcha: `kneighbors` returns positions inside the DEST-only array, so you map them back to real training indices.**

```python
dest_indices = np.where(y_train == DEST_CLASS_ENC)[0]
X_dest = X_train[dest_indices]

k = min(N_NEIGHBORS_TO_PERTURB, len(X_dest))
nn_finder = NearestNeighbors(n_neighbors=k)
nn_finder.fit(X_dest)

distances, relative_idx = nn_finder.kneighbors(X_target.reshape(1, -1))
neighbor_indices = dest_indices[relative_idx.flatten()]   # map back to real training indices

print(f"Perturbing {k} DEST-class neighbours (training indices): {neighbor_indices}")
print(f"Their distances to the target: {distances.flatten()}")
```

## Cell 8: Build the perturbation vector

Push straight across the boundary, opposite the boundary normal, scaled to `EPSILON_CROSS`. Identical logic to Section 12.

```python
push_direction = -w_diff
unit_push_direction = push_direction / np.linalg.norm(push_direction)
perturbation_vector = EPSILON_CROSS * unit_push_direction

print("Unit push direction:", unit_push_direction)
print("Perturbation vector (delta):", perturbation_vector)
```

## Cell 9: Apply the poison (features move, labels do not)

```python
X_train_poisoned = X_train.copy()
y_train_poisoned = y_train.copy()      # labels are copied and NEVER changed

print("Neighbour margins (ideally before > 0, after < 0 = crossed the line):")
for idx in neighbor_indices:
    before = f_ds(X_train[idx].reshape(1, -1))[0]
    X_train_poisoned[idx] = X_train[idx] + perturbation_vector
    after = f_ds(X_train_poisoned[idx].reshape(1, -1))[0]
    print(f"  idx {idx}: {before:+.3f} -> {after:+.3f}   crossed={before > 0 and after < 0}")

print(f"\nFeature rows changed: {len(neighbor_indices)} | Labels changed:",
      int(np.sum(y_train != y_train_poisoned)), "(must be 0 for a clean-label attack)")
```

Do not worry if a neighbour shows `crossed=False`: some of the nearest DEST points are already borderline, and what matters is the retrained result in the next cell, not that every single decoy crosses. **The one number that must read 0 is "Labels changed."**

## Cell 10: Retrain and evaluate

```python
poisoned_model = OneVsRestClassifier(
    LogisticRegression(random_state=SEED, C=1.0, solver="liblinear")   # SAME settings as baseline
)
poisoned_model.fit(X_train_poisoned, y_train_poisoned)

target_pred = poisoned_model.predict(X_target.reshape(1, -1))[0]
attack_successful = (target_pred == DEST_CLASS_ENC)

print("Target point:")
print(f"  true class:            {y_train[target_index]}")
print(f"  baseline prediction:   {baseline_model.predict(X_target.reshape(1, -1))[0]}")
print(f"  poisoned prediction:   {target_pred}   (want {DEST_CLASS_ENC})")
print(f"  ATTACK SUCCESSFUL:     {attack_successful}")

# how confident the poisoned model now is that the target is DEST (should have risen toward/over 0.5)
proba_dest = poisoned_model.predict_proba(X_target.reshape(1, -1))[0][list(poisoned_model.classes_).index(DEST_CLASS_ENC)]
print(f"  poisoned P(DEST):      {proba_dest:.3f}")

poisoned_accuracy = accuracy_score(y_test, poisoned_model.predict(X_test))
print("\nOverall accuracy on the clean test set:")
print(f"  baseline: {baseline_accuracy:.4f}   poisoned: {poisoned_accuracy:.4f}   drop: {baseline_accuracy - poisoned_accuracy:.4f}")
print("  A tiny drop = the attack stayed stealthy.")

print("\nClassification report (poisoned model on clean test set):")
print(classification_report(y_test, poisoned_model.predict(X_test)))
```

**Success looks like this pair: poisoned prediction equals DEST, and the accuracy drop is tiny.** That is the whole signature of a targeted clean-label attack.

## Cell 11: Optional auto-tune (run only if it did not flip)

If Cell 10 said `ATTACK SUCCESSFUL: False`, run this. It repeats the attack with stronger settings until the target flips, then promotes that successful run so Cell 12 plots it. **Prefer the smallest setting that works: a very large epsilon means the moved points stop being plausible DEST points, so the attack is no longer truly "clean."**

```python
def try_attack(epsilon, n_neighbors):
    k = min(n_neighbors, len(X_dest))
    nn = NearestNeighbors(n_neighbors=k).fit(X_dest)
    _, rel = nn.kneighbors(X_target.reshape(1, -1))
    nbr = dest_indices[rel.flatten()]
    pert = epsilon * (-w_diff) / np.linalg.norm(w_diff)
    Xp = X_train.copy()
    for i in nbr:
        Xp[i] = Xp[i] + pert
    m = OneVsRestClassifier(LogisticRegression(random_state=SEED, C=1.0, solver="liblinear")).fit(Xp, y_train)
    return m.predict(X_target.reshape(1, -1))[0], accuracy_score(y_test, m.predict(X_test)), nbr, Xp, m

if not attack_successful:
    found = False
    for epsilon in [EPSILON_CROSS, 0.5, 0.75, 1.0, 1.5, 2.0]:
        for n_nb in [N_NEIGHBORS_TO_PERTURB, 10, 20, 40]:
            pred, acc, nbr, Xp, m = try_attack(epsilon, n_nb)
            if pred == DEST_CLASS_ENC:
                print(f"FLIPPED at EPSILON_CROSS={epsilon}, N_NEIGHBORS_TO_PERTURB={n_nb} "
                      f"| accuracy {acc:.4f} (drop {baseline_accuracy - acc:.4f})")
                # promote this run so the plot and later cells use it
                neighbor_indices  = nbr
                X_train_poisoned  = Xp
                y_train_poisoned  = y_train.copy()
                poisoned_model    = m
                target_pred       = pred
                attack_successful = True
                found = True
                break
        if found:
            break
    if not found:
        print("Did not flip even at epsilon=2.0. This target/dataset resists a clean-label attack.")
        print("Try a target with a margin closer to 0, or accept that this dataset is a poor fit.")
else:
    print("Target already flipped in Cell 10, no tuning needed.")
```

## Cell 12: Visualize baseline vs poisoned

Real 2D plot if you have exactly two features, otherwise a PCA "shadow." **On the PCA version, trust the numbers from Cell 10 over the picture: PCA throws away dimensions, so points (especially the target) can land in visually odd spots.**

```python
target_pred_now = poisoned_model.predict(X_target.reshape(1, -1))[0]
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

if X_train.shape[1] == 2:
    def paint(ax, X_show, y_show, model, title):
        x1 = np.linspace(X_show[:, 0].min() - 1, X_show[:, 0].max() + 1, 200)
        x2 = np.linspace(X_show[:, 1].min() - 1, X_show[:, 1].max() + 1, 200)
        xx, yy = np.meshgrid(x1, x2)
        Z = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
        ax.contourf(xx, yy, Z, alpha=0.25, cmap="viridis")
        ax.scatter(X_show[:, 0], X_show[:, 1], c=y_show, s=10, cmap="viridis")
        ax.scatter(X_target[0], X_target[1], marker="P", s=260, c="white",
                   edgecolors="black", linewidths=1.6, label="target")
        ax.scatter(X_train_poisoned[neighbor_indices, 0], X_train_poisoned[neighbor_indices, 1],
                   marker="o", s=120, facecolors="none", edgecolors="magenta", linewidths=2, label="perturbed")
        ax.set_title(title); ax.legend(loc="best", fontsize=8)
    paint(axes[0], X_train, y_train, baseline_model, "Baseline (clean)")
    paint(axes[1], X_train_poisoned, y_train_poisoned, poisoned_model, f"Poisoned (target pred: {target_pred_now})")
else:
    pca = PCA(n_components=2, random_state=SEED).fit(X_train)
    def paint(ax, X_show, y_show, title):
        P  = pca.transform(X_show)
        Pt = pca.transform(X_target.reshape(1, -1))[0]
        Pn = pca.transform(X_train_poisoned[neighbor_indices])
        proxy = OneVsRestClassifier(LogisticRegression(random_state=SEED, C=1.0, solver="liblinear")).fit(P, y_show)
        x1 = np.linspace(P[:, 0].min() - 1, P[:, 0].max() + 1, 200)
        x2 = np.linspace(P[:, 1].min() - 1, P[:, 1].max() + 1, 200)
        xx, yy = np.meshgrid(x1, x2)
        Z = proxy.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
        ax.contourf(xx, yy, Z, alpha=0.25, cmap="viridis")
        ax.scatter(P[:, 0], P[:, 1], c=y_show, s=10, cmap="viridis")
        ax.scatter(Pt[0], Pt[1], marker="P", s=260, c="white",
                   edgecolors="black", linewidths=1.6, label="target")
        ax.scatter(Pn[:, 0], Pn[:, 1], marker="o", s=120, facecolors="none",
                   edgecolors="magenta", linewidths=2, label="perturbed")
        ax.set_title(title + " (PCA shadow)"); ax.legend(loc="best", fontsize=8)
    paint(axes[0], X_train, y_train, "Baseline (clean)")
    paint(axes[1], X_train_poisoned, y_train_poisoned, f"Poisoned (target pred: {target_pred_now})")

plt.tight_layout()
plt.show()
```

## What to change for a new dataset

|You want to...|Change this|
|---|---|
|Point at your dataset|`CSV_PATH`, `LABEL_COLUMN` in Cell 0|
|Choose the attack|`SOURCE_CLASS` (true class to hide), `DEST_CLASS` (class to fake), Cell 0|
|Attack a specific row, not the auto-pick|uncomment the `target_index = ...` lines at the bottom of Cell 6|
|Push harder because it did not flip|raise `EPSILON_CROSS` and/or `N_NEIGHBORS_TO_PERTURB` in Cell 0, or just run Cell 11|
|Keep it stealthy|use the smallest epsilon that flips, and watch the accuracy drop stays small|

Quick reminders baked in: ==clean-label = features move, labels never do; the target is chosen near the boundary because that is cheapest; success = target flips while overall accuracy barely moves; and it needs model internals (`w_diff`) to aim, which is what makes it stealthier but harder than label flipping.==

Want me to keep this in the same running notes format and move on to Section 14 (Trojan attacks) whenever you
