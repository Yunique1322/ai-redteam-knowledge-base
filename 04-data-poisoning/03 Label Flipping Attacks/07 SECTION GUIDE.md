
# Data Poisoning: Label-Flipping Playbook for a New Dataset

Run the cells top to bottom in one kernel. **If you restart the kernel, re-run from Cell 1, a NameError almost always means an earlier cell was not run.** Swap only the values marked `# CHANGE:`.

## Cell 1: Imports and setup

Everything you need to load, prep, train, attack, and evaluate. Run once.

```
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

SEED = 1337
np.random.seed(SEED)

print("Setup complete.")
```

## Cell 2: Load the CSV and inspect it

Look before you touch anything: how many rows/columns, what the columns are called, and how balanced the classes are. **The class counts tell you the class values you will use later for TARGET_CLASS.**

```
# CHANGE: path to your CSV, and the name of the label column
CSV_PATH = "<CSV_PATH>"
LABEL_COLUMN = "<LABEL_COLUMN>"

df = pd.read_csv(CSV_PATH)
print("Shape (rows, cols):", df.shape)
print("\nColumns:", df.columns.tolist())
print("\nFirst rows:\n", df.head())
print("\nClass balance:\n", df[LABEL_COLUMN].value_counts())
```

## Cell 3: Prepare the data

Split the answer column off from the features, make everything numeric, split into train/test, and scale. Three real-data fixes the toy data never needed are baked in here: **one-hot encoding for text feature columns, LabelEncoder for a text label, and converting to NumPy so the flip functions work.**

```
# CHANGE: nothing here unless a step does not apply (see notes below)
X_df = df.drop(columns=[LABEL_COLUMN])   # features = everything except the label
y_raw = df[LABEL_COLUMN]                  # labels = the answer column

# Turn any text/categorical FEATURE columns into numbers (safe even if there are none)
X_df = pd.get_dummies(X_df)

# Turn a text LABEL into integers 0,1,2,... (safe even if it is already numeric)
le = LabelEncoder()
y = le.fit_transform(y_raw)
print("Label mapping:", dict(zip(le.classes_, np.unique(y))))  # e.g. {'benign':0,'malignant':1}

X = X_df.to_numpy(dtype=float)

# Split: 70% train, 30% test. stratify keeps class ratio the same in both halves.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=SEED, stratify=y
)

# Scale features (fit on TRAIN only, then apply to both).
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print("Train:", X_train.shape, " Test:", X_test.shape, " Features:", X_train.shape[1])
```

Two one-line notes: scaling helps Logistic Regression and is harmless here, but **skip scaling for tree-based models (RandomForest, XGBoost), they don't need it**. And read the printed `Label mapping`, those integers are what you pass as `TARGET_CLASS` / `NEW_CLASS` later.

## Cell 4: Baseline model (your control)

Train on clean data so you have a "healthy" number to compare against. ==`max_iter=1000` because real data often needs more iterations to converge than the toy data did.==

```
# CHANGE: nothing
baseline_model = LogisticRegression(max_iter=1000, random_state=SEED)
baseline_model.fit(X_train, y_train)

baseline_accuracy = accuracy_score(y_test, baseline_model.predict(X_test))
print(f"Baseline accuracy: {baseline_accuracy:.4f}")
```

## Cell 5: Random label flipping

Flip a random fraction of all labels to degrade the model overall. Define the function, then apply, retrain, and score. **The invert (`0<->1`) assumes a binary problem; for more than two classes use the targeted function in Cell 6 instead.**

```
# CHANGE: nothing in the function
def flip_labels(y, poison_percentage, seed=SEED):
    if not 0 <= poison_percentage <= 1:
        raise ValueError("poison_percentage must be between 0 and 1.")
    n_samples = len(y)
    n_to_flip = int(n_samples * poison_percentage)
    if n_to_flip == 0:
        print("Warning: percentage too low, nothing flipped.")
        return y.copy(), np.array([], dtype=int)
    rng = np.random.default_rng(seed)
    flipped_indices = rng.choice(n_samples, size=n_to_flip, replace=False)
    y_poisoned = y.copy()
    y_poisoned[flipped_indices] = np.where(y_poisoned[flipped_indices] == 0, 1, 0)
    print(f"Flipped {n_to_flip} labels ({poison_percentage*100:.1f}%).")
    return y_poisoned, flipped_indices

# CHANGE: how much of the training data to poison (0.0 - 1.0)
POISON_PERCENTAGE = 0.30

y_train_rand, idx_rand = flip_labels(y_train, POISON_PERCENTAGE)

# Retrain on POISONED labels, but always evaluate on the CLEAN y_test
model_rand = LogisticRegression(max_iter=1000, random_state=SEED)
model_rand.fit(X_train, y_train_rand)
acc_rand = accuracy_score(y_test, model_rand.predict(X_test))
print(f"Random-poisoned accuracy: {acc_rand:.4f}  (baseline {baseline_accuracy:.4f})")
```

## Cell 6: Targeted label flipping

Flip only one chosen class into another to make the model misclassify that specific class. **Read the `Label mapping` from Cell 3 to pick the integer values for TARGET_CLASS and NEW_CLASS.**

```
# CHANGE: nothing in the function
def targeted_flip_labels(y, poison_percentage, target_class, new_class, seed=SEED):
    if not 0 <= poison_percentage <= 1:
        raise ValueError("poison_percentage must be between 0 and 1.")
    if target_class == new_class:
        raise ValueError("target_class and new_class cannot be the same.")
    unique_labels = np.unique(y)
    if target_class not in unique_labels:
        raise ValueError(f"target_class {target_class} not in y.")
    if new_class not in unique_labels:
        raise ValueError(f"new_class {new_class} not in y.")
    target_indices = np.where(y == target_class)[0]
    n_target = len(target_indices)
    if n_target == 0:
        print("Warning: no samples of target_class.")
        return y.copy(), np.array([], dtype=int)
    n_to_flip = int(n_target * poison_percentage)
    if n_to_flip == 0:
        print("Warning: percentage too low for the target class.")
        return y.copy(), np.array([], dtype=int)
    rng = np.random.default_rng(seed)
    pick = rng.choice(n_target, size=n_to_flip, replace=False)
    flipped_indices = target_indices[pick]
    y_poisoned = y.copy()
    y_poisoned[flipped_indices] = new_class
    print(f"Flipped {len(flipped_indices)} of {n_target} class-{target_class} labels to {new_class}.")
    return y_poisoned, flipped_indices

# CHANGE: which class to attack, which class to turn it into, and how much
TARGET_CLASS = 1
NEW_CLASS = 0
POISON_PERCENTAGE_TARGETED = 0.40

y_train_tgt, idx_tgt = targeted_flip_labels(
    y_train, POISON_PERCENTAGE_TARGETED, TARGET_CLASS, NEW_CLASS
)

model_tgt = LogisticRegression(max_iter=1000, random_state=SEED)
model_tgt.fit(X_train, y_train_tgt)
acc_tgt = accuracy_score(y_test, model_tgt.predict(X_test))
print(f"Targeted-poisoned accuracy: {acc_tgt:.4f}  (baseline {baseline_accuracy:.4f})")
```

## Cell 7: Evaluation

Overall accuracy hides a targeted attack, so read the per-class report and the confusion matrix. **This is where a targeted hit shows up: the target class's recall drops while the other class stays high.**

```
# CHANGE: set `model` to whichever poisoned model you want to inspect
model = model_tgt
y_pred = model.predict(X_test)

print("Accuracy:", round(accuracy_score(y_test, y_pred), 4), " Baseline:", round(baseline_accuracy, 4))
print("\nClassification report:\n", classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.show()
```

How to read the three numbers, one line each:

- **Recall (a class):** of all the real members of that class, how many the model caught. A targeted attack drives this down for the target class.
- **Precision (a class):** when the model says that class, how often it is right.
- **Confusion matrix:** rows are the truth, columns the guess, so the diagonal is correct and **off-diagonal cells are the mistakes (the target-class row is where you see the damage)**.

## Optional: decision-boundary plot (only if exactly 2 features)

The HTB-style boundary plot only makes sense in 2D. This block checks first and skips itself otherwise, so it is safe to leave in.

```
# CHANGE: set `model` to the one you want to visualize
model = model_tgt

if X_train.shape[1] == 2:
    h = 0.02
    x_min, x_max = X_train[:, 0].min() - 1, X_train[:, 0].max() + 1
    y_min, y_max = X_train[:, 1].min() - 1, X_train[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    plt.figure(figsize=(10, 6))
    plt.contourf(xx, yy, Z, alpha=0.3, cmap="coolwarm")
    plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap="coolwarm", edgecolors="k", s=30)
    plt.title("Decision Boundary")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.show()
else:
    print(f"Skipping 2D plot: dataset has {X_train.shape[1]} features, not 2.")
```

## For multiple features (uses pca to make it 2)

```
from sklearn.decomposition import PCA

# CHANGE: which labels to visualize - y_train (clean), y_train_rand, or y_train_tgt
labels_to_plot = y_train

# 1. Squash all features down to 2 dimensions so we can draw them
pca = PCA(n_components=2, random_state=SEED)
X_train_2d = pca.fit_transform(X_train)
print("Variance kept by the 2 components:", round(pca.explained_variance_ratio_.sum(), 3))

# 2. Train a small helper model in that 2D space (a VISUAL PROXY, not your real model)
viz_model = LogisticRegression(max_iter=1000, random_state=SEED)
viz_model.fit(X_train_2d, labels_to_plot)

# 3. Mesh + shaded boundary, same idea as the 2-feature version
h = 0.05
x_min, x_max = X_train_2d[:, 0].min() - 1, X_train_2d[:, 0].max() + 1
y_min, y_max = X_train_2d[:, 1].min() - 1, X_train_2d[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
Z = viz_model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

plt.figure(figsize=(10, 6))
plt.contourf(xx, yy, Z, alpha=0.3, cmap="coolwarm")
plt.scatter(
    X_train_2d[:, 0], X_train_2d[:, 1],
    c=labels_to_plot, cmap="coolwarm", edgecolors="k", s=30,
)
plt.title("Decision Boundary (2D PCA projection - approximation)")
plt.xlabel("PCA component 1")
plt.ylabel("PCA component 2")
plt.show()
```
![3. Mesh + shaded boundary, same idea as the 2-feature version](/assets/Pasted%20image%2020260919152903.png)

---
### OPTIONAL BUT RECOMMENDED Baseline vs poisoned comparison (works for any feature count)

Same honesty caveat as before: **the boundaries are proxy models drawn in PCA space, not your real N-feature model.** It's for eyeballing the shift. Needs `from sklearn.decomposition import PCA`.

```
from sklearn.decomposition import PCA

def plot_baseline_vs_poisoned(X_train, y_clean, y_poisoned, flipped_indices=None, seed=SEED):
    # Project all features down to 2D so we can draw them
    pca = PCA(n_components=2, random_state=seed)
    X2d = pca.fit_transform(X_train)

    # A proxy 2D model for each label set (clean vs poisoned)
    m_clean = LogisticRegression(max_iter=1000, random_state=seed).fit(X2d, y_clean)
    m_pois = LogisticRegression(max_iter=1000, random_state=seed).fit(X2d, y_poisoned)

    # Shared mesh so both panels line up
    h = 0.05
    x_min, x_max = X2d[:, 0].min() - 1, X2d[:, 0].max() + 1
    y_min, y_max = X2d[:, 1].min() - 1, X2d[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    grid = np.c_[xx.ravel(), yy.ravel()]
    Z_clean = m_clean.predict(grid).reshape(xx.shape)
    Z_pois = m_pois.predict(grid).reshape(xx.shape)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Left panel: baseline (clean labels)
    axes[0].contourf(xx, yy, Z_clean, alpha=0.3, cmap="coolwarm")
    axes[0].scatter(X2d[:, 0], X2d[:, 1], c=y_clean, cmap="coolwarm", edgecolors="k", s=25)
    axes[0].set_title("Baseline (clean labels)")

    # Right panel: poisoned labels, with flipped points circled
    axes[1].contourf(xx, yy, Z_pois, alpha=0.3, cmap="coolwarm")
    axes[1].scatter(X2d[:, 0], X2d[:, 1], c=y_poisoned, cmap="coolwarm", edgecolors="k", s=25)
    if flipped_indices is not None and len(flipped_indices) > 0:
        axes[1].scatter(
            X2d[flipped_indices, 0], X2d[flipped_indices, 1],
            facecolors="none", edgecolors="red", linewidths=1.5, marker="X", s=90,
            label="Flipped",
        )
        axes[1].legend()
    axes[1].set_title("Poisoned labels")

    for ax in axes:
        ax.set_xlabel("PCA 1")
        ax.set_ylabel("PCA 2")
    plt.suptitle("Baseline vs Poisoned")
    plt.show()
    print("Variance kept by 2 PCs:", round(pca.explained_variance_ratio_.sum(), 3))
```

Call it after you've made a poisoned label set. For the targeted attack (pass the flipped indices so they get circled):

```
plot_baseline_vs_poisoned(X_train, y_train, y_train_tgt, flipped_indices=idx_tgt)
```

Or for the random attack:

```
plot_baseline_vs_poisoned(X_train, y_train, y_train_rand, flipped_indices=idx_rand)
```

W

## What to change for a new dataset

|Placeholder|Where|What it is|Sensible default|
|---|---|---|---|
|`<CSV_PATH>`|Cell 2|Path to the dataset file|`"data.csv"`|
|`<LABEL_COLUMN>`|Cell 2|Column holding the answer/class|check `df.columns`|
|`POISON_PERCENTAGE`|Cell 5|Fraction of all labels to flip|`0.30`|
|`TARGET_CLASS`|Cell 6|Encoded class to attack|`1` (check `Label mapping`)|
|`NEW_CLASS`|Cell 6|Class to flip the target into|`0`|
|`POISON_PERCENTAGE_TARGETED`|Cell 6|Fraction of the target class to flip|`0.40`|
|`model`|Cell 7 / plot|Which poisoned model to inspect|`model_tgt`|

Two rules that save you every time: **run cells in order in one kernel**, and **train on poisoned labels but always score against the clean `y_test`**.

---

