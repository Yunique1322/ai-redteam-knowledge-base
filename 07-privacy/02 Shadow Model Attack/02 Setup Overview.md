# Setup Overview

## The idea in one breath

This section sets the stage. It installs the tools, loads the **Adult Census** dataset (predict whether income is above or below $50K), splits it into **three separate pools** (target members, shadow data, non-members for testing), and trains a **deliberately overfit target model** (no dropout, 100 fixed epochs). The key number is the **overfitting gap**: in HTB's figure, **93.6% accuracy on members vs 82.5% on non-members, an 11.1-point gap**. That gap is what the attack will exploit.

> **Heads-up:** No mangled formulas this time. There are three things that don't add up, and one hidden bug, all covered below:
> 
> - The course prints **three different gaps**: 5.56% (sample output), 11.1% (bar chart) and 7.1% (closing prose). Trust the **figure** (93.6% / 82.5% / 11.1%). It matches where the training curves end.
> - I installed `htb_ai_library` and read its source. A few prose claims are wrong: the data comes from **UCI, not OpenML**, it is **not pre-scaled**, it has **14 features**, and the non-member pool has **12,211** rows, not 12,210.
> - **Library bug:** `train_with_early_stopping` claims to "restore the best model weights", but it doesn't. See Part 3. It doesn't affect this section (the target uses `train_fixed_epochs`), but it will affect the shadow models next section.
> - I couldn't run the course code myself: PyTorch isn't installable in my sandbox and isn't on your PC's shell. Everything here is checked against the library source code, not executed.

---

## Part 1: Installing Dependencies

```bash
pip install torch torchvision numpy scikit-learn matplotlib tqdm safetensors opacus flask
```

|Package|Purpose|Note|
|---|---|---|
|torch, torchvision|PyTorch deep learning framework and vision utilities|Same as the evasion modules|
|numpy|Numerical computing and array operations||
|scikit-learn|Data preprocessing, metrics, and train/test splitting|`StandardScaler`, `train_test_split`, metrics|
|matplotlib|Visualization and plotting||
|tqdm|Progress bars for training loops||
|**safetensors**|Secure model weight serialization for challenge submissions|Callback to your **Pickles** module: a `.pt`/pickle file can run code when loaded, while safetensors only stores raw tensors. That's why challenges accept it.|
|**opacus**|Differential privacy for PyTorch (**DP-SGD** implementation)|Used in the DP-SGD sections|
|flask|Web API framework for a local evaluator|A local evaluator API is a common setup|

```bash
pip install --upgrade git+https://github.com/PandaSt0rm/htb-ai-library
```

This installs or updates the HTB library straight from GitHub. Use `--upgrade` so you get the version that includes the privacy functions (`MLP`, `AttackModel`, `load_adult_census`, and so on).

---

## Part 2: Imports and Configuration

### Step 1: Standard imports

```python
import os
import json
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
```

|Code|What it does|
|---|---|
|`os`, `json`|Folders/paths and saving results as JSON|
|`numpy as np`, `torch`|Arrays and the neural network framework|
|`torch.nn.functional as F`|Functions like `F.softmax`|
|`train_test_split`|Random splitting of arrays|
|`StandardScaler`|Rescales each feature to mean 0, standard deviation 1 (Part 6)|
|`accuracy_score, precision_score, recall_score, f1_score, classification_report`|Attack metrics, used in later sections to score the attack model|

### Step 2: HTB library imports

```python
from htb_ai_library import (
    set_reproducibility, use_htb_style,
    MLP, AttackModel,
    load_adult_census,
    train_fixed_epochs, train_with_early_stopping, evaluate_model,
    get_model_predictions, prepare_attack_data, create_dataloader,
    plot_training_history, plot_overfitting_gap, plot_confidence_distributions,
    plot_shadow_confidence_distributions, plot_attack_roc_curve, plot_precision_recall_curve,
    plot_attack_accuracy_comparison, analyze_attack_decision_boundary, plot_decision_boundary,
)
```

|Group|Names|Role|
|---|---|---|
|Setup|`set_reproducibility`, `use_htb_style`|Fixed seeds; HTB dark plot theme|
|Models|`MLP`, `AttackModel`|Target/shadow network; member-or-not network|
|Data|`load_adult_census`|Downloads and splits the dataset|
|Training|`train_fixed_epochs`, `train_with_early_stopping`, `evaluate_model`|Overfit training; regularized training; accuracy + probabilities|
|Attack plumbing|`get_model_predictions`, `prepare_attack_data`, `create_dataloader`|Get probabilities; build `[probs, one-hot]` rows; wrap arrays for PyTorch|
|Plots|the `plot_*` and `analyze_*` functions|Explained in Part 3|

### Step 3: Seed and device

```python
RANDOM_SEED = 1337
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
set_reproducibility(RANDOM_SEED)
use_htb_style()
```

| Code | What it does |
| ---------------------------------- | ------------------------------------------------------------------------------------ |
| `RANDOM_SEED = 1337` | One seed for everything, so the splits, weights and shuffles are identical every run |
| `DEVICE = ...` | Use the GPU if there is one, otherwise the CPU |
| `set_reproducibility(RANDOM_SEED)` | Seeds the PyTorch, NumPy and CUDA random generators |
| `use_htb_style()` | Sets the plot style |

### Step 4: Output folders and dataset config

```python
OUTPUT_DIR = "output"
MODEL_DIR = f"{OUTPUT_DIR}/models"
FIGS_DIR = "figs"
FIG_PREFIX = "Introduction_"
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(FIGS_DIR, exist_ok=True)

DATASET_CONFIG = {
    "num_classes": 2,
}
```

|Code|What it does|
|---|---|
|`OUTPUT_DIR`, `MODEL_DIR`|Where trained models get saved (`output/models`)|
|`FIGS_DIR`, `FIG_PREFIX`|Where plots get saved, with filenames starting `Introduction_`|
|`os.makedirs(..., exist_ok=True)`|Create the folder; no error if it already exists|
|`DATASET_CONFIG = {"num_classes": 2}`|Binary task (income ≤50K vs >50K). This is also why the attack input is **4** numbers: 2 probabilities + 2 one-hot slots|

---

## Part 3: Library Components (what each piece does, checked against the source)

|Component|What the course says|What the library actually does (I read the source)|
|---|---|---|
|`set_reproducibility(seed)`|Seeds PyTorch/NumPy/CUDA|✓|
|**`MLP`**|Target **and** shadow model; set hidden sizes, dropout, classes|✓ Each hidden layer is Linear -> ReLU -> Dropout. `forward` returns **raw logits**; `predict_proba` = softmax(logits)|
|**`AttackModel`**|Input `[prob_class_0, prob_class_1, label_0, label_1]`|✓ Outputs 2 logits: `[non_member, member]`|
|`load_adult_census()`|"fetches from OpenML", "scaled numpy arrays"|✗ Downloads from the **UCI repository** (`adult.data` + `adult.test`, cached in `./data`). Categories become **integer codes** (LabelEncoder), **not scaled**. **14 features**|
|`train_fixed_epochs()`|Trains N epochs, no early stopping, returns history|✓ Signature `(model, train_loader, test_loader, *, device, epochs, learning_rate)`. `device` must be passed **by name**. History keys: `train_losses`, `val_losses`, `train_accs`, `val_accs`|
|**`train_with_early_stopping()`**|"restores the best model weights"|⚠ **Bug, see the note below.** It does stop early, but the restore doesn't work|
|`evaluate_model(model, loader, device)`|Accuracy, predictions, probabilities|✓ Returns `(accuracy 0-1, predictions, probabilities)` and uses `model.eval()` + `torch.no_grad()`|
|`get_model_predictions(model, X, device)`|Probabilities from a NumPy array|✓ Also `eval()` + `no_grad()`, default batch 256, returns shape `(N, 2)`|
|`prepare_attack_data(pred_in, pred_out, labels_in, labels_out)`|Builds `[predictions, one_hot]` rows + 1/0 labels|✓ Members get label 1, non-members 0 (exactly what your Section 2 Cell 2 built by hand)|
|`create_dataloader(X, y, batch_size, shuffle=True)`|Wraps arrays as a DataLoader|✓ Converts X to float32, y to long|
|`plot_training_history`, `plot_overfitting_gap`|Loss/accuracy curves; train-vs-test bar|The two figures in this section|
|`plot_confidence_distributions`, `plot_shadow_confidence_distributions`|Histograms of confidence, members vs non-members|Later sections|
|`plot_attack_roc_curve`, `plot_precision_recall_curve`, `plot_attack_accuracy_comparison`|ROC + **AUC** (above 0.5 = better than random), precision-recall (useful when classes are imbalanced), metrics vs the 0.5 baseline|Later sections. AUC is the threshold-free metric from your Section 2 Cell 3|
|`analyze_attack_decision_boundary`, `plot_decision_boundary`|What confidence threshold the attack learned, per class|Later sections|

**Remember the evasion rule:** attacks there needed `model.eval()` but **not** `torch.no_grad()`, because you needed input gradients. Here you only read probabilities, so **`eval()` + `no_grad()` is correct**, and `evaluate_model` / `get_model_predictions` already do both. The raw method `MLP.predict_proba` does **not** call `eval()`. If you call it on a model with dropout (the shadows use 0.3), dropout randomly zeroes units and adds noise to the very confidence values the attack reads. **Use `get_model_predictions`, or call `.eval()` first.**

> **An honest note the course skips (1), library bug:** Inside `train_with_early_stopping` the "best weights" are saved with `model.state_dict().copy()`. That's a **shallow** copy: the saved tensors share memory with the live model weights, so every later optimizer step overwrites the "saved best" too. The final `load_state_dict(best_model_state)` then loads... the current weights. **The early stop happens, but you keep the weights from the stop epoch (best epoch + patience), not the best epoch.** The fix is `copy.deepcopy(model.state_dict())`. This doesn't touch the target model, but it does affect the shadow models and the attack model next section. My Cell 4 below proves the mechanism.

---

## Part 4: Configuration: Maximizing the Overfitting Gap

### Step 1: The target model (built to leak)

```python
TARGET_MODEL_CONFIG = {
    "hidden_layers": [256, 128],
    "dropout": 0.0, # No dropout to maximize overfitting
    "epochs": 100,
    "batch_size": 32,
    "learning_rate": 0.001,
}
```

### Step 2: The shadow models

```python
SHADOW_MODEL_CONFIG = {
    "num_shadow_models": 5,
    "hidden_layers": [128, 64],
    "dropout": 0.3,
    "epochs": 100,
    "batch_size": 64,
    "learning_rate": 0.001,
    "early_stopping_patience": 10,
    "shadow_data_size": 0.5,
}
```

### Step 3: The attack model

```python
ATTACK_MODEL_CONFIG = {
    "hidden_layers": [64, 32],
    "dropout": 0.2,
    "epochs": 100,
    "batch_size": 128,
    "learning_rate": 0.001,
    "early_stopping_patience": 15,
}
```

### Step 4: The three configs side by side

|Setting|Target (victim)|Shadow ×5|Attack model|Why|
|---|---|---|---|---|
|`hidden_layers`|[256, 128]|[128, 64]|[64, 32]|Target: big = memorizes more. Attack: input is only 4 numbers|
|`dropout`|**0.0**|0.3|0.2|Target: no regularization, on purpose. Attack: light, so it doesn't overfit to one shadow's quirks|
|`epochs`|100, **fixed**|up to 100|up to 100|Target trains far past the best point|
|Early stopping|**none**|patience 10|patience 15|Patience = epochs with no val-loss improvement before stopping. Attack gets longer patience so subtle patterns can emerge|
|`batch_size`|32|64|128|Bigger batches give smoother gradients for the simple attack input|
|`learning_rate`|0.001|0.001|0.001|Adam's default|
|`shadow_data_size`| - |0.5| - |Probably: each shadow trains on half of its data and holds out the other half as "out" (next section confirms)|

The course itself says that dropout 0.0 + fixed epochs + no early stopping **"in production would be mistakes"**. That's your remediation list in reverse: **add dropout, use early stopping, stop training at the best validation point**.

> **An honest note the course skips (2):** Section 2 said "match the target's architecture as closely as possible". This config does the opposite: the shadows are **smaller** ([128, 64] vs [256, 128]), **regularized** (dropout 0.3 vs 0.0), **early-stopped**, and use a different batch size. So the shadows overfit **less** than the target. The attack can still work, because members of an even-more-overfit target look even "more member-like". But it breaks the course's own rule, so expect the attack accuracy to be lower than a well-matched setup would give.

---

## Part 5: Data Splitting Strategy and Loading Data

### Step 1: The three disjoint pools

```shellsession
Total Dataset (48,842 samples)
├── Target Training (24,421) -> Members we try to identify
└── Holdout (24,421)
    ├── Shadow Training (12,210) -> Train shadow models
    └── Attack Evaluation (12,210) -> Non-members for final testing
```

|Pool|Size (real)|Role|Who "owns" it in the story|
|---|---|---|---|
|**Target training**|24,421|Members, the people we try to identify|The victim's secret training set|
|**Shadow training**|12,210|Trains the 5 shadows (the attacker's own same-population data)|Attacker|
|**Attack evaluation**|**12,211**|Known non-members, used to test the attack honestly|Grader|

**Disjoint pools prevent inflated results**: if a "non-member" were secretly also in the target's training data, the attack would get it "right" for the wrong reason.

### Step 2: Load the data

```python
print("Loading Adult Census dataset...")
X_target, y_target, X_shadow, y_shadow, X_attack_eval, y_attack_eval, num_features = load_adult_census(
    random_state=RANDOM_SEED
)

print(f"Dataset loaded: {num_features} features")
print(f" Target training (members): {len(X_target)} samples")
print(f" Shadow training: {len(X_shadow)} samples")
print(f" Attack evaluation (non-members): {len(X_attack_eval)} samples")
```

From the library source, you should see:

![Setup Overview](/assets/3%20Setup%20Overview-20261023153911717.png)

| Name | Shape | Meaning |
| -------------------------------- | --------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `X_target` | (24421, 14) | Member features (raw, unscaled) |
| `y_target` | (24421,) | 0 = income ≤50K, 1 = >50K |
| `X_shadow`, `y_shadow` | (12210, 14), (12210,) | Attacker's own data for shadows |
| `X_attack_eval`, `y_attack_eval` | (12211, 14), (12211,) | Non-members for testing |
| `num_features` | 14 | age, workclass, fnlwgt, education, education-num, marital-status, occupation, relationship, race, sex, capital-gain, capital-loss, hours-per-week, native-country |
| | | |

> **An honest note the course skips (3):**
> 
> - **Off-by-one:** `train_test_split(test_size=0.5)` on 24,421 rows rounds the test half **up**, so the non-member pool has 12,211, not 12,210.
> - **Not scaled:** the arrays are raw values (age ~38, fnlwgt ~190,000), so the `StandardScaler` step below is essential, not optional.
> - **Not one-hot:** categories are integer codes ("occupation = 7"), which is why there are only 14 features.
> - **Class imbalance:** only ~24% of people earn >50K, so a model that always says "≤50K" already scores ~76%. The 82.5% test accuracy is only ~6.5 points better than that.
> - **Watch next sections:** there are 24,421 members but only 12,211 non-members. If the attack is evaluated on all of them, a coin flip is **not** the right baseline (the balanced-baseline lesson). Balance the pool, as my Cell 3 does.

---

## Part 6: Training the Target Model

These are five separate course cells that run one after another. None of them splits a single loop or function, so I kept them separate.

### Step 1: Scale the features

```python
print("\n" + "=" * 60)
print("Training Target Model")
print("=" * 60)

scaler = StandardScaler()
X_target_norm = scaler.fit_transform(X_target)
X_attack_eval_norm = scaler.transform(X_attack_eval)
```

**Toy first** (your running normalization toy, μ = 0.2, σ = 0.4). Say one feature in the **training** data has mean μ = 0.2 and standard deviation σ = 0.4.

- `fit` **learns** μ = 0.2 and σ = 0.4 from the training data only.
- `transform` applies z = (x − μ) / σ:
    - Training value 0.6 -> (0.6 − 0.2) / 0.4 = **1.0**
    - Test value 1.0 -> (1.0 − 0.2) / 0.4 = **2.0**, using the **training** μ and σ
- If you wrongly called `fit_transform` on the test set too, it would learn new numbers (say μ = 0.5, σ = 0.5), and the same 1.0 would become (1.0 − 0.5) / 0.5 = **1.0**. The model would see the same kind of person encoded differently. **Fit on training data, then only transform everything else.**

|Symbol|Plain English|Toy|Real (age column)|
|---|---|---|---|
|x|Raw feature value|0.6 / 1.0|age 52|
|μ|Training-set mean of that feature|0.2|about 38.6|
|σ|Training-set standard deviation|0.4|about 13.7|
|z|Scaled value ("how many σ above average")|1.0 / 2.0|(52 − 38.6) / 13.7 ≈ 0.98|

**Read aloud:** "Subtract the training average, divide by the training spread."

Without scaling, `fnlwgt` (hundreds of thousands) would swamp `age` (tens) in the first layer's sums.

|Code|What it does|
|---|---|
|`print("=" * 60)`|A 60-character divider line|
|`scaler = StandardScaler()`|An empty scaler|
|`scaler.fit_transform(X_target)`|Learns μ, σ per column from the **members**, then scales them|
|`scaler.transform(X_attack_eval)`|Scales non-members with the **same** μ, σ|

### Step 2: DataLoaders

```python
train_loader = create_dataloader(X_target_norm, y_target, TARGET_MODEL_CONFIG['batch_size'])
test_loader = create_dataloader(X_attack_eval_norm, y_attack_eval,
                                TARGET_MODEL_CONFIG['batch_size'], shuffle=False)
```

|Code|What it does|
|---|---|
|`create_dataloader(X_target_norm, y_target, 32)`|Batches of 32 members, shuffled each epoch|
|`create_dataloader(X_attack_eval_norm, ..., shuffle=False)`|Batches of non-members in fixed order|

The course says "without a validation split", but `test_loader` **is** passed to training below as the validation set. That's fine: `train_fixed_epochs` only **measures** on it and never trains on it. So the "Val" curves in the figure are really **non-member curves**.

### Step 3: Create the target model

```python
target_model = MLP(
    input_size=num_features,
    hidden_layers=TARGET_MODEL_CONFIG['hidden_layers'],
    num_classes=DATASET_CONFIG['num_classes'],
    dropout=TARGET_MODEL_CONFIG['dropout']
)

print(f"Architecture: {num_features} -> {TARGET_MODEL_CONFIG['hidden_layers']} -> 2")
print(f"Training for {TARGET_MODEL_CONFIG['epochs']} epochs (no early stopping)")
```

Output (from the source):

![Setup Overview](/assets/3%20Setup%20Overview-20261023154003466.png)

|Code|What it does|
|---|---|
|`MLP(input_size=14, hidden_layers=[256,128], num_classes=2, dropout=0.0)`|14 inputs -> 256 -> 128 -> 2 logits, with ReLU between layers and no dropout|

|Layer|Shape|Weights + biases|
|---|---|---|
|Linear 1|14 -> 256|14×256 + 256 = 3,840|
|Linear 2|256 -> 128|256×128 + 128 = 32,896|
|Output|128 -> 2|128×2 + 2 = 258|
|**Total**||**36,994 parameters**|

That's 36,994 parameters for 24,421 training rows: more knobs than people, which leaves plenty of room to memorize.

### Step 4: Train for 100 fixed epochs

```python
history = train_fixed_epochs(
    target_model, train_loader, test_loader,
    device=DEVICE,
    epochs=TARGET_MODEL_CONFIG['epochs'],
    learning_rate=TARGET_MODEL_CONFIG['learning_rate']
)
```

|Code|What it does|
|---|---|
|`train_fixed_epochs(model, train_loader, test_loader, ...)`|100 epochs of Adam + cross-entropy; after each epoch, measures loss/accuracy on the non-members|
|`device=DEVICE`|Must be written **by name** (the function forces this)|
|`history`|Dict with 4 lists of 100 numbers each: `train_losses`, `val_losses`, `train_accs`, `val_accs`|

> **If this errors:** `TypeError: train_fixed_epochs() takes 3 positional arguments but 4 were given` means you passed `DEVICE` without `device=`. Fix: write `device=DEVICE`. Why: the library puts `*` before `device` in its signature, which forces keyword use.

### Step 5: Measure the overfitting gap

```python
train_acc, _, _ = evaluate_model(target_model, train_loader, DEVICE)
test_acc, _, _ = evaluate_model(target_model, test_loader, DEVICE)

print(f"\nTarget Model Performance:")
print(f" Training Accuracy: {train_acc:.4f}")
print(f" Test Accuracy: {test_acc:.4f}")
print(f" Overfitting Gap: {train_acc - test_acc:.4f}")

plot_overfitting_gap(train_acc, test_acc,
                     save_path=os.path.join(FIGS_DIR, f"{FIG_PREFIX}overfitting_gap.png"))
```

| Code | What it does |
| --------------------------------------- | ----------------------------------------------------------------------------------------------- |
| `train_acc, _, _ = evaluate_model(...)` | Keeps only the accuracy; `_` throws away predictions and probabilities |
| `train_acc - test_acc` | **Overfitting gap** = member accuracy − non-member accuracy (figure: 0.936 − 0.825 = **0.111**) |
| `plot_overfitting_gap(...)` | Draws the bar chart and saves it to `figs/Introduction_overfitting_gap.png` |

---

## Part 7: Model Architectures (`predict_proba`)

```python
def predict_proba(self, x):
    logits = self.forward(x)
    return F.softmax(logits, dim=1)
```

This is a **method inside the `MLP` class** shown for explanation. You don't run it as its own cell; it's already in the library (written there as one line, which does the same thing).

### Step 1: Toy: how a logit gap becomes confidence

For 2 classes, softmax turns logits [z0, z1] into probabilities. It raises e to each logit, then divides by the total:

- Logits [1, 3]: e¹ = 2.718 and e³ = 20.09, total 22.81 -> **[0.119, 0.881]**
- Logits [1, 5]: e¹ = 2.718 and e⁵ = 148.4, total 151.1 -> **[0.018, 0.982]**

A bigger gap between logits means confidence closer to 1. Overfitting pushes the logit gap up **on members**.

### Step 2: The course's example, run backwards

|Prediction|Logit gap (= ln(p1/p0))|Who|
|---|---|---|
|[0.05, 0.95]|ln(19) = 2.94|Member|
|[0.15, 0.85]|ln(5.67) = 1.73|Non-member, same features|

|Code|What it does|
|---|---|
|`self.forward(x)`|Raw logits, shape `(batch, 2)`|
|`F.softmax(logits, dim=1)`|Turns each row into probabilities summing to 1 (`dim=1` = across the classes)|

> **An honest note the course skips (4):** The course says this "returns calibrated probabilities". It doesn't. **Softmax outputs are not automatically calibrated**, and an overfit network is typically **overconfident**: it says 0.95 when it's right far less than 95% of the time. That overconfidence on training data **is** the leak. "Calibrated" would mean the leak wouldn't exist.

---

## Part 8: The Overfitting Gap (sample output)

```shellsession
Target Model Performance:
  Training Accuracy: 0.9012
  Test Accuracy: 0.8456
  Overfitting Gap: 0.0556
```

The course reads this as "90% of members right vs 85% of non-members", a 5.56-point gap. Consistent behaviour across tens of thousands of samples is a statistical foundation for MIA.

> **An honest note the course skips (5):** The section gives three versions of the same result:
> 
> |Source|Train|Test|Gap|
> |---|---|---|---|
> |Sample output block|90.12%|84.56%|5.56|
> |Bar chart (figure)|93.6%|82.5%|**11.1**|
> |Closing prose|90.2%|83.2%|7.1|
> 
> Only the figure matches the training curves (they end at train ≈ 0.93, val ≈ 0.825). Your own run should land near the figure. If you get something different, it's your real number. Don't chase the prose.

---

## Reading the figure/output

### Figure 1: "Target Model Training" (left: loss, right: accuracy)
![Figure 1: "Target Model Training" (left: loss, right: accuracy)](/assets/Pasted%20image%2020261023002357.png)

|What to read|Value (approx.)|What it means|
|---|---|---|
|Train loss, epoch 1 -> 100|0.35 -> 0.15, a smooth fall|The model keeps fitting its members better|
|Val (non-member) loss, lowest point|~0.32 around **epoch 12-15**|The best generalization point. **Early stopping would stop near here**|
|Val loss, epoch 100|~0.72, **more than double** the minimum|Classic overfitting signature|
|Train acc, epoch 1 -> 100|0.838 -> ~0.93|Keeps rising|
|Val acc peak|~0.850 around epoch 14-15|The best non-member accuracy ever reached|
|Val acc, epoch 100|~0.825, noisy|Only ~2.5 points worse than the peak|

**The key insight:** the non-member loss more than doubled (0.32 -> 0.72) while non-member accuracy only slipped ~2.5 points. That means the model isn't getting many _more_ answers wrong. It's getting **more confident about the ones it gets wrong**, and cross-entropy punishes confident mistakes hard. This is patient H from the Section 2 toy (loss 2.526 for one confident miss). ==Rising validation loss with flat accuracy = growing overconfidence = the membership signal.==

The course says divergence starts "around epoch 20". Its own earlier sentence said "by epoch 30-40". The figure supports ~15-20.

### Figure 2: "Overfitting Gap: The Root Cause of Membership Inference"
![Figure 2: "Overfitting Gap: The Root Cause of Membership Inference"](/assets/Pasted%20image%2020261023002402.png)

|Bar|Value|Meaning|
|---|---|---|
|Training data (members), green|93.6%|Accuracy on people the model trained on|
|Test data (non-members), red|82.5%|Accuracy on people it never saw|
|Orange arrow|Gap 11.1%|The behavioural difference the attack exploits|

The bar's 93.6% is a bit above the curve's final ~0.93 because the curve's train accuracy is averaged **during** the epoch while weights are still changing, and the bar is measured **after** training.

---

## Copy-paste cells (mine, not the course's)

These use only names from this section: `X_target`, `X_shadow`, `X_attack_eval`, `y_target`, `y_attack_eval`, `num_features`, `X_target_norm`, `X_attack_eval_norm`, `target_model`, `get_model_predictions`, `DEVICE`, `RANDOM_SEED`, `MLP`. I tested the NumPy/scikit-learn parts on stand-in arrays of the same shapes. The PyTorch parts are checked against the library source but not executed.

### Cell 1: Sanity-check the split and the scaling

```python
np.set_printoptions(suppress=True)
print("Shapes:", X_target.shape, X_shadow.shape, X_attack_eval.shape, "| features:", num_features)
print(f"Share of class 1 (>50K): members {y_target.mean():.3f} | non-members {y_attack_eval.mean():.3f}")
print("Raw column means (first 3): ", np.round(X_target.mean(axis=0)[:3], 1))
print("Scaled column means (first 3):", np.round(X_target_norm.mean(axis=0)[:3], 3))
print("Scaled column stds (first 3):", np.round(X_target_norm.std(axis=0)[:3], 3))
print("Non-member scaled means (first 3):", np.round(X_attack_eval_norm.mean(axis=0)[:3], 3))
```

Expected: shapes `(24421, 14) (12210, 14) (12211, 14)` and `features: 14`. Class-1 share about 0.239 in both pools. Raw means around `[38.6, ~3, ~189,000]` (age, workclass code, fnlwgt). Scaled means `0` and stds `1` for members. Non-member scaled means near 0 but **not exactly** 0, because they were scaled with the members' μ and σ, which is correct.

### Cell 2: The leak signals on the real target (confidence and loss, members vs non-members)

```python
probs_in = get_model_predictions(target_model, X_target_norm, DEVICE)
probs_out = get_model_predictions(target_model, X_attack_eval_norm, DEVICE)

conf_in, conf_out = probs_in.max(axis=1), probs_out.max(axis=1)
p_true_in = probs_in[np.arange(len(y_target)), y_target]
p_true_out = probs_out[np.arange(len(y_attack_eval)), y_attack_eval]
loss_in = -np.log(np.clip(p_true_in, 1e-12, 1))
loss_out = -np.log(np.clip(p_true_out, 1e-12, 1))

print(f"Mean confidence members {conf_in.mean():.3f} | non-members {conf_out.mean():.3f}")
print(f"Mean true-label prob members {p_true_in.mean():.3f} | non-members {p_true_out.mean():.3f}")
print(f"Mean loss members {loss_in.mean():.3f} | non-members {loss_out.mean():.3f}")
```

Expected: members higher on the first two lines. On the loss line, members should be around **0.15** and non-members around **0.72**. The non-member mean loss is exactly what the Val Loss curve measured at epoch 100, so it should match the figure's right end. This is the Section 2 loss attack's raw material.

### Cell 3: An honest, balanced leak check (AUC + best threshold)

This subsamples so there are **equal** numbers of members and non-members (12,211 each), making 0.500 the real coin-flip line. Run Cell 2 first.

```python
from sklearn.metrics import roc_auc_score

rng = np.random.default_rng(RANDOM_SEED)
n = min(len(p_true_in), len(p_true_out)) # balance: same number of members and non-members
idx_in = rng.choice(len(p_true_in), n, replace=False)
idx_out = rng.choice(len(p_true_out), n, replace=False)
scores = np.r_[p_true_in[idx_in], p_true_out[idx_out]]
is_mem = np.r_[np.ones(n), np.zeros(n)].astype(bool)
best_acc = max(((scores >= t) == is_mem).mean() for t in np.unique(np.round(scores, 3)))

print(f"Balanced pool: {n} members + {n} non-members (coin flip = 0.500)")
print(f"Attack AUC (true-label prob): {roc_auc_score(is_mem, scores):.3f}")
print(f"Best single-threshold accuracy: {best_acc:.3f}")
```

Expected: `Balanced pool: 12211 members + 12211 non-members`, and AUC and best accuracy both clearly **above 0.500**. I can't give your exact numbers without running PyTorch. The best threshold is chosen on the same pool it's scored on, so treat it as an **upper bound** for a simple threshold attack. That makes it the bar the shadow-model attack needs to beat in the next sections.

### Cell 4: Prove the early-stopping bug (shallow vs deep copy)

This builds a throwaway tiny `MLP`, saves its weights two ways, takes one training step, and checks which snapshot changed. It doesn't touch `target_model`.

```python
import copy
demo = MLP(num_features, [8], 2, dropout=0.0)
snap_shallow = demo.state_dict().copy() # what the library does
snap_deep = copy.deepcopy(demo.state_dict()) # the fix
opt = torch.optim.Adam(demo.parameters(), lr=0.1)
loss = F.cross_entropy(demo(torch.randn(16, num_features)), torch.randint(0, 2, (16,)))
loss.backward(); opt.step()
w_now = demo.state_dict()['output.weight']
print("Shallow snapshot silently changed with the model:", torch.equal(snap_shallow['output.weight'], w_now))
print("Deep snapshot still holds the old weights: ", not torch.equal(snap_deep['output.weight'], w_now))
```

Expected:

```txt
Shallow snapshot silently changed with the model: True
Deep snapshot still holds the old weights: True
```

The first line is the bug: the "saved best" weights moved with the model. When the shadows are trained next section, I'll give you a one-cell fixed version of `train_with_early_stopping` (same name, so your later course cells don't change) if you want real best-epoch restoring. Using it means your numbers will differ slightly from HTB's.

## Key facts to lock in

- **The module uses three disjoint pools from Adult Census (48,842 rows): 24,421 target members, 12,210 shadow-training rows, and 12,211 non-members for evaluation**. They are kept separate because any overlap between "non-members" and the target's training set would inflate the attack's success.
- **The target is made vulnerable on purpose: [256, 128] hidden layers, dropout 0.0, and 100 fixed epochs with no early stopping**. The course itself calls these production mistakes, so flipped around they become report remediations: add dropout, use early stopping, stop at the best validation point.
- ==The overfitting gap is member accuracy minus non-member accuracy, and in HTB's figure it's 93.6% − 82.5% = 11.1 points==. The section's prose quotes 5.56 and 7.1 as well, so trust the figure, which matches the curves.
- **Validation loss more than doubling (~0.32 at epochs 12-15 to ~0.72 at epoch 100) while validation accuracy drops only ~2.5 points means the model has become overconfident**. Cross-entropy punishes confident mistakes heavily, and that overconfidence is the membership signal. Softmax probabilities are not calibrated, despite what the course says.
- **Fit the StandardScaler on training data only, then use `transform` on everything else**, so every record is scaled with the same μ and σ. With μ = 0.2 and σ = 0.4, a value of 1.0 becomes 2.0. It's essential here because `load_adult_census` returns **unscaled** integer-coded data (14 features).
- **The attack model's input is 4 numbers for this binary task: [prob_class_0, prob_class_1, label_0, label_1]**. `prepare_attack_data` labels member rows 1 and non-member rows 0, and `AttackModel` outputs logits for [non_member, member].
- **Reading probabilities only needs `model.eval()` + `torch.no_grad()`, which `get_model_predictions` and `evaluate_model` already do**. That's the opposite of the evasion rule, where you needed gradients. The raw `MLP.predict_proba` doesn't call `eval()`, so dropout noise would corrupt the confidence signal on shadow models.
- **Method point:** **Before trusting any library helper or course number, check it: read the source, count the pool sizes, and balance members vs non-members before quoting an attack accuracy**. Here that turned up a broken "restore best weights" (shallow `state_dict().copy()`), an off-by-one split, wrong data-source and scaling claims, and three conflicting gap numbers.

---

