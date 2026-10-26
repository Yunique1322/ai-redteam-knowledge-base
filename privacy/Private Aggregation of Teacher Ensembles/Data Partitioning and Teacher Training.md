# Section 14: Data Partitioning and Teacher Training

## The idea in one breath

This is PATE phase 1 in code. Load MNIST as flat 784-pixel vectors and split it **48,000 private / 12,000 public / 10,000 holdout**. Scale it, then cut the private set into **250 disjoint partitions of exactly 192 images**. Train one fresh MLP teacher per partition (74-82% accuracy each). `get_teacher_votes` then turns the 250 teachers' predictions into a **vote histogram** of shape (samples, 10), which is what the noisy aggregation in the next section works on.

> **Heads-up:** no mangled formulas. "normalized to 0, 1" means to the range **[0, 1]**.
> 
> **Checked in the library source:** `train_model` uses **Adam** and evaluates on the loader you pass in every epoch, printing a line each time. `get_model_predictions` returns **softmax probabilities**, not logits. Both matter below.

---

## Part 1: Loading MNIST as numpy arrays

```python
print("\n" + "=" * 60)
print("PHASE 1: Loading MNIST Dataset")
print("=" * 60)

train_loader, test_loader = get_mnist_loaders()

X_train = train_loader.dataset.data.numpy().reshape(-1, 784).astype(np.float32) / 255.0
y_train = train_loader.dataset.targets.numpy()
X_test = test_loader.dataset.data.numpy().reshape(-1, 784).astype(np.float32) / 255.0
y_test = test_loader.dataset.targets.numpy()

num_features = 784
num_classes = 10
DATASET_CONFIG['num_features'] = num_features
DATASET_CONFIG['num_classes'] = num_classes

print(f"Dataset: {X_train.shape[0]} training samples, {num_features} features, {num_classes} classes")
print(f"Class distribution: min={np.bincount(y_train).min()}, max={np.bincount(y_train).max()}")
```

|Code|What it does|
|---|---|
|`get_mnist_loaders()`|downloads and caches MNIST, returning (train_loader, test_loader)|
|`.dataset.data`|the **raw uint8 pixel tensor (0-255)**. This **skips** the loader's own transforms, which is why we divide by 255 ourselves|
|`.reshape(-1, 784)`|flattens 28×28 into 784; `-1` means "work out the number of rows"|
|`.astype(np.float32) / 255.0`|scales pixels to [0, 1]|
|`.targets.numpy()`|the digit labels 0-9|
|`np.bincount(y_train)`|counts per digit. MNIST is ~6,000 per class, so the min/max print shows it's balanced|

|Name|Shape|Meaning|
|---|---|---|
|`X_train`|(60000, 784)|all training images, flat|
|`y_train`|(60000,)|labels|
|`X_test`|(10000, 784)|becomes the holdout|
|`y_test`|(10000,)|holdout labels|

---

## Part 2: The three-way split

```python
# Split training data into: private (for teachers) and public (for student queries)
X_private, X_public, y_private, y_public = train_test_split(
    X_train, y_train, test_size=0.2,
    random_state=RANDOM_SEED, stratify=y_train
)

# Use standard test set as holdout for evaluation
X_holdout = X_test
y_holdout = y_test

print(f"\nData splits:")
print(f" Private (teacher training): {len(X_private)} samples")
print(f" Public (student queries): {len(X_public)} samples")
print(f" Holdout (evaluation): {len(X_holdout)} samples")
```

|Code|What it does|
|---|---|
|`test_size=0.2`|20% goes to public: exactly **12,000**; private gets **48,000**|
|`stratify=y_train`|each split keeps ~10% of each digit|
|`X_holdout = X_test`|the official test set, which **nobody trains on**: neither the teachers nor the student|

|Pool|Size|Used by|Labels used?|
|---|---|---|---|
|Private|48,000|teachers only|yes (by teachers)|
|Public|12,000|teachers vote on it, student trains on it|**no, the real labels are never used for training** (y_public is only for checking label accuracy)|
|Holdout|10,000|final evaluation only|for scoring only|

---

## Part 3: Feature normalization

```python
scaler = StandardScaler()
X_private_norm = scaler.fit_transform(X_private)
X_public_norm = scaler.transform(X_public)
X_holdout_norm = scaler.transform(X_holdout)
```

|Code|What it does|
|---|---|
|`fit_transform(X_private)`|learns each pixel's mean and std **from private data**, then scales it|
|`transform(...)`|reuses those same numbers on public and holdout (the Section 3 rule: fit once, reuse)|

**Toy:** a border pixel that's almost always 0 in the private data has mean 0.0005 and std 0.004. A holdout image with that pixel at 0.5 becomes (0.5 − 0.0005) / 0.004 ≈ **125**. Normal pixels end up roughly between −3 and +3.

> **An honest note the course skips:**
> 
> 1. The course says fitting on private data is the realistic choice "without leaking information". For **privacy** it's backwards. The scaler's means and stds are statistics of the **private** data, and they flow into the student's inputs **without any DP noise**. That's a small leak outside the ε accounting. **The privacy-clean choice is to fit the scaler on public data** (or just use /255, which is already enough for MNIST).
> 2. Near-constant border pixels produce huge scaled values, like the toy. It usually trains fine, but it's why many MNIST pipelines skip StandardScaler.

---

## Part 4: Disjoint partitions

```python
print("\n" + "=" * 60)
print("PHASE 2: Training Teacher Ensemble")
print("=" * 60)

num_teachers = TEACHER_CONFIG['num_teachers']
```

```python
# Partition private data for teachers
np.random.seed(RANDOM_SEED)
indices = np.random.permutation(len(X_private_norm))
partition_size = len(X_private_norm) // num_teachers

teacher_partitions = []
for i in range(num_teachers):
    start_idx = i * partition_size
    if i == num_teachers - 1:
        partition_indices = indices[start_idx:]
    else:
        partition_indices = indices[start_idx:start_idx + partition_size]
    teacher_partitions.append(partition_indices)

print(f"Created {num_teachers} partitions, ~{partition_size} samples each")
```

**Toy:** 10 records and 3 teachers. `partition_size` = 10 // 3 = **3**. Teacher 0 gets shuffled positions 0-2, teacher 1 gets 3-5, and the last teacher gets `indices[6:]`, which is **4** records, because the last one absorbs the remainder.

|Code|What it does|
|---|---|
|`np.random.seed(RANDOM_SEED)`|seeds NumPy's **global** random generator, so the shuffle is repeatable|
|`np.random.permutation(48000)`|the numbers 0...47,999 in random order, so no partition clusters by any hidden ordering|
|`48000 // 250`|`//` is whole-number division: **192**|
|`if i == num_teachers - 1`|the last teacher takes everything left over. 48,000 mod 250 = 0, so all 250 get exactly 192|
|`teacher_partitions`|a list of 250 index arrays, one per teacher|

**Disjoint is the privacy core**: each record sits in exactly one partition, so it influences one teacher and can move at most one vote. That's Section 13's sensitivity of 2.

> **An honest note the course skips:** `stratify` balanced the private/public split, but the **250 partitions aren't stratified**. Each teacher gets about 19 of each digit on average, give or take about 4. Across all teachers, some get as few as ~5 of one digit and others ~34. That's part of why teacher accuracy varies from 74% to 82%.

---

## Part 5: Training the 250 teachers

```python
teachers = []
holdout_loader = create_dataloader(X_holdout_norm, y_holdout, 128, shuffle=False)
```

|Code|What it does|
|---|---|
|`teachers = []`|the list that will hold the 250 trained models|
|`create_dataloader(..., 128, shuffle=False)`|wraps the numpy arrays in a PyTorch loader (batch 128, fixed order). It's only used to **monitor** accuracy|

```python
for i in tqdm(range(num_teachers), desc="Training teachers"):
    partition_idx = teacher_partitions[i]
    X_teacher = X_private_norm[partition_idx]
    y_teacher = y_private[partition_idx]

    teacher = MLP(
        input_size=num_features,
        hidden_layers=TEACHER_CONFIG['hidden_layers'],
        num_classes=DATASET_CONFIG['num_classes'],
        dropout=TEACHER_CONFIG['dropout']
    )

    train_loader = create_dataloader(X_teacher, y_teacher, TEACHER_CONFIG['batch_size'])
    train_model(teacher, train_loader, holdout_loader, device=DEVICE,
                epochs=TEACHER_CONFIG['epochs'], learning_rate=TEACHER_CONFIG['learning_rate'])
    teachers.append(teacher)

print(f"\nTeacher ensemble trained: {len(teachers)} teachers")
```

|Code|What it does|
|---|---|
|`tqdm(range(250), ...)`|progress bar over the teachers|
|`X_private_norm[partition_idx]`|this teacher's 192 images, and **only** those|
|`MLP(784 → 128 → 64 → 10, dropout 0.2)`|a **fresh model each time**. No shared weights, so each teacher knows only its own partition|
|`create_dataloader(..., 64)`|192 / 64 = 3 batches per epoch|
|`train_model(...)`|from the source: **Adam** at lr 0.001 for 30 epochs, evaluating `holdout_loader` and printing a line **every epoch**|
|`teachers.append(teacher)`|keep the trained teacher|

|Name|Shape|Meaning|
|---|---|---|
|`X_teacher`|(192, 784)|one teacher's private images|
|`y_teacher`|(192,)|their labels|
|`teachers`|list of 250 MLPs|the ensemble|

> **If this errors / floods the screen:**
> 
> 1. `train_model` prints 30 lines per teacher, so **7,500 lines** in total, and it scores all 10,000 holdout images every epoch, which is most of the runtime. My cell 2 trains **the same teachers** quietly and faster.
> 2. **This loop overwrites `train_loader`.** Afterwards it's teacher 250's 192-image loader, not the MNIST loader. That's harmless here (you already have the numpy arrays), but don't reuse that name expecting MNIST.

> **An honest note the course skips:** the holdout is only **monitored** during teacher training. It isn't used to pick the best epoch, so it stays a clean evaluation set ✓.

---

## Part 6: Ensemble voting

```python
def get_teacher_votes(teachers, X, device):
    """
    Get vote counts from all teachers.

    Returns:
        np.ndarray: Vote counts of shape (num_samples, num_classes).
    """
    num_samples = X.shape[0]
    num_classes = DATASET_CONFIG['num_classes']
    votes = np.zeros((num_samples, num_classes), dtype=np.int32)

    for teacher in teachers:
        preds = get_model_predictions(teacher, X, device)
        predictions = np.argmax(preds, axis=1)

        for i, pred in enumerate(predictions):
            votes[i, pred] += 1

    return votes
```

**Toy:** one image and 5 teachers who predict [3, 3, 8, 3, 5]. The votes row starts as ten zeros, then: +1 at 3, +1 at 3, +1 at 8, +1 at 3, +1 at 5. Result: **[0, 0, 0, 3, 0, 1, 0, 0, 1, 0]**, and class 3 leads.

|Code|What it does|
|---|---|
|`np.zeros((N, 10), dtype=np.int32)`|one row of 10 counters per image|
|`get_model_predictions(teacher, X, device)`|`.eval()` (dropout off) and `no_grad`, then **softmax probabilities** of shape (N, 10)|
|`np.argmax(preds, axis=1)`|each teacher's chosen digit per image|
|`votes[i, pred] += 1`|add that teacher's vote|
|`return votes`|`votes[i, j]` = how many of the 250 teachers said image i is digit j. **Every row sums to 250**|

|Name|Shape|Meaning|
|---|---|---|
|`preds`|(N, 10)|one teacher's probabilities|
|`predictions`|(N,)|one teacher's digit per image|
|`votes`|(N, 10)|the histogram the Laplace noise gets added to next section|

> **An honest note the course skips:** the course says teachers "produce logits". `get_model_predictions` actually returns softmax **probabilities**. argmax gives the same answer either way. The inner `for i, pred` loop is plain Python (12,000 × 250 = 3 million steps), so it's a bit slow. The vectorized equivalent is `votes[np.arange(num_samples), predictions] += 1`.

---

## Reading the figure/output

There's no figure in this section. Expected prints: `Dataset: 60000 training samples, 784 features, 10 classes`, the class min/max (MNIST: about 5,421 for digit 5 up to about 6,742 for digit 1), the splits 48000 / 12000 / 10000, `Created 250 partitions, ~192 samples each`, and `Teacher ensemble trained: 250 teachers`.

---

## Copy-paste cells (mine, not the course's)

I didn't run these. They use your names from this section.

### Check the split and the partitions

```python
all_idx = np.concatenate(teacher_partitions)
print("private/public/holdout:", X_private_norm.shape, X_public_norm.shape, X_holdout_norm.shape)
print("disjoint:", len(all_idx) == len(np.unique(all_idx)), "| covers all private:", len(all_idx) == len(X_private_norm))
per_class = np.array([np.bincount(y_private[p], minlength=num_classes) for p in teacher_partitions])
print(f"digits per teacher: mean {per_class.mean():.1f}, min {per_class.min()}, max {per_class.max()}")
print("largest |scaled value| in holdout:", round(float(np.abs(X_holdout_norm).max()), 1))
```

Expected: `(48000, 784) (12000, 784) (10000, 784)`; `disjoint: True | covers all private: True`; mean 19.2, min around 5, max around 34. The largest scaled value is probably in the hundreds (the border-pixel effect).

### Quiet, faster teacher training

Run this **instead of** the course's teacher loop. It trains the same teachers: same data, seed path and settings. It only uses a smaller monitoring set and hides the 7,500 prints, and it doesn't overwrite `train_loader`.

```python
import contextlib, io
teachers = []
quick_eval_loader = create_dataloader(X_holdout_norm[:500], y_holdout[:500], 128, shuffle=False)
for i in tqdm(range(num_teachers), desc="Training teachers"):
    partition_idx = teacher_partitions[i]
    teacher = MLP(input_size=num_features, hidden_layers=TEACHER_CONFIG['hidden_layers'],
                  num_classes=DATASET_CONFIG['num_classes'], dropout=TEACHER_CONFIG['dropout'])
    teacher_loader = create_dataloader(X_private_norm[partition_idx], y_private[partition_idx], TEACHER_CONFIG['batch_size'])
    with contextlib.redirect_stdout(io.StringIO()): # hide the 30 per-epoch prints
        train_model(teacher, teacher_loader, quick_eval_loader, device=DEVICE,
                    epochs=TEACHER_CONFIG['epochs'], learning_rate=TEACHER_CONFIG['learning_rate'])
    teachers.append(teacher)
print(f"\nTeacher ensemble trained: {len(teachers)} teachers")
```

Expected: just the tqdm bar (which writes to stderr, so it still shows), then `Teacher ensemble trained: 250 teachers`.

### Teacher accuracy spread and a no-noise vote check

```python
teacher_accs = [evaluate_accuracy(t, holdout_loader, DEVICE) for t in teachers]
print(f"teacher holdout acc: mean {np.mean(teacher_accs):.1f}%, min {np.min(teacher_accs):.1f}%, max {np.max(teacher_accs):.1f}%")
holdout_votes = get_teacher_votes(teachers, X_holdout_norm[:1000], DEVICE)
print("votes shape:", holdout_votes.shape, "| each row sums to", np.unique(holdout_votes.sum(1)))
print(f"plain (no-noise) ensemble acc on 1000 holdout: {np.mean(holdout_votes.argmax(1) == y_holdout[:1000]):.1%}")
print(f"median winning votes: {np.median(holdout_votes.max(1)):.0f} / {num_teachers}")
```

Expected: mean around 78% (range ~74-82%), `votes shape: (1000, 10) | each row sums to [250]`. The ensemble should beat any single teacher by a lot, and the median winning vote shows how strong the consensus is (the course claims 200+).

---

## Quick self-check

You change to **350 teachers** with the same 48,000 private images.  
(a) What's `partition_size`, and how many images does the **last** teacher get?  
(b) After the course's teacher loop finishes, what does `train_loader` hold?  
(c) Five teachers predict **[3, 3, 8, 3, 5]** for one image. Write out its votes row.

---

## The facts worth locking in for the exam

- **The PATE data split is three pools: 48,000 private (teachers only), 12,000 public (labeled by teacher votes, its real labels never used for training) and the 10,000-image MNIST test set as an untouched holdout**. The private/public split is stratified by digit.
- **The private data is shuffled with a fixed seed and cut into 250 disjoint partitions of exactly 192 images** (48,000 // 250, remainder 0, and the last partition would absorb any remainder). Disjointness is what bounds each record's influence to one teacher and one vote.
- **Each teacher is a fresh [128, 64] MLP (dropout 0.2) trained only on its own partition**: Adam, lr 0.001, 30 epochs, batch 64. Individual teachers reach about 74-82%, partly because the partitions aren't stratified (≈19 ± 4 images per digit).
- **`get_teacher_votes` returns an (N, 10) histogram where `votes[i, j]` is the number of teachers predicting class j for image i**. Each row sums to the number of teachers (250), and this histogram is what the Laplace noise is added to.
- **`get_model_predictions` puts the model in eval mode with no_grad and returns softmax probabilities** (not logits as the course says), so dropout is off during voting.
- **Fitting the StandardScaler on private data sends un-noised private statistics into the student pipeline**, a small leak outside the DP accounting. Fitting on public data, or just dividing by 255, is the privacy-clean choice.
- **The course's teacher loop overwrites `train_loader` and prints 7,500 lines**, because `train_model` evaluates and prints every epoch. The holdout is only monitored, never used for model selection, so it stays a clean evaluation set.
- **Method point:** **when building or auditing a PATE pipeline, verify that the partitions are disjoint and cover all private data, that every vote row sums to the number of teachers, and that no private-data statistic (scalers, feature selection) reaches the student without noise**. Then check teacher accuracy spread against the no-noise ensemble accuracy before adding noise.

---

Logged to COAE-progress.md, with the self-check answer key. I made one quick read of the library source, which is where the Adam, print-flood and probabilities findings came from.