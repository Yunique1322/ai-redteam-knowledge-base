# Section 13: Private Aggregation of Teacher Ensembles (PATE)

## The idea in one breath

PATE gets its privacy from **separation** instead of noisy gradients. You split the sensitive data into many disjoint pieces and train one **teacher** per piece. The teachers then **vote** on labels for **public unlabeled data**, with **Laplace noise** added to the vote counts, and a **student** learns only from those noisy labels. The deployed student **never sees the private data**. The course runs 250 teachers on 48,000 private MNIST images (192 each), with noise scale 20 (ε₀ = 0.10 per query) and 5,000 queries, and quotes **ε ≈ 8.81** in total.

> **Heads-up:** here's how your mangled formulas should read.
> 
> - `ŷ=argmaxj(nj(x)+Lap(1ϵ))` -> **ŷ = argmax_j ( n_j(x) + Lap(1/ε) )**. See the honest note in Part 2: the scale should really be 2/ε₀.
> - `ϵ0=220=0.10` -> **ε₀ = 2 / 20 = 0.10**
> - "a bound proportional to m⋅ϵ0 instead of m⋅ϵ0" -> **√m · ε₀ instead of m · ε₀** (the square root was lost)
> - `Dprivate`, `D1,D2,...,Dn`, `T1...Tn`, `ỹ`, `x∈Dpublic` -> D_private, D₁...Dₙ, T₁...Tₙ, ỹ ("y-tilde", the noisy pseudo-label), x ∈ D_public
> 
> **Symbol clashes:** ε₀ is the **per-query** budget and ε is the **total**. PATE uses **Laplace** noise with **L1 sensitivity 2**, while DP-SGD used Gaussian noise with L2 sensitivity C. And `Lap(b)` means Laplace noise with **scale** b, not standard deviation: its std is b·√2 ≈ 28 votes at b = 20.
> 
> **Section 12 (the DP-SGD challenge) wasn't pasted.** If you want to do it hands-on,

---

## Part 1: The core idea

### Step 1: Three mechanisms

|Mechanism|What it does|
|---|---|
|**Teacher-student transfer**|n teachers on **disjoint** partitions, so each record influences only **one** teacher|
|**Aggregation with noise**|add Laplace noise to the vote counts, then take the top class|
|**Information bottleneck**|the student only ever receives a few thousand noisy labels|

### Step 2: The two phases

1. **Phase 1:** split D_private into D₁...Dₙ and train teacher Tᵢ on Dᵢ.
2. **Phase 2:** for each public x, every teacher votes. Count the votes, add noise, take the argmax to get ỹ. The student trains on the pairs (x, ỹ).

### Step 3: How it differs from DP-SGD

|Aspect|DP-SGD|PATE|
|---|---|---|
|Where the noise goes|gradients, every training step|vote counts, every labeling query|
|Noise type / sensitivity|Gaussian / L2 norm C|==Laplace / L1 = 2==|
|What gets deployed|the noisily trained model|a student that **never touched private data**|
|Extra requirement|none|**public unlabeled data** from the same distribution|

---

## Part 2: The noisy-vote formula

### Step 1: Toy, 10 teachers, 3 classes, noise scale 2

Use the same noise draws [−1.5, +2.0, +0.4] in both cases:

|Case|Votes n_j|+ noise|Noisy counts|Winner|
|---|---|---|---|---|
|Strong consensus|[7, 2, 1]|[−1.5, +2.0, +0.4]|[5.5, 4.0, 1.4]|class 0 ✓ (survives)|
|Weak consensus|[4, 3, 3]|[−1.5, +2.0, +0.4]|[2.5, 5.0, 3.4]|**class 1 (flipped)**|

**A big lead survives the noise; a close race gets decided by it.**

### Step 2: Why the sensitivity is 2

Change one private record and only **one** teacher's training changes, so at most that one teacher's vote moves, say from class 0 to class 1. The counts go from [7, 2, 1] to [6, 3, 1]: two counts change by 1 each, a total (L1) change of **2**. That's the **sensitivity**.

### Step 3: The Laplace mechanism

For the **Laplace mechanism**, per-query ε₀ = sensitivity / scale. Toy: 2/2 = 1.0. Real: **2 / 20 = 0.10**.

### Step 4: The real formula

**ŷ = argmax_j ( n_j(x) + Lap(b) )**

|Symbol|Plain English|Toy|Real|
|---|---|---|---|
|x|one public, unlabeled image| - |an MNIST digit|
|n_j(x)|number of teachers voting for class j|[7, 2, 1]|e.g. 200+ for the winner (out of 250)|
|Lap(b)|random Laplace noise, a fresh draw per class|b = 2|b = 20 (std ≈ 28 votes)|
|argmax_j|pick the class with the highest noisy count|class 0|the pseudo-label ỹ|
|ε₀|privacy cost of this one query = 2 / b|1.0|0.10|

**Read aloud:** "Count each class's votes, add a random jitter to every count, and the biggest jittered count becomes the label."

> **An honest note the course skips:** the course writes the noise as `Lap(1/ε)`, then says ε = 0.10 means scale 20. But 1/0.10 = **10**. The consistent version is ==scale = 2/ε₀== (because the sensitivity is 2), and the original paper writes it as Lap(1/γ) with ε₀ = 2γ. In the code, trust `noise_scale = 20.0` and ε₀ = 0.10.

---

## Part 3: Why aggregation protects privacy

- Each teacher overfits its own 192 images, just as Section 2's targets overfit. But for any record, only **one** teacher of 250 has memorized it. The other 249 vote from generalization, which **dilutes the membership signal**, and the noise hides what's left.
- **Strong consensus** (200+ votes): the label would be the same with or without any one record, so it reveals almost nothing about that record.

> **An honest note the course skips:** the course says that when votes are split evenly, "the label carries little information about any individual". The accounting works the other way round. **A close vote is where one teacher, and so one record, can tip the outcome**, so those queries cost the **most** privacy. That's why the improved PATE (Papernot 2018, "Confident-GNMax") simply **refuses** to answer low-consensus queries.

---

## Part 4: The data flow

|Stage|Input|Output|
|---|---|---|
|Partition|D_private (48,000)|D₁...D₂₅₀ (192 each, disjoint)|
|Teachers|each Dᵢ|250 trained models Tᵢ|
|Voting|public x -> all 250 teachers|a vote histogram over 10 classes|
|Noisy aggregation|histogram + Lap(20) per class|pseudo-label ỹ|
|Student|pairs (x, ỹ)|the deployed model S|

---

## Part 5: The information bottleneck

**The course's arithmetic, which checks out:** 48,000 × 784 × 4 bytes ≈ **150.5 MB** of private data, against 5,000 labels × ~4 bits ≈ **2.5 KB**, a ratio of about **60,000 : 1**.

> **An honest note the course skips:** "recovering megabytes from kilobytes violates information-theoretic limits" is true but beside the point. **Membership is one bit**, and a single sensitive attribute is only a few bits, so they fit easily through a 2.5 KB channel. **The protection comes from the DP noise, not from the compression.** Keep the bottleneck as intuition only; don't cite it in a report as the guarantee.

---

## Part 6: Privacy budget across 5,000 queries

### Step 1: Toy, naive vs √m growth

With ε₀ = 0.1, naive composition adds everything up: 100 queries cost 10. A √m-type bound grows more slowly: going from 100 to 10,000 queries multiplies the cost by √100 = 10, not by 100.

### Step 2: The real numbers (I checked them)

|Method|Formula|Total ε for m = 5,000, ε₀ = 0.1, δ = 10⁻⁵|
|---|---|---|
|Naive|m · ε₀|**500**|
|**Advanced composition**|ε₀·√(2m·ln(1/δ)) + m·ε₀·(e^ε₀ − 1)|33.9 + 52.6 ≈ **86.5**|
|**Data-dependent moments accountant** (course)|uses how strong each vote's consensus was|**≈ 8.81**|

> **An honest note the course skips:**
> 
> 1. The section's **last sentence** says "advanced composition gives ε ≈ 8.81". It doesn't; advanced composition gives about **86**, and even the best bound that ignores the votes is around 59. **8.81 only comes from the data-dependent moments accountant**, which gives a low cost to queries where the teachers agreed strongly.
> 2. A data-dependent ε is itself computed from the private data, so publishing it can leak a little. Papernot 2018 addresses this with "smooth sensitivity".
> 
> **In a report**, ask whether a client's PATE ε is data-dependent.

### Step 3: What's free and what isn't

- **Querying the student afterwards costs nothing**. It's post-processing: the student never touches private data again.
- **Every extra teacher-ensemble query spends budget**. So in a deployment, the teachers must stay offline or be deleted. A queryable teacher ensemble is a finding.

---

## Part 7: Implementation setup (MNIST)

The data comes in **three pools**: private (trains the teachers), public (unlabeled, labeled by the teachers for the student) and holdout (touched by nobody, used only for final evaluation).

```python
import os
import json
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from safetensors.torch import save_file
from tqdm import tqdm

from htb_ai_library import (
    set_reproducibility, use_htb_style,
    MLP, get_mnist_loaders,
    create_dataloader, train_model, evaluate_accuracy, get_model_predictions,
    HTB_GREEN, NODE_BLACK, HACKER_GREY, WHITE,
    AZURE, MALWARE_RED, AQUAMARINE, NUGGET_YELLOW,
)
```

|Code|What it does|
|---|---|
|`train_test_split`, `StandardScaler`|split MNIST into the three pools and scale the flattened pixels (same tools as the Adult/shadow sections)|
|`save_file`|safetensors again, which hints at a validator submission in the PATE challenge|
|`tqdm`|progress bar, useful when you train 250 teachers|
|`MLP`|the fully connected model from the shadow-model sections. Its `predict_proba` does **not** call `.eval()` (Section 3 note), which matters because dropout is 0.2 here|
|`get_model_predictions`|calls `.eval()` and `no_grad` for you, so it's the safe way to get teacher votes|
|`create_dataloader`, `train_model`, `evaluate_accuracy`|the usual training and evaluation helpers|
|colour constants|HTB plot colours|

```python
print("=" * 60)
print("PATE: PRIVATE AGGREGATION OF TEACHER ENSEMBLES")
print("Teacher-Student Privacy Through Knowledge Distillation")
print("=" * 60)
```

Just a banner.

```python
RANDOM_SEED = 1337
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
set_reproducibility(RANDOM_SEED)
use_htb_style()

PLOT_CONFIG = {"figsize": (10, 6), "dpi": 150}
DATASET_CONFIG = {"name": "mnist", "num_classes": 10, "num_features": 784}
```

|Code|What it does|
|---|---|
|`DEVICE`|note the capital name here (DP-SGD used `device`)|
|`DATASET_CONFIG`|10 classes, 784 = 28 × 28 flattened pixels, so the MLP input width is 784|

```python
TEACHER_CONFIG = {
    "num_teachers": 250, # Original paper configuration
    "hidden_layers": [128, 64],
    "dropout": 0.2,
    "epochs": 30,
    "batch_size": 64,
    "learning_rate": 0.001,
}
```

|Key|Value|Meaning|
|---|---|---|
|`num_teachers`|250|48,000 / 250 = **192 images per teacher** (about 19 per digit)|
|`hidden_layers`|[128, 64]|784 -> 128 -> 64 -> 10|
|`epochs`, `batch_size`|30, 64|192 / 64 = 3 batches per epoch, so 90 steps per teacher. That's cheap, and 250 teachers is fine on CPU|
|reported accuracy|74-82% each|weak on their own; the vote is what makes them strong|

```python
AGGREGATION_CONFIG = {
    "noise_scale": 20.0, # Strong privacy with high consensus
    "num_student_queries": 5000,
}

STUDENT_CONFIG = {
    "hidden_layers": [128, 64],
    "dropout": 0.2,
    "epochs": 30,
    "batch_size": 64,
    "learning_rate": 0.001,
}
```

|Key|Value|Meaning|
|---|---|---|
|`noise_scale`|20.0|Laplace scale b, so ε₀ = 2/20 = 0.10. Noise std ≈ 28 votes per class|
|`num_student_queries`|5000|public images labeled = queries spent (m in Part 6)|
|`STUDENT_CONFIG`|same as the teachers|but it trains on 5,000 noisy-labeled images (course: ~87% label accuracy)|

---

## Reading the figure/output
.
![Private Aggregation of Teacher Ensembles](/assets/13%20Private%20Aggregation%20of%20Teacher%20Ensembles-20261023190321076.png)

|Region|What it shows|Key point|
|---|---|---|
|**Sensitive Private Data** (top left)|Partition 1, 2, ... n|**disjoint**; each one feeds exactly one teacher (the lines cross, but it's one-to-one)|
|**Teacher Ensemble**|Teacher 1 ... n|each also gets the **public unlabeled data** (arrows from bottom left) to vote on|
|**Noisy Aggregation**|Vote Counts -> **+ Laplacian Noise** -> **Pseudo-Label**|the two teal boxes are **where the privacy guarantee comes from**|
|**Student Model** (green, right)|two arrows in: the pseudo-label ỹ and the public x|**no arrow from the private data to the student**. That's the whole design|
|Bottom curve|public data -> student|the student trains on (x, ỹ) pairs|

---

## Copy-paste cells (mine, not the course's)

I didn't run these. They use your config names.

### Strong vs weak consensus under the real noise

```python
rng = np.random.default_rng(RANDOM_SEED)
scale = AGGREGATION_CONFIG["noise_scale"]
def flip_rate(votes, trials=10000):
    votes = np.array(votes, dtype=float)
    noisy = votes + rng.laplace(0, scale, size=(trials, len(votes)))
    return np.mean(noisy.argmax(1) != votes.argmax())
strong = [200, 10, 10, 5, 5, 5, 5, 4, 3, 3] # 250 votes, clear winner
weak = [90, 80, 20, 20, 10, 10, 8, 5, 4, 3] # 250 votes, close race
print(f"per-query eps0 = 2/{scale} = {2/scale:.2f}")
print(f"strong consensus flip rate: {flip_rate(strong):.1%}")
print(f"weak consensus flip rate: {flip_rate(weak):.1%}")
```

Expected: `eps0 = 0.10`, strong ≈ **0.0%**, weak ≈ **40-45%**.

### Naive vs advanced composition vs the course's 8.81

```python
eps0, m, delta = 2 / AGGREGATION_CONFIG["noise_scale"], AGGREGATION_CONFIG["num_student_queries"], 1e-5
naive = m * eps0
advanced = eps0 * np.sqrt(2 * m * np.log(1 / delta)) + m * eps0 * (np.exp(eps0) - 1)
print(f"eps0={eps0:.2f} naive={naive:.1f} advanced={advanced:.1f} course (data-dependent moments accountant)=8.81")
```

Expected: `eps0=0.10 naive=500.0 advanced=86.5`.

## Key facts to lock in

- **PATE trains n teachers on disjoint partitions of the private data, labels public unlabeled data by noisy majority vote, and deploys only a student trained on those noisy labels**. The student never touches private data, which is privacy by separation rather than by noisy gradients.
- ==The aggregation is ŷ = argmax_j(n_j(x) + Lap(b)) with L1 sensitivity 2==, because one record changes one teacher's vote and so moves two counts by 1. Per-query ε₀ = 2/b, so the course's scale 20 gives ε₀ = 0.10 (the course's `Lap(1/ε)` notation contradicts this).
- **Strong consensus survives the noise and costs little privacy; close votes get flipped and cost the most**, because one teacher (one record) is pivotal there. The improved PATE (Confident-GNMax) refuses to answer low-consensus queries.
- ==Composition over 5,000 queries at ε₀ = 0.1 gives naive ε = 500 and advanced composition ≈ 86.5; the course's ε ≈ 8.81 comes only from the data-dependent moments accountant==, which rewards strong teacher consensus. Ask whether a reported PATE ε is data-dependent.
- **Querying the trained student costs no extra privacy (post-processing), but every teacher-ensemble query spends budget**. So the teachers must stay offline or be deleted, and a queryable teacher API is a finding.
- **The course setup: MNIST, 48,000 private images split across 250 teachers (192 each, 74-82% accuracy each), noise scale 20, 5,000 student queries, [128, 64] MLPs with dropout 0.2**, plus three separate pools: private, public and holdout.
- **The information bottleneck (≈150 MB to ≈2.5 KB, about 60,000 : 1) is intuition, not the guarantee**. Membership is a single bit, so the DP noise is what actually protects it.
- **Method point:** ==when recommending or auditing PATE, check for disjoint partitions, a public unlabeled in-distribution dataset, the noise scale and ε₀ = 2/b, whether ε is data-dependent, and that the teachers aren't exposed==. Then report the student's accuracy and MIA advantage against a baseline, as in DP-SGD.

---

