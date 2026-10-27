# Noisy Aggregation And Privacy Guarantees

## The idea in one breath

This is the step that makes PATE private. We take the vote histograms, add **Laplace noise** (scale 20) to every count, and use the noisy winner as the student's pseudo-label for 5,000 public images. In the course's run, 80% average teachers become an **87.8%** clean ensemble, and noise costs only **0.6 points** (87.2% noisy labels).

The big catch: the course says the total budget is ε ≈ 8.81, but **its own `compute_privacy_budget` code returns ≈ 86.5** for these settings. I worked it by hand below.

> **Heads-up:** here's how your mangled formulas should read.
> 
> - `f(x∣b)=12bexp(−|x|b)` -> **f(x | b) = (1 / 2b) · exp(−|x| / b)**
> - `202≈28.3` -> **20·√2 ≈ 28.3**
> - `P[ℳ(D)=ŷ]P[ℳ(D′)=ŷ]≤eϵ` -> **P[M(D) = ŷ] / P[M(D′) = ŷ] ≤ e^ε**
> - `ϵ=2noise_scale` -> **ε = 2 / noise_scale**
> - "grows with k instead of k" -> **grows with √k instead of k**
> - `2⋅k⋅ϵ02⋅ln(1/δ)` -> **√(2 · k · ε₀² · ln(1/δ))**
> 
> You pasted the figure twice; both copies are identical.

---

## Part 1: The Laplace mechanism

### Step 1: Toy, how big is Laplace noise at scale 20?

The Laplace curve is a sharp peak at 0 with tails that fall off exponentially. The chance a draw is bigger than t (in either direction) is **e^(−t/b)**:

|Noise bigger than|Chance (b = 20)|
|---|---|
|20 votes|e⁻¹ = **36.8%**|
|40 votes|e⁻² = **13.5%**|
|60 votes|e⁻³ = **5.0%**|

So a 200-vs-35 race (a margin of 165) essentially never flips, while a 90-vs-80 race often does. The course's example: 200 − 8.2 = 191.8 and 35 + 12.1 = 47.1, so class 3 still wins.

### Step 2: Why ε₀ = 2 / scale

The Laplace curve has a special property: sliding it over by Δ changes the height at any point by at most a factor of e^(Δ/b). One changed record moves the votes by Δ = 2 (Section 13), so the output probabilities can differ by at most **e^(2/20) = e^0.1 ≈ 1.105**. That's ε₀ = **0.10**. In Section 13's small toy, b = 2 gave e^(2/2) = e¹, so ε₀ = 1.

### Step 3: The real formulas

|Symbol|Plain English|Toy|Real|
|---|---|---|---|
|f(x \| b) = (1/2b)·e^(−\|x\|/b)|how likely each noise value is|b = 2|b = 20: peak height 1/40 = 0.025|
|std = b·√2|typical noise size|2.8|**28.3 votes**|
|n_j(x) + Lap(b)|jittered vote count|[5.5, 4.0, 1.4]|e.g. 191.8, 47.1|
|ε₀ = Δ / b = 2 / b|per-query privacy|1.0|**0.10**|

**Read aloud:** "Add Laplace jitter of scale b to every count. Since one record moves the counts by 2 at most, each query costs 2/b of privacy."

---

## Part 2: The noisy argmax

```python
def noisy_argmax(votes, noise_scale):
    """Add Laplacian noise to votes and return the winning class."""
    noise = np.random.laplace(loc=0.0, scale=noise_scale, size=votes.shape)
    noisy_votes = votes.astype(np.float64) + noise
    return np.argmax(noisy_votes, axis=1)
```

|Code|What it does|
|---|---|
|`np.random.laplace(0.0, scale, size=votes.shape)`|one independent draw per sample **per class**: a (5000, 10) matrix. Uses NumPy's **global** RNG|
|`votes.astype(np.float64) + noise`|integer votes become decimals so the noise can be added|
|`np.argmax(..., axis=1)`|the noisy winner per row, which becomes the pseudo-label|

|Name|Shape|Meaning|
|---|---|---|
|`votes`|(5000, 10)|clean histograms (rows sum to 250)|
|`noise`|(5000, 10)|Laplace(0, 20) draws|
|return value|(5000,)|pseudo-labels|

> **An honest note the course skips:** **every run of `noisy_argmax` is a new release that spends budget again**. If you re-run the cell three times and keep the "best" labels, you've made 15,000 queries, not 5,000. In a real deployment, that's a finding.

---

## Part 3: Privacy composition, and the 8.81 problem

```python
def compute_privacy_budget(num_queries, noise_scale, delta=1e-5):
    """
    Compute total privacy budget using advanced composition.

    Args:
        num_queries (int): Number of labeling queries made.
        noise_scale (float): Laplacian noise scale.
        delta (float): Failure probability.

    Returns:
        tuple: (total_epsilon, per_query_epsilon)
    """
    per_query_eps = 2.0 / noise_scale
    epsilon_sq_sum = num_queries * (per_query_eps ** 2)
    total_epsilon = np.sqrt(2 * epsilon_sq_sum * np.log(1 / delta))
    total_epsilon += num_queries * per_query_eps * (np.exp(per_query_eps) - 1)
    return total_epsilon, per_query_eps
```

### Step 1: Toy, k = 100 queries at ε₀ = 0.1, δ = 10⁻⁵ (ln(10⁵) ≈ 11.51)

- **Term 1:** √(2 × 100 × 0.01 × 11.51) = √23.0 = **4.80**
- **Term 2:** 100 × 0.1 × (e^0.1 − 1) = 10 × 0.105 = **1.05**
- **Total ≈ 5.85**, compared with naive 100 × 0.1 = 10. Better.

Now k = 10: term 1 = √2.30 = 1.52, term 2 = 0.105, total **1.62**, but naive is only **1.0**. **For few queries, advanced composition is _worse_ than naive**, so you use whichever is smaller.

### Step 2: The real numbers, line by line

|Code line|Value for k = 5000, scale 20|
|---|---|
|`per_query_eps = 2.0 / 20`|0.10|
|`epsilon_sq_sum = 5000 * 0.01`|50|
|`np.sqrt(2 * 50 * np.log(1e5))`|√1151.3 = **33.93**|
|`+= 5000 * 0.1 * (e^0.1 − 1)`|+ 52.59|
|**returned total**|****≈ 86.52****|

> **An honest note the course skips:**
> 
> - **This code prints `Total epsilon: 86.52`, not 8.81.** For what it's worth, this exact formula gives 8.81 at about **197 queries**, so the number may be left over from an older configuration.
> - Section 13 credited 8.81 to the moments accountant; this section credits it to advanced composition. Neither matches the code that ships.
> - **If the PATE challenge asks for ε, compute it yourself. Don't copy 8.81.**
> - Even 86.5 is a valid guarantee, just a weak one. Tighter, data-dependent accounting (Papernot's) is what can bring it near single digits.

|Symbol|Plain English|Real|
|---|---|---|
|k|number of labeling queries|5,000|
|ε₀|cost per query = 2/scale|0.10|
|√(2·k·ε₀²·ln(1/δ))|the main term, growing with **√k**|33.93|
|k·ε₀·(e^ε₀ − 1)|the correction term (≈ k·ε₀² when ε₀ is small)|52.59|
|δ|failure slack|10⁻⁵|

**Read aloud:** "Total cost ≈ √(2·k·ln(1/δ))·ε₀ plus k·ε₀², which here is about 86.5, not the naive 500, and not the claimed 8.81."

---

## Part 4: Generating the pseudo-labels

```python
print("\n" + "=" * 60)
print("PHASE 3: Noisy Aggregation and Privacy Budget")
print("=" * 60)

num_queries = min(AGGREGATION_CONFIG['num_student_queries'], len(X_public_norm))
query_indices = np.random.choice(len(X_public_norm), num_queries, replace=False)
X_query = X_public_norm[query_indices]
y_query_true = y_public[query_indices]
```

|Code|What it does|
|---|---|
|`min(5000, 12000)`|5,000 queries (a safety cap in case the public set is small)|
|`np.random.choice(..., replace=False)`|5,000 distinct public images, picked at random (global RNG)|
|`y_query_true`|real labels, **for scoring only**. The student never sees them, and with truly unlabeled public data you wouldn't have them at all|

```python
votes = get_teacher_votes(teachers, X_query, DEVICE)
student_labels = noisy_argmax(votes, AGGREGATION_CONFIG['noise_scale'])

label_accuracy = (student_labels == y_query_true).mean() * 100
clean_preds = votes.argmax(axis=1)
clean_ensemble_acc = (clean_preds == y_query_true).mean() * 100

print(f"Labeled {num_queries} public samples")
print(f"Clean ensemble accuracy: {clean_ensemble_acc:.2f}%")
print(f"Noisy label accuracy: {label_accuracy:.2f}%")
```

|Code|What it does|
|---|---|
|`get_teacher_votes(...)`|(5000, 10) histograms from all 250 teachers|
|`noisy_argmax(...)`|**this is the moment the privacy budget is spent**|
|`label_accuracy`|how good the student's training labels are (≈ 87%)|
|`clean_preds = votes.argmax(axis=1)`|the no-noise winner. **For measurement only**; releasing these would have no privacy at all|
|`clean_ensemble_acc`|≈ 88%. The gap to `label_accuracy` is what the noise costs|

```python
# Compute privacy budget
total_eps, per_query_eps = compute_privacy_budget(
    num_queries, AGGREGATION_CONFIG['noise_scale']
)

print(f"\nPrivacy budget:")
print(f" Per-query epsilon: {per_query_eps:.4f}")
print(f" Total epsilon: {total_eps:.2f}")
```

Expected print: `Per-query epsilon: 0.1000` and `Total epsilon: 86.52` (see Part 3).

**The tradeoff:** more noise means stronger privacy and worse labels, but **strong consensus protects label quality**. Most MNIST queries have huge margins, which is why noise only costs 0.6 points.

---

## Part 5: Visualizing it

```python
def plot_accuracy_comparison(models, accuracies, save_path=None):
    """Plot comparison of model accuracies."""
    fig, ax = plt.subplots(figsize=PLOT_CONFIG['figsize'])
    colors = [MALWARE_RED, AZURE, HTB_GREEN, AQUAMARINE]
    bars = ax.bar(models, accuracies, color=colors, edgecolor=HACKER_GREY, linewidth=2)

    for bar, acc in zip(bars, accuracies):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                f'{acc:.3f}', ha='center', va='bottom', color=WHITE, fontweight='bold')

    ax.set_ylim([0.0, 1.0])
    ax.set_ylabel('Accuracy')
    ax.set_title('Model Accuracy Comparison', color=HTB_GREEN, fontweight='bold')
    ax.grid(True, axis='y', alpha=0.3)
    if save_path:
        plt.savefig(save_path, dpi=PLOT_CONFIG['dpi'], bbox_inches='tight', facecolor=NODE_BLACK)
    plt.close()
```

|Code|What it does|
|---|---|
|`colors = [...]`|4 colours for up to 4 bars; this plot uses the first 3 (red, blue, green)|
|`ax.text(... bar.get_height() + 0.01 ...)`|writes the value just above each bar|
|`set_ylim([0, 1])`|the accuracies are fractions, which is why the next cell divides by 100|
|`plt.savefig(...)` then `plt.close()`|**saves only; nothing shows inline**. Use my cell 2 to display it|

> **If this errors / behaves oddly:** this **redefines** `plot_accuracy_comparison`. If you're in the same notebook as the DP-SGD sections, it replaces the library version, which has a different signature. Re-running DP-SGD cells afterwards would then crash. Keep PATE in its own notebook, or rename this one.

```python
avg_teacher_acc = np.mean([evaluate_accuracy(t, holdout_loader, DEVICE) for t in teachers]) / 100
os.makedirs("figs", exist_ok=True)
plot_accuracy_comparison(
    models=['Avg Teacher', 'Clean Ensemble', 'Noisy Labels'],
    accuracies=[avg_teacher_acc, clean_ensemble_acc / 100, label_accuracy / 100],
    save_path="figs/pate_accuracy_comparison.png"
)
```

|Code|What it does|
|---|---|
|`evaluate_accuracy(t, holdout_loader, ...)` for all 250|each teacher's **holdout** accuracy (a %), averaged, then / 100|
|`clean_ensemble_acc / 100`, `label_accuracy / 100`|these are measured on the **5,000 public queries**|

> **An honest note the course skips:** the bars use **different evaluation sets**: the average teacher is scored on the holdout, the other two on the public queries. Both come from the same MNIST distribution, so the comparison is fair in spirit, but it isn't the same images.

---

## Reading the figure/output
![Noisy Aggregation and Privacy Guarantees](/assets/15%20Noisy%20Aggregation%20and%20Privacy%20Guarantees-20261023190458354.png)

|Bar|Value|Change|What it means|
|---|---|---|---|
|Avg Teacher (red)|**0.802**| - |one teacher with 192 images|
|Clean Ensemble (blue)|**0.878**|**+7.6 pts**|voting fixes many individual mistakes|
|Noisy Labels (green)|**0.872**|**−0.6 pts**|the price of ε₀ = 0.1 per query: tiny, thanks to strong consensus|

**A deeper reading (mine):** if the 250 teachers made **independent** mistakes, a majority of 80%-accurate voters would be almost 100% right. Even with just 3 independent teachers: 0.8³ + 3 × 0.8² × 0.2 = 0.512 + 0.384 = **0.896**. Getting only 87.8% from 250 means **the teachers' errors are highly correlated**: they all trip on the same hard digits (the 4-vs-9 and 3-vs-8 cases the course mentioned). That puts a ceiling on how good the student's labels can get.

---

## Copy-paste cells (mine, not the course's)

I didn't run these. They use your names from this section.

### Check the budget yourself

```python
for k in [197, 1000, 5000]:
    eps, eps0 = compute_privacy_budget(k, AGGREGATION_CONFIG['noise_scale'])
    print(f"{k:>5} queries @ scale {AGGREGATION_CONFIG['noise_scale']}: eps0={eps0:.2f}, total eps={eps:.2f} (naive {k*eps0:.0f})")
```

Expected: `197 -> 8.81 (naive 20)`, `1000 -> 25.69 (naive 100)`, `5000 -> 86.52 (naive 500)`.

### Show the saved figure

```python
from IPython.display import Image, display
display(Image("figs/pate_accuracy_comparison.png"))
```

Expected: the bar chart above, with your own run's numbers.

### Where the noise actually changed labels

```python
sorted_votes = np.sort(votes, axis=1)
margin = sorted_votes[:, -1] - sorted_votes[:, -2]
flipped = student_labels != clean_preds
print(f"labels changed by noise: {flipped.sum()} of {len(flipped)} ({flipped.mean():.1%})")
print(f"median margin: {np.median(margin):.0f} | mean margin where flipped: {margin[flipped].mean():.1f} vs not flipped: {margin[~flipped].mean():.1f}")
print(f"queries with margin < 40 (~1.4 noise std): {(margin < 40).mean():.1%}")
```

Expected: only a few percent of labels change, and the changed ones have **much smaller** margins (the toy in Part 1, live). A change isn't always an error: sometimes noise flips a wrong clean label to the right one.

## Key facts to lock in

- **PATE labels each public sample with argmax_j(n_j(x) + Lap(b)), with an independent Laplace draw added to every class count**. At b = 20 the noise has std 20·√2 ≈ 28.3 votes, so large-margin votes survive and close votes can flip.
- ==Because one record can move one teacher's vote (L1 sensitivity 2), each query is ε₀ = 2/b differentially private==. At scale 20, ε₀ = 0.10, since sliding the Laplace curve by 2 changes any output probability by at most e^0.1 ≈ 1.105.
- ==Advanced composition gives total ε = √(2·k·ε₀²·ln(1/δ)) + k·ε₀·(e^ε₀ − 1), growing with √k instead of k==. For 5,000 queries at ε₀ = 0.1 and δ = 10⁻⁵ the course's own `compute_privacy_budget` returns ≈ 86.5 (33.9 + 52.6), not the 8.81 the prose claims. For very few queries, naive composition is tighter.
- **Every call to `noisy_argmax` spends budget**: re-drawing the noise to get "better" labels is extra queries. And the clean argmax must never be released, because it has no privacy at all.
- **In the course's run, 80.2% average teachers became an 87.8% clean ensemble, and noise cost only 0.6 points (87.2% noisy labels)**, because most MNIST queries have very strong teacher consensus.
- **The ensemble gains only ~8 points over its 80% members because teacher errors are highly correlated**. Independent 80% voters would give a near-perfect majority, so label quality (and the student) is capped by the shared hard cases.
- **The course's local `plot_accuracy_comparison` saves and closes the figure (nothing inline) and shadows the DP-SGD library function of the same name**. The bars also mix evaluation sets (teachers on holdout, ensemble on the public queries).
- **Method point:** **when a client reports a PATE ε, recompute it from the noise scale, number of queries and δ (and ask whether it's data-dependent), check that labels were generated exactly once, and report clean-ensemble vs noisy-label accuracy as the privacy cost to utility**.

---
