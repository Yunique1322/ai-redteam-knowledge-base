# Tuning PATE And Comparing Approaches

## The idea in one breath

This section sweeps the noise scale from 5 to 80 and shows the tradeoff. More noise means a smaller ε, and label accuracy holds near 87% up to scale 20, then drops sharply (57% at scale 80). It then compares PATE with DP-SGD, sketches three ways to deploy with confident aggregation, covers the hyperparameters and limitations, and saves a results file.

Your figure shows **ε ≈ 87 at scale 20**, which confirms my Section 15 hand calculation (86.5). Several prose numbers in this section ("ε drops from ~35 to ~9", "95% label accuracy at scale 5", "ε ≈ 3.9 / 6.2 / 8.81 / 12.5") contradict the course's own code and figure.

> **Heads-up:** the stray "ϵ" on its own line after "stronger privacy guarantees (lower" just reads "**(lower ε)**". You pasted the figure twice; both copies are identical.
> 
> **I merged one code unit:** the course split `plot_privacy_utility_tradeoff` across two blocks (the second block is the indented rest of the same function). It's given below as **one cell**, with every line unchanged.

---

## Part 1: The noise-scale sweep

### Step 1: Toy, why ε falls fast when you add noise

Doubling the noise scale **halves** ε₀. But the total ε has two parts: one grows with ε₀, and one grows with ε₀² (Section 15's correction term). At large ε₀ the ε₀² part dominates, and **halving ε₀ quarters it**. So the total falls by more than half:

|Scale change|ε₀|Total ε (code's formula)|Falls by|
|---|---|---|---|
|5 → 10|0.40 → 0.20|1119 → 289|÷3.9|
|10 → 20|0.20 → 0.10|289 → 86.5|÷3.3|
|40 → 80|0.05 → 0.025|29.8 → 11.65|÷2.6 (√k part matters more now)|

```python
print("\n" + "=" * 60)
print("PHASE 7: Privacy-Utility Tradeoff Analysis")
print("=" * 60)

noise_scales = [5, 10, 20, 40, 80]

print(f"\n{'Noise Scale':<12} {'Per-ε':<10} {'Total ε':<12} {'Label Acc':<12} {'Flip Rate':<12}")
print("-" * 58)

for ns in noise_scales:
    noisy_labels = noisy_argmax(votes, ns)

    per_eps = 2.0 / ns
    total_eps, _ = compute_privacy_budget(num_queries, ns)

    label_acc = (noisy_labels == y_query_true).mean()
    flip_rate = (noisy_labels != clean_labels).mean()

    print(f"{ns:<12} {per_eps:<10.4f} {total_eps:<12.2f} {label_acc:<12.4f} {flip_rate:<12.4f}")
```

|Code|What it does|
|---|---|
|`for ns in noise_scales`|5 fresh noisy labelings of the same 5,000 queries (5 more releases)|
|`per_eps = 2.0 / ns`|ε₀|
|`total_eps, _ = compute_privacy_budget(...)`|advanced composition for 5,000 queries. **This overwrites your scale-20 `total_eps`.**|
|`label_acc`, `flip_rate`|as in Sections 15 and 17 (fractions, 0-1)|

> **If this errors / goes wrong later:** after the loop, `total_eps` holds the **scale-80** value (11.65) and `noisy_labels` holds the scale-80 labels. The results file at the end saves that wrong ε (Part 6).

**What the code really prints for ε** (the same numbers as your figure):

|Scale|ε₀|Total ε|Label accuracy (figure)|
|---|---|---|---|
|5|0.40|**1119.4**|0.878|
|10|0.20|**289.3**|0.877|
|20|0.10|**86.52**|0.872|
|40|0.05|**29.79**|0.818|
|80|0.025|**11.65**|0.572|

> **An honest note the course skips:**
> 
> 1. The prose gives label accuracy of 95% / 92% / 87% / 78% / 65%. **Noisy labels can't beat the clean ensemble (87.8%)**: as noise shrinks toward zero, label accuracy rises _to_ 87.8%, never above it. So "95% at scale 5" is impossible. Your figure shows 87.8% there.
> 2. "ε drops from ~35 to ~9 between scale 5 and 20" is contradicted by **its own figure**, which goes from about 1,100 to about 87.

**The practical range:** the course says scale 10-40, with 20 as the elbow. On the accuracy curve that holds: flat until 20, −5.4 points by 40, then −25 points by 80. But at scale 20 the ε is about 87, not single digits.

---

## Part 2: The tradeoff plot

```python
def plot_privacy_utility_tradeoff(noise_scales, votes, y_true, num_queries, save_path=None):
    """Plot privacy vs utility across noise scales on dual axes."""
    accuracies = [(noisy_argmax(votes, ns) == y_true).mean() for ns in noise_scales]
    epsilons = [compute_privacy_budget(num_queries, ns)[0] for ns in noise_scales]

    fig, ax1 = plt.subplots(figsize=PLOT_CONFIG['figsize'])
    ax1.set_xlabel('Noise Scale')
    ax1.set_ylabel('Label Accuracy', color=HTB_GREEN)
    ax1.plot(noise_scales, accuracies, 'o-', color=HTB_GREEN, linewidth=2, markersize=10)
    ax1.tick_params(axis='y', labelcolor=HTB_GREEN)
    ax2 = ax1.twinx()
    ax2.set_ylabel('Privacy Budget (ε)', color=AZURE)
    ax2.plot(noise_scales, epsilons, 's--', color=AZURE, linewidth=2, markersize=10)
    ax2.tick_params(axis='y', labelcolor=AZURE)
    ax2.set_yscale('log')

    ax1.set_title('Privacy-Utility Tradeoff', color=HTB_GREEN, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    if save_path:
        plt.savefig(save_path, dpi=PLOT_CONFIG['dpi'], bbox_inches='tight', facecolor=NODE_BLACK)
    plt.close()
```

|Code|What it does|
|---|---|
|`accuracies = [...]`|**draws the noise again** inside the plot, so the plotted accuracies differ slightly from the table's|
|`epsilons = [... [0] ...]`|`[0]` takes total ε from the returned (total, per-query) pair|
|`'o-'` / `'s--'`|circle markers with a solid line / square markers with a dashed line|
|`ax1.twinx()`|a **second y-axis** on the right, sharing the same x-axis|
|`ax2.set_yscale('log')`|the ε axis jumps ×10 per gridline (10¹, 10², 10³)|
|`savefig` + `close`|saves only, no inline display|

> **If this errors / goes wrong later:** this name is the same as the DP-SGD library function `plot_privacy_utility_tradeoff`, but with different arguments. In a shared notebook it replaces the library version (the same clash as Section 15).

```python
os.makedirs("figs", exist_ok=True)
plot_privacy_utility_tradeoff(noise_scales, votes, y_query_true, num_queries,
                               save_path="figs/pate_privacy_utility.png")
```

Saves `figs/pate_privacy_utility.png`, which is the figure you pasted.

---

## Part 3: PATE vs DP-SGD

|Aspect|PATE|DP-SGD|
|---|---|---|
|Where the noise goes|teacher vote counts (Laplace)|per-sample gradients (Gaussian)|
|Needs public data?|**yes, matching the private distribution** (often the deal-breaker: medical, finance, personalized models)|no, trains directly on private data|
|What spends budget|each **labeling query**|each **training step**|
|Inference after training|free|**also free**|
|Compute|hundreds of teachers|2-5× slower training, per-sample clipping to tune|
|Representation|limited to what distillation transfers|can learn richer features from the private data itself|
|Pick it when|good public proxy data exists and you have high-volume inference|no public proxy exists, or you need a complex model trained on the private data|

> **An honest note the course skips:**
> 
> 1. The course says DP-SGD has "per-query privacy costs" while PATE answers queries "indefinitely". **A DP-SGD model can also be queried unlimited times for free**, because inference is post-processing for both. DP-SGD's cost is per **training step**, not per query.
> 2. "Confident aggregation lets you reject low-consensus queries without cost" is the Section 17 flaw again: a clean-vote threshold isn't DP.
> 3. The "information bottleneck as an intuitive privacy argument for stakeholders" should be used carefully, because membership is one bit (Section 13).

---

## Part 4: The three deployment strategies

### Strategy 1: Fixed privacy budget

```python
def estimate_queries_for_budget(target_epsilon, noise_scale, delta=1e-5):
    """Binary search for maximum queries within privacy budget."""
    per_query_eps = 2.0 / noise_scale
    low, high = 1, 100000
    while low < high:
        mid = (low + high + 1) // 2
        eps_sq_sum = mid * (per_query_eps ** 2)
        total_eps = np.sqrt(2 * eps_sq_sum * np.log(1 / delta))
        total_eps += mid * per_query_eps * (np.exp(per_query_eps) - 1)
        low, high = (mid, high) if total_eps <= target_epsilon else (low, mid - 1)
    return low
```

|Code|What it does|
|---|---|
|`low, high = 1, 100000`|the search range for the number of queries|
|`mid = (low + high + 1) // 2`|the midpoint, rounded up so the loop always makes progress|
|the two `total_eps` lines|the same advanced-composition formula as `compute_privacy_budget` (local variable only)|
|`(mid, high) if ... else (low, mid − 1)`|under budget: search higher; over budget: search lower|
|`return low`|the **largest number of queries that stays ≤ target ε**|

**Toy, by hand, for ε = 10 at scale 20:** 241 queries give 7.45 + 2.53 = **9.98** ≤ 10 ✓, while 242 give 7.46 + 2.55 = **10.01** > 10 ✗. So it returns **241**. ==Under this accounting, ε = 10 buys only 241 labels.==

```python
_, confident_mask = confident_aggregation(votes, noise_scale, threshold=200)
acceptance_rate = confident_mask.mean()

target_epsilon = 10.0
standard_queries = estimate_queries_for_budget(target_epsilon, noise_scale)
confident_queries = standard_queries / acceptance_rate

print(f"For ε={target_epsilon}:")
print(f" Standard aggregation: {standard_queries} queries")
print(f" Confident (threshold=200): {int(confident_queries)} queries")
print(f" Net usable labels: {int(confident_queries * acceptance_rate)} samples")
```

|Code|What it does|
|---|---|
|`confident_aggregation(..., 200)`|one more release, used only to measure the acceptance rate (~0.69 in your run)|
|`standard_queries`|**241**|
|`confident_queries = 241 / 0.69`|≈ 350 queries "allowed", because rejections are assumed free|
|`Net usable labels`|350 × 0.69 = **241**, the same as standard **by construction**. The only gain is label quality|

> **An honest note the course skips:** the "extra" queries exist only because the course assumes rejected queries cost nothing, which Section 17 showed isn't DP with a clean threshold.

### Strategy 2: Fixed label count

```python
target_labels = 5000
confident_queries = int(target_labels / acceptance_rate)

conf_eps = compute_confident_privacy_budget(
    confident_queries, noise_scale, acceptance_rate
)[0]

print(f"For {target_labels} labels:")
print(f" Queries needed: {confident_queries}")
print(f" Privacy cost: ε = {conf_eps:.2f}")
```

|Code|What it does|
|---|---|
|`int(5000 / acceptance_rate)`|about 7,250 queries at your ~69% (5,882 at the course's 85%)|
|`compute_confident_privacy_budget(...)[0]`|charges only the ~5,000 accepted queries, so it prints **ε ≈ 86.5**, the same as standard labeling of 5,000|

**So confident aggregation doesn't lower the cost of 5,000 labels**, even by the course's own accounting. It only makes them more accurate. Counting the threshold checks properly, it costs **more**.

### Strategy 3: Iterative sampling

```python
def iterative_confident_labeling(teachers, X_public, threshold, target_count, noise_scale, device):
    """Query samples until target_count high-consensus labels obtained."""
    confident_X, confident_y, queries_made = [], [], 0
    available_indices = np.random.permutation(len(X_public))

    for idx in available_indices:
        if len(confident_y) >= target_count:
            break
        x = X_public[idx:idx+1]
        labels, mask = confident_aggregation(get_teacher_votes(teachers, x, device), noise_scale, threshold)
        queries_made += 1
        if mask[0]:
            confident_X.append(x[0])
            confident_y.append(labels[0])

    return np.array(confident_X), np.array(confident_y), queries_made
```

|Code|What it does|
|---|---|
|`np.random.permutation(...)`|visits the public images in random order|
|`if len(confident_y) >= target_count: break`|stop once there are enough labels|
|`X_public[idx:idx+1]`|slicing keeps a 2-D shape (1, 784), which the teachers need|
|`get_teacher_votes(teachers, x, ...)`|all 250 teachers vote on **one** image|
|`if mask[0]`|keep it only if accepted|
|returns|images, labels, and the number of queries made|

**Expected:** with 85% acceptance, 5,000 / 0.85 ≈ 5,882 queries ✓. With your ~69%, it's about 7,250.

> **If this errors / is very slow:** it's only defined, not called. If you call it, every image triggers 250 separate `get_model_predictions` calls (each building its own DataLoader), so 7,000+ images means millions of tiny calls, which can take a very long time. Vote in batches instead (as Section 15 did) and filter afterwards.

---

## Part 5: Hyperparameters and limitations

|Knob|More of it means|Less of it means|
|---|---|---|
|Teachers|stronger consensus, but weaker individual teachers (less data each)|stronger teachers, but a weaker vote|
|Queries|more student data, more ε|less data, less ε|

> **An honest note the course skips:** the query-budget prose ("1,000 → ε ≈ 3.9, 2,500 → 6.2, 5,000 → 8.81, 10,000 → 12.5", "doubling adds ~40%") is just 8.81 scaled by √k. **The code's real values at scale 20 are 25.69 / 50.28 / 86.52 / 153.16.** Doubling the queries adds about **72-77%**, because the linear k·ε₀² term dominates here. The "+40%" rule only holds when ε₀ is very small.

|Limitation|Why it matters|Mitigation|
|---|---|---|
|Public data must match the private distribution|distribution shift hurts more than label noise|often impossible for medical, finance or personalized models, so use DP-SGD|
|Class imbalance|majority classes get stronger consensus; the minority gets noisier labels and a weaker student|stratify the query set (helps, doesn't fix it)|
|Disjoint partitions|**a person appearing in k partitions influences k teachers**: sensitivity becomes 2k, so ε is k× larger (group privacy)|deduplicate by **person** before partitioning, or account for the maximum k|

---

## Part 6: Saving results

```python
# Information bottleneck metrics
private_data_bytes = len(X_private_norm) * X_private_norm.shape[1] * 4 # float32
student_info_bytes = num_queries * 4 # 4 bits per label, ~0.5 bytes, round to int size
compression_ratio = private_data_bytes / student_info_bytes

print(f"\nInformation Bottleneck:")
print(f" Private data: {private_data_bytes / 1e6:.1f} MB")
print(f" Student labels: {student_info_bytes / 1e3:.1f} KB")
print(f" Compression ratio: {compression_ratio:.0f}:1")
```

|Code|What it does|
|---|---|
|`48000 × 784 × 4`|150,528,000 bytes = **150.5 MB**|
|`num_queries * 4`|**4 **bytes** per label** (the comment says 4 bits) = 20,000 bytes = **20.0 KB**|
|`compression_ratio`|150,528,000 / 20,000 = **7,526 : 1**|

> **An honest note the course skips:** Section 13 used 4 **bits** per label (2.5 KB, 60,000 : 1). This code uses 4 **bytes** (20 KB, 7,526 : 1). Either way it's intuition, not the guarantee.

```python
results = {
    'configuration': {'dataset': 'mnist', 'num_teachers': TEACHER_CONFIG['num_teachers'],
                      'noise_scale': AGGREGATION_CONFIG['noise_scale'], 'num_queries': num_queries},
    'privacy': {'per_query_epsilon': float(per_query_eps), 'total_epsilon': float(total_eps)},
    'utility': {'student_accuracy': float(student_test_acc), 'label_accuracy': float(label_accuracy),
                'clean_ensemble_accuracy': float(clean_ensemble_acc)},
    'information': {'private_data_bytes': int(private_data_bytes), 'student_info_bytes': int(student_info_bytes),
                    'compression_ratio': float(compression_ratio)}
}

results_path = os.path.join("figs", "pate_results.json")
with open(results_path, 'w') as f:
    json.dump(results, f, indent=2)
print(f"Results saved to {results_path}")
```

|Key|What actually gets saved|Problem|
|---|---|---|
|`noise_scale`|20.0|✓|
|`per_query_epsilon`|0.1 (from Section 15)|✓|
|`total_epsilon`|**11.65 (the sweep's scale-80 leftover)**|✗ should be **86.52**. That understates the cost about 7×|
|`student_accuracy`, `label_accuracy`|percentages (~88, ~87.2)|✓|
|`clean_ensemble_accuracy`|**0.878**, a fraction (redefined in Section 17)|✗ mixed units in one file|
|no δ, no accounting method| - |✗ an ε without δ is an incomplete claim|

Also note the file goes in `figs/`, not `output/`.

---

## Reading the figure/output

**Privacy-Utility Tradeoff:** two y-axes over one x-axis (noise scale).
![Tuning PATE and Comparing Approaches](/assets/18%20Tuning%20PATE%20and%20Comparing%20Approaches-20261023190851660.png)

|Scale|Green (label accuracy, left axis)|Blue (ε, right log axis)|What it means|
|---|---|---|---|
|5|0.878|≈ 1,100 (just above 10³)|labels as good as clean, almost no privacy|
|10|0.877|≈ 290|still no accuracy cost|
|20|0.872|≈ 87 (just under 10²)|**the accuracy elbow; ε matches my 86.5**|
|40|0.818|≈ 30|−5.4 points|
|80|0.572|≈ 11.6|**label quality collapses**|

**The asymmetry:** from 5 to 20, ε falls about 13× (1,100 → 87) for only 0.6 points of accuracy. From 20 to 80, ε falls about 7× more (87 → 11.6) but accuracy loses 30 points. Even at scale 80 (labels only 57% correct), ε is still about 11.6. **With data-independent accounting, PATE on this setup can't reach single-digit ε with useful labels.** That's why real PATE relies on data-dependent analysis and Confident-GNMax.

---

## Copy-paste cells (mine, not the course's)

I didn't run these. They're pure arithmetic plus a JSON fix, with no extra noisy releases.

### The real ε numbers

```python
print(f"{'Scale':<7}{'eps0':<7}{'eps (code formula)':<20}")
for ns in noise_scales:
    print(f"{ns:<7}{2/ns:<7.3f}{compute_privacy_budget(num_queries, ns)[0]:<20.2f}")
for k in [1000, 2500, 5000, 10000]:
    print(f"{k:>6} queries @20: eps {compute_privacy_budget(k, 20)[0]:.2f}")
print("max queries for eps=10 at scale 20:", estimate_queries_for_budget(10.0, 20))
```

Expected: 1119.40, 289.33, 86.52, 29.79, 11.65; then 25.69, 50.28, 86.52, 153.16; then **241**.

### Fix results.json (the right ε, units and δ)

```python
total_eps, per_query_eps = compute_privacy_budget(num_queries, AGGREGATION_CONFIG['noise_scale'])
results['privacy']['total_epsilon'] = float(total_eps)
results['privacy']['delta'] = 1e-5
results['privacy']['accounting'] = 'advanced composition (data-independent); confident-threshold check not charged'
results['utility']['clean_ensemble_accuracy'] = float(clean_ensemble_acc * 100 if clean_ensemble_acc <= 1 else clean_ensemble_acc)
results['information']['student_info_bytes_4bit'] = num_queries * 0.5
with open(results_path, 'w') as f:
    json.dump(results, f, indent=2)
print(json.dumps(results['privacy'], indent=2), results['utility'])
```

Expected: `total_epsilon: 86.52...`, `delta: 1e-05`, and all three accuracies as percentages (~88, ~87.2, ~87.8).

To view the figure: `display(Image("figs/pate_privacy_utility.png"))`.

---

## Quick self-check

(a) After the course's noise sweep loop, what value does `total_eps` hold, and what does `pate_results.json` therefore report as `total_epsilon`?  
(b) What is the true total ε of the deployed student (trained on Section 15's `student_labels`)?  
(c) A colleague writes in a report: "choose PATE over DP-SGD because a DP-SGD model spends privacy budget every time it's queried." True or false, and why?

---

## The facts worth locking in for the exam

- **Raising PATE's noise scale lowers ε faster than linearly but barely hurts labels until consensus margins are reached**. By the code's advanced composition for 5,000 queries, ε is about 1,119 / 289 / 86.5 / 29.8 / 11.65 at scales 5 / 10 / 20 / 40 / 80, while label accuracy goes 87.8 / 87.7 / 87.2 / 81.8 / 57.2%.
- **Noisy label accuracy can never systematically beat the clean ensemble (87.8% here), since noise only adds errors**. The course's "95% at scale 5" is impossible, and its "ε ~35 → ~9" contradicts its own plot (~1,100 → ~87).
- **Under data-independent advanced composition at scale 20, ε grows by about 72-77% each time the query count doubles** (25.7 / 50.3 / 86.5 / 153.2 for 1k / 2.5k / 5k / 10k), not the +40% the course claims. A budget of ε = 10 allows only 241 queries.
- **Confident aggregation doesn't make 5,000 labels cheaper even by the course's accounting (still ≈86.5)**; it only makes them more accurate. Its "extra queries within a fixed budget" rely on rejections being free, which isn't true without a noisy threshold.
- **Inference is free for both PATE students and DP-SGD models because it's post-processing**. DP-SGD spends budget per training step and PATE per labeling query. PATE needs public data matching the private distribution, while DP-SGD trains on the private data directly.
- **PATE's guarantee assumes each person is in exactly one partition**. Someone in k partitions multiplies the sensitivity (2k) and ε by k, so deduplicate by person before partitioning. Class imbalance gives minority classes weaker consensus and noisier labels.
- ==The course's results file saves `total_epsilon` = 11.65 (the scale-80 value left over from the sweep loop) next to `noise_scale` 20==, understating the cost about 7×, and stores accuracies in mixed units. Always recompute ε for the configuration actually deployed and record δ and the accounting method.
- **Method point:** **when tuning or auditing PATE, sweep the noise scale against label accuracy and pick the largest scale before accuracy bends, compute ε for the exact queries released (including any re-draws and threshold checks), and choose PATE vs DP-SGD on whether in-distribution public data exists, not on inference cost**.

---

Logged to COAE-progress.md, with the self-check answer key. For the PATE challenge, I've also noted the four big gotchas: the ε is 86.5 not 8.81, `results.json` saves the wrong ε, the scaler isn't saved, and the clean-vote threshold isn't DP.