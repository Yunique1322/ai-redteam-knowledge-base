# Consensus Analysis And Confident Aggregation

## The idea in one breath

Some pseudo-labels are much more reliable than others. When 200+ of the 250 teachers agree (**high consensus**), the clean vote is ~98% right and noise almost never flips it. When fewer than 150 agree (**low consensus**), it's ~65% right and noise flips it often. **Confident aggregation** only labels samples above a vote threshold, so you get fewer labels but more accurate ones, and the course claims a lower ε.

The big catch: **the course's version checks the threshold on the clean, un-noised votes, which is not differentially private**, so its "rejected queries are free" claim is wrong.

> **Heads-up:** no mangled formulas. Three things to keep straight:
> 
> - "ε Saved" in the threshold table is really **% of queries saved**, not % of ε.
> - In this section `clean_ensemble_acc` becomes a **fraction (0-1)**. In Sections 15-16 it was a **percentage**. Same name, different units (see Part 2).
> - You pasted the figure twice; both copies are identical.

---

## Part 1: The three consensus groups

|Group|Max votes (of 250)|Share|Clean accuracy (course)|Noise flips|
|---|---|---|---|---|
|**High**|≥ 200|≥ 80%|~98%|~never|
|**Medium**|150-199|60-80%|~85%|sometimes|
|**Low**|< 150|< 60%|~65%|often|

```python
print("\n" + "=" * 60)
print("PHASE 5: Teacher Consensus Analysis")
print("=" * 60)

num_teachers = len(teachers)
```

Just a banner, and the teacher count (250) read from the list.

```python
# Calculate consensus metrics
max_votes = votes.max(axis=1)

high_threshold = 200 # 80% of 250 teachers
medium_threshold = 150 # 60% of 250 teachers

high_consensus = max_votes >= high_threshold
medium_consensus = (max_votes >= medium_threshold) & ~high_consensus
low_consensus = max_votes < medium_threshold

print(f"\nConsensus Distribution ({num_queries} samples):")
print(f" High (≥80%): {high_consensus.sum()} ({100*high_consensus.mean():.1f}%)")
print(f" Medium (60-80%): {medium_consensus.sum()} ({100*medium_consensus.mean():.1f}%)")
print(f" Low (<60%): {low_consensus.sum()} ({100*low_consensus.mean():.1f}%)")
```

|Code|What it does|
|---|---|
|`votes.max(axis=1)`|the winning class's vote count for each of the 5,000 queries|
|`max_votes >= 200`|a True/False **mask**: True where consensus is high|
|`& ~high_consensus`|`~` flips True/False, so "at least 150 AND not high" = 150-199|
|`.sum()` / `.mean()`|how many True (count), and what share is True (fraction)|

Every sample falls in exactly one group.

> **An honest note the course skips:** what really decides whether noise flips a label is the **margin** (top count minus second count), not the top count alone. Max ≥ 200 does guarantee a big margin: the other 50 votes can't beat 150. But a max of 130 could be 130 vs 120 (risky) or 130 vs 40 (safe). My cell 1 checks flips by margin.

---

## Part 2: Clean accuracy by group

```python
# Calculate clean ensemble accuracy (no noise)
clean_labels = votes.argmax(axis=1)
clean_ensemble_acc = (clean_labels == y_query_true).mean()

# Calculate accuracy by consensus level
def accuracy_for_mask(mask, labels, true_labels):
    if mask.sum() == 0:
        return 0.0
    return (labels[mask] == true_labels[mask]).mean()

clean_high = accuracy_for_mask(high_consensus, clean_labels, y_query_true)
clean_medium = accuracy_for_mask(medium_consensus, clean_labels, y_query_true)
clean_low = accuracy_for_mask(low_consensus, clean_labels, y_query_true)

print(f"\nClean Ensemble Accuracy by Consensus:")
print(f" High consensus: {clean_high * 100:.2f}%")
print(f" Medium consensus: {clean_medium * 100:.2f}%")
print(f" Low consensus: {clean_low * 100:.2f}%")
print(f" Overall: {clean_ensemble_acc * 100:.2f}%")
```

|Code|What it does|
|---|---|
|`clean_labels`|the same as `clean_preds` from Section 15 (the no-noise winner)|
|`clean_ensemble_acc = (...).mean()`|**now a FRACTION** (0.878), with no `* 100`|
|`accuracy_for_mask(...)`|accuracy within one group. Returns 0.0 if the group is empty, which avoids a divide-by-zero|
|`labels[mask]`|keeps only the rows where the mask is True|

**Confidence correlates with correctness**: the more teachers agree, the more likely they're right.

> **If this errors / goes wrong later:** `clean_ensemble_acc` was a **percentage** in Sections 15-16. If you re-run Section 15's plot cell (which divides by 100) or Section 16's gap cell after this one, the numbers come out 100× too small. Run the sections in order, or rename this one.

> **An honest note the course skips:** the course calls 65% on low-consensus samples "approaching random guessing". Random for 10 classes is **10%**. 65% means hard, not random.

---

## Part 3: How noise flips labels, by group

```python
# Apply noisy aggregation
noise_scale = AGGREGATION_CONFIG['noise_scale']
noisy_labels = noisy_argmax(votes, noise_scale)

# Compare to clean predictions
noise_flipped = (noisy_labels != clean_labels)
flip_rate = noise_flipped.mean()

print(f"\nNoisy Aggregation Results (noise_scale={noise_scale}):")
print(f" Labels matching true: {(noisy_labels == y_query_true).mean() * 100:.2f}%")
print(f" Labels matching clean: {(~noise_flipped).mean() * 100:.2f}%")
print(f" Noise flip rate: {flip_rate * 100:.2f}%")
```

|Code|What it does|
|---|---|
|`noisy_argmax(votes, 20)`|a **fresh** noisy labeling of the same 5,000 queries|
|`noise_flipped`|True where the noise changed the clean winner|
|`flip_rate`|the share of labels the noise changed (a few %)|

```python
# Flip rate by consensus level
flip_high = noise_flipped[high_consensus].mean()
flip_medium = noise_flipped[medium_consensus].mean()
flip_low = noise_flipped[low_consensus].mean()

print(f"\nNoise Flip Rate by Consensus:")
print(f" High consensus: {flip_high * 100:.2f}%")
print(f" Medium consensus: {flip_medium * 100:.2f}%")
print(f" Low consensus: {flip_low * 100:.2f}%")
```

Expected pattern: high ≈ 0%, medium a little, low the most. That's Section 15's Laplace toy (a big margin survives) showing up in real data.

> **An honest note the course skips:** this is a **second noisy release** of the same queries (Section 15's "every re-draw spends budget" rule). The student was trained on `student_labels` only, so its guarantee covers that one release. But in a real system, every one of these extra analysis draws would count too.

---

## Part 4: Confident aggregation, and why the course's version isn't private

### Step 1: Toy, threshold 200

Three queries with max votes **[240, 180, 120]**:

- 240 ≥ 200: accepted, gets a noisy label (say 7)
- 180 and 120: rejected, get the **sentinel** **−1** ("no label")

Result: labels [7, −1, −1], acceptance rate 1/3. Only the 240 sample trains the student.

```python
def confident_aggregation(votes, noise_scale, threshold):
    """Label only samples where teacher consensus exceeds threshold."""
    max_votes = votes.max(axis=1)
    confident_mask = max_votes >= threshold
    labels = np.full(len(votes), -1, dtype=np.int64)

    if confident_mask.sum() > 0:
        confident_votes = votes[confident_mask]
        noise = np.random.laplace(loc=0.0, scale=noise_scale, size=confident_votes.shape)
        labels[confident_mask] = np.argmax(confident_votes.astype(np.float64) + noise, axis=1)

    return labels, confident_mask
```

|Code|What it does|
|---|---|
|`confident_mask = max_votes >= threshold`|**the decision uses the clean, un-noised votes**|
|`np.full(len(votes), -1)`|start every label as −1. Real classes are 0-9, so −1 can't be confused with one|
|`votes[confident_mask]`|keep only the accepted rows|
|`+ noise`, `argmax`|the usual noisy argmax, applied to accepted rows only|
|returns|`labels` (with −1s) and the mask (filter with `labels != -1` before training)|

### Step 2: Why this breaks DP

Take one query whose max vote is exactly **200** on dataset D: it's **accepted**. Now change one private record, so one teacher moves its vote away from the winner (neighbor D′): the max becomes **199**, and the query is **rejected**.

- P[reject | D] = **0**
- P[reject | D′] = **1**
- Ratio = **∞**, so no ε can bound it

**The "−1" itself is an output, and it depends on private data with no noise.** That's why "rejected queries release no information and cost nothing" is wrong here.

> **An honest note the course skips:** the real version, **Confident-GNMax** (Papernot 2018), adds noise to the threshold check too: _accept if max votes + noise ≥ T_. That check is **charged for every query, including rejected ones** (at a smaller cost than labeling), and it's tracked with an RDP accountant. So `compute_confident_privacy_budget` below **undercounts**. In a report: **confident aggregation must use a noisy threshold, and the check itself costs budget**. My cell 3 shows the noisy check.

### Step 3: Threshold sweep

```python
print("\n" + "=" * 60)
print("PHASE 6: Confident Aggregation Analysis")
print("=" * 60)

thresholds = [150, 175, 200, 215, 225]
```

```python
print(f"\nConfident Aggregation Analysis:")
print(f"{'Threshold':<12} {'Accepted':<12} {'Accuracy':<12} {'ε Saved':<12}")
print("-" * 48)

for threshold in thresholds:
    labels, mask = confident_aggregation(votes, noise_scale, threshold)
    accepted_rate = mask.mean()

    if mask.sum() > 0:
        accuracy = (labels[mask] == y_query_true[mask]).mean() * 100
    else:
        accuracy = 0.0

    epsilon_saved = (1 - accepted_rate) * 100

    print(f"{threshold:<12} {100*accepted_rate:<12.1f}% {accuracy:<12.2f}% {epsilon_saved:<12.1f}%")
```

|Code|What it does|
|---|---|
|`for threshold in thresholds`|runs confident aggregation 5 times, which is 5 more noisy releases|
|`accepted_rate = mask.mean()`|the share of queries labeled|
|`accuracy`|label accuracy on the **accepted** rows only, so it rises with the threshold|
|`epsilon_saved = (1 − accepted_rate) * 100`|**really "queries saved"**. ε doesn't fall in proportion (Part 5)|

**The tradeoff:** a higher threshold gives fewer labels with higher accuracy. A lower threshold gives more labels that are noisier.

---

## Part 5: The budget function

```python
def compute_confident_privacy_budget(num_queries, noise_scale, acceptance_rate, delta=1e-5):
    """Compute privacy budget considering only accepted (labeled) queries."""
    per_query_eps = 2.0 / noise_scale
    effective_queries = int(num_queries * acceptance_rate)
    epsilon_sq_sum = effective_queries * (per_query_eps ** 2)
    total_epsilon = np.sqrt(2 * epsilon_sq_sum * np.log(1 / delta))
    total_epsilon += effective_queries * per_query_eps * (np.exp(per_query_eps) - 1)
    return total_epsilon, per_query_eps, effective_queries
```

This is Section 15's advanced composition, applied to accepted queries only. `int(...)` rounds down. It returns 3 values this time.

**Worked numbers (the course's 85% acceptance):** k = 4,250. Term 1 = √(2 × 4250 × 0.01 × 11.51) = √978.6 = **31.28**; term 2 = 4250 × 0.1 × 0.1052 = **44.70**; total ≈ **75.98**. Compare 86.52 for all 5,000.

|Measure|Value|
|---|---|
|Queries saved|15%|
|ε actually saved|(86.52 − 75.98) / 86.52 = **12.2%**|

> **An honest note the course skips:** (1) ε doesn't fall "proportionally", because the √k term shrinks more slowly than k. (2) This number **ignores the cost of the accept/reject check** (Part 4), so the real ε is higher than this function says.

---

## Part 6: The consensus histogram and student accuracy by group

```python
def plot_consensus_distribution(votes, save_path=None):
    """Plot histogram of maximum vote counts with threshold markers."""
    fig, ax = plt.subplots(figsize=PLOT_CONFIG['figsize'])
    max_votes = votes.max(axis=1)
    ax.hist(max_votes, bins=range(100, 251, 10), color=AZURE, edgecolor=HACKER_GREY, alpha=0.8)
    ax.axvline(200, color=HTB_GREEN, linestyle='--', linewidth=2, label='High (≥200)')
    ax.axvline(150, color=AQUAMARINE, linestyle='--', linewidth=2, label='Medium (≥150)')
    ax.set_xlabel('Maximum Votes (out of 250 teachers)')
    ax.set_ylabel('Number of Samples')
    ax.set_title('Teacher Consensus Distribution', color=HTB_GREEN, fontweight='bold')
    ax.legend(framealpha=0.8)
    ax.grid(True, alpha=0.3)
    if save_path:
        plt.savefig(save_path, dpi=PLOT_CONFIG['dpi'], bbox_inches='tight', facecolor=NODE_BLACK)
    plt.close()
```

|Code|What it does|
|---|---|
|`bins=range(100, 251, 10)`|bars from 100 to 250 in steps of 10. **Samples below 100 votes aren't drawn at all**|
|`axvline(200 / 150)`|the dashed green (high) and cyan (medium) threshold lines|
|`savefig` + `close`|saves only; nothing shows inline (same display cell as before)|

```python
os.makedirs("figs", exist_ok=True)
plot_consensus_distribution(votes, save_path="figs/pate_consensus_distribution.png")
```

```python
# Evaluate student on samples grouped by original consensus
student_preds = get_model_predictions(student_model, X_query, DEVICE).argmax(axis=1)

student_high = (student_preds[high_consensus] == y_query_true[high_consensus]).mean() * 100
student_medium = (student_preds[medium_consensus] == y_query_true[medium_consensus]).mean() * 100
student_low = (student_preds[low_consensus] == y_query_true[low_consensus]).mean() * 100

print(f"\nStudent Accuracy by Original Consensus:")
print(f" High consensus samples: {student_high:.2f}%")
print(f" Medium consensus samples: {student_medium:.2f}%")
print(f" Low consensus samples: {student_low:.2f}%")
```

|Code|What it does|
|---|---|
|`get_model_predictions(student_model, X_query, ...)`|the student's predictions on **its own 5,000 training images**|
|`student_high/medium/low`|the student's accuracy against the **true** labels within each group|

The idea: the student is best where the teachers agreed, because that's where its labels were clean. **It can't learn what the teachers themselves couldn't agree on.**

> **An honest note the course skips:** (1) `X_query` is the student's **training set**, so this measures how well it fit its labels, not how well it generalizes. For generalization, group the **holdout** by the teachers' votes instead. (2) "Low consensus approaches random (10%)" contradicts the course's own 65% clean accuracy on that group. Expect something well above 10%. (3) These lines have no empty-group guard, so an empty group gives `nan` with a warning.

---

## Reading the figure/output

**Teacher Consensus Distribution** (my approximate reading of the bars):
![Consensus Analysis and Confident Aggregation](/assets/17%20Consensus%20Analysis%20and%20Confident%20Aggregation-20261023190715147.png)

|Range|Bars (approx. samples each)|Total|Share of 5,000|
|---|---|---|---|
|< 100|not drawn|~135|~3%|
|100-149 (low)|70, 90, 145, 135, 160|~600|~12%|
|150-199 (medium)|165, 135, 160, 195, 175|~830|~17%|
|200-249 (high)|245, 300, 350, 530, **2,010**|~3,435|**~69%**|

- **The tall last bar** (240-250, about 2,010 samples): **about 40% of queries are near-unanimous**. 230-250 together hold about half of all queries.
- **The green line at 200:** everything to its right is the high group, about 69%.
- **The long flat tail** from 100 to 200: about 1 in 4 queries is ambiguous to some degree.

> **An honest note the course skips:**
> 
> 1. The course says threshold 200 accepts "~85%". The figure shows **~69%**. 85% is about what **threshold 150** accepts (high + medium ≈ 4,265 / 5,000).
> 2. "The vast majority exceed even the strict threshold" and "low consensus is rare" are overstated: it's ~69% high and ~15% low (counting the off-chart samples).
> 3. The course's per-group accuracies (98 / 85 / 65), weighted by the figure's group sizes, give about **91%** overall, not the 87.8% from Section 15. The numbers come from different runs.

---

## Copy-paste cells (mine, not the course's)

I didn't run these. They use your names from this section.

### Real acceptance rates, and flips by margin

```python
max_votes = votes.max(axis=1)
sorted_votes = np.sort(votes, axis=1)
margin = sorted_votes[:, -1] - sorted_votes[:, -2]
print(f"below 100 votes (not on the histogram): {(max_votes < 100).sum()}")
for t in [150, 175, 200, 215, 225]:
    print(f"threshold {t}: accepts {np.mean(max_votes >= t):.1%}")
print(f"flip rate by MARGIN <40: {noise_flipped[margin < 40].mean():.1%} | >=40: {noise_flipped[margin >= 40].mean():.1%}")
```

Expected: about 135 below 100; threshold 200 accepts ~69% and 150 ~85%. Small-margin samples flip far more often than the rest.

### Honest ε per threshold (ε saved vs queries saved)

```python
full_eps, _ = compute_privacy_budget(num_queries, noise_scale)
for t in [150, 175, 200, 215, 225]:
    rate = np.mean(votes.max(axis=1) >= t)
    eps_t, _, k = compute_confident_privacy_budget(num_queries, noise_scale, rate)
    print(f"threshold {t}: {k} queries, eps {eps_t:.2f} (queries saved {1-rate:.1%}, eps saved {1-eps_t/full_eps:.1%}) "
          f"[ignores the cost of the accept/reject check]")
```

Expected: full ε 86.52. At ~85% acceptance, ε ≈ 76 (about 12% saved, not 15%).

### Confident aggregation with a noisy check (Confident-GNMax style)

This shows the mechanism only. Its budget needs an RDP accountant, which I haven't computed here.

```python
def noisy_threshold_aggregation(votes, noise_scale, threshold, threshold_noise=40.0):
    """Confident-GNMax-style: the accept/reject check itself is noisy (still costs budget - not computed here)."""
    rng = np.random.default_rng(RANDOM_SEED)
    noisy_max = votes.max(axis=1) + rng.normal(0, threshold_noise, size=len(votes))
    accept = noisy_max >= threshold
    labels = np.full(len(votes), -1, dtype=np.int64)
    if accept.any():
        v = votes[accept].astype(np.float64)
        labels[accept] = np.argmax(v + rng.laplace(0, noise_scale, size=v.shape), axis=1)
    return labels, accept

nt_labels, nt_mask = noisy_threshold_aggregation(votes, noise_scale, 200)
print(f"noisy check accepts {nt_mask.mean():.1%}, label accuracy {(nt_labels[nt_mask] == y_query_true[nt_mask]).mean():.1%}")
```

Expected: an acceptance rate near the clean threshold's (~69%), but now a 199-vote sample can sometimes pass and a 201 can sometimes fail. That randomness is what makes the check private. Label accuracy should still be high (~95%+).

To view the histogram: `display(Image("figs/pate_consensus_distribution.png"))`, using the same import as before.

## Key facts to lock in

- **Teacher consensus (the winning class's vote count) predicts label reliability**: in the course's run high consensus (≥200 of 250) was ~98% correct, medium (150-199) ~85%, low (<150) ~65%. Noise almost never flips high-consensus labels because the margin is huge.
- **Whether noise flips a label depends on the margin between the top two classes**, not on the top count alone. Max ≥ 200 guarantees a margin of at least 150, but a low max can still have a safe or a risky margin.
- **Confident aggregation labels only queries whose max vote reaches a threshold, marking the rest −1 and filtering them out**. That gives fewer labels with higher accuracy, and the course claims a lower ε.
- **The course's version decides acceptance from the clean, un-noised votes, which is not differentially private**: a neighbor dataset can turn 200 votes into 199 and flip accept to reject with certainty. The "−1" leaks, so "rejected queries cost nothing" is false.
- **Confident-GNMax fixes this with a noisy threshold check (max + noise ≥ T) that is charged on every query, including rejected ones**, at a smaller cost than labeling, and tracked with an RDP accountant.
- **The "ε Saved" column is really queries saved, and ε doesn't fall proportionally**: 4,250 instead of 5,000 queries gives ε ≈ 76.0 vs 86.5 (12.2% saved, not 15%), and that still ignores the threshold-check cost.
- **In the figure, about 69% of queries reach 200+ votes (about 40% are near-unanimous at 240-250), about 17% are medium and about 15% low**, including ~135 below 100 that the histogram doesn't draw. The course's "threshold 200 accepts ~85%" matches threshold 150 instead.
- **Method point:** **when auditing confident or selective labeling, check that the accept/reject decision itself is noised and budgeted, recompute ε for the accepted queries with the √k composition rather than proportionally, and evaluate the student by consensus group on held-out data, not on its own training queries**.

---
