# Analyzing The Privacy-Utility TradeoffUntitled

## The idea in one breath

This section puts the three models side by side: a table, two plots, and a JSON file. Then it interprets the results. **Lower ε means less leakage and less memorization, but lower accuracy**, and the extra privacy gets **more expensive per unit** as ε drops. It ends with DP-SGD's limits (2-5× compute, big accuracy loss below ε = 1, no protection against evasion or model stealing) and the alternatives: **PATE**, **local DP** and **federated learning**.

> **Heads-up:** no mangled formulas. Two things to know before reading the figures:
> 
> - **The figures come from a different run than the prose.** The bars say 64.2 / 58.5 / 52.3, while the text says 67 / 58 / 53. Neither is wrong; they're separate runs.
> - **You pasted each figure twice.** They're identical, so I read each one once.

---

## Part 1: The two helper functions

```python
def print_comparison_table(test_acc_baseline, test_acc_dp10, test_acc_dp3,
                           mia_adv_baseline, mia_adv_dp10, mia_adv_dp3,
                           final_epsilon_10, final_epsilon_3):
    """Print a comparison table of all model results."""
    print("\n" + "=" * 70)
    print(" COMPARISON TABLE")
    print("=" * 70)
    print(f"\n{'Model':<20} {'Test Acc':<12} {'MIA Adv':<12} {'Epsilon':<10}")
    print("-" * 55)
    print(f"{'Baseline':<20} {f'{test_acc_baseline:.2f}%':<12} {mia_adv_baseline:<12.4f} {'∞':<10}")
    print(f"{'DP (ε=10)':<20} {f'{test_acc_dp10:.2f}%':<12} {mia_adv_dp10:<12.4f} {final_epsilon_10:<10.2f}")
    print(f"{'DP (ε=3)':<20} {f'{test_acc_dp3:.2f}%':<12} {mia_adv_dp3:<12.4f} {final_epsilon_3:<10.2f}")
```

|Code|What it does|
|---|---|
|`{'Model':<20}`|left-align (`<`) the text in a column 20 characters wide, which is how the columns line up|
|`{f'{test_acc_baseline:.2f}%':<12}`|an f-string inside an f-string: first build "67.13%", then pad it to 12 wide. It works because the inner one uses `'` and the outer uses `"`|
|`{mia_adv_baseline:<12.4f}`|4 decimals, left-aligned, 12 wide|
|`'∞'`|the baseline has no DP guarantee, so its ε is **infinite**|

```python
def save_results(test_acc_baseline, test_acc_dp10, test_acc_dp3,
                 mia_adv_baseline, mia_adv_dp10, mia_adv_dp3,
                 final_epsilon_10, final_epsilon_3, save_path="output/results.json"):
    """Save all results to a JSON file."""
    results = {
        "baseline": {
            "test_accuracy": float(test_acc_baseline),
            "mia_advantage": float(mia_adv_baseline),
        },
        "dp_eps10": {
            "test_accuracy": float(test_acc_dp10),
            "mia_advantage": float(mia_adv_dp10),
            "epsilon": float(final_epsilon_10),
        },
        "dp_eps3": {
            "test_accuracy": float(test_acc_dp3),
            "mia_advantage": float(mia_adv_dp3),
            "epsilon": float(final_epsilon_3),
        },
    }
    with open(save_path, "w") as f:
        json.dump(results, f, indent=2)
```

|Code|What it does|
|---|---|
|`results = {...}`|nested dictionary, one block per model|
|`float(...)`|makes every value a plain Python float before saving (the same "convert at the border" rule as your Shadow API sheet)|
|`json.dump(..., indent=2)`|writes readable JSON to `output/results.json`|

> **An honest note the course skips:** here the values are already Python floats or NumPy `float64` (a subclass of float), so JSON would cope even without `float()`. It's defensive, and still good habit. The bigger gap: **the JSON has no δ and no overfitting gap**. ε without δ is an incomplete privacy claim (Section 7). My cell 3 adds both.

---

## Part 2: The comparison table

```python
print_comparison_table(
    test_acc_baseline, test_acc_dp10, test_acc_dp3,
    mia_adv_baseline, mia_adv_dp10, mia_adv_dp3,
    final_epsilon_10, final_epsilon_3
)
```

This just calls the helper with the 8 values from Section 10. The course's reading: accuracy 67 → 58 → 53, MIA advantage 0.019 → 0.008 → 0.004, each step cutting the advantage by roughly 50-60% (0.019→0.008 is −58%, 0.008→0.004 is −50%).

> **An honest note the course skips:** from Section 10, a zero-leak model already scores about **0.004**, and up to about 0.008 by luck. So the second "halving" (0.008 → 0.004) isn't real evidence; it's inside the noise. Your figure (Part 3) shows it even more clearly.

---

## Part 3: The visualizations

```python
plot_accuracy_comparison(
    test_acc_baseline, test_acc_dp10, test_acc_dp3,
    save_path="figs/dp_sgd_accuracy_comparison.png"
)
print("Saved accuracy comparison to figs/dp_sgd_accuracy_comparison.png")
```

```python
plot_privacy_utility_tradeoff(
    test_acc_baseline, test_acc_dp10, test_acc_dp3,
    mia_adv_baseline, mia_adv_dp10, mia_adv_dp3,
    final_epsilon_10, final_epsilon_3,
    save_path="figs/dp_sgd_privacy_utility_tradeoff.png"
)
print("Saved privacy-utility tradeoff to figs/dp_sgd_privacy_utility_tradeoff.png")
```

|Code|What it does|
|---|---|
|`plot_accuracy_comparison(...)`|3-bar chart of test accuracy, saved as a PNG|
|`plot_privacy_utility_tradeoff(...)`|two panels (accuracy vs ε, MIA advantage vs ε), saved as a PNG|

> **If this errors / nothing shows:** the shadow-model plot helpers in this library saved the file and closed the figure, so nothing appeared inline. If that happens here too, use my cell 1 to display the PNGs.

---

## Part 4: Saving results and the final banner

```python
save_results(
    test_acc_baseline, test_acc_dp10, test_acc_dp3,
    mia_adv_baseline, mia_adv_dp10, mia_adv_dp3,
    final_epsilon_10, final_epsilon_3,
    save_path="output/results.json"
)
print("Results saved to output/results.json")
```

```python
print("\n" + "=" * 80)
print(" DEMONSTRATION COMPLETE")
print("=" * 80)
print("\nFindings:")
print(f" Baseline MIA advantage: {mia_adv_baseline:.4f}")
print(f" DP (ε=10) MIA advantage: {mia_adv_dp10:.4f} ({improvement_10:+.4f})")
print(f" DP (ε=3) MIA advantage: {mia_adv_dp3:.4f} ({improvement_3:+.4f})")
print("=" * 80)
```

|Code|What it does|
|---|---|
|`save_results(...)`|writes the JSON from Part 1|
|`improvement_10`, `improvement_3`|defined back in Section 10 (baseline advantage − DP advantage). `:+.4f` always shows the sign|

---

## Part 5: Interpreting the results, and the diminishing-returns math

### Step 1: Toy, the price of privacy

Think of privacy as something you buy with accuracy:

- **First purchase** (baseline → ε = 10): 0.011 less advantage costs 9 points, so **9 / 11 = 0.82 points per 0.001**.
- **Second purchase** (ε = 10 → ε = 3): 0.004 less advantage costs 5 points, so **5 / 4 = 1.25 points per 0.001**.

The price per unit went **up by about 50%**. That's what **diminishing returns** means here.

### Step 2: The course's version

> **An honest note the course skips:** the course says the first step costs "roughly 1.2 points per 0.001", but 9 / 0.011 = **0.82**. It then concludes "the marginal cost stays roughly constant". With the right number, the marginal cost **rises**, which makes its own diminishing-returns point stronger. On your figure's run the effect is extreme (see the self-check).

### Step 3: The overfitting gap

10 → 3 → 1 points. **This is the cleanest evidence DP-SGD works**, sturdier than MIA numbers that sit near the noise floor.

### Step 4: The end point

Once the measured advantage reaches the noise floor, **more noise buys no measurable privacy (with this attack) but still costs accuracy**. The worst-case DP bound keeps tightening, though: ε = 3 is still a much stronger _guarantee_ than ε = 10 (Section 7's table: 95.3% vs ~100% worst-case cap).

---

## Part 6: Limitations (what goes in a report's "residual risk")

|Limitation|Detail|
|---|---|
|Compute|2-5× slower; memory grows with batch size (per-sample gradients). Gradient accumulation and mixed precision help|
|Utility at ε < 1|ε = 3 already cost 14 points; ε = 1 could cost 20-30+. Public pretraining and better architectures help|
|**Protects**|individual records: membership inference, and extracting a single record's details|
|**Doesn't protect**|**aggregate patterns** (property inference, e.g. "average age"), **model stealing** via API queries, **adversarial examples** (your evasion modules)|

**For the report:** DP-SGD is the fix for a _privacy_ finding only. It's no fix for an evasion or extraction finding on the same model.

---

## Part 7: Alternatives

|Approach|Where the privacy comes from|Strength|Catch|
|---|---|---|---|
|**PATE** (next sections)|noisy **voting** by teachers trained on disjoint data splits|often better utility at the same ε|trains many teachers, needs clean partitions, **and a public unlabeled dataset for the student** (the course omits this)|
|**Local DP**|each user adds noise **before** sending data, like Section 7's coin-flip survey|you don't have to trust the curator|needs far more data for the same accuracy|
|**Federated learning**|raw data stays on devices; only updates are shared|data minimization|communication cost; struggles with non-IID data|

> **An honest note the course skips:** federated learning plus secure aggregation hides individual updates from the server, but it is **not** a formal privacy guarantee. Gradient-inversion attacks exist, and the final model can still leak membership. For a real guarantee you combine it with DP (e.g. DP-FedAvg).

**How to choose:** threat model (who are you protecting against?), data shape (can it be partitioned? is it IID?), and the accuracy budget.

---

## Reading the figure/output

### Figure 1: Bar chart, test accuracy
*(Figure not available in source notes.)*

| Model | Value on bar | Drop from baseline |
| -------------------------------------------------------------------------------------- | ------------ | ------------------ |
| Baseline (green) | **64.2%** | - |
| DP-SGD ε = 10 (blue) | **58.5%** | −5.7 pts |
| DP-SGD ε = 3 (red)*(Figure not available in source notes.)* | **52.3%** | −11.9 pts |

All three bars stay above the **50% line**, which is the challenge's accuracy target.

### Figure 2: The two panels


![Analyzing the Privacy-Utility TradeoffUntitled](/assets/11%20Analyzing%20the%20Privacy-Utility%20TradeoffUntitled-20261023190134127.png)

- **X-axis:** log scale and **inverted**: 10² on the left, 10¹ in the middle, and the unlabeled ε ≈ 3 point at the far right. **The baseline is drawn at ε = 100 only as a stand-in for ∞.** It doesn't mean the baseline has ε = 100.
- **Left panel (accuracy):** 64.2 → 58.5 → ~52.3. The dashed green line marks the baseline.
- **Right panel (MIA advantage):** **0.0220** → **~0.0037** → **~0.0034**. A steep drop, then nearly flat.

|Reading|Value|What it means|
|---|---|---|
|Baseline advantage|0.022|~5× the noise floor = real leakage|
|ε = 10 advantage|~0.0037|**below the ~0.004 zero-leak level**|
|ε = 3 advantage|~0.0034|also below it; the 0.0003 difference from ε = 10 is noise|

> **An honest note the course skips:** (1) the course says "both curves slope downward left-to-right as epsilon increases". With the inverted axis, **ε decreases** from left to right; the curves fall as privacy gets _stronger_. (2) It calls this a "dual-axis plot", but it's two side-by-side panels. (3) In this run there's no "cut in half again" from ε = 10 to ε = 3. **Both DP models are already at the floor**, so ε = 3 bought about 6 accuracy points' worth of nothing measurable.

---

## Copy-paste cells (mine, not the course's)

I didn't run these. They use your Section 10 names.

### Show the two saved figures

```python
from IPython.display import Image, display
display(Image("figs/dp_sgd_accuracy_comparison.png"))
display(Image("figs/dp_sgd_privacy_utility_tradeoff.png"))
```

Expected: the two figures above, drawn from your own run.

### Report-ready table with gap, cost and price per 0.001

```python
rows = [("Baseline", train_acc_baseline, test_acc_baseline, mia_adv_baseline, float("inf")),
        ("DP eps=10", train_acc_dp10, test_acc_dp10, mia_adv_dp10, final_epsilon_10),
        ("DP eps=3", train_acc_dp3, test_acc_dp3, mia_adv_dp3, final_epsilon_3)]
print(f"{'Model':<11}{'Test%':>7}{'Gap pp':>8}{'Cost pp':>9}{'MIA adv':>9}{'eps':>7} delta={DELTA}")
for name, tr, te, adv, eps in rows:
    print(f"{name:<11}{te:>7.2f}{tr - te:>8.2f}{test_acc_baseline - te:>9.2f}{adv:>9.4f}{eps:>7.2f}")
pairs = [("Baseline -> eps10", test_acc_baseline - test_acc_dp10, mia_adv_baseline - mia_adv_dp10),
         ("eps10 -> eps3", test_acc_dp10 - test_acc_dp3, mia_adv_dp10 - mia_adv_dp3)]
for name, cost, gain in pairs:
    per = cost / (gain * 1000) if gain > 0 else float("nan")
    print(f"{name:<18} costs {cost:.2f} pp for {gain:.4f} less advantage -> {per:.2f} pp per 0.001")
```

Expected (with the course's numbers): about 0.82 and 1.29 points per 0.001. You'll get `nan` if ε = 3 happened to score higher than ε = 10, which is pure noise.

### Add δ and the overfitting gap to results.json

```python
with open("output/results.json") as f:
    res = json.load(f)
res["delta"] = DELTA
res["baseline"]["overfit_gap"] = float(train_acc_baseline - test_acc_baseline)
res["dp_eps10"]["overfit_gap"] = float(train_acc_dp10 - test_acc_dp10)
res["dp_eps3"]["overfit_gap"] = float(train_acc_dp3 - test_acc_dp3)
with open("output/results.json", "w") as f:
    json.dump(res, f, indent=2)
print(json.dumps(res, indent=2))
```

Expected: the same JSON plus `"delta": 1e-05` and an `overfit_gap` for each model.

---

## Quick self-check

Use **your figure's** numbers: baseline 64.2% / 0.022, ε = 10 58.5% / 0.0037, ε = 3 52.3% / 0.0034.  
(a) What's the accuracy cost per 0.001 of advantage reduction for each step (baseline → ε = 10, then ε = 10 → ε = 3)?  
(b) The zero-leak noise floor is about 0.004. Which privacy claims can you honestly make in a report?  
(c) Which model would you submit to a validator wanting test accuracy above 50% and MIA advantage below 5%, and why?

---

## The facts worth locking in for the exam

- **Lower ε gives less membership leakage and less memorization, but lower accuracy**. In the course's run: test accuracy 67 → 58 → 53%, overfitting gap 10 → 3 → 1 point, MIA advantage 0.019 → 0.008 → 0.004 for baseline → ε = 10 → ε = 3.
- **The privacy-utility tradeoff shows diminishing returns: each extra unit of privacy costs more accuracy**. Baseline → ε = 10 cost 0.82 points per 0.001 of advantage (not the course's 1.2), and ε = 10 → ε = 3 cost 1.25, so the marginal cost rises rather than staying constant.
- **Once the measured MIA advantage reaches the noise floor (~0.004 here), more noise buys no measurable empirical privacy but still costs accuracy**. In the figure's run, ε = 10 (0.0037) and ε = 3 (0.0034) are both at the floor. The worst-case guarantee still gets stronger with lower ε, though.
- **The overfitting gap is the clearest evidence DP-SGD works** (10 → 3 → 1 point), because it doesn't depend on a weak attack or a noisy threshold search.
- **DP-SGD protects individual records only**: it limits membership inference, but aggregate patterns (property inference), model stealing via API queries and adversarial examples are unaffected. It costs 2-5× compute and can lose 20-30+ points below ε = 1.
- **Alternatives: PATE (noisy teacher voting; needs disjoint partitions plus public unlabeled data), local DP (noise added by each user; no trusted curator but needs far more data), federated learning (data stays on devices)**. Federated learning and secure aggregation alone are not a formal guarantee without DP.
- **A results file or report must record δ alongside ε, plus the overfitting gap**. The course's `results.json` stores neither δ nor gap, and the baseline's ε is ∞ (shown at 100 on the plot only as a stand-in).
- **Method point:** **to choose a DP setting, pick the largest ε (best accuracy) whose measured MIA advantage is already at the null noise floor and which meets the accuracy target**. Then report the before/after table, the accuracy cost in percentage points, the spent ε with δ, and what DP does _not_ cover.

---

Logged to COAE-progress.md, with the self-check answer key. Next is the DP-SGD challenge, which we'll do hands-on. The validator wants `models/dp_model.safetensors` (the ε = 10 model saved with `._module`).