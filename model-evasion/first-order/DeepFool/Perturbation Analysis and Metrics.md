# Perturbation Analysis And Metrics

## The idea in one breath

Section 19 stored 20 attack results. This section turns them into evidence: a **perturbation heatmap** showing where pixels changed, histograms of **perturbation size** and **iterations**, a **per-class success** bar chart, and a printed **summary with the most common class flips** (9 → 4 three times). The numbers match Section 19: 20/20, average L2 4.87, range 0.60 to 7.72, average 3.0 iterations.

> **Heads-up:** no mangled formulas. Two things in the prose don't match the figures:
> 
> - The course says raw perturbations are "often [−0.5, 0.5]". The top-row colour bars show ±1.5, ±0.9 and ±1.2.
> - The course says the L2 values "cluster around the 4-5 average". The histogram is spread out, with one group near 4 and another near 7 (Reading the figures).
> 
> **Overwrite warning:** this section creates a **list** called `iterations`, which replaces Section 18's single number `iterations` (4). Re-running a Section 18 cell afterwards would print the list.

---

## Part 1: Where DeepFool changes pixels (spatial analysis)

### Step 1: Diverging colour maps (toy)

A perturbation has a **sign**: + means brightened, − means darkened. A **diverging colormap** like `RdBu_r` shows that: red = positive, blue = negative, white = zero.

Toy: three pixels changed by **[−0.8, 0, +0.4]**.

|Colour limits|−0.8 shows as|0 shows as|+0.4 shows as|
|---|---|---|---|
|**Symmetric** ±0.8 (`vmin=-vmax, vmax=vmax`)|deep blue|**white** ✓|light red|
|Default (−0.8 to +0.4)|deep blue|pale red ✗ (zero is not the middle anymore)|deep red|

The middle of the range always gets the middle colour (white), so the limits must be symmetric for **white to mean "no change"**. That's why the code uses `vmin=-vmax, vmax=vmax`. (In Section 18's `inferno` panel, zero wasn't the middle colour, which is why that panel was confusing.)

### Step 2: The ×10 amplification trick and its trap (toy)

The bottom row shows (adversarial − original) × 10, with colour limits fixed at ±0.5:

|Real change|× 10|What you see (limits ±0.5)|
|---|---|---|
|0.01|0.1|faint red|
|0.03|0.3|medium red|
|0.05|0.5|**full dark red**|
|0.20|2.0 → capped at 0.5|full dark red|
|1.50|15.0 → capped at 0.5|full dark red|

Anything above 0.05 looks **identical**. This is called **saturation**. The course's own example (0.03 → 0.3) works, but DeepFool's real changes here go up to 1.6, so most of the digit area hits the cap.

### Step 3: The code

```python
def visualize_perturbation_analysis(results, save_dir='output'):
    """
    Analyze and visualize perturbation characteristics across samples.

    Creates two-row visualization: top shows raw perturbation heatmaps,
    bottom shows amplified differences overlaid on originals.

    Args:
        results (list): Attack results
        save_dir (str): Output directory
    """
    print("\nGenerating perturbation analysis...")

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.patch.set_facecolor(NODE_BLACK)

    for ax in axes.flatten():
        ax.set_facecolor(NODE_BLACK)
        for spine in ax.spines.values():
            spine.set_edgecolor(HACKER_GREY)

    # Select first 3 successful attacks
    successful_attacks = [r for r in results if r['success']][:3]

    for idx, result in enumerate(successful_attacks):
        # Top row: Raw perturbation heatmap
        ax_top = axes[0, idx]
        pert = result['perturbation'].squeeze().numpy()
        vmax = np.abs(pert).max() or 1e-6
        im_top = ax_top.imshow(pert, cmap='RdBu_r', vmin=-vmax, vmax=vmax)
        ax_top.set_title(f'Perturbation (L2={result["l2_norm"]:.3f})',
                        color=HTB_GREEN, fontsize=10)
        ax_top.axis('off')

        cbar_top = plt.colorbar(im_top, ax=ax_top, fraction=0.046, pad=0.04)
        cbar_top.outline.set_edgecolor(HACKER_GREY)
        cbar_top.ax.tick_params(colors=WHITE)

        # Bottom row: Amplified difference visualization
        ax_bottom = axes[1, idx]
        orig_img = result['original_image'].squeeze().numpy()
        adv_img = result['perturbed_image'].squeeze().detach().numpy()
        diff_amplified = (adv_img - orig_img) * 10 # 10x amplification for visibility

        im_bottom = ax_bottom.imshow(diff_amplified, cmap='RdBu_r', vmin=-0.5, vmax=0.5)
        ax_bottom.set_title(f"{result['original_label']} → {result['adversarial_label']} "
                           f"({result['iterations']} iters)",
                           color=NUGGET_YELLOW, fontsize=10)
        ax_bottom.axis('off')

        cbar_bottom = plt.colorbar(im_bottom, ax=ax_bottom, fraction=0.046, pad=0.04)
        cbar_bottom.outline.set_edgecolor(HACKER_GREY)
        cbar_bottom.ax.tick_params(colors=WHITE)

    plt.suptitle('DeepFool Perturbation Analysis', color=HTB_GREEN, fontsize=16, y=0.98)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'deepfool_perturbations.png'),
                facecolor=NODE_BLACK, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Perturbation analysis saved to {save_dir}/deepfool_perturbations.png")

# Generate the perturbation analysis visualization
visualize_perturbation_analysis(results, save_dir='output')
```
![Perturbation Analysis and Metrics](/assets/20%20Perturbation%20Analysis%20and%20Metrics-20261017171302442.png)

|Code|What it does|
|---|---|
|`plt.subplots(2, 3, ...)`|2 rows × 3 columns: 3 attacks, two views each.|
|`[r for r in results if r['success']][:3]`|Keep only successful attacks, then take the first 3 (the 7, the 2 and the 1).|
|`enumerate(successful_attacks)`|`idx` = 0, 1, 2 = which column.|
|`result['perturbation'].squeeze().numpy()`|The raw `r_tot` (no overshoot), (1,1,28,28) → (28,28) → NumPy.|
|`vmax = np.abs(pert).max() or 1e-6`|Biggest absolute change. The `or 1e-6` is a safety net: if every change were exactly 0, use a tiny number so the colour limits aren't both 0.|
|`imshow(pert, cmap='RdBu_r', vmin=-vmax, vmax=vmax)`|Symmetric diverging map, so white = zero (Step 1).|
|`plt.colorbar(...)`, `.outline.set_edgecolor`, `.tick_params(colors=WHITE)`|Colour bar styled for the dark theme.|
|`result['perturbed_image'].squeeze().detach().numpy()`|The attacked image. `.detach()` makes sure no gradient tracking is attached before converting.|
|`(adv_img - orig_img) * 10`|Actual change (which includes the 2% overshoot), × 10.|
|`vmin=-0.5, vmax=0.5`|**Fixed** limits, which causes the saturation in Step 2.|
|`f"{...original_label} → {...adversarial_label} ({...iterations} iters)"`|Title like "7 → 2 (4 iters)".|
|`plt.savefig(...)` / `plt.close()`|Saves `output/deepfool_perturbations.png` and **doesn't show it inline** (same as Section 19).|

> **If this errors (or nothing appears):** it's saved, not shown. Use Cell 3 below to display it. `RuntimeError: Can't call numpy() on Tensor that requires grad` on the top row would mean `perturbation` still carries gradient tracking. Add `.detach()` before `.numpy()`, as the bottom row does.

---

## Part 2: Statistical distributions

### Step 1: Histograms (toy)

A **histogram** cuts a range into equal-width **bins** and counts how many values fall in each.

Toy: values [1, 1, 2, 3, 3, 3] with bins [1,2), [2,3), [3,4] give bar heights **2, 1, 3**.

`range(1, max(iterations)+2)` builds the integer bin edges. With max 5, `range(1, 7)` = 1, 2, 3, 4, 5, 6, so there are 5 bins: [1,2), [2,3), [3,4), [4,5), [5,6]. `range` stops **before** its end number, which is why the code adds 2 (one for "include 5", one more so 5 gets a bin of its own).

### Step 2: Counting with a dictionary (toy)

Per-class success uses a nested dictionary. Toy with 3 results:

|Result|Class (original_label)|success|Dictionary after|
|---|---|---|---|
|1|9|True|{9: total 1, success 1}|
|2|4|True|{9: 1/1, 4: 1/1}|
|3|9|False|{9: total 2, success 1, 4: 1/1}|

Success rates: 9 → 1/2 × 100 = **50%**, 4 → 1/1 = **100%**.

### Step 3: The code

```python
print("\nGenerating attack metrics visualization...")

# Setup three-panel figure
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.patch.set_facecolor(NODE_BLACK)

for ax in axes:
    ax.set_facecolor(NODE_BLACK)
    for spine in ax.spines.values():
        spine.set_edgecolor(HACKER_GREY)
    ax.tick_params(colors=WHITE)
    ax.grid(True, alpha=0.3, color=HACKER_GREY, linestyle='--')

# Panel 1: L2 Norm Distribution
l2_norms = [r['l2_norm'] for r in results]
axes[0].hist(l2_norms, bins=15, color=HTB_GREEN, alpha=0.7, edgecolor=HACKER_GREY)
axes[0].set_xlabel('L2 Norm', color=WHITE)
axes[0].set_ylabel('Frequency', color=WHITE)
axes[0].set_title('Perturbation Magnitude Distribution', color=HTB_GREEN)
print(f"L2 norm range: [{min(l2_norms):.4f}, {max(l2_norms):.4f}]")

# Panel 2: Iteration Count Distribution
iterations = [r['iterations'] for r in results]
axes[1].hist(iterations, bins=range(1, max(iterations)+2),
            color=AZURE, alpha=0.7, edgecolor=HACKER_GREY)
axes[1].set_xlabel('Iterations', color=WHITE)
axes[1].set_ylabel('Frequency', color=WHITE)
axes[1].set_title('Iterations Required', color=HTB_GREEN)
print(f"Iteration range: [{min(iterations)}, {max(iterations)}]")

# Panel 3: Per-Class Success Rates
class_success = {}
for r in results:
    orig = r['original_label']
    if orig not in class_success:
        class_success[orig] = {'total': 0, 'success': 0}
    class_success[orig]['total'] += 1
    if r['success']:
        class_success[orig]['success'] += 1

classes = sorted(class_success.keys())
success_rates = [
    class_success[c]['success'] / class_success[c]['total'] * 100
    if class_success[c]['total'] > 0 else 0
    for c in classes
]

bars = axes[2].bar(classes, success_rates, color=NUGGET_YELLOW,
                   alpha=0.7, edgecolor=HACKER_GREY)
axes[2].set_xlabel('Original Class', color=WHITE)
axes[2].set_ylabel('Success Rate (%)', color=WHITE)
axes[2].set_title('Attack Success by Class', color=HTB_GREEN)
axes[2].set_ylim(0, 105)

# Add percentage labels on bars
for bar, rate in zip(bars, success_rates):
    height = bar.get_height()
    ax_x = bar.get_x() + bar.get_width() / 2.0
    axes[2].text(ax_x, height + 1, f'{rate:.0f}%',
                 ha='center', va='bottom', color=WHITE, fontsize=8)

# Save visualization
plt.suptitle('DeepFool Attack Metrics', color=HTB_GREEN, fontsize=16, y=1.02)
plt.tight_layout()
plt.savefig('output/deepfool_metrics.png',
            facecolor=NODE_BLACK, dpi=150, bbox_inches='tight')
plt.close()

print("Metrics visualization saved to output/deepfool_metrics.png")
```
![Perturbation Analysis and Metrics](/assets/20%20Perturbation%20Analysis%20and%20Metrics-20261017171331908.png)

|Code|What it does|
|---|---|
|`plt.subplots(1, 3, ...)`|One row, 3 charts.|
|`ax.grid(True, alpha=0.3, ...)`|Faint dashed grid lines.|
|`l2_norms = [r['l2_norm'] for r in results]`|All 20 L2 values in one list.|
|`axes[0].hist(l2_norms, bins=15, ...)`|Histogram with 15 equal bins from 0.596 to 7.721, each about 0.475 wide.|
|`alpha=0.7`|70% opaque bars. `edgecolor` = bar outline.|
|`iterations = [...]`|All 20 iteration counts. **This overwrites Section 18's `iterations`.**|
|`bins=range(1, max(iterations)+2)`|One bin per whole number (Step 1).|
|`class_success = {}`|Empty dictionary for the per-class counts.|
|`if orig not in class_success: ... {'total': 0, 'success': 0}`|First time a class appears, create its two counters.|
|`['total'] += 1` / `['success'] += 1`|Count images and flips (Step 2).|
|`sorted(class_success.keys())`|Classes in order 0, 1, 2... (only the ones that appeared).|
|`success / total * 100 if total > 0 else 0`|Percentage, protected against dividing by 0.|
|`axes[2].bar(classes, success_rates, ...)`|One bar per class.|
|`set_ylim(0, 105)`|Extra room above 100 for the labels.|
|`bar.get_x() + bar.get_width() / 2.0`|Horizontal centre of each bar.|
|`axes[2].text(ax_x, height + 1, f'{rate:.0f}%', ...)`|Writes "100%" just above each bar.|
|`plt.savefig('output/deepfool_metrics.png', ...)` / `plt.close()`|Saved, not shown inline.|

**Expected printed lines:**

```txt
Generating attack metrics visualization...
L2 norm range: [0.5960, 7.7214]
Iteration range: [1, 5]
Metrics visualization saved to output/deepfool_metrics.png
```

---

## Part 3: The summary printout

```python
def print_summary_statistics(results):
    """
    Print summary statistics for attack results.

    Computes and displays success rate, perturbation statistics, iteration
    statistics, and common class transitions.

    Args:
        results (list): Attack results
    """
    print("\n" + "="*60)
    print("Attack Summary Statistics")
    print("="*60)

    successful_attacks = [r for r in results if r['success']]

    if successful_attacks:
        avg_l2 = np.mean([r['l2_norm'] for r in successful_attacks])
        avg_iterations = np.mean([r['iterations'] for r in successful_attacks])
        min_l2 = min([r['l2_norm'] for r in successful_attacks])
        max_l2 = max([r['l2_norm'] for r in successful_attacks])

        print(f"Success Rate: {len(successful_attacks)}/{len(results)} "
              f"({100*len(successful_attacks)/len(results):.1f}%)")
        print(f"Average L2 Norm: {avg_l2:.4f}")
        print(f"L2 Range: [{min_l2:.4f}, {max_l2:.4f}]")
        print(f"Average Iterations: {avg_iterations:.1f}")

        # Class transition analysis
        transitions = {}
        for r in successful_attacks:
            key = f"{r['original_label']}→{r['adversarial_label']}"
            transitions[key] = transitions.get(key, 0) + 1

        print(f"\nMost Common Misclassifications:")
        for trans, count in sorted(transitions.items(), key=lambda x: x[1], reverse=True)[:5]:
            print(f" {trans}: {count} times")
    else:
        print("No successful attacks generated")

    print("="*60)

# Generate summary
print_summary_statistics(results)
```

|Code|What it does|
|---|---|
|`"="*60`|A line of 60 "=" characters.|
|`successful_attacks = [...]`|Only the flipped ones. The averages below are over **successful** attacks, which is all 20 here.|
|`if successful_attacks:`|An empty list counts as False, so it skips the stats if nothing flipped.|
|`np.mean`, `min`, `max`|Average, smallest and largest.|
|`key = f"{...}→{...}"`|A text label like "9→4".|
|`transitions.get(key, 0) + 1`|"Current count, or 0 if this pair is new, plus 1." The standard dictionary-counting pattern.|
|`sorted(transitions.items(), key=lambda x: x[1], reverse=True)[:5]`|Turn the dictionary into (pair, count) pairs, sort by the count (`x[1]`) from high to low, and keep the top 5.|
|`lambda x: x[1]`|A tiny throwaway function: "given a pair, return its second item".|

![Perturbation Analysis and Metrics](/assets/20%20Perturbation%20Analysis%20and%20Metrics-20261017171405232.png)
**Reading it:** 9 → 4 happened 3 times out of the 4 nines. A 9 and a 4 share an upper loop or bar and a vertical stroke, so their boundary is close. That's DeepFool's "nearest boundary" choice showing up as a pattern.

> **An honest note:** only 9 → 4 happened more than once. The four "1 time" rows are **tied with many other one-off flips**. `sorted` keeps ties in the order they were first seen, so those four are simply the earliest ones, not more important than the rest.

---

## Reading the figures

### Figure 1: DeepFool Perturbation Analysis
![Perturbation Analysis and Metrics](/assets/20%20Perturbation%20Analysis%20and%20Metrics-20261016194837505.png)

|Panel|What you see|What it means|
|---|---|---|
|Top-left, L2 = 7.721 (7 → 2)|red and blue pixels along the 7's bar and stroke, plus a cluster at the bottom-left; colour bar ±1.5|changes sit on the strokes; the bottom-left cluster is where a 2's base gets drawn|
|Top-middle, L2 = 4.099 (2 → 6)|changes scattered over the upper loop and the bottom bar; colour bar ±0.9|changes along the 2's own shape|
|Top-right, L2 = 5.160 (1 → 8)|a narrow vertical band of changes along the 1; colour bar ±1.2|building 8-like loops around the stroke|
|Bottom row (× 10, ±0.5)|almost solid dark red and dark blue blocks over the digit area, white further out, a few coloured edge pixels|**saturated**: every pixel that changed by more than 0.05 is full colour, so it only shows where something changed and in which direction|

|Key value|Where|Meaning|
|---|---|---|
|7.721, 4.099, 5.160|top titles|L2 per attack (model space, without overshoot)|
|±1.5, ±0.9, ±1.2|top colour bars|biggest single-pixel change in each, so the raw range is well past the course's "[−0.5, 0.5]"|
|4, 3, 3 iters|bottom titles|iterations per attack|

> **Honest notes on Figure 1:**
> 
> - Each top panel has its own colour scale, so the same red means 1.5 in the first panel but 0.9 in the second. **Don't compare intensity across panels.**
> - The course says the background "remains white (unchanged)". In the top row, the far background is near zero but not exactly zero (faint tints). The bottom row's coloured corner pixels show some edge pixels changed by more than 0.05.
> - The bottom row is better read as a "changed vs not changed" map than as a strength map. Cell 2 below draws an unsaturated version.

### Figure 2: DeepFool Attack Metrics
![Figure 2: DeepFool Attack Metrics](/assets/Pasted%20image%2020261016194845.png)

|Panel|What you see|What it means|
|---|---|---|
|L2 histogram|bars spread from 0.6 to 7.7: 2 near 0.6-1.0, a tall bar of 4 around 4.0-4.4, and a second group of 7 between 6.3 and 7.7|**wide spread, with two groups** (about 4 and about 7), not a tight cluster at 4-5 as the course says|
|Iterations histogram|2, 3, **10**, 4, 1|1 iteration: 2 images, 2: 3, **3: 10**, 4: 4, 5: 1. That totals 20, and the average is (2 + 6 + 30 + 16 + 5) ÷ 20 = 59 ÷ 20 = **2.95**, which prints as 3.0.|
|Success by class|100% bars for 0-7 and 9, **no bar for 8**|all flipped; there's no 8 among the first 20 test images|

> **Honest notes on Figure 2:**
> 
> - **Iteration bars sit between whole numbers.** The tall bar between 3 and 4 means "exactly 3 iterations". That's because of how the bins are built.
> - **Per-class success is uninformative for DeepFool.** It's always 100% (no budget), and each class has only 1-4 images (0: 3, 1: 3, 2: 1, 3: 1, 4: 3, 5: 2, 6: 1, 7: 2, 9: 4). The course says it shows "which digits prove more vulnerable", but all bars are equal. For DeepFool, vulnerability per class means **average L2 per class** (Cell 1 below).
> - The class axis uses `original_label`, the model's prediction, not the true digit (same caveat as Section 19).

---

## Copy-paste cells (mine, not the course's)

These use `results`, `np` and `plt`, which already exist.

### Cell 1: Per-class vulnerability the useful way (average L2)

```python
from collections import defaultdict
by_class = defaultdict(list) # a dict whose missing keys start as empty lists
for r in results:
    by_class[r['true_label']].append(r['l2_norm'])
for c in sorted(by_class):
    v = by_class[c]
    print(f"digit {c}: n={len(v)} mean L2={np.mean(v):.3f} min={min(v):.3f} max={max(v):.3f}")
```

**Expected:** 9 lines (no digit 8), with n = 3, 3, 1, 1, 3, 2, 1, 2, 4 for digits 0-7 and 9. The digit with the **lowest** mean L2 is the most vulnerable. Keep in mind n is tiny.

### Cell 2: An unsaturated view of the 7 → 2 change

```python
res = results[0] # the 7 -> 2 attack
diff = (res['perturbed_image'] - res['original_image']).squeeze().numpy()
lim = np.abs(diff).max() # symmetric limits from the real data
plt.imshow(diff, cmap='RdBu_r', vmin=-lim, vmax=lim); plt.colorbar()
plt.title("7 -> 2 real change (model space, incl. overshoot)"); plt.axis('off'); plt.show()
print("share of pixels changed by more than 0.05:", round(float((np.abs(diff) > 0.05).mean()), 3))
```

**Expected:** a picture like the top-left panel, with a colour bar of about ±1.6. Then a fraction between 0 and 1: every pixel in that fraction shows as full colour in the course's bottom row.

### Cell 3: Show both saved figures inline

```python
from IPython.display import Image, display
for f in ['output/deepfool_perturbations.png', 'output/deepfool_metrics.png']:
    display(Image(f))
```

**Expected:** the two figures above.

### Cell 4: Exact iteration counts (no histogram guessing)

```python
from collections import Counter
print(sorted(Counter(r['iterations'] for r in results).items()))
```

**Expected:** `[(1, 2), (2, 3), (3, 10), (4, 4), (5, 1)]`

---

## How important is this for the exam?

Medium-high for interpretation. You need to read these charts correctly in a report and avoid the traps (saturation, per-panel colour scales, meaningless 100% bars). The plotting code itself is low priority, since it's copy-paste.

## Quick self-check

In the bottom row (×10, colour limits ±0.5), one pixel shows **full dark red**. What is the smallest real change (model space) that pixel could have had, and can you tell from the picture whether it was 0.1 or 1.5?

---

## The facts worth locking in for the exam

- **The summary for the first 20 test images is 20/20 success, average L2 4.8661, L2 range [0.5960, 7.7214] and average iterations 3.0** (exactly 59 ÷ 20 = 2.95), with 9 → 4 the most common flip at 3 times.
- ==A diverging colormap such as `RdBu_r` needs symmetric limits (`vmin=-vmax, vmax=vmax`) so that white means zero change==, red means brightened and blue means darkened. Separate panels with their own limits can't be compared by colour intensity.
- **Amplifying by 10 with fixed ±0.5 limits saturates at a real change of 0.05**. Since DeepFool's changes here reach about ±1.6, the bottom row only shows where pixels changed and in which direction, not by how much.
- **DeepFool's changes concentrate on the digit's strokes and the features that separate the two classes**, such as drawing a 2's base onto a 7 or loops around a 1. The background is mostly near zero, but not exactly zero.
- **The L2 sizes are widely spread (0.6 to 7.7, with groups near 4 and 7), not tightly clustered**, and most images need 3 iterations (10 of 20), with a range of 1 to 5.
- **Per-class success rate is uninformative for DeepFool because it always reaches 100%**, and with 1-4 images per class (none for 8 among the first 20) it carries no statistical weight. Per-class average L2 is the meaningful vulnerability measure.
- **Frequent transitions like 9 → 4 show which class pairs share a nearby boundary**, but one-off flips are ties, and their order in the top-5 list is arbitrary.
- **Method point:** **before drawing conclusions from a chart, check its colour limits for saturation, check whether panels share a scale, and check the sample size behind every bar**. Prefer metrics that can actually vary, such as per-class L2 rather than a success rate that is always 100%.

The progress file is updated: Section 20 is done, the course's chart mistakes, your notebook names (including the `iterations` overwrite) and the self-check answer are saved, and Section 21 is next.