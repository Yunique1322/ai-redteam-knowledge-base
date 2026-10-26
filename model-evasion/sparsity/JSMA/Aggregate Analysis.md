Logged. Here's Section 26.

---

# Section 26: Aggregate Analysis

## The idea in one breath

Two aggregate plots: an **L0 histogram** showing the spread of sparsity across all attacks, and a **5-row comparison grid** putting single-pixel and pairwise side by side for all 10 samples. The grid makes the 70%→90% story visual - but watch out, the prose in this section contradicts both the data and its own figure.

## Part 1 - The L0 histogram

```python
def visualize_l0_distribution(l0_values, output_dir):
    from htb_ai_library.visualization.styles import HTB_GREEN, NUGGET_YELLOW, AZURE

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.hist(l0_values, bins=30, color=HTB_GREEN, alpha=0.7)

    mean_l0 = np.mean(l0_values)
    median_l0 = np.median(l0_values)
    min_l0 = np.min(l0_values)
    max_l0 = np.max(l0_values)
```

```python
    ax.axvline(mean_l0, color=NUGGET_YELLOW, linestyle='--', linewidth=2, label=f'Mean: {mean_l0:.1f}')
    ax.axvline(median_l0, color=AZURE, linestyle='--', linewidth=2, label=f'Median: {median_l0:.1f}')
    ax.set_xlabel('L0 Norm (Pixels Modified)', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title('L0 Norm Distribution', color=HTB_GREEN, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3); ax.legend()
```

|Code|What it does|
|---|---|
|`ax.hist(..., bins=30, alpha=0.7)`|30 bins over the ~10-120 range; transparency lets gridlines show|
|`ax.axvline(mean, yellow)` / `axvline(median, blue)`|mark central tendency with dashed verticals|

Fed with both batches combined:

```python
all_l0_values = results['pixels_modified'] + results_pairs['pixels_modified']
visualize_l0_distribution(all_l0_values, output_dir)
```

```
Mean L0: 58.2
Median L0: 55.0
Min L0: 12
Max L0: 118
```

**Reading it:** **mean (58.2) > median (55.0), so the distribution is right-skewed** - the cluster of low-L0 pairwise successes sits on the left, and a tail of high-L0 attacks (the ~100- and 118-pixel hard/failed ones) drags the mean rightward past the median.

> **Honest note - the x-axis is mislabeled.** It says "L0 Norm" but it's fed `pixels_modified`, the _modification counter_, not `count_nonzero`. As we've seen, that counter overcounts true L0 (68 vs 66, 118 vs 116). So this histogram is really a "modifications" distribution. Ironically, the comparison grid below gets this right (uses `count_nonzero`), so the two plots in _this same section_ use different definitions of L0.

## Part 2 - Comparison grid: data prep (true L0 here)

```python
def prepare_comparison_data(original_images, single_adversarial, pairwise_adversarial):
    n_samples = len(original_images)
    data = []
    for idx in range(n_samples):
        orig = original_images[idx].cpu().numpy().squeeze()
        single_adv = single_adversarial[idx].cpu().numpy().squeeze()
        pair_adv = pairwise_adversarial[idx].cpu().numpy().squeeze()

        single_pert = np.abs(single_adv - orig)
        pair_pert = np.abs(pair_adv - orig)
        single_pert_norm = single_pert / (single_pert.max() + 1e-8)
        pair_pert_norm = pair_pert / (pair_pert.max() + 1e-8)

        data.append({
            'orig': orig, 'single_adv': single_adv, 'pair_adv': pair_adv,
            'single_pert_norm': single_pert_norm, 'pair_pert_norm': pair_pert_norm,
            'l0_single': int((single_pert > 0).sum()), # true L0 = count_nonzero
            'l0_pair': int((pair_pert > 0).sum())
        })
    return data
```

Note `(pert > 0).sum()` - the **honest** L0 here, which is why the grid shows sample 1 pairwise as **L0=66** (matching Section 25) while the counter said 68.

## Part 3 - The three row helpers

```python
def plot_original_row(axes, data, original_labels, HTB_GREEN):
    for idx, sample_data in enumerate(data):
        axes[0, idx].imshow(sample_data['orig'], cmap='gray', interpolation='nearest')
        if idx == 0:
            axes[0, idx].set_ylabel('Original', color=HTB_GREEN, fontsize=11, fontweight='bold')
        axes[0, idx].text(0.02, 0.98, f'#{idx+1}\nTrue: {original_labels[idx]}',
                         transform=axes[0, idx].transAxes, verticalalignment='top', fontsize=9,
                         bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))
        axes[0, idx].axis('off')
```

```python
def plot_perturbation_row(axes, row, data, label, HTB_GREEN):
    key = 'single_pert_norm' if 'Single' in label else 'pair_pert_norm'
    l0_key = 'l0_single' if 'Single' in label else 'l0_pair'
    for idx, sample_data in enumerate(data):
        axes[row, idx].imshow(sample_data[key], cmap='hot', interpolation='nearest')
        if idx == 0:
            axes[row, idx].set_ylabel(label, color=HTB_GREEN, fontsize=11, fontweight='bold')
        axes[row, idx].text(0.5, 0.02, f'L0={sample_data[l0_key]}', transform=axes[row, idx].transAxes,
                           horizontalalignment='center', fontsize=8,
                           bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))
        axes[row, idx].axis('off')
```

```python
def plot_result_row(axes, row, data, label, predictions, targets, HTB_GREEN, MALWARE_RED):
    key = 'single_adv' if 'Single' in label else 'pair_adv'
    for idx, sample_data in enumerate(data):
        axes[row, idx].imshow(sample_data[key], cmap='gray', interpolation='nearest')
        if idx == 0:
            axes[row, idx].set_ylabel(label, color=HTB_GREEN, fontsize=11, fontweight='bold')
        success = (predictions[idx] == targets[idx])
        result_color = HTB_GREEN if success else MALWARE_RED
        axes[row, idx].text(0.5, 0.02, f'Pred: {predictions[idx]}', transform=axes[row, idx].transAxes,
                           horizontalalignment='center', fontsize=8, color=result_color,
                           bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))
        axes[row, idx].axis('off')
```

|Helper|Trick|
|---|---|
|all three|`transform=axes[...].transAxes` → text positioned in 0-1 axis coords, consistent across panels regardless of image values|
|perturbation / result|`'Single' in label` picks the right dict key, so one function serves both the single and pairwise rows|
|result|green/red title by `prediction == target` → pass/fail at a glance|

## Part 4 - Assemble the 5-row grid

```python
def visualize_attack_comparison(original_images, single_adversarial, pairwise_adversarial,
                                original_labels, predicted_single, predicted_pairs,
                                target_labels, output_dir):
    from htb_ai_library.visualization.styles import HTB_GREEN, MALWARE_RED
    n_samples = len(original_images)
    data = prepare_comparison_data(original_images, single_adversarial, pairwise_adversarial)

    fig, axes = plt.subplots(5, n_samples, figsize=(2.5 * n_samples, 12))
    if n_samples == 1:
        axes = axes.reshape(-1, 1)

    plot_original_row(axes, data, original_labels, HTB_GREEN)
    plot_perturbation_row(axes, 1, data, 'Single-Pixel\nPerturbation', HTB_GREEN)
    plot_result_row(axes, 2, data, 'Single-Pixel\nResult', predicted_single, target_labels, HTB_GREEN, MALWARE_RED)
    plot_perturbation_row(axes, 3, data, 'Pairwise\nPerturbation', HTB_GREEN)
    plot_result_row(axes, 4, data, 'Pairwise\nResult', predicted_pairs, target_labels, HTB_GREEN, MALWARE_RED)

    plt.tight_layout()
    save_path = output_dir / 'jsma_attack_comparison.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight'); plt.close()
    print(f"Attack comparison visualization saved to {save_path}")
```

Five rows: originals, single-pixel perturbation, single-pixel result, pairwise perturbation, pairwise result.

## Reading the grid - and the prose you must not trust

What the **figure** actually shows:

- **Row 3 (single-pixel results):** mostly **green** - #1 (Pred 7) and #9 (Pred 5) are red failures, the rest succeed. That's ~7/10, matching Section 19's **70%**.
- **Row 4 (pairwise perturbations):** sparser hot spots, L0 labels **12-68**, except sample 2 at **L0=116**.
- **Row 5 (pairwise results):** 9 of 10 **green**; only sample 2 red. The **90%** from Section 23.
- Sample 4's pairwise L0=**12** is the standout - a flip from 12 pixels.

> **Major course inconsistency - flag it hard.** The section's prose claims single-pixel had "**0% success**," "**universal failure: all 10 red**," and "**average 93.9 pixels**." That contradicts **both** Section 19 (70%, 7/10) **and the figure printed right above it** (row 3 is mostly green). The prose is describing some _other_ run - a θ=0.25 **lock-on** single-pixel configuration that failed everything (like Section 20's behavior) - not the `results` data the code actually plots. **Trust the figure and Section 19's 70%; ignore the "0%" narrative.** This is the clearest example yet of the course's text drifting from its own code.

## Copy-paste cell (mine, not the course's)

A self-contained L0 histogram that plots **true L0** (count_nonzero), so the x-axis label is honest, and takes the adversarial images directly:

```python
# === Section 26: honest L0 histogram ===
import numpy as np, torch, matplotlib.pyplot as plt

def l0_histogram(originals, adv_lists, labels=None, bins=30):
    """adv_lists: list of (name, [adv_tensors]) to overlay; originals aligned by index."""
    plt.figure(figsize=(8, 4.5))
    for name, advs in adv_lists:
        l0 = [int(np.count_nonzero((a - o).detach().cpu().numpy()))
              for o, a in zip(originals, advs)]
        plt.hist(l0, bins=bins, alpha=0.6, label=f"{name} (mean {np.mean(l0):.1f}, med {np.median(l0):.0f})")
    plt.xlabel('L0 (true pixels changed)'); plt.ylabel('Count')
    plt.title('L0 Distribution'); plt.legend(); plt.grid(alpha=0.3); plt.tight_layout(); plt.show()

# l0_histogram(original_images,
# [("single", results['adversarial']), ("pairwise", results_pairs['adversarial'])])
```

## Quick self-check

The L0 histogram reports **mean 58.2** and **median 55.0**.

1. What does mean > median tell you about the distribution's shape?
2. Which samples are responsible for that gap?

I'll confirm next message.

## The facts worth locking in for the exam

- **Mean > median signals a right-skewed distribution** - a tail of high-L0 attacks pulls the average above the typical (median) case.
- **Mark mean and median with `axvline`** on a histogram to expose skew at a glance.
- **Compute aggregate L0 with `count_nonzero`**, not the modification counter - the grid does this correctly (sample 1 = 66), while the histogram here mistakenly uses the counter (68).
- **The comparison grid is 5 rows× N samples**: originals, single perturbation, single result, pairwise perturbation, pairwise result, with green/red titles for pass/fail.
- ==`transform=axes.transAxes` places annotations in 0-1 axis coordinates==, keeping labels consistent across panels of differing content.
- **The figure shows single-pixel ~70% and pairwise 90%** (only sample 2 fails pairwise) - the prose's "single-pixel 0% / universal failure" is wrong and contradicts the data.
- **Pairwise is visibly sparser**: successful flips at L0 12-68 vs single-pixel's scattered 14-51, with sample 4 flipping on just 12 pixels.
- **Method point:** **aggregate views (distribution + side-by-side grid) turn per-sample anecdotes into the population story** - but always confirm the narrative against the actual plotted numbers, since here the text and the figure disagree.

Paste Section 27 when ready - that's almost certainly the **JSMA Challenge** (interactive). I can help you work through it hands-on; just paste the challenge text and any server/API details and we'll solve it together.