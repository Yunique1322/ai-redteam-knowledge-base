# Attack Visualization

## The idea in one breath

Two pictures that make the numbers real. The **attack-process grid** shows original → perturbation (amplified 10×) → adversarial for 10 digits, proving the change is tiny but the prediction flips. The **four histograms** show the spread of L1, squared-L2, L∞ and elastic distortion across the 20 images. The one relationship to carry: ==mean elastic = mean squared-L2 + β × mean L1== (4.38 = 4.18 + 0.01 × 20.20).

> **Heads-up:** no mangled formulas. One thing to reconcile up front, covered in the honest note below: the intro's "average L2 distortion 2.36" and the figure's "squared L2 mean 4.18" are _different_ quantities, and the 2.36 doesn't line up with this run's data.

---

## Part 1 - The attack-process grid

```python
print("\nGenerating visualizations...")

# Visualization: Attack process (original → perturbation → adversarial)
print("Creating attack process visualization...")

fig, axes = plt.subplots(3, 10, figsize=(20, 6))

# Show first 10 examples
num_display = min(10, batch_size)

for i in range(num_display):
    # Original image
    orig_img = original_images[i].detach().cpu().squeeze()
    axes[0, i].imshow(orig_img, cmap="gray", vmin=0, vmax=1)
    axes[0, i].axis("off")
    axes[0, i].set_title(f"True: {attack_targets[i].item()}", fontsize=10)

    # Perturbation (amplified for visibility)
    pert = (best_adv[i] - original_images[i]).detach().cpu().squeeze()
    pert_display = pert * 10 # Amplify by 10x for visibility
    axes[1, i].imshow(pert_display, cmap="seismic", vmin=-1, vmax=1)
    axes[1, i].axis("off")
    axes[1, i].set_title("Perturbation", fontsize=10)

    # Adversarial image
    adv_img = best_adv[i].detach().cpu().squeeze()
    axes[2, i].imshow(adv_img, cmap="gray", vmin=0, vmax=1)
    axes[2, i].axis("off")
    axes[2, i].set_title(f"Pred: {adv_predictions[i].item()}", fontsize=10, color=MALWARE_RED)

# Row labels
fig.text(0.02, 0.80, "Original", rotation=90, fontsize=14, weight="bold", ha="center", va="center")
fig.text(0.02, 0.50, "Perturbation\n(10× amplified)", rotation=90, fontsize=14, weight="bold", ha="center", va="center")
fig.text(0.02, 0.20, "Adversarial", rotation=90, fontsize=14, weight="bold", ha="center", va="center")

plt.tight_layout(rect=[0.03, 0, 1, 1])
plt.savefig(output_dir / "ead_attack_process.png", dpi=150, bbox_inches="tight")
plt.close()

print(f" Saved to {output_dir}/ead_attack_process.png")
```

|Code|What it does|
|---|---|
|`plt.subplots(3, 10)`|3 rows (original, perturbation, adversarial) × 10 columns|
|`original_images[i].detach().cpu().squeeze()`|one image, off the GPU, 2-D for display|
|`cmap="gray", vmin=0, vmax=1`|show as a 0-1 grayscale digit|
|`pert * 10`|**amplify the change 10×** so the eye can see it|
|`cmap="seismic", vmin=-1, vmax=1`|==red = positive change, blue = negative, white = zero==|
|`set_title(f"Pred: ...", color=MALWARE_RED)`|the fooled prediction, in red|
|`plt.savefig(output_dir / "...")`|write the PNG to `output/`|

### The colormap difference worth noting

This uses **`seismic`, where zero change is _white_** and red/blue show the direction. That's clearer than the DeepFool module's `inferno` map, where zero was a _middle_ color and you had to remember that dark meant "darkened," not "unchanged." On `seismic`, white = untouched, so the sparsity is visible as white space.

---

## Part 2 - Reading the attack-process figure
![Part 2 - Reading the attack-process figure](/assets/Pasted%20image%2020261017204427.png)


The grid shows 10 digits, and **every one is misclassified**:

|Column|True|Pred|Flip|
|---|---|---|---|
|1|7|2|7→2|
|2|2|1|2→1|
|3|1|7|1→7|
|4|0|6|0→6|
|5|4|9|4→9|
|6|1|7|1→7|
|7|4|8|4→8|
|8|9|4|9→4|
|9|5|6|5→6|
|10|9|4|9→4|

|Row|What you see|
|---|---|
|Original|clean, perfectly recognizable digits|
|Perturbation (10×)|**red/blue speckle concentrated on the strokes and edges**, with white (untouched) background - the change follows the digit's shape, not random noise|
|Adversarial|looks essentially identical to the original, yet the model reads it as a different digit|

The takeaway: **the change clusters where the ink is** (e.g. around curves and loops), because those are the high-gradient pixels that move the decision boundary. The bottom row proves the point - a human still reads "7," the model says "2."

---

## Part 3 - The distortion histograms

```python
# Visualization 2: Distortion distributions for L1, L2, L∞, Elastic
print("Creating distortion analysis...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Calculate distortion metrics
l1_values = l1_dist.detach().cpu().numpy()
l2_values = l2_dist.detach().cpu().numpy()
linf_values = linf_dist.detach().cpu().numpy()
elastic_values = elastic_dist.detach().cpu().numpy()

# L1 distribution
axes[0, 0].set_title(r"$L_1$ Distortion Distribution", fontsize=14)
axes[0, 0].set_xlabel(r"$L_1$ Distance")
axes[0, 0].set_ylabel("Frequency")
axes[0, 0].hist(l1_values, bins=15, color=AZURE, alpha=0.7, edgecolor=NODE_BLACK)
axes[0, 0].axvline(l1_values.mean(), color=MALWARE_RED, linestyle="--",
                   linewidth=2, label=f"Mean: {l1_values.mean():.2f}")
axes[0, 0].legend(frameon=False, fontsize=10)

# L2 distribution
axes[0, 1].set_title(r"Squared $L_2$ Distortion Distribution", fontsize=14)
axes[0, 1].set_xlabel(r"Squared $L_2$ Distance")
axes[0, 1].set_ylabel("Frequency")
axes[0, 1].hist(l2_values, bins=15, color=VIVID_PURPLE, alpha=0.7, edgecolor=NODE_BLACK)
axes[0, 1].axvline(l2_values.mean(), color=MALWARE_RED, linestyle="--",
                   linewidth=2, label=f"Mean: {l2_values.mean():.2f}")
axes[0, 1].legend(frameon=False, fontsize=10)

# L∞ distribution
axes[1, 0].set_title(r"$L_\infty$ Distortion Distribution", fontsize=14)
axes[1, 0].set_xlabel(r"$L_\infty$ Distance")
axes[1, 0].set_ylabel("Frequency")
axes[1, 0].hist(linf_values, bins=15, color=NUGGET_YELLOW, alpha=0.7, edgecolor=NODE_BLACK)
axes[1, 0].axvline(linf_values.mean(), color=MALWARE_RED, linestyle="--",
                   linewidth=2, label=f"Mean: {linf_values.mean():.2f}")
axes[1, 0].legend(frameon=False, fontsize=10)

# Elastic-net distribution
axes[1, 1].set_title("Elastic-Net Distortion Distribution", fontsize=14)
axes[1, 1].set_xlabel("Elastic Distance")
axes[1, 1].set_ylabel("Frequency")
axes[1, 1].hist(elastic_values, bins=15, color=AQUAMARINE, alpha=0.7, edgecolor=NODE_BLACK)
axes[1, 1].axvline(elastic_values.mean(), color=MALWARE_RED, linestyle="--",
                   linewidth=2, label=f"Mean: {elastic_values.mean():.2f}")
axes[1, 1].legend(frameon=False, fontsize=10)

plt.tight_layout()
plt.savefig(output_dir / "ead_distortion_analysis.png", dpi=150, bbox_inches="tight")
plt.close()

print(f" Saved to {output_dir}/ead_distortion_analysis.png")
```

|Code|What it does|
|---|---|
|`l1_dist.detach().cpu().numpy()`|pull each metric's 20 values into numpy for plotting|
|`hist(..., bins=15, ...)`|a 15-bin histogram per metric|
|`axvline(values.mean(), ...)`|the **red dashed mean line**|
|`legend(...)`|shows the mean value in the corner|

---

## Part 4 - Reading the histograms


![Attack Visualization](/assets/11%20Attack%20Visualization-20261017204447573.png)

|Metric|Mean|Shape|What it says|
|---|---|---|---|
|L1|20.20|cluster ~10-14, tail out to ~62|total change is moderate for most, but a few images need much more|
|Squared L2|4.18|most below 5, tail at 14/16/21|**a few hard examples need far more energy** (the right tail)|
|L∞|0.45|centered 0.3-0.45, up to ~0.9|**the biggest single pixel moves ~half the full 0-1 range** - not a subtle per-pixel attack|
|Elastic|4.38|tracks squared-L2 closely|the composite the optimizer actually minimized|

### The relationship to lock in (verified)

The elastic mean isn't a coincidence next to the squared-L2 mean:

```
mean(elastic) = mean(squared-L2) + β · mean(L1)
             = 4.18 + 0.01 × 20.20
             = 4.18 + 0.202 = 4.382 ≈ 4.38 ✓
```

==Because β = 0.01 is tiny, the L1 contribution is only ~0.2, so elastic ≈ squared-L2.== This is the numeric face of "β = 0.01 weights L2 far more than L1" - sparsity is barely being pushed here. Crank β to 0.1 and the L1 term would matter much more, and re-running the attack would shift the whole distribution toward sparser (fewer-pixel) solutions.

The L∞ reading is the most striking for the report: **even though EAD is a "sparse" attack, its biggest per-pixel change averages 0.45** - it changes few-ish pixels but moves them a lot, exactly the "concentrated changes" trade-off from Section 2.

> **An honest note the course skips (the 2.36 doesn't fit):** the intro says "19/20 = 95% success, average L2 distortion 2.36," but the figure reports squared-L2 mean 4.18. These are different quantities (one squared, one not), but 2.36 _can't_ be the true-L2 mean of this same data: by Jensen's inequality, the mean of the per-image square roots is at most √(4.18) = **2.045**, which is below 2.36. So the 2.36 is a stale or different-run figure - **trust the histogram's 4.18 (squared) and an implied true-L2 average ≤ 2.045**. The 95% success rate (19 of 20) is fine.

---

## Copy-paste cells (mine, not the course's)

### Full summary stats behind the histograms

The figures show shape; this prints the actual spread. Uses your names.

```python
import numpy as np
for name, t in [("L1", l1_dist), ("L2(sq)", l2_dist), ("Linf", linf_dist), ("Elastic", elastic_dist)]:
    v = t.detach().cpu().numpy()
    print(f"{name:8} mean {v.mean():6.3f} | median {np.median(v):6.3f} | "
          f"min {v.min():6.3f} | max {v.max():6.3f} | std {v.std():6.3f}")
```

Expected (numbers match your run; shape like):

```txt
L1 mean 20.200 | median 13.x | min 2.x | max 62.x | std 1x.x
L2(sq) mean 4.180 | median 1.x | min 0.x | max 21.x | std 5.x
...
```

The big gap between median and max (and the large std) is the **long right tail** the histograms show.

### Verify the elastic identity yourself

```python
beta = 0.01
lhs = elastic_dist.mean().item()
rhs = l2_dist.mean().item() + beta * l1_dist.mean().item()
print(f"mean elastic = {lhs:.4f}")
print(f"mean L2sq + beta*meanL1 = {rhs:.4f}") # should match
```

---

## Quick self-check

These same 20 perturbations have mean squared-L2 = 4.18 and mean L1 = 20.20 (at β = 0.01). If you **recompute the elastic distance** on these _same_ perturbations but with **β = 0.1**, what would the mean elastic distance be? (This is recomputing the metric, not re-running the attack.)

---

## The facts worth locking in for the exam

- ==The perturbation panel amplifies the change 10× and uses `seismic` (white = no change, red = brighter, blue = darker)==, revealing that EAD concentrates its changes on the digit's strokes and edges, not the background.
- **All displayed digits are misclassified with changes imperceptible to a human** (7→2, 0→6, 9→4, ...), the visual proof that a crossed decision boundary needn't look like a damaged image.
- ==mean(elastic) = mean(squared-L2) + β·mean(L1)==, verified as 4.18 + 0.01×20.20 = 4.38, so with a small β the elastic distortion nearly equals the squared-L2 distortion.
- **The distributions have long right tails**: most images need modest distortion (squared-L2 under 5) while a few hard examples need far more (up to ~21), which is why reporting the spread matters more than the mean alone.
- **The L∞ mean of ~0.45 shows EAD moves its chosen pixels a lot** (nearly half the 0-1 range), confirming the sparse-attack trade-off of few pixels changed by large amounts rather than many pixels changed slightly.
- **Four norms give four complementary views**: L1 (total change), squared-L2 (energy), L∞ (worst single pixel), elastic (the optimized composite) - the same multi-ruler discipline from Section 5.
- ==The intro's "average L2 = 2.36" is inconsistent with this run's squared-L2 mean of 4.18== (Jensen caps the true-L2 mean at √4.18 = 2.045), so rely on the figure's 4.18 and the 95% (19/20) success rate.
- **Method point:** **plot the full distribution, not just the mean**, because a mean of 4.18 hides that most examples sit under 5 while a few outliers reach 21 - the shape is what tells you about heterogeneous robustness.

Logged Section 11, the 2.36 inconsistency, and the self-check answer key. Paste Section 12 (Sparsity Analysis) when ready - that's the last ElasticNet teaching section before the challenge.