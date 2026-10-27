# Section 12: Sparsity Analysis

## The idea in one breath

The last ElasticNet section looks at **how sparse the perturbations really are and where they land**. Two views: a scatter of L1 vs squared-L2 (plus sparsity), and heatmaps showing the change concentrates **on the digit's edges and strokes**. Measured mean sparsity is 52.6% (about half the pixels untouched), and **100% success (20/20)**. I have to flag this section carefully: the course gets the **L1-vs-sparsity direction backwards twice**, and the data in its own figures proves it.

> **Heads-up: mangled formulas.** Read them as:
> 
> - `L1≈L2⋅k` -> the course means L1 ≈ √(k · squared-L2) for k equal-magnitude pixels (its own later formula makes this precise).
> - `k≈L1²/(squaredL2)` -> k ≈ L1² / (squared-L2). The example `k≈60²/9=400`.
> - `L1 ∝ √L2` -> L1 proportional to √(squared-L2), the reference curve.
> - sparsity = `(1 − nonzero/784) × 100` = **percent of pixels left unchanged**.

---

## Part 1 - The mixed-norm scatter code

```python
# Visualization 3: Mixed-norm relationship and sparsity-distortion tradeoff
print("Creating mixed-norm analysis...")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Prepare data
l1_values = l1_dist.detach().cpu().numpy()
l2_values = l2_dist.detach().cpu().numpy()
perturbations = best_adv - original_images
nonzero_mask = torch.abs(perturbations) > 1e-6
sparsity_per_example = (1 - nonzero_mask.float().sum(dim=(1, 2, 3)) / 784) * 100
sparsity_values = sparsity_per_example.detach().cpu().numpy()

# Color by success (green=success, red=failed)
colors = [HTB_GREEN if s else MALWARE_RED for s in final_success.cpu().numpy()]

# Plot 1: L1 vs L2 relationship
axes[0].set_title(r"$L_1$ vs Squared $L_2$ Distortion Relationship", fontsize=14)
axes[0].set_xlabel(r"Squared $L_2$ Distance", fontsize=12)
axes[0].set_ylabel(r"$L_1$ Distance", fontsize=12)
axes[0].scatter(l2_values, l1_values, c=colors, s=100, alpha=0.7,
                edgecolors=NODE_BLACK, linewidth=1.5)

# Add reference line showing L1 = sqrt(L2) relationship
l2_range = np.linspace(l2_values.min(), l2_values.max(), 100)
axes[0].plot(l2_range, np.sqrt(l2_range) * 8, color=AZURE, linestyle="--",
             linewidth=2, alpha=0.5, label=r"Reference: $L_1 \propto \sqrt{L_2}$")
axes[0].legend(frameon=False, fontsize=10)
axes[0].grid(True, alpha=0.3, color=HACKER_GREY)

# Add success legend manually
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor=HTB_GREEN, edgecolor=NODE_BLACK, label=f'Success ({final_success.sum()}/{batch_size})'),
    Patch(facecolor=MALWARE_RED, edgecolor=NODE_BLACK, label=f'Failed ({(~final_success).sum()}/{batch_size})')
]
axes[0].legend(handles=legend_elements, loc='upper left', frameon=False, fontsize=10)

# Plot 2: Sparsity vs L2 distortion
axes[1].set_title("Sparsity vs Distortion Tradeoff", fontsize=14)
axes[1].set_xlabel(r"Squared $L_2$ Distance", fontsize=12)
axes[1].set_ylabel("Sparsity (%)", fontsize=12)
axes[1].scatter(l2_values, sparsity_values, c=colors, s=100, alpha=0.7,
                edgecolors=NODE_BLACK, linewidth=1.5)

# Add mean lines
axes[1].axvline(l2_values.mean(), color=VIVID_PURPLE, linestyle="--",
                linewidth=2, alpha=0.7, label=f"Mean $L_2$: {l2_values.mean():.2f}")
axes[1].axhline(sparsity_values.mean(), color=AQUAMARINE, linestyle="--",
                linewidth=2, alpha=0.7, label=f"Mean Sparsity: {sparsity_values.mean():.1f}%")
axes[1].legend(frameon=False, fontsize=10)
axes[1].grid(True, alpha=0.3, color=HACKER_GREY)

plt.tight_layout()
plt.savefig(output_dir / "ead_success_analysis.png", dpi=150, bbox_inches="tight")
plt.close()

print(f" Saved to {output_dir}/ead_success_analysis.png")
```

|Code|What it does|
|---|---|
|`nonzero_mask = torch.abs(perturbations) > 1e-6`|which pixels actually changed|
|`(1 - nonzero.sum(dim=(1,2,3)) / 784) * 100`|==sparsity % = fraction of pixels left unchanged==, per image|
|`colors = [HTB_GREEN if s else MALWARE_RED ...]`|green = fooled, red = not|
|left scatter|L1 (y) vs squared-L2 (x)|
|`np.sqrt(l2_range) * 8`|the reference curve L1 = 8·√(squared-L2)|
|right scatter|sparsity (y) vs squared-L2 (x), with mean lines|

---

## Part 2 - The L1 / L2 / sparsity relationship (and the course's two sign errors)

### The real relationship

If a perturbation changes **k** pixels each by magnitude **a**:

- L1 = k·a
- squared-L2 = k·a²
- therefore **L1 = √(k · squared-L2)**, and **k = L1² / squared-L2**

So for a **fixed** squared-L2 energy, **more changed pixels -> larger L1**. Concretely (same energy = 9):

|Perturbation|changed pixels|L1|squared-L2|
|---|---|---|---|
|sparse (1 pixel of 3.0)|1|**3**|9|
|dense (9 pixels of 1.0)|9|**9**|9|

**The sparse one has the _smaller_ L1.** That single fact flips both of the course's interpretations.

> **An honest note the course skips (two backwards claims - verified):** the course says the reference-curve reading is "points **above** the curve indicate **sparser** perturbations... increasing L1 relative to L2," and that k = L1²/L2 is "an **upper bound** on k... actual k is often **smaller** (higher sparsity)." Both are backwards:
> 
> 1. **Above the reference = _denser_, not sparser.** For a fixed squared-L2, a bigger L1 means _more_ changed pixels (the table above). So high points are the _least_ sparse. The figures prove it: the high-squared-L2 outliers (≈13, ≈16) sit _farthest above_ the reference curve on the left, and on the right they have the _lowest_ sparsity (~30%, ~38%). Same points, both plots - above = dense.
> 2. **k = L1²/L2 is a _lower_ bound on the changed-pixel count, not an upper bound.** By the Cauchy - Schwarz inequality, the true number of changed pixels is always ≥ L1²/squared-L2 (equal only when all magnitudes match). I checked numerically: random varied-magnitude perturbations with true k = 120 give k_est ≈ 63-68, always below the truth. Using the reported means, k_est = 20.20²/4.18 ≈ 98 pixels -> an _estimated_ 87.5% sparsity, but the _measured_ sparsity is only 52.6%. **Real (varying-magnitude) perturbations change _more_ pixels than the equal-magnitude estimate, so they're _less_ sparse than the formula suggests**, the opposite of the course's "actual k smaller, higher sparsity."
> 
> The course's _arithmetic_ (k ≈ 60²/9 = 400 -> ~49% sparsity) is fine as an equal-magnitude calculation; only the direction of the "bound" and the "above = sparser" reading are wrong. ==Correct rule: lower L1 at a given squared-L2 = sparser = below the reference curve.==

---

## Part 3 - Reading the mixed-norm figure

![Part 3 - Reading the mixed-norm figure](/assets/Pasted%20image%2020261017204729.png)

**Left plot (L1 vs squared-L2):**

|What|Reading|
|---|---|
|All markers green|**100% success, 20/20**|
|Curve shape|L1 rises like √(squared-L2), as the k = L1²/L2 relationship predicts|
|Cluster|most points at squared-L2 < 5 (mean 4.18), L1 in the low-teens to ~26|
|Outliers|a few at squared-L2 ≈ 13/16/21 with L1 up to ~62 - these sit far above the reference, i.e. **denser (most changed pixels)**|

**Right plot (sparsity vs squared-L2):**

|What|Reading|
|---|---|
|Most points|45-60% sparsity (mean line at 52.6%) at low squared-L2|
|Outliers|the high-squared-L2 points drop to ~30-38% sparsity - **harder images need denser changes**|
|Trend|**higher squared-L2 <-> lower sparsity** (the correct direction)|

The two plots agree: the expensive (high-L2) examples are both far above the reference line _and_ the least sparse, which is exactly the dense-attack signature.

> **An honest note (success count):** Section 11's intro said "95% (19/20)," but this section's figures clearly say **20/20 = 100%**, and the squared-L2 mean (4.18) and L1 mean (20.20) match Section 11's. So the run fooled all 20 - the "19/20" was the stale figure I flagged last section.

---

## Part 4 - The sparsity heatmap code

```python
# Visualization 4: Sparsity analysis
print("Creating sparsity analysis...")

fig = plt.figure(figsize=(16, 8))

# Create grid: 2 rows × 5 columns for examples, 1 row for statistics
gs = fig.add_gridspec(3, 5, height_ratios=[1, 1, 0.6], hspace=0.3, wspace=0.3)

# Show perturbation heatmaps for first 10 examples
num_display_sparse = min(10, batch_size)

for i in range(num_display_sparse):
    row = i // 5
    col = i % 5
    ax = fig.add_subplot(gs[row, col])

    pert = (best_adv[i] - original_images[i]).detach().cpu().squeeze()

    # Show perturbation with colorbar
    im = ax.imshow(torch.abs(pert), cmap="hot", vmin=0, vmax=pert.abs().max())
    ax.axis("off")
    ax.set_title(f"Example {i+1}", fontsize=10)

# Compute sparsity statistics
perturbations = best_adv - original_images
nonzero_mask = torch.abs(perturbations) > 1e-6
sparsity_per_example = (1 - nonzero_mask.float().sum(dim=(1, 2, 3)) / 784) * 100
sparsity_values = sparsity_per_example.detach().cpu().numpy()

# Statistics subplot
ax_stats = fig.add_subplot(gs[2, :])
ax_stats.set_title("Sparsity Distribution Across Examples", fontsize=14)
ax_stats.set_xlabel("Example Index")
ax_stats.set_ylabel("Sparsity (%)")
ax_stats.bar(range(len(sparsity_values)), sparsity_values, color=AQUAMARINE,
             alpha=0.7, edgecolor=NODE_BLACK)
ax_stats.axhline(sparsity_values.mean(), color=MALWARE_RED, linestyle="--",
                 linewidth=2, label=f"Mean: {sparsity_values.mean():.2f}%")
ax_stats.legend(frameon=False, fontsize=10)
ax_stats.set_ylim([0, 100])

plt.tight_layout()
plt.savefig(output_dir / "ead_sparsity_analysis.png", dpi=150, bbox_inches="tight")
plt.close()

print(f" Saved to {output_dir}/ead_sparsity_analysis.png")
```

|Code|What it does|
|---|---|
|`gs = fig.add_gridspec(3, 5, height_ratios=[1,1,0.6])`|2 rows of 5 heatmaps + a shorter stats row|
|`imshow(torch.abs(pert), cmap="hot", vmin=0, vmax=pert.abs().max())`|==magnitude heatmap: black = no change -> white = max change==|
|`ax_stats.bar(...)`|one sparsity bar per image|
|`axhline(sparsity_values.mean())`|the mean sparsity line|

---

## Part 5 - Reading the sparsity figure

![Sparsity Analysis](/assets/12%20Sparsity%20Analysis-20261017204716358.png)

**Heatmaps (`hot`: black = 0, red/orange = moderate, white = max):** the bright pixels **cluster along the digit's strokes and edges**, not the background. Example 3 lights up the central region; Example 6 the branching strokes; Example 5 runs along the vertical stroke.

Why edges? **Gradient-based optimization finds the steepest directions, and a CNN's early layers extract edge features**, so edge pixels are the most influential. Background pixels barely move the prediction (the model learned to ignore uniform regions), so they stay dark.

**Bar chart:** mean sparsity 52.6%, bars ranging ~30% to ~73%. The variation is **robustness heterogeneity**: some digits flip with few changed pixels (high bars), tough ones need denser changes (low bars). Note the lowest bars correspond to the high-squared-L2 outliers from the scatter - the same hard examples.

---

## Copy-paste cells (mine, not the course's)

### Estimate changed pixels and check it against the truth

Shows that k = L1²/L2 under-counts changed pixels (so over-states sparsity). Uses your names.

```python
import numpy as np
perts = (best_adv - original_images).detach().cpu()
true_k = (perts.abs() > 1e-6).float().sum(dim=(1,2,3)).numpy() # real changed-pixel count
l1 = l1_dist.detach().cpu().numpy(); l2sq = l2_dist.detach().cpu().numpy()
k_est = l1**2 / l2sq # equal-magnitude estimate
print("mean true changed pixels:", true_k.mean().round(1))
print("mean k_est (L1^2/L2sq) :", k_est.mean().round(1), " (always <= true_k by Cauchy-Schwarz)")
print("mean true sparsity % :", (100*(1-true_k/784)).mean().round(1))
print("mean k_est sparsity % :", (100*(1-k_est/784)).mean().round(1), " (over-states sparsity)")
```

Expected: `k_est` below `true_k`, and the k_est sparsity well above the true 52.6%.

### Rank two perturbations by sparsity

```python
import numpy as np
for name, L1, L2sq in [("P", 10, 4), ("Q", 20, 4)]:
    print(f"{name}: k_est = {L1**2/L2sq:.0f} changed pixels -> {'sparser' if L1**2/L2sq < 50 else 'denser'}")
```

## Key facts to lock in

- **Sparsity is the percent of pixels left unchanged, (1 − nonzero/784)×100**, and this run's mean is 52.6% with 100% (20/20) success - so EAD leaves about half the pixels untouched here.
- **For a fixed squared-L2, fewer changed pixels (sparser) means _lower_ L1**, because L1 = √(k · squared-L2); so sparse perturbations sit _below_ the reference curve and dense ones above - the opposite of what the course's text states.
- ==k = L1²/squared-L2 is a _lower_ bound on the number of changed pixels (Cauchy-Schwarz), reached only when all magnitudes are equal==, so with varying magnitudes the real perturbation changes _more_ pixels and is _less_ sparse than the formula implies (estimate 87.5% vs measured 52.6%).
- **Higher squared-L2 coincides with lower sparsity**: the hard-example outliers (squared-L2 ≈ 13-21) are both farthest above the reference line and the least sparse (~30-38%), the signature of dense modifications.
- **Perturbations concentrate on the digit's edges and strokes, not the background**, because gradient descent targets the high-influence edge pixels a CNN's early layers rely on, while uniform background pixels barely move the prediction.
- **Sparsity varies widely across images (~30% to ~73%, mean 52.6%)**, reflecting heterogeneous robustness - easy digits flip with few changed pixels, hard ones need denser changes.
- ==The equal-magnitude model L1 = k·a, squared-L2 = k·a² is the bridge between the two norms and the pixel count==, and it's the quickest way to sanity-check a scatter point's implied sparsity in the report.
- **Method point:** ==when a text claims a direction ("above = sparser"), test it against the extreme points in its own figure== - here the high-distortion outliers are visibly the least sparse, which immediately exposes the reversed interpretation.

