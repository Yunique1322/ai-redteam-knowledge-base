# Visualizations

## The idea in one breath

Three plots that let you *see* a JSMA attack: a 4-panel transformation (original → adversarial → perturbation heatmap → binary mask), a dual-axis progress curve (confidence vs pixels over iterations), and a saliency heatmap showing which pixels JSMA wanted most. No new attack logic - just matplotlib on results you already have.

## Part 1 - Prepare the data

```python
def prepare_attack_process_data(original, adversarial):
    # Convert to numpy
    if isinstance(original, torch.Tensor):
        orig = original.cpu().numpy()
    else:
        orig = original
    if isinstance(adversarial, torch.Tensor):
        adv = adversarial.cpu().numpy()
    else:
        adv = adversarial

    # Squeeze and compute perturbations
    orig = np.squeeze(orig)
    adv = np.squeeze(adv)
    pert = adv - orig
    pert_mag = np.abs(pert)
    mod_map = (pert_mag > 0)

    # Compute norms
    l0 = int(mod_map.sum())
    l2 = float(np.linalg.norm(pert))
    linf = float(pert_mag.max() if pert_mag.size else 0.0)

    return {'orig': orig, 'adv': adv, 'pert_mag': pert_mag, 'mod_map': mod_map,
            'l0': l0, 'l2': l2, 'linf': linf}
```

| Code | What it does |
|---|---|
| `isinstance(..., torch.Tensor)` | accept either a tensor or a numpy array |
| `np.squeeze(...)` | `(1,1,28,28)` → `(28,28)` for plotting |
| `pert = adv - orig`, `pert_mag = np.abs(pert)` | the change and its magnitude |
| `mod_map = (pert_mag > 0)` | boolean "did this pixel change" map |
| `l0 = mod_map.sum()` | **true L0 - count of pixels actually changed** (`count_nonzero`, not the attack's step counter) |
| `l2`, `linf` | Euclidean size, biggest single change |

## Part 2 - The 4-panel figure

```python
def visualize_attack_process(original, adversarial, original_class, target_class,
                             predicted_class, output_dir):
    from htb_ai_library.visualization.styles import HTB_GREEN, MALWARE_RED

    data = prepare_attack_process_data(original, adversarial)
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.8))
    success = (predicted_class == target_class)

    # Original
    axes[0].imshow(data['orig'], cmap='gray', interpolation='nearest')
    axes[0].set_title(f'Original\nTrue: {original_class}', color=HTB_GREEN, fontsize=12)
    axes[0].axis('off')

    # Adversarial
    axes[1].imshow(data['adv'], cmap='gray', interpolation='nearest')
    axes[1].set_title(f'Adversarial\nPred: {predicted_class} Target: {target_class} ' +
                      ('Success' if success else 'Fail'),
                      color=HTB_GREEN if success else MALWARE_RED, fontsize=12)
    axes[1].axis('off')

    # Perturbation magnitude
    pm_norm = data['pert_mag'] / (data['pert_mag'].max() + 1e-8)
    axes[2].imshow(pm_norm, cmap='hot', interpolation='nearest')
    axes[2].set_title(f'Perturbation |Δ|\nL0={data["l0"]} L2={data["l2"]:.3f} L∞={data["linf"]:.3f}',
                      color=HTB_GREEN, fontsize=12)
    axes[2].axis('off')

    # Modified pixels
    axes[3].imshow(data['mod_map'], cmap='hot', interpolation='nearest')
    total = int(data['orig'].size)
    percent = 100.0 * data['l0'] / max(1, total)
    axes[3].set_title(f'Modified Pixels\n{data["l0"]}/{total} ({percent:.2f}%)',
                      color=HTB_GREEN, fontsize=12)
    axes[3].axis('off')

    plt.tight_layout()
    save_path = output_dir / 'jsma_attack_process.png'
    plt.savefig(save_path, dpi=175, bbox_inches='tight')
    plt.close()
    print(f"Attack process visualization saved to {save_path}")
```

| Choice | Why |
|---|---|
| `cmap='gray'` for panels 0-1 | keep the digit readable |
| title green on success, red on fail | instant pass/fail read |
| `pm_norm = pert_mag / (max + 1e-8)` | normalize so faint changes are still visible; `+1e-8` avoids divide-by-zero |
| `cmap='hot'` for panels 2-3 | yellow/red = big change, black = untouched |

## Part 3 - Reading the 4-panel figure

For sample 1 (7 → 2): the **Original** is a clean "7"; the **Adversarial** reads "2" to the model (green "Success"); the **Perturbation |Δ|** heatmap lights up the top bar and upper vertical stroke (hot yellow) with scattered red lower down - **concentrated, not uniform noise, the L0 signature**; the **Modified Pixels** mask shows the white changed pixels on black.

> **The number to notice - and a callback.** The figure says **L0 = 66** and **66/784 (8.42%)**, but back in Section 22 the attack counter reported `pixels_modified = 68`. That gap is the Section 19/23 lesson made visual: **`count_nonzero` (66) is the true sparsity; the step counter (68) overcounts** because 2 of the modifications left the pixel unchanged after clamping (a net-zero move). The course prose here says "68 ... L0=68 ... L∞=1.0," but the actual figure shows **66, 8.42%, L∞=0.996** - trust the figure. (L∞ is 0.996 not 1.0 because the clean "7" had a faint grey pixel, so the largest change fell just short of a full 1.0.)

## Part 4 - Progress curve (dual axis)

```python
def visualize_attack_progress(stats, output_dir):
    from htb_ai_library.visualization.styles import HTB_GREEN, NUGGET_YELLOW

    its = stats['iterations']
    conf = stats['target_confidence']
    pix = stats['pixels_modified']

    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    ax2 = ax.twinx()

    ax.plot(its, conf, color=HTB_GREEN, linewidth=2.2, label='Target Confidence')
    ax2.plot(its, pix, color=NUGGET_YELLOW, linewidth=2.0, label='Pixels Modified')

    ax.set_xlabel('Iteration'); ax.set_ylabel('Target Confidence'); ax2.set_ylabel('Pixels Modified')
    ax.grid(True, alpha=0.35)

    lines = ax.get_lines() + ax2.get_lines()
    ax.legend(lines, [l.get_label() for l in lines], loc='upper left', framealpha=0.9)
    plt.tight_layout()
    save_path = output_dir / 'jsma_progress.png'
    plt.savefig(save_path, dpi=175, bbox_inches='tight'); plt.close()
    print(f"Progress visualization saved to {save_path}")
```

`ax.twinx()` is the key trick: a **second y-axis sharing the x-axis**, so confidence (left, green) and pixel count (right, yellow) sit on one timeline despite wildly different scales.

> **Read the figure carefully - it's a *failure*.** This plot is fed `stats` from the **single-pixel** attack (Sections 17-18), not the pairwise success. The green confidence line crawls to only ~0.00027 over 100 iterations and never approaches 0.5; the yellow pixel line climbs linearly to 100. So you're looking at the single-pixel attack that *failed* on this sample - flat confidence, full budget spent. (The healthy shape would be the Section 22 curve: flat, then a sharp climb across the boundary.)

## Part 5 - Saliency heatmap

```python
def visualize_saliency_map(alpha, beta, search_space, original_shape, output_dir):
    from htb_ai_library.visualization.styles import HTB_GREEN

    inc_scores = score_increase_saliency(alpha, beta)
    dec_scores = score_decrease_saliency(alpha, beta)
    combined_scores = np.maximum(inc_scores, dec_scores)
    combined_scores = combined_scores * search_space

    h, w = original_shape[-2:]
    saliency_map = combined_scores.reshape(h, w)

    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(saliency_map, cmap='hot', interpolation='nearest')
    ax.set_title('Saliency Map\n(Higher = More Important)', color=HTB_GREEN, fontsize=14, fontweight='bold')
    ax.axis('off')
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Saliency Score')
    plt.tight_layout()
    save_path = output_dir / 'jsma_saliency_map.png'
    plt.savefig(save_path, dpi=175, bbox_inches='tight'); plt.close()
    print(f"Saliency map saved to {save_path}")
```

| Code | What it does |
|---|---|
| `np.maximum(inc_scores, dec_scores)` | per pixel, keep the better of its two directions |
| `* search_space` | zero out unavailable pixels |
| `.reshape(h, w)` | the length-784 score vector back to 28×28 for display |

The heatmap (hot spots up to ~0.63) shows the high-saliency pixels on the **clean** image at iteration 0. Note it uses the **single-pixel** saliency scores, not pairwise - it's an "importance map," so the brightest cells are where JSMA would reach first.

## Copy-paste cell (mine, not the course's)

A self-contained 4-panel viz (no HTB-library colors) that prints the honest L0 vs step-count gap - drop in any `(orig, adv)` pair:

```python
# === Section 25: self-contained attack-process viz ===
import numpy as np, torch, matplotlib.pyplot as plt

def show_attack_process(original, adversarial, true_cls, target_cls, pred_cls,
                        pixels_modified=None, save=None):
    o = np.squeeze(original.cpu().numpy() if isinstance(original, torch.Tensor) else original)
    a = np.squeeze(adversarial.cpu().numpy() if isinstance(adversarial, torch.Tensor) else adversarial)
    pert = a - o; mag = np.abs(pert)
    l0, l2, linf = int(np.count_nonzero(pert)), float(np.linalg.norm(pert)), float(mag.max())
    ok = (pred_cls == target_cls)

    fig, ax = plt.subplots(1, 4, figsize=(14, 3.8))
    ax[0].imshow(o, cmap='gray'); ax[0].set_title(f"Original\nTrue: {true_cls}")
    ax[1].imshow(a, cmap='gray'); ax[1].set_title(f"Adv\nPred {pred_cls} Tgt {target_cls} "
                                                  + ("OK" if ok else "FAIL"),
                                                  color='green' if ok else 'red')
    ax[2].imshow(mag/(mag.max()+1e-8), cmap='hot'); ax[2].set_title(f"|Δ| L0={l0} L2={l2:.2f} L∞={linf:.3f}")
    ax[3].imshow(mag > 0, cmap='hot'); ax[3].set_title(f"Modified {l0}/{o.size} ({100*l0/o.size:.2f}%)")
    for a_ in ax: a_.axis('off')
    plt.tight_layout()
    if pixels_modified is not None and pixels_modified != l0:
        print(f"note: counter said {pixels_modified} modifications, true L0={l0} (gap = net-zero moves)")
    if save: plt.savefig(save, dpi=150, bbox_inches='tight')
    plt.show()
```

## Quick self-check

The attack counter reported `pixels_modified = 68`, but the perturbation panel's title shows **L0 = 66**.

1. What does the 2-pixel gap mean?
2. Which number is the honest sparsity to report?

I'll confirm next message.

## The facts worth locking in for the exam

- **The 4-panel figure tells the whole story**: original, adversarial (title colored by success), normalized perturbation heatmap, and binary modified-pixel mask with the L0 percentage.
- **Measure sparsity for the figure with `count_nonzero`**, not the attack's step counter - here they differ (66 vs 68) because some modifications were net-zero after clamping.
- **Normalize the perturbation magnitude before plotting** (`mag/(max+1e-8)`) or small changes render as uniform black; the `1e-8` guards against divide-by-zero.
- **JSMA perturbations concentrate on semantic strokes, scattered rather than uniform** - the visual fingerprint of an L0-minimizing attack vs FGSM's even noise.
- **`ax.twinx()` overlays two different-scale metrics on one timeline** - confidence (left) against cumulative pixels (right).
- **The progress plot shown is a single-pixel *failure*** (confidence maxes ~0.0003, never crossing the boundary) - a healthy run would be flat-then-sharp; read which attack a plot came from.
- **The saliency heatmap reshapes the length-784 score vector to 28×28** using `np.maximum(inc, dec) * search_space`, highlighting the pixels JSMA would pick first on the clean image.
- **Method point:** **these visualizations are diagnostics, not new attack logic** - use the 4-panel for a single result, the progress curve to spot stalls/failures, and the saliency map to sanity-check that high-importance pixels land on meaningful features.

Paste Section 26 when ready - that's the aggregate visualizations (L0 distributions, single-pixel vs pairwise across the whole test set), so expect comparison plots and summary statistics.