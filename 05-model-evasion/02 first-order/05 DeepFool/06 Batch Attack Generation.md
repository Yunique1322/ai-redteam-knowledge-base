# Batch Attack Generation

## The idea in one breath

This section runs `deepfool` on the first **20 test images** one at a time and saves each result in a list of dictionaries. Then it summarizes them: **20/20 success, average L2 4.87, average 3.0 iterations**, and a **13× spread** between the easiest image (L2 0.60) and the hardest (L2 7.72). One image tells you almost nothing. Twenty start to show how unevenly robust a model is across its inputs.

> **Heads-up:** no mangled formulas. One wording trap. The course calls the easiest pair (3->5, L2 0.60) both "well-separated" and "structurally similar" in the same paragraph. Only "similar" is right. A tiny perturbation means the boundary is **close**. See the honest note in Part 3.

---

## Part 1: Set up the batch run

```python
num_examples = 20
print(f"\nGenerating {num_examples} adversarial examples using DeepFool...")

_, test_loader = get_mnist_loaders(batch_size=1, normalize=True)
model.eval()

results = []
success_count = 0

print(f"Test loader ready with {len(test_loader.dataset)} samples")
print(f"Will process first {num_examples} samples")
print("Starting batch attack generation...")
```

|Code|What it does|
|---|---|
|`num_examples = 20`|How many test images to attack.|
|`get_mnist_loaders(batch_size=1, normalize=True)`|One image per batch, which is the only shape `deepfool` handles.|
|`model.eval()`|Dropout off before any attack.|
|`results = []`|An empty list that will hold one **dictionary** per image.|
|`success_count = 0`|Counter for flips.|
|`len(test_loader.dataset)`|Size of the whole test set (10,000).|

**Expected:**

```txt
Generating 20 adversarial examples using DeepFool...
Test loader ready with 10000 samples
Will process first 20 samples
Starting batch attack generation...
```

**Why one image at a time?** The course says that in a batch, every image would wait for the slowest. True, but the simpler reason is that Section 17's `deepfool` only reads `fs[0, ...]`, so it **only works on one image** anyway.

---

## Part 2: The attack loop

```python
for idx, (data, target) in enumerate(test_loader):
    if idx >= num_examples:
        break

    data = data.to(device)

    # Execute DeepFool attack
    r, iterations, orig_label, adv_label, pert_image = deepfool(
        data, model, num_classes=10, overshoot=0.02, max_iter=50, device=device
    )

    # Track success and store metrics
    success = (orig_label != adv_label)
    if success:
        success_count += 1

    results.append({
        'original_image': data.cpu(),
        'perturbation': r.cpu(),
        'perturbed_image': pert_image.cpu(),
        'original_label': orig_label,
        'adversarial_label': adv_label,
        'iterations': iterations,
        'true_label': target.item(),
        'l2_norm': torch.norm(r.cpu()).item(),
        'success': success
    })

    # Progress feedback
    print(f" Example {idx+1}: True={target.item()}, Orig={orig_label}, "
          f"Adv={adv_label}, Iter={iterations}, L2={torch.norm(r.cpu()).item():.4f}")
```

|Code|What it does|
|---|---|
|`enumerate(test_loader)`|Gives a counter `idx` (0, 1, 2...) plus each image `data` and its true label `target`.|
|`if idx >= num_examples: break`|Stop after 20.|
|`deepfool(data, model, ...)`|Attack this one image. Same 5 returns as Section 17, renamed.|
|`success = (orig_label != adv_label)`|"Did the model's prediction change?" True or False.|
|`results.append({...})`|Store this image's record as a dictionary.|
|`.cpu()`|Moves tensors off the GPU so 20 images' worth of data doesn't fill GPU memory.|
|`target.item()`|The true digit as a plain number.|
|`torch.norm(r.cpu()).item()`|L2 size of the perturbation as a plain number.|
|`print(f" Example {idx+1}: ...")`|Live progress, one line per image.|

### What each dictionary holds

|Key|Type|Meaning|
|---|---|---|
|`original_image`|tensor (1,1,28,28)|the clean image (model space)|
|`perturbation`|tensor (1,1,28,28)|`r_tot`, without the 2% overshoot|
|`perturbed_image`|tensor (1,1,28,28)|the attacked image (with overshoot)|
|`original_label`|number|the model's **prediction** before the attack|
|`adversarial_label`|number|the prediction after|
|`iterations`|number|rounds needed|
|`true_label`|number|the real digit|
|`l2_norm`|number|size of the perturbation (L2, model space)|
|`success`|True/False|did the prediction change|

A **dictionary** is a labelled box: `results[0]['l2_norm']` means "image 0's L2". A list of dictionaries is basically a spreadsheet where each dictionary is a row.

**What the printout looks like (course's description):** fast cases show 1-2 iterations and L2 around 2-3, and slow cases show 5-6 iterations and L2 around 8-10.

---

## Part 3: Summary statistics

```python
print(f"\nAttack Success Rate: {success_count}/{num_examples} "
      f"({100*success_count/num_examples:.1f}%)")
print(f"Average L2 norm: {np.mean([r['l2_norm'] for r in results]):.4f}")
print(f"Average iterations: {np.mean([r['iterations'] for r in results]):.1f}")
```

|Code|What it does|
|---|---|
|`100*success_count/num_examples`|Success as a percentage.|
|`[r['l2_norm'] for r in results]`|A **list comprehension**: "pull `l2_norm` out of every dictionary into a new list".|
|`np.mean(...)`|Average of that list.|

Output:

![Batch Attack Generation](/assets/19%20Batch%20Attack%20Generation-20261017170935407.png)
### Reading the numbers

|Value|Meaning|
|---|---|
|20/20 (100%)|DeepFool always finds a boundary eventually. That's expected, because it has no budget limit.|
|Average L2 4.8661|Typical size in model space. × 0.3081 = **1.50 in pixel space**.|
|Average iterations 3.0|Most images flip in 2-4 rounds.|
|Smallest L2 0.60 (3 -> 5)|This 3 sat very close to the 5 boundary.|
|Largest L2 7.72 (7 -> 2)|The Section 18 image.|
|7.72 ÷ 0.60 ≈ 12.9, about 13×|**Robustness varies hugely from image to image.**|

**Why 100% isn't impressive here:** FGSM's success rate depended on ε. DeepFool keeps going until it flips, so 100% success is normal. **For DeepFool, the meaningful number is the size (L2 or ρ_adv), not the success rate.**

> **An honest note the course gets wrong:** it calls 3->5 (L2 0.60, 1 iteration) a "well-separated class" and 4->7 (5 iterations) "structurally similar". That's backwards. A tiny, fast flip means the image sat **close** to the boundary, i.e. the two digits look alike to the model (a 3 and a 5 share their lower curve). Well-separated classes need **larger** perturbations and usually more iterations. The same paragraph later calls 3->5 "structurally similar", which is the correct reading.

> **An honest note (2):** `success = (orig_label != adv_label)` checks whether the **prediction** changed, not whether the model was right to begin with. If the model had already misread an image, a flip still counts as success. Section 7's rule says to count success only among **originally correct** images. On these 20 it probably makes no difference (about 99% accuracy), but in a report it matters. Cell 2 below does it properly.

> **An honest note (3):** with a 13× spread, the **average** gets pulled up by a few hard images. The **median** (the middle value) is a fairer "typical" figure. Report both.

---

## Part 4: The grid visualization

```python
def visualize_attack_grid(results, save_dir='output'):
    """
    Create grid visualization showing original and adversarial images side-by-side.

    Args:
        results (list): Attack results from batch generation
        save_dir (str): Directory to save visualization
    """
    print("\nGenerating attack grid visualization...")

    num_examples = min(10, len(results))
    fig, axes = plt.subplots(4, 5, figsize=(15, 12))
    fig.patch.set_facecolor(NODE_BLACK)

    for ax in axes.flatten():
        ax.set_facecolor(NODE_BLACK)
        for spine in ax.spines.values():
            spine.set_edgecolor(HACKER_GREY)

    for idx in range(num_examples):
        row = idx // 5
        col = idx % 5

        # Original image (top row for this column)
        ax_original = axes[row * 2, col]
        img = mnist_denormalize(results[idx]['original_image'].squeeze()).numpy()
        ax_original.imshow(img, cmap='gray', vmin=0, vmax=1)
        ax_original.set_title(f"Original: {results[idx]['original_label']}",
                            color=HTB_GREEN, fontsize=10)
        ax_original.axis('off')

        # Adversarial image (bottom row for this column)
        ax_adv = axes[row * 2 + 1, col]
        adv_img = mnist_denormalize(results[idx]['perturbed_image'].squeeze()).numpy()
        ax_adv.imshow(adv_img, cmap='gray', vmin=0, vmax=1)

        title_color = MALWARE_RED if results[idx]['success'] else HACKER_GREY
        ax_adv.set_title(f"Adversarial: {results[idx]['adversarial_label']}",
                        color=title_color, fontsize=10)
        ax_adv.axis('off')

    plt.suptitle('DeepFool Attack: Original vs Adversarial Examples',
                color=HTB_GREEN, fontsize=16, y=0.98)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'deepfool_examples.png'),
                facecolor=NODE_BLACK, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Grid visualization saved to {save_dir}/deepfool_examples.png")

# Generate the grid visualization
visualize_attack_grid(results, save_dir='output')
```

|Code|What it does|
|---|---|
|`num_examples = min(10, len(results))`|Shows only the **first 10** of the 20 (the grid has room for 10 pairs).|
|`plt.subplots(4, 5, ...)`|4 rows × 5 columns = 20 panels = 10 original/adversarial pairs.|
|`axes.flatten()`|Turns the 4×5 grid into one list so a single loop can style every panel.|
|`row = idx // 5`|**Integer division**: which pair of rows (0 for images 0-4, 1 for images 5-9).|
|`col = idx % 5`|**Remainder**: which column (0-4).|
|`axes[row * 2, col]`|Original goes in row 0 or row 2.|
|`axes[row * 2 + 1, col]`|Its adversarial goes directly below, in row 1 or row 3.|
|`mnist_denormalize(...).numpy()`|Model space -> 0-1 pixels for display.|
|`MALWARE_RED if ... else HACKER_GREY`|Red title = flipped, grey = attack failed.|
|`plt.savefig(os.path.join(save_dir, 'deepfool_examples.png'), ...)`|Saves the figure as a PNG in `output/`.|
|`plt.close()`|Closes the figure **without showing it**.|

### The grid position arithmetic (toy)

|idx|idx // 5 (row pair)|idx % 5 (column)|Original at|Adversarial at|
|---|---|---|---|---|
|0|0|0|axes[0, 0]|axes[1, 0]|
|4|0|4|axes[0, 4]|axes[1, 4]|
|5|1|0|axes[2, 0]|axes[3, 0]|
|7|1|2|axes[2, 2]|axes[3, 2]|
|9|1|4|axes[2, 4]|axes[3, 4]|

> **If this errors (or nothing appears):** `plt.close()` means the grid is **never shown in the notebook**, only saved to `output/deepfool_examples.png`. To see it inline, use Cell 4 below, or change `plt.close()` to `plt.show()`. If you get `NameError: mnist_denormalize`, add `from htb_ai_library import mnist_denormalize` (it's missing from Section 16's imports).

---

## Reading the figure
![Reading the figure](/assets/Pasted%20image%2020261016194700.png)
The figure shows the first 10 test images: original on top, adversarial directly below.

|Pair|Original -> Adversarial|What changed visually|
|---|---|---|
|1|7 -> 2|Grey haze around the strokes, plus a bright smudge at the bottom-left where a 2's base would be. The most visible one, matching its L2 of 7.72.|
|2|2 -> 6|Faint speckle around the lower loop. Hard to spot.|
|3|1 -> 8|A faint crossing stroke and speckle in the middle, hinting at an 8's loops.|
|4|0 -> 6|Speckle inside and around the ring.|
|5|4 -> 9|Faint brightening near the top, partly closing the 4 into a 9's loop.|
|6|1 -> 7|Grey noise along the left of the stroke.|
|7|4 -> 8|Light speckle, almost no visible change.|
|8|9 -> 4|The top of the loop is roughened, partly opening a 9 toward a 4.|
|9|5 -> 6|Almost identical to the original, so this is likely one of the smallest perturbations.|
|10|9 -> 4|Speckle around the loop.|

**Patterns worth noticing:**

|Observation|What it means|
|---|---|
|Every adversarial title is red|10/10 flipped, matching the 20/20 summary.|
|The flips are to look-alike digits (9<->4, 1->7, 5->6, 0->6)|DeepFool goes to the **closest boundary**, which is usually the most similar-looking digit. This is the Section 15 argmin at work.|
|Noise sits around the strokes, not spread across the whole image|The concentration from Section 18, seen on 10 images.|
|Some changes are clearly visible (7->2, 1->8)|On MNIST's black-and-white digits, "minimal" is still noticeable, as the course admits.|

> **An honest note:** the titles say "Original: 7", but that's `original_label`, the model's **prediction**, not the true digit. They match here because the model is about 99% accurate, but on a weaker model they could differ.

---

## Copy-paste cells (mine, not the course's)

These use this section's names: `results`, `np`.

### Cell 1: Spread of the perturbation sizes

```python
l2s = np.array([r['l2_norm'] for r in results])
print("mean L2:", round(l2s.mean(), 4), "| median L2:", round(float(np.median(l2s)), 4))
print("min:", round(l2s.min(), 4), "| max:", round(l2s.max(), 4),
      "| spread x", round(l2s.max() / l2s.min(), 1))
print("mean L2 in pixel space:", round(l2s.mean() * 0.3081, 4))
```

**Expected:** mean `4.8661`, min about `0.60`, max about `7.72`, spread about `13`, pixel-space mean about `1.50`. The median will probably be a bit below the mean.

### Cell 2: Success rate by the Section 7 rule, plus ρ_adv

```python
correct = [r for r in results if r['original_label'] == r['true_label']]
flipped = [r for r in correct if r['success']]
print(f"originally correct: {len(correct)}/{len(results)}")
print(f"success among correct: {len(flipped)}/{len(correct)} = {len(flipped)/len(correct):.1%}")
rho = np.mean([(r['perturbation'].norm() / r['original_image'].norm()).item() for r in results])
print(f"rho_adv over these {len(results)} images: {rho:.2%}")
```

**Expected:** probably `20/20` correct and `100.0%`. ρ_adv is a percentage, smaller than the 7's 32.48%, because most images were easier.

### Cell 3: Easiest to hardest

```python
for r in sorted(results, key=lambda r: r['l2_norm']):
    print(f"true {r['true_label']}: {r['original_label']} -> {r['adversarial_label']} "
          f"iter={r['iterations']} L2={r['l2_norm']:.3f}")
```

**Expected:** 20 lines from smallest to largest L2, starting near `0.60` (a 3 -> 5) and ending with the `7 -> 2` at about `7.72`.

### Cell 4: Show the saved grid in the notebook

```python
from IPython.display import Image, display
display(Image('output/deepfool_examples.png'))
```

**Expected:** the 4×5 grid above, displayed inline.

---

## How important is this in general?

High. Running an attack over many samples, storing per-sample metrics and reporting success rate, average/median size and iterations is exactly the evidence a client report needs. Remember to apply the originally-correct filter.

## Key facts to lock in

- **On the first 20 MNIST test images, DeepFool succeeded 20/20 with an average model-space L2 of 4.87 (about 1.50 in pixel space) and an average of 3.0 iterations**. Individual sizes ranged from 0.60 (3 -> 5) to 7.72 (7 -> 2), about a 13× spread.
- **Each image is attacked separately (batch size 1)** because the course's `deepfool` only handles one image and each image needs a different number of iterations. Results are stored as a list of dictionaries with the images, labels, iterations, L2 and a success flag.
- **For DeepFool, a 100% success rate is expected and not informative, because it has no budget**. The meaningful robustness numbers are the perturbation sizes (average, median, ρ_adv).
- **A small, fast flip means the image sat close to the boundary of a look-alike class**, as with 3 -> 5. The course mislabels this as "well-separated", but well-separated classes need larger perturbations.
- ==The course's success flag checks `orig_label != adv_label`, so a misclassified image that changes still counts==. Section 7's rule counts success only among originally correct images.
- **DeepFool's flips go to visually similar digits (9 <-> 4, 5 -> 6, 1 -> 7)** because it always steps toward the nearest boundary, and the noise clusters around the strokes.
- `visualize_attack_grid` shows the first 10 of the 20 results, **saves the PNG to `output/deepfool_examples.png` and calls `plt.close()`, so nothing appears inline** unless you display the saved file or swap in `plt.show()`.
- **Method point:** **with large per-image variation, report the distribution, not just the mean**: give the median, the minimum and the maximum next to the average, convert sizes to pixel space, and compute success only over originally correct samples.
