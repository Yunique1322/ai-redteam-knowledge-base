# Section 8: Visualization

## The idea in one breath

Numbers say the attack worked; pictures show **how**. This section builds `visualize_attack`, which draws four panels for one image: the original, the adversarial version, the perturbation (amplified so you can see it), and a bar chart comparing class probabilities before and after. It's general-purpose: you hand it any attack function, so the same plotting code will work for I-FGSM and DeepFool later. `visualize_fgsm_attack` is a thin FGSM wrapper around it.

## Part 1: Imports and the styling helper

```python
import matplotlib.pyplot as plt
import numpy as np

# Colors imported from library
from htb_ai_library import (
    HTB_GREEN, NODE_BLACK, HACKER_GREY, WHITE,
    AZURE, NUGGET_YELLOW, MALWARE_RED, VIVID_PURPLE, AQUAMARINE
)

def _style_axes(ax: plt.Axes) -> None:
    """Apply Hack The Box dark theme to an axes instance.

    Args:
        ax: Matplotlib axes to style
    """
    ax.set_facecolor(NODE_BLACK)
    ax.tick_params(colors=HACKER_GREY)
    for spine in ax.spines.values():
        spine.set_color(HACKER_GREY)
    ax.grid(True, color=HACKER_GREY, linestyle="--", alpha=0.25)
```

|Code|What it does|
|---|---|
|`matplotlib.pyplot as plt`|Python's standard plotting library|
|`HTB_GREEN, NODE_BLACK, ...`|colour codes from the HTB library (just named colour strings)|
|`ax`|one **axes** = one panel/plot area inside a figure|
|`set_facecolor(NODE_BLACK)`|dark background|
|`tick_params(colors=...)`|grey tick marks and numbers|
|`ax.spines`|the four border lines of the panel, coloured grey|
|`grid(..., alpha=0.25)`|dashed grid; `alpha` is transparency (0 = invisible, 1 = solid), so 0.25 is faint|
|`-> None`|the function returns nothing; it just changes the panel's look|

This is pure cosmetics. It doesn't affect the attack at all.

## Part 2: The main function, `visualize_attack`

The course builds this **one** function across six consecutive blocks (prediction setup, figure layout, original panel, adversarial panel, perturbation panel, probability chart). Only the first piece starts with `def`; the rest are indented continuations that can't run alone. I've reassembled them into the single real function, every line unchanged and in order. Run it as one cell.

```python
def visualize_attack(model: nn.Module,
                    image: Tensor,
                    label: Tensor,
                    make_adv,
                    title: str,
                    num_classes: int = 10,
                    targeted: bool = False,
                    target_class: int | None = None) -> None:
    """HTB-styled visualization for adversarial examples.

    Args:
        model: Classifier in evaluation mode
        image: Single image in normalized space, shape (C,H,W)
        label: Scalar true label tensor
        make_adv: Callable (model, image_batch, label_batch) -> adv_batch in normalized space
        title: Figure title
        num_classes: Number of classes to show in probability bars
        targeted: Whether the attack is targeted
        target_class: Optional target class to annotate
    """
    model.eval()
    dev = next(model.parameters()).device
    image_dev = image.to(dev)
    label_dev = label.to(dev)

    # Compute clean predictions
    with torch.no_grad():
        clean_probs = F.softmax(model(image_dev.unsqueeze(0)), dim=1).squeeze(0)
        clean_pred = int(clean_probs.argmax().item())

    # Generate adversarial example
    x_adv_dev = make_adv(model, image_dev.unsqueeze(0), label_dev.unsqueeze(0)).squeeze(0)
    perturbation_dev = x_adv_dev - image_dev

    # Compute adversarial predictions
    with torch.no_grad():
        adv_probs = F.softmax(model(x_adv_dev.unsqueeze(0)), dim=1).squeeze(0)
        adv_pred = int(adv_probs.argmax().item())

    # Denormalize for visualization
    image_vis = mnist_denormalize(image_dev.unsqueeze(0)).squeeze(0).detach().cpu()
    x_adv_vis = mnist_denormalize(x_adv_dev.unsqueeze(0)).squeeze(0).detach().cpu()
    perturbation_vis = (x_adv_vis - image_vis)

    # Create figure with grid layout
    fig = plt.figure(figsize=(16, 10), facecolor=NODE_BLACK)
    gs = fig.add_gridspec(2, 3, hspace=0.35, wspace=0.35)

    # Original image panel
    ax1 = fig.add_subplot(gs[0, 0])
    _style_axes(ax1)
    if image_vis.shape[0] == 1:
        ax1.imshow(image_vis.squeeze(0), cmap='gray', vmin=0, vmax=1)
    else:
        ax1.imshow(image_vis.permute(1, 2, 0))
    ax1.set_title(f"Original | class={clean_pred} | p={clean_probs[clean_pred]:.2%}",
                  color=HTB_GREEN, fontweight="bold")
    ax1.set_xticks([])
    ax1.set_yticks([])

    # Adversarial image panel
    ax2 = fig.add_subplot(gs[0, 1])
    _style_axes(ax2)
    if x_adv_vis.shape[0] == 1:
        ax2.imshow(x_adv_vis.squeeze(0), cmap='gray', vmin=0, vmax=1)
    else:
        ax2.imshow(x_adv_vis.permute(1, 2, 0))
    title_color = MALWARE_RED if adv_pred != int(label.item()) else HTB_GREEN
    adv_title = f"Adversarial | class={adv_pred} | p={adv_probs[adv_pred]:.2%}"
    if targeted and target_class is not None:
        adv_title += f" | target={target_class}"
    ax2.set_title(adv_title, color=title_color, fontweight="bold")
    ax2.set_xticks([])
    ax2.set_yticks([])

    # Perturbation panel (scaled for visibility)
    ax3 = fig.add_subplot(gs[0, 2])
    _style_axes(ax3)
    pert_scaled = (perturbation_vis * 10 + 0.5).clamp(0, 1)
    if pert_scaled.shape[0] == 1:
        ax3.imshow(pert_scaled.squeeze(0), cmap='gray', vmin=0, vmax=1)
    else:
        ax3.imshow(pert_scaled.permute(1, 2, 0))
    ax3.set_title("Perturbation (x10)", color=NUGGET_YELLOW, fontweight="bold")
    ax3.set_xticks([])
    ax3.set_yticks([])

    # Class probability comparison
    ax4 = fig.add_subplot(gs[1, :])
    _style_axes(ax4)
    x = np.arange(num_classes)
    width = 0.4
    ax4.bar(x - width/2, clean_probs[:num_classes].cpu(), width,
            color=AZURE, label="clean")
    ax4.bar(x + width/2, adv_probs[:num_classes].cpu(), width,
            color=MALWARE_RED, label="adv")
    ax4.set_xlabel("Class", color=WHITE)
    ax4.set_ylabel("Probability", color=WHITE)
    legend = ax4.legend(facecolor=NODE_BLACK, edgecolor=HACKER_GREY)
    for text in legend.get_texts():
        text.set_color(WHITE)
    ax4.set_title("Class probabilities", color=HTB_GREEN, fontweight="bold")
    for text in ax4.get_xticklabels() + ax4.get_yticklabels():
        text.set_color(HACKER_GREY)

    # Add main title and display
    fig.suptitle(title, color=HTB_GREEN, fontweight="bold", fontsize=24, y=0.98)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    plt.show()
```

> **If this errors:** `target_class: int | None` needs Python 3.10 or newer. On older Python you get `TypeError: unsupported operand type(s) for |`. Fix it by adding `from typing import Optional` and writing `target_class: Optional[int] = None`. Same meaning, older syntax.




The rest of this section walks through the function in the course's six pieces.

### Piece 1: Predictions, attack, and denormalizing

#### Toy first: adding and removing the batch dimension

The model always expects a **batch**. One MNIST image has shape [1, 28, 28] (channel, height, width), so we wrap it as a batch of one and unwrap it afterwards:

```
image [1, 28, 28]
.unsqueeze(0) → [1, 1, 28, 28] add a "batch of 1" dimension at the front
model(...) → [1, 10] 10 scores for that one image
.squeeze(0) → [10] drop the batch dimension again
```

**`unsqueeze(0)` adds a size-1 dimension at position 0; `squeeze(0)` removes it.**

#### The lines

|Code|What it does|
|---|---|
|`make_adv` (a parameter)|an **attack function passed in as an input**. The visualizer doesn't care which attack it is, so it can be reused for I-FGSM and DeepFool later|
|`next(model.parameters()).device`|finds which device (CPU/GPU) the model's weights are on, so the image can go to the same place|
|`.to(dev)`|moves the image and label there (avoids the "same device" error)|
|clean prediction inside `no_grad()`|measuring only, so no gradients needed|
|`make_adv(...)` **outside** `no_grad()`|**the attack needs gradients, so it is deliberately not wrapped** (the Section 4 rule again)|
|`clean_probs.argmax()`|the predicted class; `int(...item())` turns it into a plain Python number|
|`perturbation_dev = x_adv_dev - image_dev`|δ in **normalized** units (computed but not plotted)|
|`mnist_denormalize(...)`|back to [0, 1] pixels for display (x_norm·σ + μ, Section 5)|
|`.detach().cpu()`|matplotlib can't draw GPU tensors or tensors with gradient tracking|
|`perturbation_vis = x_adv_vis - image_vis`|**δ in **pixel** units, the actual visual change**|

Without denormalizing, normalized values like −0.424 or 2.821 would be drawn on a [0, 1] scale and the digit would be unrecognizable.

### Piece 2: The figure layout

|Code|What it does|
|---|---|
|`plt.figure(figsize=(16, 10), ...)`|a 16×10-inch canvas with a dark background|
|`add_gridspec(2, 3, ...)`|splits it into a grid of **2 rows × 3 columns**|
|`hspace`, `wspace`|spacing between rows and between columns|

The layout: row 0 holds three images, `gs[0, 0]`, `gs[0, 1]` and `gs[0, 2]`. Row 1 holds the bar chart in `gs[1, :]`, where the `:` means "span all columns."

### Pieces 3 and 4: The original and adversarial panels

|Code|What it does|
|---|---|
|`image_vis.shape[0] == 1`|one channel means greyscale (MNIST); otherwise it's colour|
|`imshow(..., cmap='gray', vmin=0, vmax=1)`|draws in greys with **0 fixed as black and 1 fixed as white**|
|`.permute(1, 2, 0)`|colour images: PyTorch stores [C, H, W] but matplotlib wants [H, W, C], so this reorders the dimensions|
|`f"... p={...:.2%}"`|formats a probability as a percentage, e.g. 1.0 → `100.00%`|
|`set_xticks([])`, `set_yticks([])`|hides axis numbers on images|
|`MALWARE_RED if adv_pred != label else HTB_GREEN`|red title = fooled, green title = still correct|
|`adv_title += f"|target=..."`|

**Why `vmin=0, vmax=1` matters:** without it, matplotlib stretches **each** image's own darkest pixel to black and brightest to white. A faint perturbation could then look dramatic, or a real one could be hidden. **Fixing the scale makes the original and adversarial images directly comparable.**

**A small catch:** for **targeted** attacks, the title turns red whenever the prediction isn't the true label, even if the model landed on some _other_ wrong class rather than your target. Compare `class=` with `target=` in the title to see whether the targeted attack actually hit.

### Piece 5: The perturbation panel and the ×10 scaling

#### Toy first: mapping signed changes onto a 0-1 display

The perturbation has negative values (darkened pixels) and positive ones (brightened), and a display only shows 0 to 1. The formula `δ × 10 + 0.5`, then clamping to [0, 1]:

|Pixel change δ|× 10|+ 0.5|After clamp|Shown as|
|---|---|---|---|---|
|0 (unchanged)|0|0.5|0.5|mid-grey|
|+0.03|0.3|0.8|0.8|light grey|
|−0.03|−0.3|0.2|0.2|dark grey|
|**+0.2465**|2.465|2.965|**1.0**|pure white|
|**−0.2465**|−2.465|−1.965|**0.0**|pure black|

- ****0.5 means "no change,"** above 0.5 means brightened, below 0.5 means darkened.**
- The ×10 makes tiny changes visible; the +0.5 centres zero at mid-grey.

**Important for your ε:** the course's example uses ±0.03, but your attack uses ε = 0.8 normalized, which is **±0.2465** in pixels. ×10 pushes that far past the limits, so **every fully changed pixel shows as pure white or pure black**. The panel becomes nearly two-colour (saturated). The course's own warning applies: the panel shows **direction**, not true size.

### Piece 6: The probability bar chart

|Code|What it does|
|---|---|
|`np.arange(num_classes)`|positions 0, 1, ..., 9 along the x-axis|
|`x - width/2` and `x + width/2`|shift the clean bars left and the adversarial bars right so each class has a side-by-side pair|
|`clean_probs[:num_classes].cpu()`|the 10 probabilities, moved to CPU for plotting|
|`legend`, `set_color(...)`|the "clean / adv" key, styled for the dark theme|
|`fig.suptitle(...)`|the big title at the top|
|`tight_layout(rect=(0, 0, 1, 0.93))`|packs the panels neatly, leaving the top 7% for the title|
|`plt.show()`|displays the figure|

## Part 3: The FGSM wrapper

```python
def visualize_fgsm_attack(model: nn.Module,
                         image: Tensor,
                         label: Tensor,
                         epsilon: float,
                         num_classes: int = 10,
                         targeted: bool = False,
                         target_class: int | None = None) -> None:
    """Wrapper for visualize_attack using FGSM.

    Args:
        model: Classifier model
        image: Single image tensor
        label: True label
        epsilon: Perturbation budget
        num_classes: Classes to display
        targeted: If True, targeted attack
        target_class: Target class for targeted attack
    """
    def _make_adv(m, xb, yb):
        if targeted and target_class is None:
            raise ValueError("target_class must be provided when targeted=True")
        y_used = yb if not targeted else torch.full_like(yb, target_class)
        return fgsm_attack(m, xb, y_used, epsilon, targeted=targeted)

    mode = "Targeted" if targeted else "Untargeted"
    visualize_attack(model, image, label, _make_adv,
                    title=f"FGSM {mode}",
                    num_classes=num_classes,
                    targeted=targeted,
                    target_class=target_class)
```

|Code|What it does|
|---|---|
|`def _make_adv(m, xb, yb):`|a function defined **inside** another function. It "remembers" `epsilon`, `targeted` and `target_class` from the outer function (this is called a **closure**)|
|`raise ValueError(...)`|refuses a targeted attack with no target|
|`torch.full_like(yb, target_class)`|builds a label tensor the same shape as `yb`, filled with the target class (e.g. `[3]`)|
|`y_used = yb if not targeted else ...`|untargeted uses the **true** label; targeted uses the **target** label|
|`fgsm_attack(..., targeted=targeted)`|the Section 6 attack (`step_dir = −1` when targeted)|
|`visualize_attack(..., _make_adv, ...)`|passes the attack function into the general visualizer|

This is the Section 6 caution handled for you: **for targeted mode, the wrapper swaps in the target labels automatically.**

## Part 4: Running it

```python
# Assume images, labels from test_loader (from Setup)
# Assume epsilon from Core Implementation (epsilon=0.8)
_ = visualize_fgsm_attack(model, images[0].detach().cpu(),
                         labels[0].detach().cpu(), epsilon)
```

- `images[0]` is the first image in the batch, shape [1, 28, 28].
- `.detach().cpu()` hands over a clean CPU copy; the function moves it to the model's device itself.
- `_ = ...` means "ignore the return value" (it returns `None` anyway).
![Visualization](/assets/8%20Visualization-20261015220846666.png)
## Part 5: Reading the figure ("FGSM Untargeted")

### Top row

- **Original | class=7 | p=100.00%:** the clean "7," predicted with essentially total confidence.
- **Adversarial | class=3 | p=51.45%:** red title, so **the model is fooled and now says "3."** The background is speckled with grey blocks: those are black pixels pushed **brighter** by +0.2465 (about 63/255 grey). The black blocks are pixels the attack pushed **darker**, which the clamp stopped at black (no change, Section 7). The stroke of the 7 looks slightly dimmer.
- **Perturbation (x10):** the screenshot uses a heat-style colour map, while the code as written uses `'gray'`, so **your** version will be black/white/grey. Reading the screenshot: **white** = brightened, **orange** (the mid value, 0.5) = **no change**, **black** = darkened. Notice the 7's shape shows up in black (the stroke was dimmed), and a lot of background is orange (pushes downward were clamped away). The few red and yellow pixels are partial changes, where the clamp cut part of the step.

### Bottom row (class probabilities)

|Class|Clean (blue)|Adversarial (red)|
|---|---|---|
|2|≈ 0|≈ 0.05|
|3|≈ 0|**≈ 0.51**|
|7|**≈ 1.00**|≈ 0.43|

**It barely flipped: "3" wins 51% to 43%.** This is Section 7's point that a flip can be marginal. The model still gives the true class 43%. The untargeted attack didn't choose "3"; it pushed "away from 7," and 3 happened to be the nearest rival. That's the "nearest boundary" idea DeepFool formalizes later.

## Copy-paste cells (mine, not the course's)

These use names you already have: `model`, `images`, `labels`, `epsilon`, `flipped`, `x_adv`.

### Visualize an image the attack definitely flipped

Picks the first flipped image from the Section 6 batch.

```python
i = flipped.nonzero().flatten()[0].item()
visualize_fgsm_attack(model, images[i].detach().cpu(), labels[i].detach().cpu(), epsilon)
```

Expected: the same four-panel figure with a **red** adversarial title.

### Watch ε grow on the same image

Draws three figures for image 0 at increasing budgets.

```python
for eps in [0.1, 0.3, 0.8]:
    visualize_fgsm_attack(model, images[0].detach().cpu(), labels[0].detach().cpu(), eps)
```

Expected: at 0.1 the title probably stays **green** (no flip) and the noise is faint. By 0.8 it flips (red) with obvious background speckle.

### Try a targeted attack

Asks FGSM to turn image 0 into a "3."

```python
visualize_fgsm_attack(model, images[0].detach().cpu(), labels[0].detach().cpu(),
                      epsilon, targeted=True, target_class=3)
```

Expected: title "FGSM Targeted" and `target=3` in the adversarial panel title. Check whether `class=` equals 3. Single-step targeted FGSM often misses, which is the motivation for I-FGSM (Sections 10-12).

### Print the top-3 probabilities without plotting

Useful in labs where you just need the numbers for a report.

```python
with torch.no_grad():
    p_clean = F.softmax(model(images[:1]), dim=1)[0]
    p_adv = F.softmax(model(x_adv[:1]), dim=1)[0]
for name, p in [("clean", p_clean), ("adv", p_adv)]:
    vals, idx = p.topk(3)
    print(name, [(int(c), round(float(v), 4)) for c, v in zip(idx, vals)])
```

Expected: two lines like `clean [(7, 1.0), ...]` and `adv [(3, 0.51...), (7, 0.43...), ...]`. Your exact numbers may differ.

## Quick self-check

The perturbation panel uses `δ × 10 + 0.5`, then clamps to [0, 1]. What display value does a pixel-space change of **−0.02** get, and what does your real FGSM change of **+0.2465** get?

Reply with both numbers and I'll check them.

## The facts worth locking in for the exam

- **Images must be denormalized back to [0, 1] pixel space (x_norm·σ + μ) before display**, and the visual perturbation should be computed as the difference of the denormalized images. Otherwise normalized values like −0.424 or 2.821 make the digit unrecognizable.
- ==`imshow(..., vmin=0, vmax=1)` fixes the brightness scale so clean and adversarial images are comparable==. Without it, matplotlib auto-stretches each image's own range and can exaggerate or hide the perturbation.
- **The perturbation panel maps signed changes with δ×10 + 0.5, clamped to [0, 1], so 0.5 means "no change," brighter means increased and darker means decreased**. At ε = 0.8 normalized (±0.2465 pixel) it saturates to pure black and white, so it shows direction, not true size.
- **`visualize_attack` takes the attack as a function argument (`make_adv`), so the same plotting code works for FGSM, I-FGSM and DeepFool**. The attack call sits outside `torch.no_grad()` because it needs gradients, while the predictions sit inside it.
- ==The FGSM wrapper builds target labels with `torch.full_like(yb, target_class)` when `targeted=True`==, so the gradient is computed toward the target class and `step_dir = −1` steps toward it.
- **In HTB's example a 100%-confident "7" became a "3" at only 51.45%, with the true class still at about 43%**, showing a flip can be marginal. The untargeted attack lands on whichever rival class is nearest, not one it chose.
- **A red adversarial title only means the prediction no longer matches the true label**. For targeted attacks, check that `class=` equals `target=` to confirm the target was actually hit.
- **Method point:** **pair every attack figure with its probability bars, and report how far the true-class probability fell, not just the new label**, because a 51%-vs-43% flip is much weaker evidence than a confident misclassification.