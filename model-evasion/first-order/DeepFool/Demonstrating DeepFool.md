# Demonstrating DeepFool

## The idea in one breath

This section runs `deepfool` end to end on one test image, the first MNIST test digit, a **7**. It flips the 7 to a **2 in 4 iterations** with a model-space L2 of **7.72**, an L∞ of **1.58** and a relative size of **32.48%**, while the confidence drops from 1.000 to 0.404. It then draws a 4-panel figure showing that DeepFool **concentrates** its change on the digit's strokes instead of spreading it evenly like FGSM.

> **Heads-up: one formula the course gets wrong in the prose**
> 
> |Course says|Correct reading|
> |---|---|
> |"uniform baseline L∞ ≈ L2/784 ≈ 7.72/28 ≈ 0.276"|L∞ ≈ L2 ÷ **√784** = L2 ÷ **28** = 0.276. Dividing by 784 would give 0.0098. The course wrote 784 but actually divided by 28, which is the right number.|
> 
> **Symbol note:** the course calls the image's own size ‖x‖₂ and the change's size ‖r‖₂, the same as ρ_adv in Section 15.

---

## Part 1: Load the model and one test image

```python
# Load trained model
model_path = 'output/mnist_model.pth'
if os.path.exists(model_path):
    model_data = load_model(model_path)
    model = model_data['model'].to(device)
    model.eval()
else:
    raise FileNotFoundError("Model not found.")

# Get single test sample
_, test_loader = get_mnist_loaders(batch_size=1, normalize=True)
dataiter = iter(test_loader)
image, true_label = next(dataiter)
image = image.to(device)

print(f"True label: {true_label.item()}")
```

|Code|What it does|
|---|---|
|`if os.path.exists(model_path):`|Uses the cached model from Section 16.|
|`model_data = load_model(model_path)`|Reads the saved dictionary.|
|`model = model_data['model'].to(device)`|Takes the model out and puts it on the device.|
|`model.eval()`|**Dropout off**, so gradients are stable from one iteration to the next.|
|`raise FileNotFoundError(...)`|Stops with a clear error if you skipped Section 16's training.|
|`get_mnist_loaders(batch_size=1, ...)`|Test data, **one image per batch**, which is exactly the (1, 1, 28, 28) shape `deepfool` needs.|
|`iter(test_loader)` / `next(dataiter)`|Takes the first batch. The test set isn't shuffled, so it's always the same image (a 7).|
|`image.to(device)`|Puts the image on the same device as the model.|
|`true_label.item()`|Turns the one-number tensor into a plain number for printing.|

**Expected:** `True label: 7`

> **If this errors:** `FileNotFoundError: Model not found.` means Section 16's caching cell never saved the model. Re-run Section 16's train-and-save cell.

---

## Part 2: Baseline, then attack

```python
# Baseline classification
with torch.no_grad():
    original_output = model(image)
    original_pred = original_output.argmax(dim=1).item()
    original_confidence = F.softmax(original_output, dim=1).max().item()

print(f"Original: class {original_pred} (confidence: {original_confidence:.3f})")

# Execute DeepFool attack
r_total, iterations, orig_label, pert_label, pert_image = deepfool(
    image, model, num_classes=10, overshoot=0.02, max_iter=50, device=device
)

print(f"Attack: {orig_label} → {pert_label} in {iterations} iterations")
```

|Code|What it does|
|---|---|
|`with torch.no_grad():`|Just reading the prediction, so no gradients are needed. This is fine **here** because it isn't the attack.|
|`original_output = model(image)`|10 logits.|
|`.argmax(dim=1).item()`|The highest-scoring class.|
|`F.softmax(..., dim=1).max().item()`|Turns logits into probabilities and takes the biggest one = the **confidence** in the predicted class.|
|`deepfool(image, model, num_classes=10, overshoot=0.02, max_iter=50, device=device)`|The Section 17 function, called with its default settings spelled out.|
|`r_total, iterations, orig_label, pert_label, pert_image`|The 5 returns, renamed (Section 17 called them r_tot, loop_i, label, k_i, pert_image).|

**Expected (from the course's numbers):**

![Terminal output showing the original image classified as class 7 with confidence 1.000, then the attack flipping it from 7 to 3 in 2 iterations](/assets/18%20Demonstrating%20DeepFool-20261017170622796.png)

The course's guide: well-separated digits flip in 1-3 iterations, and ambiguous ones need 4-6. This 7 took 4, so it sits further from its nearest boundary than a typical digit.

> **An honest note:** the course says high confidence predicts many iterations. That's only a loose trend. As Section 14 warned, deep networks can be 100% confident and still close to a boundary. Confidence is not distance.

---

## Part 3: Measuring the perturbation

### Step 1: The code

```python
# Compute perturbation norms
perturbation_norm_l2 = torch.norm(r_total).item()
perturbation_norm_linf = torch.abs(r_total).max().item()
relative_perturbation = perturbation_norm_l2 / torch.norm(image).item()

# Evaluate adversarial confidence
with torch.no_grad():
    adv_output = model(pert_image)
    adv_confidence = F.softmax(adv_output, dim=1).max().item()

# Display results
print(f"\n=== Attack Results ===")
print(f"L2 norm: {perturbation_norm_l2:.4f}")
print(f"L∞ norm: {perturbation_norm_linf:.4f}")
print(f"Relative perturbation: {relative_perturbation:.2%}")
print(f"Original confidence: {original_confidence:.3f}")
print(f"Adversarial confidence: {adv_confidence:.3f}")
```

|Code|What it does|
|---|---|
|`torch.norm(r_total)`|**L2**: square every pixel change, add them up, take the square root. The overall size, which is what DeepFool minimizes.|
|`torch.abs(r_total).max()`|**L∞**: the single biggest pixel change.|
|`perturbation_norm_l2 / torch.norm(image)`|**Relative size**: change ÷ image, the per-image piece of ρ_adv from Section 15.|
|`model(pert_image)` + softmax max|Confidence in whatever class the attacked image is now predicted as.|
|`:.2%`|Prints 0.3248 as `32.48%`.|

**Expected (from the course):**

![Demonstrating DeepFool](/assets/18%20Demonstrating%20DeepFool-20261017170642585.png)

### Step 2: Concentrated or spread out? (toy first)

Take 4 pixels and two different changes with the **same L2**:

|Change|L2|L∞|Shape|
|---|---|---|---|
|Spread evenly: [0.5, 0.5, 0.5, 0.5]|√(4 × 0.25) = **1.0**|**0.5**|uniform|
|All on one pixel: [1.0, 0, 0, 0]|√1 = **1.0**|**1.0**|concentrated|

With the same L2, the concentrated change has twice the L∞. So:

- if a change is perfectly even over n pixels, L∞ = L2 ÷ √n. With 4 pixels, √4 = 2, so 1.0 ÷ 2 = 0.5.
- **the bigger the real L∞ compared with L2 ÷ √n, the more concentrated the change**

### Step 3: The formula

**Uniform baseline:** L∞_uniform = ‖r‖₂ ÷ √n **Concentration:** L∞_real ÷ L∞_uniform

|Symbol|Plain English|Toy|Real (HTB)|
|---|---|---|---|
|‖r‖₂|overall size of the change|1.0|7.72|
|n|number of pixels|4|784|
|√n|square root of the pixel count|2|28|
|L∞_uniform|biggest change if spread perfectly evenly|0.5|7.72 ÷ 28 = **0.276**|
|L∞_real|the actual biggest change|1.0|**1.58**|
|ratio|how concentrated it is|2×|1.58 ÷ 0.276 ≈ **5.7×**|

**Read aloud:** "Divide the L2 by the square root of the pixel count to get what each pixel would change if it were spread evenly. Compare the real biggest change with that."

On the real digit, the biggest change is about **5.7 times** the even-spread value. DeepFool piles its change onto a few pixels that matter (the strokes) instead of spreading it everywhere, which FGSM does by design.

### Step 4: Relative perturbation

The course's toy: a bright image with ‖x‖ = 20 and a dark one with ‖x‖ = 10, each with a 15% relative change, means changes of 3.0 and 1.5. The absolute sizes are different, but proportionally the attacks are the same. The 7 needed **32.48%**, which is high, because turning a 7 into a 2 needs a real structural change.

### Step 5: What the confidence numbers mean

1.000 → 0.404 means the model went from completely sure it's a 7 to 40.4% sure it's a **2**. The `.max()` picks the top class, which is now 2.

> **An honest note the course gets wrong:** it says 0.404 is "below the 0.5 decision threshold". There is **no 0.5 threshold with 10 classes**. The prediction is just the argmax. 0.404 is the probability of class 2, and the 7 is probably close behind (Cell 2 below shows it). The real lesson is still right: a winning probability barely above the runner-up means the image **just crossed** the boundary, which is what a minimal attack should do. Compare FGSM's weak 51% flip vs I-FGSM's 99.7% in the FGSM part.

> **An honest note (2):** the norms use `r_total`, but the image actually received 1.02 × `r_total` (Section 17: overshoot is applied to the total). So the true change is 2% bigger: L2 ≈ 7.88, not 7.72.

> **An honest note (3), and the big one:** these are **model-space** numbers. Convert with × 0.3081 (Section 5): L2 7.72 → **2.38** in pixel space, and L∞ 1.58 → **0.49** in pixel space, about **124/255**. That's a change of half the brightness range on some pixels, which is **clearly visible** (look at the grey speckle in panel 4). On MNIST, "minimal in L2" does **not** mean invisible.

### Step 6: DeepFool vs FGSM on the two rulers

Using the FGSM part's ε = 0.8 model space:

|Ruler (model space)|FGSM, ε = 0.8|DeepFool, this 7|
|---|---|---|
|L∞ (biggest pixel)|**0.8**|1.58|
|L2 (overall size)|up to 0.8 × 28 = 22.4|**7.72**|

This is the Section 14 ruler lesson on real data. DeepFool's overall change is about 3 times smaller, but its biggest single pixel change is about twice as big.

---

## Part 4: The 4-panel visualization

```python
# Prepare images for visualization
original_img = mnist_denormalize(image.squeeze()).cpu().numpy()
adversarial_img = mnist_denormalize(pert_image.squeeze()).cpu().numpy()
perturbation = r_total.cpu().squeeze().numpy()

# Normalize perturbation for visibility (amplify minimal changes)
pert_display = perturbation - perturbation.min()
if pert_display.max() > 0:
    pert_display = pert_display / pert_display.max()

# Create four-panel visualization
fig, axes = plt.subplots(1, 4, figsize=(15, 5))
fig.patch.set_facecolor(NODE_BLACK)

for ax in axes:
    ax.set_facecolor(NODE_BLACK)
    for spine in ax.spines.values():
        spine.set_edgecolor(HACKER_GREY)

# Panel 1: Original clean image
axes[0].imshow(original_img, cmap='gray', vmin=0, vmax=1)
axes[0].set_title(f"Original\nClass: {original_pred}",
                  color=HTB_GREEN, fontweight='bold')
axes[0].axis('off')

# Panel 2: Amplified perturbation pattern
axes[1].imshow(pert_display, cmap='inferno')
axes[1].set_title("Perturbation\n(amplified)",
                  color=NUGGET_YELLOW, fontweight='bold')
axes[1].axis('off')

# Panel 3: Perturbation magnitude heatmap
im = axes[2].imshow(np.abs(perturbation), cmap='viridis')
axes[2].set_title(f"Magnitude\nL2: {perturbation_norm_l2:.4f}",
                  color=AZURE, fontweight='bold')
axes[2].axis('off')
plt.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)

# Panel 4: Adversarial result
title_color = HTB_GREEN if pert_label != original_pred else MALWARE_RED
axes[3].imshow(adversarial_img, cmap='gray', vmin=0, vmax=1)
axes[3].set_title(f"Adversarial\nClass: {pert_label}",
                  color=title_color, fontweight='bold')
axes[3].axis('off')

# Summary metrics
metrics_text = (
    f"Iterations: {iterations} | "
    f"Relative pert: {relative_perturbation:.2%} | "
    f"Confidence: {original_confidence:.3f} → {adv_confidence:.3f}"
)
fig.text(0.5, 0.02, metrics_text, ha='center', fontsize=10, color=WHITE)

plt.suptitle("DeepFool Attack Visualization", fontsize=14,
             color=HTB_GREEN, fontweight='bold', y=1.02)
plt.tight_layout()
plt.show()
```

|Code|What it does|
|---|---|
|`mnist_denormalize(image.squeeze())`|Model space → normal 0-1 pixels for display. `.squeeze()` drops the size-1 dimensions: (1,1,28,28) → (28,28).|
|`.cpu().numpy()`|Matplotlib needs NumPy arrays on the CPU.|
|`perturbation = r_total...`|The raw change, still in model space, with signs (+ brightened, − darkened).|
|`pert_display = perturbation - perturbation.min()`|Shift so the most negative value becomes 0.|
|`pert_display / pert_display.max()`|Stretch so the most positive becomes 1 (**min-max scaling**, purely for display).|
|`plt.subplots(1, 4, ...)`|One row of 4 panels.|
|`NODE_BLACK`, `HACKER_GREY`|HTB dark theme.|
|Panel 1 `cmap='gray', vmin=0, vmax=1`|Original in greyscale, with fixed black = 0 and white = 1.|
|Panel 2 `cmap='inferno'`|The scaled, signed change: dark → red → yellow.|
|Panel 3 `np.abs(perturbation)`, `cmap='viridis'`|Size of the change at each pixel, ignoring direction, with a colour bar.|
|`plt.colorbar(...)`|The scale bar next to panel 3.|
|`title_color = HTB_GREEN if pert_label != original_pred else MALWARE_RED`|Green title when the attack worked. Note: this is the reverse of the FGSM plots, where red meant "fooled".|
|`fig.text(...)`|The summary line at the bottom.|

> **If this errors:** `NameError: name 'mnist_denormalize' is not defined`. Section 16's import list leaves it out. Add `from htb_ai_library import mnist_denormalize`. Same for `F`, `plt` and `np` if you started a fresh notebook: `import torch.nn.functional as F`, `import matplotlib.pyplot as plt`, `import numpy as np`.

---

## Reading the figure

![Reading the figure](/assets/Pasted%20image%2020261016194416.png)

|Panel|What you see|What it means|
|---|---|---|
|Original, Class: 7|clean white 7 on black|confidence 1.000|
|Perturbation (amplified)|**red-orange background**, yellow and orange blobs along the top bar and the middle, a few near-black spots, and a yellow and black cluster at the bottom|after min-max scaling, **zero change sits at the middle colour (red)**. Yellow = brightened most, black or dark purple = darkened most.|
|Magnitude, L2: 7.7214|dark purple background; teal, green and yellow pixels shaped roughly like a 2 or 7|brightness = how much each pixel changed. The colour bar tops out around **1.5**, which matches L∞ ≈ 1.58.|
|Adversarial, Class: 2|the 7 is still readable, with grey speckle around it and a smudge where a 2's bottom curve would go|the change is **visible** on MNIST, and it adds 2-like strokes at the top-left and bottom|
|Bottom line|Iterations: 4, Relative pert: 32.48%, Confidence 1.000 → 0.404|the course's headline numbers|

**Key values read off the figure:**

|Value|Where|Meaning|
|---|---|---|
|L2 7.7214|panel 3 title|overall change, model space (2.38 pixel space)|
|colour bar max ≈ 1.5|panel 3|biggest single-pixel change ≈ L∞ 1.58|
|4 iterations|bottom|an "ambiguous" digit, by the course's 4-6 rule|
|32.48%|bottom|the change is about a third of the image's own size|
|0.404|bottom|the new class (2) wins narrowly|

> **An honest note the course gets wrong:** it says "dark purple dominates background regions, showing minimal modification". Because of min-max scaling, **the background in panel 2 is the middle colour (red-orange), not dark purple**. Dark means the **most negative** change, i.e. pixels darkened the most. Panel 3 is the one where dark purple means "barely changed". Also, FGSM's version of panel 2 wouldn't be "uniform yellow". Every FGSM pixel moves exactly +ε or −ε, so it would be two-tone (all yellow or all black).

What the course gets right: the change sits on the digit's strokes, not the background, with more brightening than darkening. Turning a 7 into a 2 mostly means **adding** the 2's curve.

---

## Copy-paste cells (mine, not the course's)

These use this section's names: `model`, `image`, `pert_image`, `r_total`, `perturbation_norm_l2`.

### Cell 1: The real size, with overshoot, in both spaces

```python
delta = pert_image - image # what was really added (includes the 2% overshoot)
print("L2 reported (r_total):", round(perturbation_norm_l2, 4), "| real (delta):", round(delta.norm().item(), 4))
print("L2 pixel space:", round(delta.norm().item() * 0.3081, 4))
linf_px = delta.abs().max().item() * 0.3081
print("Linf pixel space:", round(linf_px, 4), "=", round(linf_px * 255), "/255")
uniform = delta.norm().item() / 28 # even-spread baseline: L2 / sqrt(784)
print("uniform baseline:", round(uniform, 4), "| concentration x", round(delta.abs().max().item() / uniform, 1))
```

**Expected:** real L2 ≈ 7.88 (2% above 7.72), L2 pixel ≈ 2.43, L∞ pixel ≈ 0.50 ≈ 127/255, concentration ≈ 5.7×.

### Cell 2: See who's actually winning after the attack

```python
with torch.no_grad():
    p_clean = F.softmax(model(image), dim=1).squeeze()
    p_adv = F.softmax(model(pert_image), dim=1).squeeze()
top = p_adv.topk(3) # the 3 most likely classes after the attack
for prob, cls in zip(top.values, top.indices):
    print(f"class {cls.item()}: clean {p_clean[cls].item():.3f} -> adversarial {prob.item():.3f}")
```

**Expected:** class 2 first at about 0.404, with class 7 close behind. That shows a narrow crossing, not a "below 0.5" story.

### Cell 3: Is the attacked image still valid?

```python
print("pixel range:", round(pert_image.min().item(), 3), "to", round(pert_image.max().item(), 3),
      "| valid: -0.424 to 2.821")
with torch.no_grad():
    still = model(pert_image.clamp(-0.424, 2.821)).argmax(dim=1).item()
print("after clipping, prediction:", still)
```

**Expected:** the range likely goes a bit past the valid limits, because `deepfool` doesn't clip (Section 17). After clipping, the prediction is usually still 2.

---

## How important is this for the exam?

High for the measuring part. You'd compute L2, L∞ and relative size, convert to pixel space, and report them. Reading the confidence correctly (it's the **new** class's probability, and there's no 0.5 threshold) is the kind of detail a report reviewer catches.

## Quick self-check

A DeepFool run on a 784-pixel image gives **L2 = 5.6** and **L∞ = 0.8** (model space). What would L∞ be if the change were spread perfectly evenly, and how many times more concentrated is the real change?

---

## The facts worth locking in for the exam

- On the first MNIST test image, **DeepFool flipped a 7 into a 2 in 4 iterations with a model-space L2 of 7.72, an L∞ of about 1.58, and a relative perturbation of 32.48%**, while confidence went from 1.000 to 0.404.
- **Run the attack on one image of shape (1, 1, 28, 28) with `model.eval()` on**. Use `torch.no_grad()` only for reading predictions, never around the attack itself.
- ==L2 measures the overall change (what DeepFool minimizes), L∞ the single biggest pixel change, and relative perturbation = ‖r‖₂ ÷ ‖x‖₂==, which compares images of different brightness fairly and is one term of ρ_adv.
- ==An evenly spread change would have L∞ = L2 ÷ √n (÷ 28 for MNIST, not ÷ 784)==. The real 1.58 is about 5.7 times that 0.276 baseline, which shows DeepFool concentrates on stroke pixels while FGSM moves every pixel by the same ε.
- **After the attack, the max softmax is the probability of the NEW class**. 0.404 means "2" narrowly wins. With 10 classes there is no 0.5 threshold, and a narrow win is exactly what a minimal attack should produce.
- **The course's norms use `r_total`, which leaves out the 2% overshoot**. The image really received 1.02 × `r_total`, so measure `pert_image − image` when reporting.
- **Model-space numbers must be converted for reporting**: × 0.3081 gives L2 ≈ 2.38 and L∞ ≈ 0.49 (about 124/255) in pixel space. That is clearly visible, so minimal in L2 does not mean imperceptible on MNIST.
- **Method point:** **when reading a min-max scaled perturbation plot, zero change sits at the middle colour, not the darkest one**. Use an absolute-magnitude map to find untouched pixels, and always state the norm, the space and whether overshoot is included when you report a perturbation size.

The progress file is updated: Section 18 is done, your notebook names, the course's mistakes and the self-check answer are saved, and Section 19 is next.