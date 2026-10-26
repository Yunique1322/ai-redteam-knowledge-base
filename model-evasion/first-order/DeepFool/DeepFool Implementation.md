# DeepFool Implementation

## The idea in one breath

This is Section 15's algorithm written as one Python function, `deepfool`. In each round it takes the gradient of every rival class's score and of the current class's score, computes the distance to each rival's boundary (gap ÷ steepness), steps toward the closest one, and stops as soon as the prediction changes. It returns the total perturbation, the number of rounds, the original class, the new class and the attacked image. **Overshoot 0.02 is applied to the total perturbation**, which answers the question left open in Section 15.

> **Heads-up:** no mangled formulas in this section. Two naming notes:
> 
> - The code's `w_k` is my **w′_k** from Section 15 (rival gradient minus current gradient), and `f_k` is my **f′_k** (rival score minus current score).
> - The code's `w` (no subscript) is "the w′ of the closest rival found so far". It is not a weight vector.

---

## Part 1: The whole function (merged)

The course split one function into 9 blocks, and none of them runs on its own. **I merged them into one cell. Every line, comment and docstring is unchanged**, only joined in order.

```python
def deepfool(image: torch.Tensor,
             net: nn.Module,
             num_classes: int = 10,
             overshoot: float = 0.02,
             max_iter: int = 50,
             device: str = 'cuda') -> Tuple[torch.Tensor, int, int, int, torch.Tensor]:
    """
    Generate minimal adversarial perturbation using DeepFool algorithm.

    Args:
        image (torch.Tensor): Input image tensor of shape (1, C, H, W)
        net (nn.Module): Target neural network in evaluation mode
        num_classes (int): Number of top-scoring classes to consider (default: 10)
        overshoot (float): Overshoot parameter for boundary crossing (default: 0.02)
        max_iter (int): Maximum iterations before terminating (default: 50)
        device (str): Computation device ('cuda' or 'cpu')

    Returns:
        Tuple containing:
            - r_tot (torch.Tensor): Total accumulated perturbation
            - loop_i (int): Number of iterations performed
            - label (int): Original predicted class
            - k_i (int): Final adversarial class
            - pert_image (torch.Tensor): Final perturbed image
    """
    image = image.to(device)
    net = net.to(device)

    # Original prediction and class ordering (descending score)
    f_image = net(image).data.cpu().numpy().flatten()
    I = f_image.argsort()[::-1]
    label = I[0]

    # Working tensors and accumulators
    input_shape = image.shape
    pert_image = image.clone()
    r_tot = torch.zeros(input_shape).to(device)
    loop_i = 0

    # Iterate until a successful perturbation is found or the limit is reached
    while loop_i < max_iter:
        x = pert_image.clone().requires_grad_(True)
        fs = net(x)
        # Current top prediction at x
        k_i = fs.data.cpu().numpy().flatten().argsort()[::-1][0]

        # Stop when the prediction changes
        if k_i != label:
            break

        # Initialize the best candidate step for this iteration
        pert = float('inf')
        w = None

        # Search minimal step among candidate classes
        for k in range(1, num_classes):
            if I[k] == label:
                continue

            # Compute gradient for candidate class
            if x.grad is not None:
                x.grad.zero_()
            fs[0, I[k]].backward(retain_graph=True)
            grad_k = x.grad.data.clone()

            # Compute gradient for original class
            if x.grad is not None:
                x.grad.zero_()
            fs[0, label].backward(retain_graph=True)
            grad_label = x.grad.data.clone()

            # Direction and distance under linearization
            w_k = grad_k - grad_label
            f_k = (fs[0, I[k]] - fs[0, label]).data.cpu().numpy()
            pert_k = abs(f_k) / (torch.norm(w_k.flatten()) + 1e-10)

            if pert_k < pert:
                pert = pert_k
                w = w_k

        # Minimal step for the selected direction
        r_i = (pert + 1e-4) * w / (torch.norm(w.flatten()) + 1e-10)
        r_tot = r_tot + r_i

        # Apply with overshoot to ensure crossing
        pert_image = image + (1 + overshoot) * r_tot
        loop_i += 1

    return r_tot, loop_i, label, k_i, pert_image
```

> **If this errors:**
> 
> - `NameError: name 'Tuple' is not defined` means the type label needs `from typing import Tuple`. Also needed: `import numpy as np`, `from torch import nn`. These come from the FGSM-part imports.
> - `AssertionError: Torch not compiled with CUDA enabled` means you're on CPU but `device` defaults to `'cuda'`. Always pass `device=device`.
> - `IndexError` or wrong results usually mean you passed a batch of many images. This function handles **one image at a time** with shape (1, 1, 28, 28). Use `imgs[0:1]`, not `imgs[0]` (which drops the batch dimension).

### Shapes

|Name|Shape|Meaning|
|---|---|---|
|`image`|(1, 1, 28, 28)|one normalized MNIST image, with a batch of 1|
|`f_image`|(10,)|10 starting logits as a NumPy array|
|`I`|(10,)|class numbers sorted from highest to lowest score|
|`label`|single number|the original predicted class|
|`fs`|(1, 10)|logits at the current perturbed point|
|`grad_k`, `grad_label`, `w_k`, `w`|(1, 1, 28, 28)|one slope per pixel|
|`f_k`|single number|rival score minus current score (negative)|
|`pert_k`, `pert`|single number|distance to a boundary|
|`r_i`, `r_tot`, `pert_image`|(1, 1, 28, 28)|this round's step, the running total, the attacked image|

---

## Part 2: Setup, before the loop

### Ranking the classes

|Code|What it does|
|---|---|
|`image = image.to(device)` / `net = net.to(device)`|Puts the image and the model on the same device.|
|`net(image)`|10 logits for the original image.|
|`.data.cpu().numpy().flatten()`|Detach from gradient tracking → move to CPU → NumPy → a flat list of 10.|
|`f_image.argsort()`|Class numbers ordered from **lowest** to highest score.|
|`[::-1]`|Reverse it, so highest comes first.|
|`label = I[0]`|The top class = the model's original prediction.|

**Toy (the course's example):** logits `[2.1, 5.3, 0.8, 3.7]` for classes 0-3.

- `argsort()` → `[2, 0, 3, 1]` (class 2 is lowest at 0.8, ..., class 1 is highest at 5.3)
- `[::-1]` → **`[1, 3, 0, 2]`**
- `label = 1`

`argsort` returns **class numbers, not scores**. That's the most common thing to misread here.

### Working variables

|Code|What it does|
|---|---|
|`input_shape = image.shape`|Remembers (1, 1, 28, 28).|
|`pert_image = image.clone()`|The image being attacked, starting as an exact copy.|
|`r_tot = torch.zeros(input_shape).to(device)`|The running total of all steps, starting at zero change.|
|`loop_i = 0`|Round counter.|

---

## Part 3: One round of the loop

### Re-check the prediction

|Code|What it does|
|---|---|
|`while loop_i < max_iter:`|Keep going for at most 50 rounds.|
|`x = pert_image.clone().requires_grad_(True)`|A fresh copy of the current image with gradient tracking on, so we can get slopes for its pixels.|
|`fs = net(x)`|Logits at the current point. This is the **re-linearization** from Section 14: fresh scores, fresh slopes.|
|`k_i = ...argsort()[::-1][0]`|Today's top class.|
|`if k_i != label: break`|**Stop the moment the prediction changes.** This is DeepFool's "minimal" stopping rule.|
|`pert = float('inf')`|"Closest distance so far" starts at infinity, so the first real distance always wins.|
|`w = None`|"Direction to the closest boundary" is empty for now.|

### Loop over rivals

|Code|What it does|
|---|---|
|`for k in range(1, num_classes):`|Positions 1 to 9 in the ranking `I` (position 0 is the current class itself).|
|`if I[k] == label: continue`|Skip if this "rival" is the current class.|
|`x.grad.zero_()`|Wipe old slopes. PyTorch **adds** new gradients onto old ones unless you clear them.|
|`fs[0, I[k]].backward(retain_graph=True)`|Slopes of **this rival's** score with respect to every pixel. `[0, I[k]]` = image 0 in the batch, class `I[k]`.|
|`grad_k = x.grad.data.clone()`|Save a copy before the next wipe erases it.|
|`fs[0, label].backward(retain_graph=True)`|Slopes of the **current class's** score.|
|`grad_label = x.grad.data.clone()`|Save it.|
|`retain_graph=True`|Keep PyTorch's record of the forward pass, because we call `.backward()` many times on the same `fs`. Without it, the second call errors.|

Compare with the FGSM part: there, **one** backward pass on the loss gave one gradient. Here we need a **separate gradient for each class score**, because each rival has its own boundary.

### Direction, gap, distance

|Code|Section 15 name|Plain English|
|---|---|---|
|`w_k = grad_k - grad_label`|w′_k|the direction that raises the rival and lowers the current class: the "tug-of-war" net pull|
|`f_k = (fs[0, I[k]] - fs[0, label])...`|f′_k|rival minus current score, negative because the current class is ahead|
|`pert_k = abs(f_k) / (torch.norm(w_k.flatten()) + 1e-10)`|\|f′_k\| ÷ ‖w′_k‖₂|distance to this rival's boundary|
|`+ 1e-10`| - |a tiny number so you never divide by zero|
|`if pert_k < pert: pert = pert_k; w = w_k`|argmin|remember the closest rival so far|

**The course's numbers:** gap 5.3 with steepness 2.0 gives distance 2.65. With steepness 4.0 it gives 1.325 (twice as steep, half as far). With steepness 1.0 it gives 5.3. This is the same "height ÷ steepness" rule as Section 14.

### Take the step

|Code|What it does|
|---|---|
|`w / (torch.norm(w.flatten()) + 1e-10)`|Turns `w` into a **direction of length 1** (a unit vector).|
|`(pert + 1e-4) * ...`|Walks a distance of `pert` along that direction, plus a tiny 0.0001 nudge.|
|`r_tot = r_tot + r_i`|Adds this round's step to the running total.|
|`pert_image = image + (1 + overshoot) * r_tot`|Builds the attacked image from the **original** image plus 1.02 × the total.|
|`loop_i += 1`|Count the round.|
|`return r_tot, loop_i, label, k_i, pert_image`|The 5 outputs.|

**Is this the Section 15 formula?** Yes. distance × unit direction = (|f′| ÷ ‖w′‖) × (w′ ÷ ‖w′‖) = |f′| ÷ ‖w′‖² × w′. The two ‖w′‖ multiply into the square.

**Overshoot, settled:** the code applies 1.02 to the **total** `r_tot`, starting from the original image every time. That's the original paper's convention, not "each step × 1.02" as the Section 15 text said.

---

## Part 4: Running the code on the Section 15 toy

Pretend our 3-class, 2-pixel toy is the network: x = [1, 1], A [3, 1] → 4, B [1, 2] → 3, C [−4, 6] → 2.

|Code step|Value|
|---|---|
|`f_image`|[4, 3, 2]|
|`I`|[0, 1, 2] (A, B, C)|
|`label`|0 (A)|
|Round 0: `k_i`|0, same as label, so continue|
|k = 1 (B): `w_k`, `f_k`, `pert_k`|[−2, 1], −1, **0.447**, so pert = 0.447|
|k = 2 (C): `w_k`, `f_k`, `pert_k`|[−7, 5], −2, **0.233** < 0.447, so pert = 0.233 and w = [−7, 5]|
|unit direction `w / ‖w‖`|[−0.814, 0.581]|
|`r_i` = (0.233 + 0.0001) × that|[−0.189, 0.135]|
|`pert_image` = x + 1.02 × r_tot|[0.807, 1.138] → scores A 3.559, B 3.083, **C 3.600**|
|Round 1: `k_i`|2 (C) ≠ label, so **break**|
|Returns|r_tot [−0.189, 0.135], loop_i **1**, label 0, k_i 2, pert_image [0.807, 1.138]|

A linear model flips in **one round** because its boundary is flat. A real CNN usually needs 2-5 rounds because its boundaries curve.

---

## Part 5: `num_classes`, the speed knob

The loop only checks the top `num_classes` classes from the initial ranking. For MNIST (10 classes) the default checks everything. For ImageNet (1,000 classes), checking only the top 10 saves about 99% of the gradient work, and the closest boundary is almost always one of the runners-up anyway.

---

## Honest notes the course skips

> **No clipping to a valid image.** Unlike your FGSM code, `deepfool` never clamps `pert_image` to the valid range (−0.424 to 2.821 in model space). Pixels can drift slightly past black or white. For a measurement that's fine. For a real submission (like Section 13), you'd clamp, then re-check that the class still flipped.

> **No `net.eval()` inside.** The docstring says the net must be "in evaluation mode", but the function doesn't enforce it. Since Section 16's training branch never calls `model.eval()`, call it yourself, or dropout will make every round's gradients random.

> **The skip check never fires.** `I` is the original ranking and `label = I[0]`. For k ≥ 1, `I[k]` can never equal `I[0]`, so `if I[k] == label: continue` is harmless dead code. Also, `I` is **never re-sorted** during the loop. Rivals are always taken from the starting ranking, which is fine for MNIST because it checks all 9.

> **Wasted work.** `grad_label` is the same for every rival, but it's recomputed inside the loop, 9 times per round. That's 18 backward passes instead of 10. It's correct, just slower.

> **`k_i` can be stale.** If the loop stops because it hit `max_iter` (50), the returned `k_i` comes from the image **before** the last step. Always re-check the prediction on `pert_image` yourself (Cell 2 below).

> **Units.** The images are normalized, so `r_tot`'s size is in **model space**. Multiply by 0.3081 for pixel space (the Section 5 rule).

---

## Copy-paste cells (mine, not the course's)

These use the names from Section 16: `model`, `test_loader`, `device`.

### Cell 1: Attack one test image

```python
model.eval() # dropout OFF (the function doesn't do this)
imgs, labs = next(iter(test_loader))
img = imgs[0:1].to(device) # [0:1] keeps the batch dim -> (1, 1, 28, 28)
r_tot, n_iter, orig, adv, pert_image = deepfool(img, model, device=device)
print("true:", labs[0].item(), "| original pred:", orig, "| adversarial pred:", adv, "| rounds:", n_iter)
```

**Expected:** the original prediction equals the true digit, the adversarial prediction is a different digit, and rounds is usually a small number (about 2-5).

### Cell 2: Re-check and measure the perturbation

```python
with torch.no_grad():
    final_pred = model(pert_image).argmax(dim=1).item() # don't trust k_i blindly
delta = pert_image - img # what was actually added (incl. overshoot)
print("re-checked prediction:", final_pred)
print("L2 model space:", round(delta.norm().item(), 4),
      "| pixel space:", round(delta.norm().item() * 0.3081, 4))
print("Linf in /255 levels:", round(delta.abs().max().item() * 0.3081 * 255))
print("relative size (rho for this image):", round((delta.norm() / img.norm()).item(), 4))
print("pixel range:", round(pert_image.min().item(), 3), "to", round(pert_image.max().item(), 3),
      "| valid: -0.424 to 2.821")
```

**Expected:** `re-checked prediction` matches `adv` from Cell 1. The L2 is small, and the relative size is a few percent. The range may slightly exceed the valid limits, which shows there's no clipping.

### Cell 3: Clip to a valid image and check it still fools

```python
pert_clipped = pert_image.clamp(-0.424, 2.821) # keep pixels between black and white
with torch.no_grad():
    print("after clipping:", model(pert_clipped).argmax(dim=1).item(), "| original:", orig)
```

**Expected:** usually still the adversarial digit. If it flips back to the original, the overshoot was too small to survive clipping. Re-run Cell 1 with `overshoot=0.05`.

---

## How important is this for the exam?

High. This is the code you'd actually run. Know how to call it (one image of shape (1, C, H, W), `model.eval()` first, `device=device`), what the 5 outputs are, and the two practical gaps: no clipping, and re-check the final prediction yourself.

## Quick self-check

The model's starting logits are **[1.0, 6.0, 4.5, 2.0]** for classes 0-3. In the first round, the gradient-difference lengths ‖w_k‖ are: class 2 → **1.0**, class 3 → **4.0**, class 0 → **2.0**. Which class does DeepFool step toward in this round?

---

## The facts worth locking in for the exam

- ==`deepfool(image, net, num_classes=10, overshoot=0.02, max_iter=50, device)` attacks ONE image of shape (1, C, H, W)== and returns 5 things: the total perturbation `r_tot`, the number of rounds `loop_i`, the original class `label`, the final class `k_i`, and `pert_image`.
- `I = f_image.argsort()[::-1]` gives **class numbers sorted from highest to lowest score, not the scores themselves**, and `label = I[0]` is the original prediction. Logits [2.1, 5.3, 0.8, 3.7] give I = [1, 3, 0, 2].
- **Each round needs a separate backward pass per class score**: `fs[0, I[k]].backward(retain_graph=True)` for each rival and one for the current class. `retain_graph=True` keeps the graph alive for repeated backward calls, and `x.grad.zero_()` stops gradients from piling up.
- ==Distance per rival = |f_k| ÷ ‖w_k‖₂, with w_k = grad_k − grad_label==. The smallest distance wins, and the step is that distance along the unit direction of w, which equals |f′| ÷ ‖w′‖² × w′.
- ==The loop stops the moment the prediction changes (`k_i != label`)==. That's what makes DeepFool minimal rather than budget-driven, and a linear model flips in one round.
- ==Overshoot is applied to the total: `pert_image = image + (1 + overshoot) * r_tot`==, always rebuilt from the original image, which settles the per-step vs total question.
- The course's code **does not clip to the valid pixel range, does not call `net.eval()`, and can return a stale `k_i` if it hits `max_iter`**. Call `model.eval()` first, re-check the prediction on `pert_image`, and clamp if you need a valid image.
- **Method point:** **after any attack, re-verify independently**. Re-run the model on the returned image, measure the perturbation in the stated norm and space (model-space L2 × 0.3081 = pixel space), and confirm the image is still valid, instead of trusting the attack's own return values.

