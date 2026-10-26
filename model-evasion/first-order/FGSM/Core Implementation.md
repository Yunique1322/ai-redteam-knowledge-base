# Core Implementation

## The idea in one breath

This is where the formula becomes code. Three small functions do the work. `_forward_and_loss` computes how wrong the model is. `_input_gradient` gets the slope of that wrongness for every pixel. `fgsm_attack` takes the sign, scales it by ε, adds it to the image and clamps. On one batch, ε = 0.8 (normalized, about 0.25 in pixels) flips **71%** of correct predictions. A second version, `fgsm_pixel_space`, does the same attack when your images and ε are in plain [0, 1] pixel units.

## Part 1: The loss helper

```python
def _forward_and_loss(model: nn.Module, x: Tensor, y: Tensor) -> tuple[Tensor, Tensor]:
    """Forward pass and cross-entropy loss without side effects.

    Args:
        model: Neural network classifier
        x: Input images tensor
        y: Target labels tensor

    Returns:
        tuple[Tensor, Tensor]: Model logits and scalar loss value
    """
    if getattr(model, "training", False):
        raise RuntimeError("Expected model.eval() for attack computations to avoid BN/Dropout state updates")
    logits = model(x)
    loss = F.cross_entropy(logits, y)
    return logits, loss
```

### Line by line

|Code|What it does|
|---|---|
|`def _forward_and_loss(...)`|Defines a function. The leading `_` is a Python convention meaning "internal helper."|
|`model: nn.Module, x: Tensor`|**Type hints**: labels saying what kind of thing each input is. Python doesn't enforce them; they're for readers.|
|`-> tuple[Tensor, Tensor]`|"This returns a pair of tensors."|
|`"""..."""`|The **docstring**: built-in documentation. `help(_forward_and_loss)` prints it.|
|`getattr(model, "training", False)`|Reads `model.training` safely (returns `False` if it doesn't exist). It's `True` after `model.train()` and `False` after `model.eval()`.|
|`raise RuntimeError(...)`|Stops with an error if you forgot `model.eval()`. **A built-in guard against the train-mode mistake.**|
|`logits = model(x)`|The forward pass: images in, 10 scores per image out.|
|`F.cross_entropy(logits, y)`|Cross-entropy loss (`−log p_y`) averaged over the batch into **one number**.|

### Shapes

|Name|Shape|Meaning|
|---|---|---|
|`x`|[128, 1, 28, 28]|128 images, 1 channel, 28×28|
|`logits`|[128, 10]|10 class scores for each image|
|`loss`|[] (a single number)|average wrongness for the whole batch|

It returns **both** logits and loss because sometimes you need the predictions, sometimes the loss, sometimes both.

## Part 2: The input gradient

```python
def _input_gradient(model: nn.Module, x: Tensor, y: Tensor) -> Tensor:
    """Return gradient of loss with respect to input tensor x.

    Args:
        model: Neural network in evaluation mode
        x: Input images to compute gradients for
        y: True labels for loss computation

    Returns:
        Tensor: Gradient tensor with same shape as x
    """
    x_req = x.clone().detach().requires_grad_(True)
    _, loss = _forward_and_loss(model, x_req, y)
    model.zero_grad(set_to_none=True)
    loss.backward()
    return x_req.grad.detach()
```

This is the "backprop to inputs" recipe in five lines.

|Code|Plain English|Why|
|---|---|---|
|`x.clone()`|make a copy|so we never change the caller's original images|
|`.detach()`|cut the copy off from any earlier calculations|so gradients start fresh from here|
|`.requires_grad_(True)`|"track slopes for these pixels" (the `_` means it changes the tensor in place)|**without this, `x_req.grad` would be `None`**|
|`_, loss = ...`|run the model and get the loss; `_` throws away the logits we don't need||
|`model.zero_grad(set_to_none=True)`|clear any old **weight** gradients|`backward()` also computes weight gradients, and PyTorch **adds** to old ones; clearing keeps things clean|
|`loss.backward()`|compute every slope, including **∂loss/∂pixel**, and store it in `x_req.grad`||
|`x_req.grad.detach()`|return the gradient as a plain tensor|shape [128, 1, 28, 28], one slope per pixel|

**The weights are never updated here.** There's no `optimizer.step()`. We only read slopes, which is exactly Section 1's "freeze the weights, take the gradient with respect to the input."

## Part 3: The FGSM attack

```python
def fgsm_attack(model: nn.Module,
                images: Tensor,
                labels: Tensor,
                epsilon: float,
                targeted: bool = False) -> Tensor:

    # Valid normalized range for MNIST
    MNIST_NORM_MIN = (0.0 - 0.1307) / 0.3081
    MNIST_NORM_MAX = (1.0 - 0.1307) / 0.3081

    if epsilon < 0:
        raise ValueError("epsilon must be non-negative")
    if not images.is_floating_point():
        raise ValueError("images must be floating point tensors")

    grad = _input_gradient(model, images, labels)
    step_dir = -1.0 if targeted else 1.0
    x_adv = images + step_dir * epsilon * grad.sign()
    x_adv = torch.clamp(x_adv, MNIST_NORM_MIN, MNIST_NORM_MAX)
    return x_adv.detach()
```

### Toy first: one pixel through the four operations

Take one normalized pixel at **0.5**, ε = 0.8, untargeted, and say its gradient is **−3.7**.

```
1. grad.sign() sign(−3.7) = −1
2. × epsilon × step_dir −1 × 0.8 × (+1.0) = −0.8
3. add to image 0.5 + (−0.8) = −0.3
4. clamp to [−0.424, 2.821] = −0.3 (already inside, unchanged)
```

Now a black background pixel at **−0.424** with gradient **−0.2**:

```
sign = −1 → step = −0.8 → −0.424 − 0.8 = −1.224 → clamp → −0.424
```

**The clamp stops a pixel going "darker than black."** Without it you'd feed the model impossible images.

### Line by line

|Code|What it does|
|---|---|
|`MNIST_NORM_MIN / MAX`|the normalized versions of black (−0.424) and white (2.821) from Section 5|
|`if epsilon < 0: raise ...`|rejects a negative budget|
|`is_floating_point()`|rejects integer images (0-255 whole numbers); gradients need decimal numbers|
|`grad = _input_gradient(...)`|one slope per pixel|
|`step_dir = -1.0 if targeted else 1.0`|==**+1** = untargeted (push loss _up_ on the true label); **−1** = targeted (push loss _down_ on the target label)==|
|`grad.sign()`|every slope becomes +1, −1 (or 0 if exactly zero, which is rare)|
|`epsilon * ...`|the scalar ε is **broadcast** (copied) across all 128×784 values, so every pixel's change is exactly ±ε|
|`torch.clamp(x_adv, MIN, MAX)`|keeps every pixel inside the valid normalized range|
|`.detach()`|returns a plain tensor with no gradient tracking|

**Two cautions for labs:**

- ==With `targeted=True`, the `labels` you pass must be the **target** labels, not the true ones==, because the gradient is computed on whatever labels you give it.
- The clamp bounds are **hardcoded for MNIST**. On another dataset, change them to that dataset's normalized min and max.

## Part 4: Testing the attack

```python
images, labels = next(iter(test_loader))
images, labels = images.to(device), labels.to(device)

model.eval()
# Epsilon in normalized space (≈0.25 in pixel space)
epsilon = 0.8
with torch.no_grad():
    clean_pred = model(images).argmax(dim=1)

x_adv = fgsm_attack(model, images, labels, epsilon)
with torch.no_grad():
    adv_pred = model(x_adv).argmax(dim=1)

originally_correct = (clean_pred == labels)
flipped = (adv_pred != labels) & originally_correct
success = flipped.sum().item() / max(int(originally_correct.sum().item()), 1)
print(f"FGSM flips (first batch): {success:.2%}")
```

Output:

![Core Implementation](/assets/6%20Core%20Implementation-20261016163456553.png)

### Notice where `no_grad()` is, and isn't

`torch.no_grad()` wraps only the two **prediction** steps. The `fgsm_attack(...)` call sits **outside** it, because the attack needs gradients. **That placement is exactly the eval-yes / no_grad-no rule from Section 4.**

### Toy first: the success calculation on 5 images

```
labels = [7, 2, 1, 0, 4]
clean_pred = [7, 2, 1, 0, 9] ← the model already got the last one wrong
adv_pred = [3, 2, 8, 6, 4]

originally_correct = clean_pred == labels = [T, T, T, T, F] → 4 correct
adv_pred != labels = [T, F, T, T, F]
flipped = (adv_pred != labels) & originally_correct
                                                = [T, F, T, T, F] → 3 flipped

success = 3 / 4 = 75%
```

|Piece|Meaning|
|---|---|
|`==`, `!=`|compare position by position, giving True/False lists|
|`&`|"and", position by position|
|`.sum().item()`|count the Trues as a plain Python number|
|`max(..., 1)`|avoids dividing by zero if nothing was correct|
|`:.2%`|format as a percentage with 2 decimals|

**Only originally-correct images count**, because flipping a prediction that was already wrong proves nothing. (In the toy, image 5 was wrong to begin with, so it's excluded.)

### Reading the real number

71.09% = **91 out of 128**. ε_norm = 0.8 is 0.8 × 0.3081 ≈ **0.25 in pixel space**, about **64/255** brightness levels. One step of invisible-to-moderate noise flips about 7 in 10 correct predictions.

**Name note:** this cell uses `model`. In Section 4 you created `trained_model = train_model(model, ...)`. If `train_model` trains in place they're the same object. The first cookbook cell below checks.

## Part 5: The pixel-space variant

### Why it exists

`fgsm_attack` expects **normalized** images and a **normalized** ε. Some workflows give you plain [0, 1] images and a pixel-space budget like "8/255." This variant accepts those, normalizes internally for the model, and converts the gradient back to pixel units.

### Toy first: why the gradient gets divided by σ

Use the Section 5 toy, σ = 0.4. Normalization is x_norm = (x − μ) / σ, so:

```
change the plain pixel x by 1 → x_norm changes by 1/σ = 1/0.4 = 2.5
```

Say the loss rises by **2** for every unit of x_norm (normalized gradient = 2). Then per unit of plain x:

```
loss change = 2 × 2.5 = 2 / 0.4 = 5 → pixel gradient = normalized gradient ÷ σ
```

That's the **chain rule**: when one quantity feeds into another, you multiply the rates. In formula form:

```
∇x ℒ = ∇x_norm ℒ / σ
```

|Symbol|Plain English|Toy|
|---|---|---|
|`∇x_norm ℒ`|slope of loss per unit of normalized pixel|2|
|`σ`|the normalization spread|0.4|
|`∇x ℒ`|slope of loss per unit of plain pixel|2 / 0.4 = 5|

HTB's example: normalized gradient 2.5 with σ = 0.229 gives 2.5 / 0.229 ≈ **10.9**.

> **An honest note the course skips:** σ is always positive, so dividing by it **never changes the sign**. sign(2) = sign(5) = +1. For sign-based FGSM, the step direction is identical either way. What actually makes this variant "pixel-space" is that ε is added to the **plain [0, 1] image** and the clamp is **[0, 1]**. The ÷σ only changes the result in attacks that use the gradient's **size** (like the L2 version), not FGSM's sign. It's still the mathematically correct gradient, so it's good practice to keep.

### The helper that shapes mean and std

```python
def _norm_params(images: Tensor, mean: list, std: list) -> tuple[Tensor, Tensor]:
    """Convert normalization parameters to broadcastable tensors.

    Args:
        images: Input images tensor with shape (N, C, H, W)
        mean: Normalization mean per channel as list
        std: Normalization std per channel as list

    Returns:
        tuple[Tensor, Tensor]: Mean and std tensors with shape (1, C, 1, 1)
    """
    device, dtype, C = images.device, images.dtype, images.shape[1]
    mean_t = torch.tensor(mean, device=device, dtype=dtype).view(1, -1, 1, 1)
    std_t = torch.tensor(std, device=device, dtype=dtype).view(1, -1, 1, 1)
    if mean_t.shape[1] != C or std_t.shape[1] != C:
        raise ValueError("mean/std channels must match images")
    return mean_t, std_t
```

|Code|What it does|
|---|---|
|`images.device, images.dtype`|match the images' device (CPU/GPU) and number type, to avoid "same device" errors|
|`images.shape[1]`|C = number of colour channels (1 for MNIST, 3 for RGB)|
|`torch.tensor(mean, ...)`|turns the Python list `[0.1307]` into a tensor|
|`.view(1, -1, 1, 1)`|reshapes it to [1, C, 1, 1]; `-1` means "work out this size yourself"|
|channel check|errors if you pass 3 means for a 1-channel image|

**Why [1, C, 1, 1]? So it broadcasts**: PyTorch automatically stretches a size-1 dimension to match. A [1, 1, 1, 1] mean subtracts from every image, every row and every column of a [128, 1, 28, 28] batch at once. For RGB, `[0.485, 0.456, 0.406]` becomes [1, 3, 1, 1], one mean per colour channel.

### The pixel-space attack

```python
def fgsm_pixel_space(model: nn.Module,
                     images: Tensor,
                     labels: Tensor,
                     epsilon: float,
                     mean: list,
                     std: list,
                     targeted: bool = False) -> Tensor:
    """FGSM for pixel-space inputs attacking normalized models.

    This variant accepts images in [0,1] pixel space rather than normalized
    space. It normalizes inputs internally for the model, converts gradients
    back to pixel space, and returns adversarials in [0,1] pixel space.

    Args:
        model: Model expecting normalized inputs
        images: Clean images in [0,1] pixel space (unnormalized)
        labels: Target labels
        epsilon: Max perturbation in pixel space (e.g., 8/255)
        mean: Normalization mean per channel
        std: Normalization std per channel
        targeted: If True, minimize loss towards labels

    Returns:
        Tensor: Adversarial images in [0,1] pixel space (unnormalized)
    """
    mean_t, std_t = _norm_params(images, mean, std)
    x = images.clone().detach()
    x_norm = (x - mean_t) / std_t
    x_norm.requires_grad_(True)

    _, loss = _forward_and_loss(model, x_norm, labels)
    model.zero_grad(set_to_none=True)
    loss.backward()

    # Convert gradient from normalized space to image space
    grad_img = x_norm.grad / std_t
    step_dir = -1.0 if targeted else 1.0
    x_adv = torch.clamp(x + step_dir * epsilon * grad_img.sign(), 0.0, 1.0)
    return x_adv.detach()
```

### The flow, step by step

|Step|Code|Space|
|---|---|---|
|1. copy the plain image|`x = images.clone().detach()`|pixel [0, 1]|
|2. normalize for the model|`x_norm = (x - mean_t) / std_t`|normalized|
|3. track slopes on the **normalized** tensor|`x_norm.requires_grad_(True)`|normalized|
|4. forward, loss, backward|`_forward_and_loss`, `loss.backward()`|normalized|
|5. convert the slope back (chain rule)|`grad_img = x_norm.grad / std_t`|pixel|
|6. step on the **plain** image, clamp to [0, 1]|`torch.clamp(x + ... , 0.0, 1.0)`|pixel|

Compare with `fgsm_attack`: there the step and clamp happen in normalized space with bounds [−0.424, 2.821]. Here they happen in pixel space with bounds [0, 1]. **Each function clamps in the space its image lives in.**

## Part 6: Using the pixel-space variant

```python
# Example: Starting with pixel-space images
epsilon_px = 8 / 255 # pixel-space epsilon (≈0.031)
mean, std = [0.1307], [0.3081]

# Denormalize existing normalized images to get pixel-space images
mean_t, std_t = _norm_params(images, mean, std)
pixel_images = images * std_t + mean_t
pixel_images = torch.clamp(pixel_images, 0.0, 1.0)

# Attack in pixel space
x_adv_pixel = fgsm_pixel_space(model, pixel_images, labels, epsilon_px, mean, std)

# x_adv_pixel is in [0,1] and can be displayed or saved directly
# If you need to pass to the model again, normalize it first:
x_adv_norm = (x_adv_pixel - mean_t) / std_t
```

|Line|What it does|
|---|---|
|`epsilon_px = 8 / 255`|"change each pixel by at most 8 brightness levels" ≈ 0.031|
|`images * std_t + mean_t`|**denormalize**: x = x_norm·σ + μ (Section 5 Step 3)|
|`torch.clamp(..., 0.0, 1.0)`|tidies tiny rounding errors back into [0, 1]|
|`fgsm_pixel_space(...)`|attack with a pixel-space budget|
|`(x_adv_pixel - mean_t) / std_t`|**re-normalize before giving the result back to the model**, which only understands normalized input|

**When to use which:** your loaders give normalized batches (`normalize=True`), so use `fgsm_attack` directly. Use `fgsm_pixel_space` only when the images arrive as plain [0, 1] pixels, for example from an external API or a challenge endpoint.

## Copy-paste cells (mine, not the course's)

Run these after the course cells above. They use only names that already exist.

### Confirm `model` and `trained_model` are the same object

```python
print(trained_model is model)
```

Expected: `True`. If you get `False`, use `trained_model` in the attack cells instead.

### Check the perturbation really is capped at ε

This reads the biggest single-pixel change, which is the L∞ size (Section 2).

```python
delta = (x_adv - images).abs().max().item()
print(round(delta, 4), round(delta * 0.3081, 4), round(delta * 0.3081 * 255))
```

Expected: `0.8 0.2465 63`. That's the normalized ε, its pixel size, and the /255 level count. It can't be larger, because of the clamp.

### Show the two FGSM versions agree

A pixel ε of 8/255 is the same as a normalized ε of (8/255) / 0.3081 ≈ 0.102. This compares the two outputs in normalized space.

```python
eps_norm = (8 / 255) / 0.3081
x_adv_a = fgsm_attack(model, images, labels, eps_norm)
diff = (x_adv_a - x_adv_norm).abs().gt(1e-4).float().mean().item()
print(f"fraction of pixels that differ: {diff:.4f}")
```

Expected: `0.0000` or very close. A handful of pixels can differ where a gradient is almost exactly zero and rounding flips its sign. This confirms the "÷σ doesn't change the sign" note.

### Show what got flipped

This lists the first few successful flips as true label → adversarial prediction.

```python
idx = flipped.nonzero().flatten()[:10]
for i in idx.tolist():
    print(f"image {i}: {labels[i].item()} -> {adv_pred[i].item()}")
```

Expected: up to 10 lines like `image 0: 7 -> 3`. Your exact digits will vary.

## Quick self-check

Using the toy (σ = 0.4), one pixel's normalized-space gradient is **−2**. What is its **pixel-space** gradient, and does dividing by σ change FGSM's step **direction** for that pixel?

Reply with your answer and I'll check it.

## The facts worth locking in for the exam

- **The input gradient comes from `x.clone().detach().requires_grad_(True)`, a forward pass, `loss.backward()`, then reading `x_req.grad`**. Forgetting `requires_grad_(True)` leaves the gradient as `None`, and the weights are never updated because there's no optimizer step.
- **`fgsm_attack` is four operations: `grad.sign()`, multiply by ε (and by `step_dir`), add to the images, then `torch.clamp` to the valid range**, giving every pixel a change of exactly ±ε before clamping.
- ==`step_dir = +1` is untargeted (raise the loss on the true labels) and `step_dir = −1` is targeted (lower the loss on the target labels)==, so for a targeted attack the `labels` argument must hold the target classes.
- **`fgsm_attack` works in normalized space and clamps to MNIST's normalized bounds [−0.424, 2.821]**, while `fgsm_pixel_space` takes [0, 1] images and a pixel ε (like 8/255 ≈ 0.031) and clamps to [0, 1]. Each clamps in the space its images are in.
- ==Converting the gradient from normalized to pixel space uses the chain rule ∇xℒ = ∇x_normℒ / σ==. Because σ is positive this never changes the sign, so for sign-based FGSM a pixel ε is equivalent to a normalized ε of ε/σ.
- ==Attack success counts only images the model originally classified correctly: flipped = (adv_pred != labels) & originally_correct==. On the first batch, ε_norm = 0.8 (≈ 0.25 pixel, ≈ 64/255) flipped 71.09% (91 of 128).
- **`_forward_and_loss` raises an error if the model is still in training mode**. Attacks must run with `model.eval()` and must not be wrapped in `torch.no_grad()`, while clean and adversarial _predictions_ can be.
- **Method point:** **always verify an attack's real budget by measuring `(x_adv − images).abs().max()` and converting it to pixel units**, so the ε you report matches what was actually applied.