# Loss Gradients And Optimization

## The idea in one breath

This section has **no new code** - it's the intuition for what the gradients in FISTA actually do. Every pixel feels **two forces**: the adversarial gradient (pushing it to fool the model) and the L2 gradient (a restoring spring pulling it back to the original, equal to `2·δ`). The trade-off constant **c** scales the first against the second, and on a single pixel it can literally **flip which way that pixel moves**.

> **Heads-up: mangled formulas.** Read them as:
> 
> - `∇x′ℒtotal` -> the gradient of the total loss with respect to the adversarial **image** x′ (the pixels), not the weights.
> - `Z=Wout⋅h+bout` -> Z = W_out · h + b_out (logits = weights × hidden + bias).
> - `∂f/∂h=WoutT⋅∂f/∂Z` -> ∂f/∂h = W_outᵀ · ∂f/∂Z (gradient flows back through the transpose of the weights).
> - `η≤1/L` -> step size η must be ≤ 1/L, where L is the Lipschitz constant.
> - "saturated (reached its minimum value of −κ)" - see the honest note in Part 4; Section 6's code actually floors at 0.

---

## Part 1 - Gradients with respect to pixels, not weights

Training a network computes the gradient of the loss with respect to the **weights**. An attack flips this: **freeze the weights, turn gradients on for the image**, and `loss.backward()` gives the gradient with respect to each **pixel**. That tells you how each pixel affects the loss, which is exactly what FISTA needs to nudge pixels.

### Two forces on every pixel

|Force|Where it comes from|What it does|
|---|---|---|
|Adversarial gradient|backprop through the whole network|points each pixel toward misclassification|
|L2 gradient = 2·δᵢ|the squared-L2 term|a spring pulling the pixel back to its original value|

The L2 force is dead simple: if a pixel's change is δᵢ = 0.15, its L2 gradient is 2 × 0.15 = **0.30**, pulling it back toward the original. **It grows linearly with how far the pixel has strayed**, so the further you push, the harder it pulls back.

The adversarial force is the black box: FISTA just asks "give me a loss and a gradient" and follows it, **never needing to know the architecture**. That's what makes ElasticNet work on any differentiable model.

### The margin example (this one's correct)

True class 3, logits `[0.2, 0.5, 1.8, 2.1, 0.9, ...]`:

- `real` = Z₃ = **2.1** (the true class, index 3)
- `other` = **1.8** (class 2, the best competitor)
- margin = 2.1 − 1.8 = **0.3**, so adversarial loss = 0.3 (κ = 0)

If increasing a pixel raises class 2's logit and lowers class 3's, that pixel gets a **negative gradient** (increase it -> shrink the margin -> lower the loss). Big gradient magnitude = that pixel is very sensitive.

---

## Part 2 - How c tips a single pixel (the key applied idea)

### Toy

One pixel, currently perturbed by δ = 0.15. Its two forces:

- L2 gradient: 2 × 0.15 = **+0.30** (restoring, toward original)
- adversarial gradient: **−2.5** (pushing toward misclassification)

The combined gradient is `L2 + c · adversarial`. Watch c flip the result:

|c|combined gradient = 0.30 + c·(−2.5)|gradient descent moves the pixel...|
|---|---|---|
|0.1|0.30 − 0.25 = **+0.05**|down slightly (toward original); margin still drops overall|
|1.0|0.30 − 2.50 = **−2.20**|**up, hard (toward misclassification)**|

Remember **gradient descent steps _opposite_ the gradient**: a positive combined gradient (+0.05) means step the pixel down; a negative one (−2.20) means step it up. So small c lets the L2 spring win (stay near original, attack may fail); large c lets the adversarial force win (fool the model, bigger change). **This per-pixel tug-of-war is exactly the global trade-off binary search tunes.**

### The magnitude story over the run

|Iterations|Adversarial gradient|L2 gradient|Phase|
|---|---|---|---|
|1-100|5-20 (dominates)|0.1-0.5|bold - drive toward misclassification|
|100-500|1-5|1-5|balanced - adv saturating, L2 pulls back excess|
|500-1000|~0 (saturated)|0.1-1.0|refine - only distortion left to shrink|

And c's effect on the whole attack:

|c|Adversarial gradient contribution|Outcome|
|---|---|---|
|0.001|0.001-0.05|too weak, attack may fail|
|10|50-500|succeeds easily, but perturbation is bloated|
|0.01-1.0|balanced|**the MNIST sweet spot binary search aims for**|

---

## Part 3 - The loss landscape and gradient flow

### Mixed curvature

The adversarial term makes **steep cliffs near decision boundaries** (small change, big loss swing); the L2 term is a **smooth bowl** (gradient proportional to how far you've moved). Together the surface is steep toward misclassification and shallow sideways - curved, not flat.

### The automatic mode switch

This is the elegant part: **when the adversarial loss saturates to its floor, it contributes zero gradient**, so optimization silently becomes pure L2 minimization plus soft thresholding. The hinge handles the switch from "achieve misclassification" to "minimize distortion" with **no explicit if-statement**. Before saturation, both forces pull at once, bending the trajectory.

### Where the gradient goes through the layers

|Layer|What it does to the backward gradient|
|---|---|
|Output: Z = W_out·h + b_out|flows back via the transpose: ∂f/∂h = W_outᵀ · ∂f/∂Z; hidden units with big weights to relevant classes get stronger signal|
|ReLU|**active (input > 0): pass through; inactive (≤ 0): block entirely**, creating the piecewise, non-convex landscape|
|Convolution (3×3)|spreads each output gradient across a 3×3 **receptive field** of input pixels|
|Max pooling|**only the max cell in each window gets gradient; the rest get zero**|

The consequence: ReLU and max-pooling mean **many pixels receive weak or zero gradient**, so soft thresholding zeros them, which _helps_ sparsity but can miss pixels that might matter with more signal. FISTA's many iterations let gradients **build up over time**, sometimes activating pixels that started weak.

---

## Part 4 - Convergence and stopping

### The theory vs reality

FISTA's O(1/k²) guarantee needs three things: f has **Lipschitz-continuous gradients**, h is **convex**, and the step size **η ≤ 1/L** (L = the gradient's Lipschitz constant, bigger for deeper/heavier networks). Neural nets are **non-convex, so the guarantee technically doesn't hold** - but FISTA works well empirically, finding good _local_ minima. ElasticNet always **starts from the original image (zero perturbation)** for a consistent start, and binary search over c acts like multiple restarts with different adversarial pressure.

### When to stop

|Stopping rule|How it works|
|---|---|
|Fixed iterations (e.g. 1000)|simple, but wastes time on easy examples and may under-optimize hard ones|
|Objective change < ε|stop when `|
|Success + saturation|stop when it's misclassified _and_ the adversarial loss has bottomed out|
|**HTB's hybrid**|max iterations per binary-search step + periodic success checks; if the whole batch is fooled early, end that step|

> **An honest note the course skips (the saturation floor):** here the text says the adversarial loss saturates at its "minimum value of −κ." But Section 6's actual code was `clamp(..., min=0)`, which floors at **0**, not −κ. These are two equivalent C&W conventions - `max(margin + κ, 0)` floors at 0, while `max(margin, −κ)` floors at −κ - and both mean "stop once a competitor leads by κ." Since you're running Section 6's code, **your adversarial loss bottoms out at 0**, not −κ.

---

## Copy-paste cell (mine, not the course's)

The section has no code, but this makes the "two forces" real. It computes the input gradient on one image and shows the magnitude ranges the section talks about. Uses your names: `model`, `test_loader`, `device`.

### Inspect pixel gradients on one image

```python
import torch, torch.nn.functional as F

model.eval() # dropout off
img, label = next(iter(test_loader)) # one batch; take image 0
x = img[0:1].to(device).clone().requires_grad_(True) # gradients ON for the IMAGE
logit = model(x)
adv_loss = F.cross_entropy(logit, torch.tensor([label[0].item()], device=device)) # stand-in loss
adv_loss.backward() # fills x.grad (per-pixel)

g = x.grad.abs()
print(f"adversarial grad per pixel: min {g.min():.4f} max {g.max():.4f} mean {g.mean():.4f}")
print(f"pixels with near-zero grad (<1e-4): {(g < 1e-4).sum().item()} / {g.numel()}")
```

Expected (numbers vary): a spread of per-pixel magnitudes and a chunk of near-zero pixels - **the weak-gradient pixels that soft thresholding will zero out**. (This uses cross-entropy as a quick stand-in; the real attack uses the C&W margin loss from Section 6.)

## Key facts to lock in

- **Attacks compute the gradient with respect to the input pixels, not the weights**, by freezing the network and enabling gradients on the image, so `loss.backward()` reveals how each pixel affects the loss.
- **Every pixel feels two forces: the adversarial gradient (toward misclassification) and the L2 gradient 2·δᵢ (a linear restoring spring toward the original)**, and the trade-off constant c scales the first against the second.
- **On a single pixel, raising c can flip the net gradient's sign**: with δ = 0.15 and adversarial gradient −2.5, the combined gradient is +0.05 at c = 0.1 (drift back to original) but −2.20 at c = 1.0 (push hard to misclassify) - the local image of the global trade-off.
- **When the adversarial loss saturates it contributes zero gradient, so optimization automatically becomes pure L2 minimization + soft thresholding**, switching from "fool it" to "shrink it" with no explicit mode switch.
- **ReLU blocks gradient where its input ≤ 0 and max-pooling passes gradient only through the winning cell**, so many pixels get weak or zero signal, which soft thresholding then zeros - helping sparsity but sometimes missing useful pixels.
- **FISTA's O(1/k²) guarantee requires a convex, Lipschitz-smooth problem with η ≤ 1/L, conditions neural nets violate**, yet it still finds effective local minima; ElasticNet starts from zero perturbation and uses binary search over c as de-facto restarts.
- **HTB's stopping is hybrid: a max iteration count per binary-search step plus periodic success checks**, ending a step early if the whole batch is already fooled, which avoids wasting FISTA iterations on easy examples.
- **Method point:** **watch gradient magnitudes as a health signal** - adversarial gradients over ~100/pixel early mean overshoot and oscillation, and gradients under ~0.001/pixel before iteration 100 mean premature stagnation in a poor local minimum.

