# Distance Metrics

## The idea in one breath

A small helper that **measures** the change three ways - L1 (total change), squared L2 (energy), and the elastic-net blend `L2² + β·L1` - once per image in a batch. It makes no decisions; it just computes rulers. The one thing to burn in: **what this course calls "L2 distance" is the sum of squares with no square root**, so numbers won't match a true Euclidean distance.

> **Heads-up: your paste dropped the superscripts.** Read them as:
> 
> - `100×0.152=2.25` → 100 × 0.15² = 2.25 (squared).
> - `50×0.302=4.50` → 50 × 0.30² = 4.50 (squared).
> - `∥δ∥22+β∥δ∥1` → ‖δ‖₂² + β·‖δ‖₁ (squared L2 plus β times L1).
> - `∥x′−x∥22` → ‖x′−x‖₂², the squared L2.

---

## Part 1 - The three rulers, with numbers

### L1: total change, pixels weighted by how much they moved

L1 adds up the absolute change of every pixel. Two examples that tie:

|Change|L1|
|---|---|
|100 pixels × 0.15|100 × 0.15 = **15.0**|
|50 pixels × 0.30|50 × 0.30 = **15.0**|

Same L1. **Unlike L0 (which would say 100 vs 50), L1 doesn't care how many pixels - only the total amount of movement.**

### Squared L2: energy, punishes big individual changes

L2 here squares each pixel's change before summing. Same two perturbations:

|Change|squared L2|
|---|---|
|100 pixels × 0.15|100 × 0.0225 = **2.25**|
|50 pixels × 0.30|50 × 0.09 = **4.50**|

Now they split. The 50-pixel version has **twice the squared-L2 even though its L1 is identical**, because squaring makes one 0.30 change count for more than two 0.15 changes (0.09 vs 0.045). **L2 punishes concentration; L1 doesn't.**

> **An honest note the course skips:** that "twice" only holds for _squared_ L2. If you took the real Euclidean norm (with the square root), the ratio would be √2 ≈ 1.41, not 2 (I checked: 1.5 vs 2.12). The course calls `l2_dist` the "L2 distance" throughout but the code never takes a square root, so **treat every "L2 distance" in this module as ‖δ‖₂²**. That's why the elastic formula is `L2² + β·L1`, not `L2 + β·L1`.

### Elastic-net: the weighted blend

```
elastic distance = ‖δ‖₂² + β · ‖δ‖₁
```

Same knob as Section 2: ==β = 0 is pure squared-L2; bigger β pulls toward sparsity.== It has no clean geometric shape (it's not a true norm's circle or diamond), but it captures the balance neither pure norm can.

---

## Part 2 - Norm trade-offs (the intuition recap)

|Ruler|What it rewards|Failure mode|
|---|---|---|
|Pure L2|smooth, spread-out changes|**no sparsity** - can nudge _every_ pixel, shifting global brightness/texture|
|Pure L1|sparsity via the zero-kink|needs to "overcome a barrier" to turn a pixel on, so small ones stay 0|
|Elastic (L2²+β·L1)|**few pixels (L1), none huge (L2)**|the middle ground; β tunes where it sits|

The course's stated heuristic: for a fixed L1 budget, 50 pixels at 0.2 ties 100 pixels at 0.1, but the **sparse version often fools the classifier more decisively** because a few concentrated changes can push harder across a local decision boundary. And the β guidance: small β (0.01) → nearly-smooth with mild sparsity (when imperceptibility matters); large β (1.0) → aggressive sparsity (fewest pixels, possibly visible clusters).

---

## Part 3 - The code

```python
def compute_distances(adv_images, original_images, beta):
    """
    Compute L1, L2, and elastic-net distances.

    Returns all three distance metrics used in optimization
    and decision-making.

    Parameters:
        adv_images (torch.Tensor): Adversarial images (batch_size, C, H, W)
        original_images (torch.Tensor): Original images (batch_size, C, H, W)
        beta (float): Weight for L1 in elastic-net distance

    Returns:
        tuple: (l1_dist, l2_dist, elastic_dist) each shape (batch_size,)
    """
    l1_dist = torch.sum(torch.abs(adv_images - original_images), dim=(1, 2, 3))
    l2_dist = torch.sum((adv_images - original_images) ** 2, dim=(1, 2, 3))
    elastic_dist = l2_dist + beta * l1_dist

    return l1_dist, l2_dist, elastic_dist
```

|Code|What it does|
|---|---|
|`adv_images - original_images`|the perturbation δ, pixel by pixel|
|`torch.abs(...)`|absolute value of each change (for L1)|
|`(...) ** 2`|square each change (for squared L2)|
|`torch.sum(..., dim=(1, 2, 3))`|**add up over channel, height, width - but not batch**|
|`elastic_dist = l2_dist + beta * l1_dist`|the blend, same β as the proximal operator|

Shapes:

|Name|Shape|Meaning|
|---|---|---|
|`adv_images`, `original_images`|(B, C, H, W) e.g. (20, 1, 28, 28)|a batch of images|
|`adv_images - original_images`|(20, 1, 28, 28)|the per-pixel change|
|after `sum(dim=(1,2,3))`|(20,)|**one scalar per image**|

### Why keep the batch dimension

`dim=(1, 2, 3)` collapses the three image axes and **leaves axis 0 (the batch) intact**, giving 20 separate distances for 20 images. The reason is binary search: each image gets its **own** trade-off constant c - an easy one near the boundary might need c = 0.01, a stubborn one c = 10. **Without per-image distances you'd be forced into one shared c**, over-perturbing the easy images and under-perturbing the hard ones.

### Why squared L2 (not true L2)

Same reason as Section 4: the gradient of ‖x′−x‖₂² is the clean **2(x′−x)**, whereas the true norm's gradient has an ugly normalized form. Squaring keeps the problem **strictly convex** and yields the same optimal perturbations. It's a pure measurement function - **no thresholding, no constraints, no success decisions** - other functions consume these numbers later.

---

## Copy-paste cells (mine, not the course's)

### Reproduce the course's two perturbations

Confirms the 15.0 / 2.25 / 4.50 numbers and that `l2_dist` is squared.

```python
import torch

def compute_distances(adv_images, original_images, beta):
    l1_dist = torch.sum(torch.abs(adv_images - original_images), dim=(1, 2, 3))
    l2_dist = torch.sum((adv_images - original_images) ** 2, dim=(1, 2, 3))
    return l1_dist, l2_dist, l2_dist + beta * l1_dist

orig = torch.zeros(2, 1, 10, 20) # two blank 200-pixel "images"
adv = orig.clone()
adv[0, 0].reshape(-1)[:100] = 0.15 # image 0: 100 pixels changed by 0.15
adv[1, 0].reshape(-1)[:50] = 0.30 # image 1: 50 pixels changed by 0.30
l1, l2, elastic = compute_distances(adv, orig, beta=1.0)
print("L1 :", l1.tolist()) # same for both
print("L2 (sq) :", l2.tolist()) # second is 2x
print("elastic :", elastic.tolist())
```

Expected:

```txt
L1 : [15.0, 15.0]
L2 (sq) : [2.25, 4.5]
elastic : [17.25, 19.5]
```

Identical L1, but the concentrated 50-pixel change costs twice the squared-L2 and so loses on the elastic ruler.

---

## Quick self-check

A perturbation changes three pixels - [0.4, −0.2, 0.1] - and leaves the rest at 0. With **β = 0.5**, what does `compute_distances` return for `l1_dist`, `l2_dist`, and `elastic_dist`? (Remember what `l2_dist` actually computes.)

---

## The facts worth locking in for the exam

- ==`compute_distances` returns L1, squared-L2, and elastic (= l2 + β·l1), each as one scalar per image==, using `dim=(1,2,3)` to sum over channel/height/width while keeping the batch axis.
- **In this module "L2 distance" means the sum of squares ‖δ‖₂² with no square root**, so an elastic distance is `‖δ‖₂² + β‖δ‖₁`; expecting a true Euclidean value will make your numbers disagree with the labs.
- **L1 ties 100 pixels × 0.15 with 50 pixels × 0.30 (both 15.0), but squared-L2 splits them (2.25 vs 4.50)**, because squaring punishes concentrated changes while L1 only tracks total movement and L0 only counts pixels.
- **Per-image distances exist so binary search can give each image its own trade-off constant c**, from c ≈ 0.01 for an easy near-boundary image to c ≈ 10 for a resistant one, avoiding the over/under-perturbation a single shared c would cause.
- **Squared L2 is used instead of true L2 because its gradient 2(x′−x) is simple and it stays strictly convex**, producing the same optimal perturbations as the normalized form would.
- **β sets the sparsity - smoothness balance: 0.01 gives nearly-smooth perturbations with mild sparsity, 1.0 gives aggressive sparsity** that may form visible clusters but minimizes the modified-pixel count.
- **`compute_distances` is pure measurement - no thresholding, no constraints, no success checks**; downstream functions use its outputs for loss gradients and attack-success decisions.
- **Method point:** **when writing a batched metric, reduce only the feature dimensions and keep the batch dimension**, so every example carries its own number and per-example logic (like per-image binary search) stays possible.

Logged Section 5 and the self-check answer key. Paste Section 6 (Adversarial Loss) when ready - that's where the course nails down the exact targeted-vs-untargeted margin I flagged back in Section 4.