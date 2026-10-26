# Proximal Operators And FISTA Framework

## The idea in one breath

The L1 term has a **sharp corner at zero where the derivative doesn't exist**, so ordinary gradient descent can't produce exact zeros - it just hovers near zero forever. The fix is the **proximal operator**, and for L1 it turns out to be one dead-simple rule called **soft thresholding**: zero out anything smaller than a threshold, shrink everything else toward zero. **FISTA** is the full optimizer: gradient step on the smooth part, soft-threshold the L1 part, plus a momentum "look-ahead" that speeds it up from O(1/k) to O(1/k²).

> **Heads-up: your paste flattened every fraction, superscript and square root.** Here is each formula read correctly:
> 
> - `proxλh(z)=argminx{12∥x−z∥22+λh(x)}` → **prox_{λh}(z) = argmin over x of [ ½·‖x−z‖₂² + λ·h(x) ]**. The "12" is the fraction ½; "22" is "L2, squared."
> - Soft threshold `𝒮λ(z)i` = three cases: **zᵢ−λ if zᵢ>λ; 0 if |zᵢ|≤λ; zᵢ+λ if zᵢ<−λ**.
> - L2 prox `max(0,1−λ/∥z∥2)⋅z` → **max(0, 1 − λ/‖z‖₂) · z**.
> - `tk+1=1+1+4tk22` → **t_{k+1} = ( 1 + √(1 + 4·t_k²) ) / 2**. (This one was mangled hardest - it's "one plus the square root of (one plus four t-k-squared), all over two.")
> - `y(k+1)=x(k+1)+tk−1tk+1(...)` → **y_{k+1} = x_{k+1} + ((t_k − 1)/t_{k+1})·(x_{k+1} − x_k)**.
> - `f(x′)=c⋅max(Zy(x′)−maxj≠yZj(x′)+κ,0)+∥x′−x∥22` → **f(x′) = c·max( Z_y(x′) − max_{j≠y} Z_j(x′) + κ, 0 ) + ‖x′−x‖₂²**.
> - `∇x′∥x′−x∥22=2(x′−x)` → the gradient of the squared-L2 term is **2(x′−x)**.
> - `proxηβ∥⋅∥1(z)=x+𝒮ηβ(z−x)` → **x + 𝒮_{ηβ}(z − x)**. η is the step size (eta), β the sparsity weight, κ (kappa) a confidence margin.

---

## Part 1 - Why L1 is hard

### The idea

FGSM and DeepFool had **closed-form** moves: a sign operation, a projection formula. Done in one shot. ElasticNet's objective has the L1 term, and L1 has no such shortcut.

### The kink at zero (toy)

Picture the graph of |x|. For x > 0 its slope is +1; for x < 0 its slope is −1. At **x = 0** the slope jumps from −1 to +1 - there's a sharp V-corner, and **no single derivative exists there**.

Gradient descent only knows how to move "downhill by the slope." At the exact point where sparsity happens (a value landing on 0), there's no slope to read. So what actually happens with naive gradient descent on L1:

|Step|value|
|---|---|
|...|0.014|
|next|−0.009|
|next|0.006|
|next|−0.004|

It **oscillates around zero, getting tiny but never landing on it**. The course's name for that is **pseudo-sparsity** (many small values) instead of **true sparsity** (many exact zeros). You met this exact warning in Section 2.

---

## Part 2 - The proximal operator (the general tool)

### The idea in plain words

A projection asks: _"what's the closest allowed point to where I am?"_ A proximal operator generalizes that to: _"what point is a good compromise between staying near where I am and keeping some penalty small?"_

### Toy

Say you're standing at **z = 0.12** on a number line. Two forces act on you:

- A **spring to z** (the ½(x−z)² term): pulls you back toward 0.12. The further you stray, the harder it pulls.
- A **penalty pulling to 0** (the λ·h term): pulls you toward the origin.

The proximal operator finds where these two forces **balance**. With the right penalty, that balance point for z = 0.12 turns out to be 0.02 - close to 0.12 but nudged toward zero.

### The real formula

```
prox_{λh}(z) = argmin over x of [ ½·‖x − z‖₂² + λ·h(x) ]
```

|Symbol|Plain English|In the toy|
|---|---|---|
|z|where you are now|0.12|
|x|candidate answer we're solving for|0.02 at the balance|
|½‖x−z‖₂²|"stay close to z" (the spring)|pulls toward 0.12|
|λ·h(x)|"obey the penalty h"|pulls toward 0|
|λ|how strongly the penalty matters (step size × weight)|0.1|
|argmin over x|"the x that makes the whole thing smallest"|the balance point|

**Read aloud:** "The proximal operator returns the point that is both close to z and low in penalty, balanced by λ."

### Three penalties, three behaviours

|Penalty h|Proximal operator becomes|
|---|---|
|indicator (0 inside a set, ∞ outside)|**ordinary projection** (the FGSM/DeepFool case)|
|smooth, like ‖x‖₂²|a closed form, basically a gradient step|
|non-smooth L1, ‖x‖₁|**soft thresholding** (the one we need)|

So the proximal operator **is** the generalization of the projections you already used; L1 is just the case that needs the new rule.

---

## Part 3 - Soft thresholding: the L1 proximal operator

### The rule (toy first)

Pick a threshold λ = 0.1. Soft thresholding looks at each number on its own and applies:

|Your value zᵢ|Region|Action|Result|
|---|---|---|---|
|0.12|above +λ|subtract λ|0.12 − 0.1 = **0.02**|
|0.08|within ±λ|zero it|**0**|
|−0.25|below −λ|add λ|−0.25 + 0.1 = **−0.15**|

So for z = [0.12, 0.08, −0.25], **𝒮₀.₁(z) = [0.02, 0.0, −0.15]** - exactly the course's check. **Values smaller than λ die; values larger than λ get pulled λ closer to zero.**

### The real formula

```
             ⎧ zᵢ − λ if zᵢ > λ (shrink a positive value)
𝒮_λ(z)ᵢ = ⎨ 0 if |zᵢ| ≤ λ (kill a small value → sparsity)
             ⎩ zᵢ + λ if zᵢ < −λ (shrink a negative value)
```

One-line equivalent (the form you'll code): `sign(z) · max(|z| − λ, 0)`.

**Read aloud:** "Push every value λ toward zero; if that would cross zero, stop at zero."

### Why it's "soft" vs "hard"

| Thresholding | Small values (|z| ≤ λ) | Large values | Analogy |  
|---|---|---|---|  
| Hard | set to 0 | **left unchanged** | scissors (cut or keep) |  
| Soft | set to 0 | **shrunk by λ** | gently squeezing everything inward |

### Why this is the right answer (the 1-D derivation)

For one coordinate, the proximal problem is: minimize `½(xᵢ − zᵢ)² + λ|xᵢ|`. Split by the sign of zᵢ:

- **zᵢ > λ:** guess the answer is positive, so |xᵢ| = xᵢ. The derivative is (xᵢ − zᵢ) + λ. Set it to 0 → **xᵢ = zᵢ − λ**. ✓ (matches case 1)
- **zᵢ < −λ:** guess negative, |xᵢ| = −xᵢ. Derivative is (xᵢ − zᵢ) − λ. Set to 0 → **xᵢ = zᵢ + λ**. ✓ (case 3)
- **|zᵢ| ≤ λ:** the pull toward 0 (constant λ) beats the spring's pull toward zᵢ, so the minimum sits at **xᵢ = 0**. ✓ (case 2)

I verified this numerically: minimizing that 1-D objective by brute force for zᵢ = 0.12, 0.08, −0.25 lands on exactly 0.02, 0, −0.15. The intuition from Section 2 returns: **near zero, L1's constant pull wins and snaps the value to exactly 0; far from zero, the spring wins but L1 still offsets it by λ.**

### Why it's cheap

Because ‖·‖₁ is a _sum_ of absolute values and ‖·‖₂² is a _sum_ of squares, the big multi-pixel problem **splits into independent one-pixel problems**. Each pixel is thresholded on its own, with no reference to the others - so it's a single vectorized operation over all 784 pixels.

---

## Part 4 - The L2 proximal operator (contrast)

For comparison, the L2 (not squared) proximal operator is:

```
prox_{λh}(z) = max(0, 1 − λ/‖z‖₂) · z
```

This multiplies the **whole vector** by one shared shrink factor. It **scales every coordinate by the same fraction** - it never singles out small coordinates to zero. That's the geometric reason L2 gives dense perturbations and L1 gives sparse ones: L1 treats pixels individually (can zero some), L2 treats them as one unit (shrinks all together).

---

## Part 5 - From proximal gradient to FISTA

### Proximal gradient: the basic loop

For a problem `minimize f(x) + h(x)` where f is smooth and h is not, each iteration is two moves:

```
x_{k+1} = prox_{ηh}( x_k − η·∇f(x_k) )
```

In words: **take a normal gradient step on the smooth part f (step size η), then soft-threshold the result for h.** This provably converges, but slowly - the error shrinks like **O(1/k)** (after k steps the error is proportional to 1/k).

### FISTA: add momentum (the look-ahead)

FISTA computes the gradient not at the current point but at an **extrapolated point y_k that guesses where the optimization is heading**, using the direction of the last step. If steps keep pointing the same way, this look-ahead lets it move faster. Same cost per iteration, but the error now shrinks like **O(1/k²)** - a quadratic speed-up.

```
x_{k+1} = prox_{ηh}( y_k − η·∇f(y_k) ) ← gradient at the look-ahead point y_k
t_{k+1} = ( 1 + √(1 + 4·t_k²) ) / 2 ← grows the momentum strength
y_{k+1} = x_{k+1} + ((t_k − 1)/t_{k+1})·(x_{k+1} − x_k) ← extrapolate past the new point
```

|Symbol|Plain English|
|---|---|
|x_k|the actual answer at step k|
|y_k|the look-ahead point where the gradient is measured|
|t_k|a counter (start t₀ = 1) that grows and controls momentum strength|
|(t_k − 1)/t_{k+1}|the momentum coefficient - how far to extrapolate|
|η (eta)|gradient step size|

### The t-sequence and the shortcut (verified)

Starting from t₀ = 1, the t's grow and the momentum coefficient climbs toward 1:

|k|t_k|exact coef (t_k−1)/t_{k+1}|shortcut k/(k+3)|
|---|---|---|---|
|1|1.000|0.000|0.250|
|3|2.194|0.434|0.500|
|5|3.295|0.599|0.625|
|10|5.942|0.765|0.769|

The course notes that many implementations skip the exact t-recursion and just use **k/(k+3)** as the momentum coefficient. As the table shows, the two differ early (k=1: 0.0 vs 0.25) but **track closely once k is large** (k=10: 0.765 vs 0.769), which is why the course says it's valid "for large k" and slightly more conservative. This is the version the lab code will use.

---

## Part 6 - FISTA applied to ElasticNet

### The split: smooth f vs non-smooth h

ElasticNet's objective gets carved into the part gradients can handle and the part soft thresholding handles.

**Smooth part f** (adversarial loss + L2 distance):

```
f(x′) = c · max( Z_y(x′) − max_{j≠y} Z_j(x′) + κ , 0 ) + ‖x′ − x‖₂²
```

|Piece|Plain English|
|---|---|
|Z_y(x′)|the **logit** (raw score) of the true class y|
|max_{j≠y} Z_j(x′)|the best competitor's logit|
|Z_y − max_{j≠y} Z_j|"how much the true class still leads by"|
|+ κ|**confidence margin**: demand the lead be beaten by κ, not just tied|
|max( ... , 0 )|a **hinge**: once misclassified with margin, this is 0 (stop pushing)|
|c ·|the trade-off constant from Section 2's binary search|
|‖x′−x‖₂²|the smooth L2 distance; its gradient is just **2(x′−x)**|

This is the **Carlini & Wagner margin**. As written it's the **untargeted** form (drive the true class _down_ below a rival).

> **An honest note the course skips:** this exact margin pushes the _true_ class down, which is untargeted. The module intro and Section 2 described EAD as usually _targeted_ (force a chosen class T). The targeted version flips the comparison - reward the target logit for beating the best _other_ class. Section 6 (Adversarial Loss) is where the course pins down the precise targeted form, so don't over-anchor on this untargeted hinge.

**Non-smooth part h** (the L1 penalty):

```
h(x′) = β · ‖x′ − x‖₁
```

### Sparsity is enforced on the _change_, not the image

The proximal step is applied to the perturbation δ = x′ − x:

```
prox_{ηβ‖·‖₁}(z) = x + 𝒮_{ηβ}(z − x)
```

The reasoning matters: **we want a sparse perturbation (few pixels changed), not a sparse image (few pixels bright).** So you subtract x to get the change, soft-threshold _that_, then add x back.

### One FISTA iteration, start to finish

1. Compute ∇f at the look-ahead point y_k (backprop through the net for the margin, plus 2(x′−x) for L2).
2. Take a gradient step with learning rate η.
3. **Apply soft thresholding with threshold η·β** to the perturbation (this is where pixels die).
4. Update the momentum point with coefficient k/(k+3).
5. Repeat for **100-1000 iterations**.

### The 2-D example (verified)

After a gradient step, suppose the perturbation is δ = [0.12, −0.07] with threshold 0.1. Soft thresholding gives **𝒮₀.₁(δ) = [0.02, 0]** - the first survives (shrunk), **the second is small, so it dies** (sparsity in action). At k = 10 the momentum coefficient is 10/(10+3) = 10/13 ≈ **0.769**, extrapolating the next evaluation point along the recent direction.

---

## Part 7 - Why FISTA (the contrast)

|Aspect|FGSM|DeepFool|ElasticNet (FISTA)|
|---|---|---|---|
|Iterations|1|5-20 closed-form|**100-1000**|
|Each step does|sign of one gradient|one geometric projection|full backprop + soft threshold + momentum|
|Closed form?|yes|yes (per step)|**no** - the L1 kink forbids it|
|Produces sparsity?|no|no|**yes, true zeros**|

The reason for all this machinery: ElasticNet's mixed L1/L2 objective has **no closed-form solution** (the L1 corners break standard gradients) and the misclassification constraint ties the perturbation to the network's decision boundary in a way geometry can't shortcut. FISTA splits it: gradients for smooth, proximal for non-smooth, momentum for speed.

---

## Copy-paste cells (mine, not the course's)

Pure numpy - run anywhere to build intuition before the lab code arrives.

### The soft-threshold operator (you'll reuse this idea in the lab)

```python
import numpy as np

def soft_threshold(z, lam):
    """L1 proximal operator: shrink toward 0 by lam, zero out anything <= lam."""
    return np.sign(z) * np.maximum(np.abs(z) - lam, 0.0) # the one-line three-region rule

z = np.array([0.12, 0.08, -0.25])
print("S_0.1(z) =", soft_threshold(z, 0.1)) # course's check
```

Expected:

```txt
S_0.1(z) = [ 0.02 0. -0.15]
```

### Watch the threshold create sparsity

Higher λ zeros out more coordinates - the sparsity knob in action.

```python
d = np.array([0.30, -0.05, 0.12, 0.02, -0.40]) # a toy perturbation
for lam in [0.0, 0.1, 0.2, 0.35]:
    out = soft_threshold(d, lam)
    print(f"lam={lam:<4} -> {out} L0={np.count_nonzero(out)}")
```

Expected:

```txt
lam=0.0 -> [ 0.3 -0.05 0.12 0.02 -0.4 ] L0=5
lam=0.1 -> [ 0.2 0. 0.02 0. -0.3 ] L0=3
lam=0.2 -> [ 0.1 0. 0. 0. -0.2 ] L0=2
lam=0.35 -> [ 0. 0. 0. 0. -0.05] L0=1
```

Each bump in λ kills the smallest survivor - exactly how β controls sparsity in the real attack (where the threshold is η·β).

---

## Quick self-check

After a gradient step, a FISTA iteration has produced the point z, and the original image x gives a perturbation **z − x = [0.30, −0.05, 0.12, 0.00]**. With step size **η = 1** and **β = 0.2** (so the soft-threshold level is η·β = 0.2), what is the thresholded perturbation, and what is its L0?

---

## The facts worth locking in for the exam

- **L1 can't be minimized by plain gradient descent because |x| has a corner at zero with no derivative**, so naive descent gives pseudo-sparsity (tiny non-zero values) instead of true sparsity (exact zeros); the proximal operator is the fix.
- ==The proximal operator prox_{λh}(z) = argmin ½‖x−z‖₂² + λh(x) generalizes projection==, balancing "stay near z" against "keep penalty h small," and reduces to ordinary projection when h is an indicator function.
- **For L1 the proximal operator is soft thresholding: zero out any value with |zᵢ| ≤ λ, and pull every larger value λ toward zero**, written compactly as sign(z)·max(|z|−λ, 0) - the single operation that creates exact zeros.
- **Soft thresholding (shrink everything) differs from hard thresholding (zero small values, leave the rest untouched)**, and it decomposes coordinate-by-coordinate, so all 784 pixels are thresholded independently in one cheap vectorized step.
- ==FISTA = proximal gradient + Nesterov momentum: gradient step on smooth f, soft-threshold the L1 part, then extrapolate to a look-ahead point==, which lifts convergence from O(1/k) to O(1/k²) at the same per-iteration cost; the momentum coefficient is approximated by k/(k+3).
- ==EAD splits into smooth f = c·max(Z_y − max_{j≠y}Z_j + κ, 0) + ‖x′−x‖₂² and non-smooth h = β‖x′−x‖₁==, where the L2 gradient is simply 2(x′−x) and κ is the confidence margin in the C&W hinge.
- ==Soft thresholding is applied to the perturbation δ = x′−x, not the image==, as prox(z) = x + 𝒮_{ηβ}(z−x), because the goal is a sparse _change_ (few pixels modified), not a sparse _image_ (few pixels bright).
- **Method point:** **when an objective has a smooth part and a non-smooth part, split them - gradient-step the smooth part and apply the matching proximal operator to the non-smooth part**, rather than forcing one method onto the whole thing; this split is the whole reason FISTA can do what plain gradient descent cannot.

Logged Section 4 and the self-check answer key. That was the hardest math in the ElasticNet half - soft thresholding is the one operation to really nail. Paste Section 5 (Distance Metrics) when ready.