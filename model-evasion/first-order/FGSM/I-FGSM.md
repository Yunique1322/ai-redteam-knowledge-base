# I-FGSM

## The idea in one breath

FGSM takes **one big step** based on the slope at the starting point. **I-FGSM** (also called the **Basic Iterative Method, BIM**; Kurakin et al., 2016) takes **many small steps**, re-reading the slope after each one. After every step it pulls the image back inside the same ε budget around the **original** image (the **projection**). Same budget, better aim: in HTB's illustration, 10 small steps raised the loss **53% more** than one FGSM step with the same ε.

> **Heads-up: the formulas got flattened in your paste.** Read them like this:
> 
> - `x(0)=x` means **x⁽⁰⁾ = x**. The superscript **(t)** is a **step counter** ("the image after step t"), _not_ a power. x⁽³⁾ means "after 3 steps," not "x cubed."
> - `x(t+1)=Πℬ∞(x,ϵ)(x(t)+αsign(∇x(t)ℒ(θ,x(t),y)))` means **x⁽ᵗ⁺¹⁾ = Π_B∞(x,ε)( x⁽ᵗ⁾ + α·sign(∇ₓℒ(θ, x⁽ᵗ⁾, y)) )**
> - `α=ϵT` means **α = ε / T**
> - `Πℬ∞(x,ϵ)(x′)=x+clip(x′−x,−ϵ,ϵ)` means **Π(x′) = x + clip(x′ − x, −ε, ε)**, followed by **x′ ← clip(x′, x_min, x_max)**

## Part 1: Why one big step can go wrong (toy)

### Step 1: A curved "hill" for one pixel

Imagine the loss for one pixel is a hill with its top at x = 0.3:

```
L(x) = 1 − (x − 0.3)² slope(x) = −2(x − 0.3)

start x = 0: L = 1 − 0.09 = 0.91, slope = +0.6 → sign +1 ("go right")
```

The attacker wants the loss as high as possible, so the top of the hill (L = 1.0 at x = 0.3) is the best spot.

### Step 2: FGSM, one big stride with ε = 0.8

```
x = 0 + 0.8 × (+1) = 0.8
L = 1 − (0.8 − 0.3)² = 1 − 0.25 = 0.75 ← LOWER than the start (0.91)!
```

The slope at the start said "go right," and FGSM went right by the **whole** budget. It walked straight over the hilltop and down the other side. **A single step trusts the slope at the starting point all the way, even after the ground has curved.** That's the "flat plane approximating a curved landscape" problem from Section 3.

### Step 3: I-FGSM, eight small steps of α = 0.1

Same budget ε = 0.8 and T = 8 steps, so α = 0.8 / 8 = 0.1. Re-read the slope every time:

|Step t|Slope at current x|Sign|New x|Loss|
|---|---|---|---|---|
|start|+0.6| - |0.0|0.91|
|1|+0.6|+1|0.1|0.96|
|2|+0.4|+1|0.2|0.99|
|3|+0.2|+1|0.3|**1.00** (the top)|
|4-8|≈ 0, tiny ±|flips|jitters 0.3 ↔ 0.2|0.99-1.00|

**Because it re-checks the slope after every step, I-FGSM stops climbing at the top instead of overshooting.** FGSM ended at 0.75, while I-FGSM ends at about 0.99-1.00, with the same ε. The little back-and-forth at the end is normal: once you're at the top, any step is a bit downhill, so the sign keeps flipping.

**Analogy:** crossing a dark room toward a door. FGSM points once and walks blind for 8 metres. I-FGSM takes a step, feels around, re-aims, and repeats.

## Part 2: The projection, "stay inside the budget" (toy)

### Step 4: Why you need it

Many small steps all pointing the same way could add up to more than ε. The **projection** stops that: after every step, **measure how far you've drifted from the ORIGINAL image and cap it at ε.**

One normalized pixel, original value **0.5**, ε = **0.25**, α = **0.1**, and suppose the gradient signs over 4 steps are **+, +, +, −**:

|Step|After the α step|Drift from original (x′ − x)|clip(drift, −0.25, 0.25)|Projected value|
|---|---|---|---|---|
|1|0.5 + 0.1 = 0.6|+0.1|+0.1|**0.6**|
|2|0.6 + 0.1 = 0.7|+0.2|+0.2|**0.7**|
|3|0.7 + 0.1 = 0.8|+0.3|**+0.25** (capped)|**0.75**|
|4|0.75 − 0.1 = 0.65|+0.15|+0.15|**0.65**|

At step 3 the pixel tried to drift 0.3 from the original, more than ε = 0.25, so it was pulled back to 0.5 + 0.25 = 0.75. At step 4 the gradient changed its mind, and the pixel stepped back down. **Unlike FGSM, I-FGSM pixels can move back and forth, and end anywhere inside [x − ε, x + ε], not only at the corners ±ε.**

### Step 5: The second clip, the valid range

After the ε-cap, the pixel must also stay a real image value. In normalized MNIST that's [−0.424, 2.821] (Section 5). A black pixel at −0.424 pushed down stays at −0.424. That's the `x′ ← clip(x′, x_min, x_max)` part.

> **The classic bug:** clipping relative to the **previous** step (x⁽ᵗ⁾) instead of the **original** x. Each step would then be allowed its own ε, and after 10 steps the image could drift 10ε. **Always measure drift from the original image.**

### Step 6: The real formulas

```
x⁽⁰⁾ = x
x⁽ᵗ⁺¹⁾ = Π_B∞(x,ε)( x⁽ᵗ⁾ + α · sign( ∇ₓ ℒ(θ, x⁽ᵗ⁾, y) ) )

Π(x′) = x + clip(x′ − x, −ε, ε) then clip(·, x_min, x_max)
α = ε / T (typical choice)
```

|Symbol|Plain English|Toy (Part 2)|Real (HTB)|
|---|---|---|---|
|`x`|the **original** clean image (never changes)|0.5|an MNIST digit|
|`x⁽ᵗ⁾`|the image after t steps|0.6, 0.7, ...||
|`T`|number of steps (iterations)|4|10|
|`α` (alpha)|size of each small step|0.1|0.8 / 10 = **0.08**|
|`ε`|total budget per pixel (same meaning as FGSM)|0.25|0.8 (normalized)|
|`∇ₓℒ(θ, x⁽ᵗ⁾, y)`|the gradient **re-computed at the current image**|||
|`B∞(x, ε)`|the L∞ "box" of all images within ε of x per pixel (Section 2's square)|[0.25, 0.75]||
|`Π` (capital pi)|**projection**: "snap back into the box"|the cap at step 3||
|`clip(a, lo, hi)`|anything below lo becomes lo, above hi becomes hi|clip(0.3, −0.25, 0.25) = 0.25||
|`x_min, x_max`|valid pixel range||−0.424, 2.821|

Read aloud: **"Take a small sign step using the gradient at where you are now, then snap back inside the ε-box around the original image and the valid pixel range; repeat T times."**

**Targeted I-FGSM** is the same idea as Section 9, applied every step: use the target label yₜ and **subtract** the step:

```
x⁽ᵗ⁺¹⁾ = Π( x⁽ᵗ⁾ − α · sign( ∇ₓ ℒ(θ, x⁽ᵗ⁾, yₜ) ) )
```

> **An honest note the course skips:** with α = ε/T exactly, T steps can move a pixel **at most** T·α = ε in total, so the **ε-box clip can never actually trigger**. Only the valid-range clip can. The ε-projection starts to matter when α·T > ε, which is common in practice (people often pick α a bit larger so the attack can explore and backtrack). My toy used α·T = 0.4 > ε = 0.25 on purpose so you could see it work. Keep the projection in your code regardless: it's what guarantees the budget.

## Part 3: Why iteration helps, in the course's words

- The **first** step from x is FGSM's step, the best move **for the flat-plane approximation** at x (Section 3).
- After moving, that flat plane no longer fits the real curved loss. **Recomputing the gradient at x⁽ᵗ⁾ re-fits the plane to where you are now.**
- Over several steps, this follows the bends of the loss surface and pushes the image closer to (and across) the decision boundary.
- **The gains come from better directions, not a bigger budget**, because the projection holds the same ε.

**The price:** FGSM needs **1** gradient computation; I-FGSM needs **T** (10 steps = 10 forward and backward passes, about 10× the time). FGSM is fast, while I-FGSM is stronger.

**Related vocabulary:** **PGD (Projected Gradient Descent, Madry et al.)** is essentially I-FGSM plus a **random starting point** inside the ε-box. It's the standard strong first-order attack, and you'll meet the name often.

## Part 4: Reading HTB's loss example

The course compares FGSM and I-FGSM with the same ε = 0.8, using 10 iterations at α = 0.08:

|Attack|Loss path|Final|Increase|
|---|---|---|---|
|FGSM (1 step)|0.3 → 1.2|1.2|**0.9**|
|I-FGSM (10 steps)|0.3 → 0.45 → 0.62 → 0.81 → 0.98 → 1.15 → 1.31 → 1.44 → 1.55 → 1.63 → 1.68|1.68|**1.38**|

```
extra gain = 1.38 / 0.9 = 1.53 → 53% more loss increase with the same budget
```

Look at the per-step gains: 0.15, 0.17, 0.19, 0.17, 0.17, 0.16, 0.13, 0.11, 0.08, 0.05. They **shrink at the end**: **diminishing returns** as the attack runs out of room inside the ε-box, just like the hill toy flattening at the top. The course says the loss "might evolve" this way, so treat these as illustrative numbers, not a measured run.

## Copy-paste cells (mine, not the course's)

The I-FGSM code itself comes in Section 11. These two cells just let you see the mechanics with tiny numbers.

### Watch the projection cap a pixel

Replays the Part 2 toy with real `torch.clamp` calls, the same functions the implementation will use.

```python
x = torch.tensor([0.5])
eps, alpha = 0.25, 0.1
x_t = x.clone()
for s in [+1, +1, +1, -1]:
    x_t = x_t + alpha * s
    x_t = x + torch.clamp(x_t - x, -eps, eps)
    x_t = torch.clamp(x_t, -0.424, 2.821)
    print(round(x_t.item(), 2))
```

Expected:

```
0.6
0.7
0.75
0.65
```

### FGSM vs I-FGSM on the hill

Replays the Part 1 toy: one big step overshoots, small steps find the top.

```python
def L(x): return 1 - (x - 0.3) ** 2
def slope(x): return -2 * (x - 0.3)

x0, eps = 0.0, 0.8
x_fgsm = x0 + eps * np.sign(slope(x0))
print("FGSM:", round(x_fgsm, 2), round(L(x_fgsm), 3))

x, alpha = x0, eps / 8
for t in range(8):
    x = x + alpha * np.sign(slope(x))
    x = x0 + np.clip(x - x0, -eps, eps)
print("I-FGSM:", round(float(x), 2), round(float(L(x)), 3))
```

Expected (I checked this one by running it):

```
FGSM: 0.8 0.75
I-FGSM: 0.2 0.99
```

I-FGSM ends at 0.2 rather than 0.3 because, after reaching the top, it jitters 0.3 ↔ 0.2 and stops on an even step. Either way its loss (0.99) beats FGSM's 0.75.

## Quick self-check

One normalized pixel starts at **0.5**, with **ε = 0.3** and **α = 0.2**. The gradient signs over three steps are **+, +, −**. Using the projection (drift measured from the **original** 0.5), what is the pixel's value after each step, and where does it end?

Reply with your three values and I'll check them.

## The facts worth locking in for the exam

- **I-FGSM (also called BIM, Kurakin et al. 2016) repeats small FGSM steps of size α, recomputing the gradient at the current image x⁽ᵗ⁾ each time, and projects back into the ε-box after every step**, giving stronger attacks than one FGSM step with the same budget.
- ==The update is x⁽ᵗ⁺¹⁾ = Π_B∞(x,ε)(x⁽ᵗ⁾ + α·sign(∇ₓℒ(θ, x⁽ᵗ⁾, y))), with α typically ε/T==, and the superscript (t) is a step counter, not an exponent.
- **The projection is per-pixel clipping relative to the ORIGINAL image, x + clip(x′ − x, −ε, ε), followed by clipping to the valid input range**. Clipping relative to the previous step instead would let the total perturbation grow to T·α and break the budget.
- **Iteration helps because one first-order step trusts a flat-plane approximation that stops fitting once you move**. Re-reading the slope tracks the curved loss and avoids overshooting, so the gains come from better direction, not a larger ε.
- **Unlike FGSM, whose pixels always land at exactly ±ε, I-FGSM pixels can move back and forth and finish anywhere inside [x − ε, x + ε]**.
- ==Targeted I-FGSM uses the target label yₜ and subtracts the step each iteration, x⁽ᵗ⁺¹⁾ = Π(x⁽ᵗ⁾ − α·sign(∇ₓℒ(θ, x⁽ᵗ⁾, yₜ)))==, the same two changes as targeted FGSM, applied repeatedly.
- ==In HTB's illustration, 10 steps of α = 0.08 at ε = 0.8 raised the loss by 1.38 versus 0.9 for single-step FGSM, about 53% more==, with per-step gains shrinking near the end (diminishing returns). The cost is T gradient computations instead of one.
- **Method point:** **when comparing attacks, hold ε fixed and compare loss increase or success rate, and always verify that the final (x_adv − x).abs().max() ≤ ε**, so any improvement is proven to come from the method rather than a bigger budget.

