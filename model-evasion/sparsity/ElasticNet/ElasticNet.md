# ElasticNet

## The idea in one breath

ElasticNet (**EAD**, Elastic-net Attacks to Deep neural networks, Chen et al. 2018) measures the change with **two rulers at once**: L2, which keeps it smooth and easy to optimize, plus L1, which pushes most pixel changes to exactly zero. A knob called **β** sets the balance between them. A second knob, **c**, sets how hard the attack pushes to fool the model, and **binary search** finds the smallest c that still works for each image. The section's key example: two changes with the same L2² of 1.00, where L1 (10.0 vs 3.16) is the only ruler that prefers the one touching fewer pixels.

> **Heads-up:** your paste lost the superscripts and subscripts:
> 
> - `∥x∥1=∑i|xi|` → ‖x‖₁ = Σᵢ |xᵢ|, meaning "L1 = add up the absolute value of every coordinate."
> - `100×0.102=1.00` → 100 × 0.10² = 1.00 (0.10 **squared**, not 0.102).
> - `10×0.3162≈1.00` → 10 × 0.316² ≈ 1.00 (0.316 **squared**).
> - "squared L2" means ‖δ‖₂², the L2 norm with the square root left off: just the sum of squared changes.

---

## Part 1 - Single-norm limitations

### Step 1: what each attack's ruler does to the change

|Attack|Ruler|What the change looks like|
|---|---|---|
|FGSM|L∞ (cap each pixel at ε)|**every** pixel moves the full ±ε, so it's visibly noisy|
|DeepFool / **C&W**|L2 (smallest overall size)|smooth, but **dense**: every pixel moves a little|
|Pure L1|sum of absolute changes|**sparse**: few pixels move, the rest stay exactly 0|

**Carlini & Wagner (C&W)** is the classic pure-L2 optimization attack. EAD is built on top of its framework, which is why the course names it.

### Step 2: why L1 zeroes things out and L2 doesn't (toy)

Think of each ruler as a **pull back toward zero** on every changed pixel. The question is how strong that pull is when a change is already small.

|Size of a pixel's change|L2² pull (2 × size)|L1 pull (constant β, say 1)|
|---|---|---|
|0.5|1.0|1|
|0.1|0.2|1|
|0.01|0.02 (almost nothing)|1 (still full strength)|

- **L2's pull fades as the change shrinks.** Near zero it barely pulls at all, so small changes linger at 0.01 or 0.001 forever. That's why L2 attacks are dense.
- **L1's pull never fades.** A tiny change still gets yanked with full force, so it goes all the way to **exactly 0**, unless that pixel is important enough for the attack to push back harder.

That's the whole secret: **L1 keeps pulling at full strength right down to zero, so unimportant pixels get dropped completely.**

### Step 3: the geometry picture the course mentions

Draw all 2-pixel changes with the same "size":

- With **L2** they form a **circle**, which is round everywhere.
- With **L1** they form a **diamond** with sharp corners sitting **on the axes**, at points like (1, 0) and (0, 1).

An optimizer sliding along that shape tends to stop at a corner, and every corner has one coordinate equal to **zero**. That's what the course means by **the "diamond-shaped constraint set with sharp corners along coordinate axes"**. It's the picture version of Step 2.

> **An honest note the course skips:** the course says "the gradient pushes toward solutions where most xᵢ = 0 exactly." With plain gradient descent that isn't quite true. At zero, L1 has a kink (no single slope), so plain steps **overshoot back and forth around 0** and land on 0.003, then −0.002, never exactly 0. The exact zeros come from FISTA's special **shrinkage step** (Sections 4 and 7), which snaps small values to 0 on purpose. The course gets to this later, when it says FISTA creates "true sparsity, not just near-zero."

---

## Part 2 - The sparsity - smoothness trade-off

### Toy first: two changes, same L2², different L1

A 4-pixel image, two candidate changes:

|Case|Change|L0|L2² (sum of squares)|L1 (sum of sizes)|
|---|---|---|---|---|
|A (dense)|[0.1, 0.1, 0.1, 0.1]|4|4 × 0.01 = **0.04**|4 × 0.1 = **0.4**|
|B (sparse)|[0.2, 0, 0, 0]|1|0.04 = **0.04**|**0.2**|

L2² **can't tell them apart** (both 0.04). L1 can: B scores half of A.

### The ElasticNet distance: add them with a knob

EAD's distance is **L2² + β × L1**:

|β|A: 0.04 + β × 0.4|B: 0.04 + β × 0.2|Winner|
|---|---|---|---|
|0|0.04|0.04|tie (pure L2 doesn't care)|
|1|0.44|0.24|**B** (sparse wins)|

==Bigger β means fewer pixels; β = 0 is just a pure L2 attack.==

### The real formula

```
elastic-net distance = ‖δ‖₂² + β · ‖δ‖₁ where δ = x′ − x
```

|Symbol|Plain English|Toy|Real (course)|
|---|---|---|---|
|δ = x′ − x|the change (adversarial minus original)|A or B|28×28 image changes|
|‖δ‖₂²|sum of squared changes (smoothness, energy)|0.04 for both|1.00 for both cases|
|‖δ‖₁|sum of absolute changes (the sparsity ruler)|0.4 vs 0.2|10.0 vs 3.16|
|β|sparsity knob|0 or 1|tuned in the lab|

**Read aloud:** "The ElasticNet distance is the change's squared size plus β times its total absolute size."

### HTB's real example (arithmetic checked: correct)

|Case|Pixels changed|Each by|L2²|L1|
|---|---|---|---|---|
|A|100|0.10|100 × 0.01 = **1.00**|100 × 0.10 = **10.0**|
|B|10|0.316|10 × 0.0999 ≈ **1.00**|10 × 0.316 = **3.16**|

Same story as the toy, scaled up. B uses **10× fewer pixels at the same energy**, and only the L1 term notices. B's pixels move **more** each (0.316 vs 0.10). That's the trade the course keeps describing: **a few larger edits instead of many tiny ones**.

> **An honest note the course skips:** the course says "the L2 component provides smooth gradients for stable optimization" as if L2 fixes L1's kink. It doesn't: adding L2 leaves the L1 term just as kinked at zero. What solves it is FISTA, which **splits** the problem into the smooth part (handled by an ordinary gradient step) and the L1 part (handled by shrinkage). L2's real jobs are keeping the total energy down and stopping a few pixels from getting enormous.

---

## Part 3 - The full ElasticNet attack

### Three goals in one objective (preview; built later in code)

Following the C&W framework, EAD minimizes:

```
c · f(x′) + ‖x′ − x‖₂² + β · ‖x′ − x‖₁ with x′ kept inside [0, 1]
```

|Goal|Term|What it does|
|---|---|---|
|Fool the model|c · f(x′)|f is the **adversarial loss** (Section 6). It's zero once the target class wins with a margin, and positive otherwise|
|Small, sparse change|‖x′−x‖₂² + β‖x′−x‖₁|the elastic-net distance from Part 2|
|Valid image|x′ ∈ [0, 1]|the box constraint, same as every attack so far|

### FISTA in one line

**FISTA (Fast Iterative Shrinkage-Thresholding Algorithm)** is the solver. Each iteration it (1) takes a normal gradient step on the smooth parts, (2) applies **shrinkage** to the L1 part, which snaps small changes to exactly 0, and (3) adds **momentum**, a push in the direction it was already going, to converge faster. The name says it all: _iterative_, _shrinkage_, _thresholding_. Section 4 opens it up with a toy example.

---

## Part 4 - Binary search on c

### What c does

|c|Effect|
|---|---|
|Large|"fool it at any cost" → succeeds, but the change is bigger than needed|
|Small|"keep it tiny" → change is small, but the attack may **fail**|

You want the **smallest c that still succeeds**, and that's different for every image.

### Toy: guessing c on one image (range 0 to 10)

|Try|c|Result|New bounds (lower, upper)|
|---|---|---|---|
|1|5|success|(0, **5**): success, so try smaller|
|2|2.5|fail|(**2.5**, 5): fail, so need bigger|
|3|3.75|success|(2.5, **3.75**)|
|4|3.125|...|each try halves the range|

The rule: ==success → lower the upper bound; failure → raise the lower bound; next c = the middle==. After about 9 tries, the range is under 1% of where it started.

### Why this beats FGSM's fixed ε

FGSM uses **one ε for every image**: too big for images near a boundary, too small for far ones. Binary search **finds each image's own vulnerability threshold**. Near-boundary images settle on a small c, and far ones get a bigger c, with no manual tuning.

> **An honest note the course skips:** real C&W and EAD code doesn't bisect from the start. While it has **never** succeeded (no upper bound yet), it **multiplies c by 10** each round, and only bisects once a success gives it an upper bound. Watch for that `× 10` line when Section 9's code arrives.

---

## Part 5 - Contrasts with previous attacks

|Aspect|FGSM|DeepFool|ElasticNet (EAD)|
|---|---|---|---|
|Steps|1|a few (linearize, hop, repeat)|many (FISTA iterations) × binary-search rounds|
|Ruler|L∞ (fixed ε)|L2 (minimal)|L2² + β·L1 (mixed)|
|Sparse?|no, every pixel moves ±ε|no, dense|**yes**, most pixels exactly 0|
|Strength tuning|one ε for all|none needed (stops at boundary)|per-image c via binary search|
|Best used for|speed, **adversarial training** data|**precise robustness measurement** (L2 lower bound)|**evading detection, transfer, finding which pixels matter**|

---

## Copy-paste cells (mine, not the course's)

These are standalone and need only numpy.

### Measure the course's Case A and Case B

Builds both 28×28 changes and prints L0, L1 and L2².

```python
import numpy as np
a = np.zeros(784); a[:100] = 0.10 # Case A: 100 pixels changed by 0.10
b = np.zeros(784); b[:10] = np.sqrt(0.1) # Case B: 10 pixels changed by 0.316
for name, d in [("A", a), ("B", b)]:
    print(f"{name} | L0 {np.count_nonzero(d)} | L1 {np.abs(d).sum():.2f} | L2^2 {(d**2).sum():.2f}")
```

Expected:

```txt
A | L0 100 | L1 10.00 | L2^2 1.00
B | L0 10 | L1 3.16 | L2^2 1.00
```

### Watch β flip the preference

Computes the elastic-net distance for several β (run after the cell above).

```python
for beta in [0.0, 0.01, 0.1, 1.0]:
    dA = (a**2).sum() + beta * np.abs(a).sum() # elastic-net distance of A
    dB = (b**2).sum() + beta * np.abs(b).sum() # elastic-net distance of B
    pick = "tie" if np.isclose(dA, dB) else ("B (sparse)" if dB < dA else "A (dense)")
    print(f"beta={beta:<4} | A {dA:.3f} | B {dB:.3f} | prefers {pick}")
```

Expected:

```txt
beta=0.0 | A 1.000 | B 1.000 | prefers tie
beta=0.01 | A 1.100 | B 1.032 | prefers B (sparse)
beta=0.1 | A 2.000 | B 1.316 | prefers B (sparse)
beta=1.0 | A 11.000 | B 4.162 | prefers B (sparse)
```

Even a tiny β of 0.01 is enough to break the tie in favour of the sparse change.

---

## Quick self-check

Same 4-pixel toy. A new Case C changes **2 pixels by 0.15 each**: [0.15, 0.15, 0, 0]. With **β = 1**, what is C's elastic-net distance, and where does it rank against A (0.44) and B (0.24)?

---

## The facts worth locking in for the exam

- **EAD (Chen et al., 2018) adds an L1 term to the C&W L2 framework**, minimizing c·f(x′) + ‖x′−x‖₂² + β·‖x′−x‖₁ inside the [0, 1] box, so the change is both smooth (L2²) and sparse (L1).
- **L1 produces exact zeros because its pull toward zero stays at full strength (β) no matter how small a change gets**, while L2's pull (2 × size) fades near zero, which is why pure L2 attacks like C&W and DeepFool are dense.
- **Two changes with the same L2² can have very different L1**, as in the course's example: 100 pixels × 0.10 and 10 pixels × 0.316 both have L2² = 1.00, but L1 is 10.0 vs 3.16, so only the L1 term prefers the sparse one.
- **β is the sparsity knob**: β = 0 makes EAD a pure L2 attack, and a larger β tilts it toward fewer, larger pixel changes (even β = 0.01 already favours the sparse case in the example).
- **c trades attack success against change size, and binary search finds the smallest c that still succeeds for each image**: success lowers the upper bound, failure raises the lower bound, and in real code c is multiplied by 10 until a first success gives an upper bound.
- **Exact zeros come from FISTA's shrinkage step, not from plain gradient descent**, which only oscillates around zero at L1's kink. FISTA does a gradient step on the smooth parts, shrinkage on the L1 part, and adds momentum.
- **Pick the attack by its purpose**: FGSM for speed and adversarial-training data, DeepFool for precise L2 robustness measurement, and EAD for sparse changes that evade noise-based detectors, transfer between models, or reveal which pixels matter.
- **Method point:** **when comparing two candidate changes, measure them with every ruler (L0, L1, L2², L∞) before judging**, because two changes can look identical under one ruler (here L2²) and completely different under another (L1 and L0).

I've logged Section 2 and the self-check answer key to your progress file. Paste Section 3 (Environment Setup) when you're ready.