# Understanding Norms

## The idea in one breath

A **norm** is a ruler that turns "how much did I change the image?" into one number. Different rulers measure in different ways: **L0** counts how many pixels you touched, **L1** adds up all the changes, **L2** gives the straight-line distance, and **L∞** reports only the single biggest change. The ruler you choose shapes what the attack looks like, so FGSM (L∞) and DeepFool (L2) produce very different perturbations.
![The idea in one breath](/assets/Pasted%20image%2020261013151032.png)


## Part 1: The toy example, built up one ruler at a time

### Step 1: What gets measured is the change (δ), not the image

We're back to a tiny image, this time with **4 pixels**:

```
original x = [0.2, 0.6, 0.5, 0.9]
attacked   = [0.5, 0.2, 0.5, 0.9]
```

The **perturbation δ (delta)** is just "attacked minus original", one pixel at a time:

```
δ = [0.5−0.2,  0.2−0.6,  0.5−0.5,  0.9−0.9]
  = [+0.3,     −0.4,     0,        0]
```

Every norm below measures **δ**, the change you made, not the picture itself. In plain words: pixel 1 got brighter by 0.3, pixel 2 got darker by 0.4, and pixels 3 and 4 weren't touched.

### Step 2: L0 asks "how many pixels did I touch?"
![Step 2: L0 asks "how many pixels did I touch?"](/assets/Pasted%20image%2020261013151229.png)
Count the entries that aren't zero:

```
δ = [+0.3, −0.4, 0, 0]
      ✓     ✓    ✗  ✗    → L0 = 2
```

The size of each change doesn't matter at all. A change of 0.001 counts as 1, and so does a change of 255.

### Step 3: L1 asks "what's my total change?"
![Understanding Norms](/assets/2%20Understanding%20Norms-20261013151240341.png)
Drop the minus signs (take the **absolute value**, written |...|) and add everything up:

```
L1 = |0.3| + |−0.4| + |0| + |0| = 0.3 + 0.4 = 0.7
```

**Why drop the signs?** Brightening by 0.3 and darkening by 0.3 are both changes. Without absolute values they would cancel out to 0, and the ruler would claim you changed nothing.

### Step 4: L2 asks "how far is it in a straight line?"
![Step 4: L2 asks "how far is it in a straight line?"](/assets/Pasted%20image%2020261013151253.png)
Square each change, add the squares, then take the square root:

```
L2 = √(0.3² + 0.4² + 0² + 0²) = √(0.09 + 0.16) = √0.25 = 0.5
```

This is Pythagoras: legs of 0.3 and 0.4 give a hypotenuse of 0.5 (the classic 3-4-5 triangle, scaled down).

### Step 5: L∞ asks "what's the single biggest change?"
![Understanding Norms](/assets/2%20Understanding%20Norms-20261013151303272.png)
```
L∞ = max(|0.3|, |−0.4|, 0, 0) = 0.4
```

### Step 6: All four rulers on the same change

|Ruler|Question it answers|Our δ|
|---|---|---|
|L0|How many pixels did I touch?|**2**|
|L1|What's the total change?|**0.7**|
|L2|Straight-line size?|**0.5**|
|L∞|Biggest single change?|**0.4**|

**It's the same perturbation, but it gets four different "sizes" depending on the ruler.** When a paper or a lab says "ε = 0.1", you always need to ask which norm it's measured in.

---
## Part 2: The p-norm formula (one recipe for L1, L2 and L∞)

### Step 7: One formula with a dial

L1 and L2 are the same recipe with a different setting on a dial called **p**. Correct form (your paste dropped the superscripts):

```
‖δ‖p = ( Σ |δi|^p )^(1/p)
```

|Symbol|Plain English|
|---|---|
|`‖δ‖p`|"the size of δ, measured with ruler p"|
|`δi`|the change to pixel number i|
|`\|δi\|`|that change without its minus sign|
|`^p`|raise it to the power p|
|`Σ`|add them up across all pixels i = 1 ... n|
|`^(1/p)`|take the p-th root (undo the power)|

Read aloud: **"Raise each change to the power p, add them all up, then take the p-th root."**

### Step 8: Turn the dial and watch what happens

Using δ = [0.3, −0.4, 0, 0]:

|p|Calculation|Result|
|---|---|---|
|1|(0.3 + 0.4)^1|**0.700** (that's L1)|
|2|(0.09 + 0.16)^(1/2)|**0.500** (that's L2)|
|4|(0.0081 + 0.0256)^(1/4)|**0.428**|
|10|(0.0000059 + 0.000105)^(1/10)|**0.402**|
|∞|just the biggest one|**0.400** (that's L∞)|

As p grows, the answer slides toward the biggest single change, 0.4.

**Why?** Raising to a power makes the big number dominate. At p = 10, 0.4¹⁰ is about **18 times** larger than 0.3¹⁰, so the 0.3 hardly counts anymore. ==As p increases, the norm cares more and more about the largest change; at p = ∞ only the largest change matters.== At p = 1 every change counts equally, and at p = 2 the squaring already gives big changes extra weight.

### Step 9: Why L0 is an "imposter"

Try plugging p = 0 into the formula: the last step needs the "1/0-th root," which doesn't exist. So L0 gets its own separate definition:

```
‖δ‖0 = Σ 𝟙[δi ≠ 0]
```

|Symbol|Plain English|
|---|---|
|`𝟙[...]`|**indicator function**: gives 1 if the statement in brackets is true, 0 if it's false|
|`δi ≠ 0`|"pixel i was changed"|
|`Σ`|add up all the 1s, which gives the count of changed pixels|

## Part 3: The three rules a real norm must follow

We'll test them on our toy δ.

|Rule|What it says|Toy check|
|---|---|---|
|1. Zero means zero|Size 0 only if nothing changed|δ = [0,0,0,0] → every ruler gives 0 ✓|
|2. Doubling means doubling|2× the change gives 2× the size|2δ = [0.6, −0.8, 0, 0] → L1 1.4, L2 1.0, L∞ 0.8, all doubled ✓. **But L0 stays 2** ✗|
|3. Triangle inequality|A direct path is never longer than a detour|Do pixel 1 alone (size 0.3), then pixel 2 alone (size 0.4): 0.3 + 0.4 = 0.7 ≥ the direct L2 of 0.5 ✓|

**L0 fails the "doubling" rule, so it isn't a true norm**: doubling your changes doesn't double the number of pixels you touched. Everyone still calls it a norm because it captures **sparsity** (how many things you touched), which no p-norm measures.

## Part 4: How the norms relate to each other

### Step 10: The size-ordering chain

Correct form (your paste dropped the √ signs):

```
‖δ‖∞  ≤  ‖δ‖2  ≤  ‖δ‖1  ≤  √n · ‖δ‖2  ≤  n · ‖δ‖∞
```

Here **n** is the number of pixels. Check it with the toy (n = 4):

```
0.4  ≤  0.5  ≤  0.7  ≤  √4 × 0.5 = 1.0  ≤  4 × 0.4 = 1.6   ✓
```

What it means in plain words: **"if you cap one ruler, the others are automatically capped too."** The cap gets looser as n grows.

And for L0 (again with the √ restored):

```
‖δ‖0 ≤ n          and          ‖δ‖2 ≤ √(‖δ‖0) · ‖δ‖∞
```

Toy check: √2 × 0.4 = 0.566 ≥ 0.5 ✓. The first part just says you can't touch more pixels than exist. The second says a count alone limits nothing. **L0 only limits the size when you combine it with a per-pixel cap (L∞ or the valid [0,1] range)**, because "I changed 2 pixels" says nothing about _how much_.

### Step 11: Why this matters for FGSM (the real numbers)

Apply the chain to MNIST: n = 784 pixels and an FGSM budget of ε = 0.1 in L∞:

```
L2  ≤ √784 × 0.1 = 28 × 0.1 = 2.8
L1  ≤  784 × 0.1 = 78.4
```

FGSM moves **every** pixel by exactly ε, so it actually hits these maximums. **A perturbation that's "tiny" under L∞ (0.1) can be large under L2 (2.8) and L1 (78.4).** Always report which norm you're quoting.

## Part 5: The shapes of the budgets

### Step 12: Two pixels and a budget of 1

With only 2 pixels you can draw every allowed change (a, b) on paper:

|Norm|Rule for (a, b)|Shape|Is (0.7, 0.7) allowed?|Is (1, 0) allowed?|
|---|---|---|---|---|
|L1|\|a\| + \|b\| ≤ 1|◆ diamond|1.4 > 1 → **no**|1 → on the edge|
|L2|√(a² + b²) ≤ 1|● circle|0.99 → **yes**, just|1 → on the edge|
|L∞|max(\|a\|, \|b\|) ≤ 1|■ square|0.7 → **yes**|1 → on the edge|

Three insights fall out of that table:

- **L1's diamond has its corners on the axes**, at (1, 0) and (0, 1). The cheapest way to spend an L1 budget is often "all on one pixel, zero on the others." That's why **L1 tends to give **sparse** perturbations** (many pixels left exactly 0).
- **L2's circle** has no corners, so it treats every direction the same. The squaring punishes any single big change, so L2 spreads the change thinly across many pixels.
- **L∞'s square has its corners at (±1, ±1)**, where every pixel is at its maximum. **FGSM's `ε · sign(gradient)` always lands exactly on a corner of the L∞ square.** That's the geometric picture behind "every pixel moves the full ε."

## Part 6: How each norm behaves during optimization

### Step 13: Can a gradient "flow" through it?

Attacks follow slopes, so the question is whether each ruler has a usable slope.

|Norm|Slope behaviour|Toy illustration|How attacks handle it|
|---|---|---|---|
|L0|No slope; it jumps 0 → 1|Nudging 0 → 0.001 jumps the count from 0 to 1|Greedy or combinatorial search (like your GoodWords greedy search)|
|L1|A "kink" at 0|\|x\| is a V shape: slope −1 on the left, +1 on the right, undefined at 0|Soft-thresholding, proximal gradient methods|
|L2|Smooth everywhere (squared L2)|The slope of δ² is just 2δ, no gaps|Plain gradient descent; **the "vanilla ice cream" norm**|
|L∞|Slope only on the biggest entry|Nudge the 0.3 → L∞ stays 0.4 (slope 0). Nudge the −0.4 → L∞ changes. Only one pixel "has a gradient"|**Projected gradient descent**: take a step, then **clip back** into the square|

That "clip back into the square" is exactly what I-FGSM does after each step later in this module.

## Part 7: The real examples (HTB's figures)

### Figure 1: "Navigating the City"

The trip goes from the **green** dot to the **red** dot: **5 blocks east and 4 blocks north**.

|Panel|Calculation|HTB's number|
|---|---|---|
|L0 (Sparsity)|Two directions were used (east and north)|**2**|
|L1 (Manhattan)|Walk the streets: 5 + 4|**9.00**|
|L2 (Euclidean)|Fly straight: √(5² + 4²) = √41|**6.40**|
|L∞ (Chebyshev)|Longest single leg: max(5, 4)|**5.00**|

This is exactly our toy ordering: 5 ≤ 6.40 ≤ 9, which is L∞ ≤ L2 ≤ L1. The purple box in the L∞ panel shows that everything inside it costs at most 5.

### Figure 2: "The Shape of Constraints"

Each panel shows the original point in white, with budgets of 0.5 (inner) and 1.3 (outer).

- **L0:** teal dots are spots you can reach by changing 1 coordinate (they sit on the lines through the original). Green dots need 2 coordinates changed. Notice there's no "distance" ring, only how many coordinates moved.
- **L1:** a diamond, **L2:** a circle, **L∞:** a square, exactly matching the Step 12 table. As the budget grows, each shape just gets bigger.

### Figures 3 and 4: L0 vs L1 on the letter "H"

- **L0 perturbation:** about 20 pixels, all at **full strength** (solid red = brightened, solid blue = darkened). In the adversarial image you can see bright specks in the background and dark holes inside the H. That's the "sore thumb" drawback.
- **L1 perturbation:** somewhat more pixels, with **varying strengths** (faint and strong reds and blues). It's still sparse, but more of a "salt-and-pepper" look.

### Figure 5: "Gradient Flow"
![Figure 5: "Gradient Flow"](/assets/Pasted%20image%2020261013151620.png)
The white path goes from the centre toward the red target.

- **L0:** jumps between separate blobs, which is combinatorial search.
- **L1:** goes along the x₁ axis first, then turns up x₂. It "follows the creases" because of the diamond's corners.
- **L2:** one smooth straight line.
- **L∞:** goes diagonally (all coordinates move together, like FGSM), hits the square's edge, then **slides along the boundary**. That slide is the projection step.

### Figure 6: "Visual Effect"
![Figure 6: "Visual Effect"](/assets/Pasted%20image%2020261013151625.png)
- **L0 and L1:** a few scattered specks; the rest of the image is untouched.
- **L2 and L∞:** a fine grain or haze over the **whole** image, with every pixel changed a little. This is what FGSM output looks like.

## Quick self-check

Take the doubled change **δ = [0.6, 0, −0.8, 0]**. What are its **L0, L1, L2 and L∞**?

Reply with all four and I'll check them.

## The facts worth locking in for the exam

- **A norm measures the size of the perturbation δ (adversarial minus original), not the image itself**. The same δ gets different sizes under different norms, so "ε = 0.1" means nothing until you say which norm it's in.
- ==The p-norm is ‖δ‖p = (Σ|δi|^p)^(1/p): p = 1 gives the total absolute change, p = 2 gives the Euclidean straight-line size, and p = ∞ gives only the largest single change==. As p increases, the largest entries dominate more and more.
- ==L0 counts the number of changed elements, ‖δ‖0 = Σ𝟙[δi ≠ 0], and is not a true norm because it breaks the scaling ("doubling") rule==. We still use it because it captures pure sparsity, and it only limits size when paired with a per-pixel cap like L∞ or the valid pixel range.
- **The norms are ordered ‖δ‖∞ ≤ ‖δ‖2 ≤ ‖δ‖1 ≤ √n·‖δ‖2 ≤ n·‖δ‖∞**. On MNIST (n = 784) an L∞ budget of 0.1 allows an L2 of up to 2.8 and an L1 of up to 78.4, and FGSM actually reaches those maximums.
- **The constraint shapes are an L1 diamond, an L2 circle (ball) and an L∞ square (box)**. The diamond's corners on the axes are why L1 gives sparse perturbations. The square's corners are where FGSM lands, since every pixel is at ±ε.
- **L2 perturbations spread a faint haze across all pixels and are the easiest to optimize (squared L2 is smooth everywhere)**, while L0 and L1 give a few visible specks, and L∞ changes many pixels by similar amounts up to the cap.
- **L0 needs greedy or combinatorial search, L1 needs special methods for its kink at zero (soft-thresholding or proximal gradient), and L∞ has a gradient only on its largest element, so it is handled with projected gradient descent** (step, then clip back into the box).
- **Method point:** ==when reporting an evasion finding, state the norm and the budget together (for example "L∞, ε = 0.1"), and if you can, report the other norms too==, because a perturbation that's small under one ruler can be large under another.