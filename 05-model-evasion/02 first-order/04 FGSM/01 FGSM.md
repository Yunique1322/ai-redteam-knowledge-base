# FGSM

## The idea in one breath

FGSM asks the model "which way should each pixel move to make you most wrong?" It keeps only the **direction** (+ or −) for each pixel, and moves every pixel by exactly ε in that direction, in one step. This section proves that move is the **best possible** single step when your budget is "no pixel changes by more than ε" (L∞). It also explains why such small moves add up to a flipped prediction.

> **Heads-up: a few formulas were mangled when you pasted them.** Read them like this:
> 
> - `w⊤x` means w-transpose-x, the **dot product** (explained below)
> - `1p+1q=1` means **1/p + 1/q = 1**
> - `1∞+11` means **1/∞ + 1/1**
> - `δ2*=ϵg||g||2` means **δ₂* = ε · g / ‖g‖₂**

## Part 1: Why FGSM works, part 1: small changes add up (the ε‖w‖₁ bound)

### Step 1: The dot product, which is all "w⊤x" means

The section writes the toy model's score as `f(x) = w⊤x`. That's just our familiar recipe: multiply each weight by its pixel and add everything up.

```
w = [2, −3, 1]      x = [0.5, 0.2, 0.4]
w⊤x = 2(0.5) + (−3)(0.2) + 1(0.4) = 1.0 − 0.6 + 0.4 = 0.8
```

==`w⊤x` = "multiply pair by pair, then add" = the dot product.== The score is called a **logit**: the model's raw score for a class, before it gets turned into a probability.

### Step 2: How much does a perturbation δ move the score?

Add a change δ and the score becomes w⊤(x + δ) = w⊤x + w⊤δ. The original 0.8 is still there, so **the **change** in score is just w⊤δ**:

```
δ = [−0.1, +0.1, −0.1]
w⊤δ = 2(−0.1) + (−3)(+0.1) + 1(−0.1) = −0.2 − 0.3 − 0.1 = −0.6
```

### Step 3: What's the most any δ inside the budget can do?

Our rule is that each pixel moves at most ε = 0.1. Try some options:

|Choice of δ|Calculation|Score change|
|---|---|---|
|[−0.1, +0.1, 0]|−0.2 − 0.3 + 0|−0.5|
|[−0.05, +0.1, −0.1]|−0.1 − 0.3 − 0.1|−0.5|
|[+0.1, +0.1, +0.1]|+0.2 − 0.3 + 0.1|0.0|
|**[−0.1, +0.1, −0.1]** (FGSM)|−0.2 − 0.3 − 0.1|**−0.6** <- the biggest|

The biggest possible move is ε × (|2| + |−3| + |1|) = 0.1 × 6 = 0.6. That "6" is the **L1 norm of the weights**, ‖w‖₁, from Section 2.

### Step 4: The real formula

```
|f(x+δ) − f(x)| = |w⊤δ| ≤ ε‖w‖₁
```

|Symbol|Plain English|In our toy|
|---|---|---|
|`f(x+δ) − f(x)`|how much the score moved|−0.6|
|`\|...\|`|ignore the sign, just the size|0.6|
|`‖δ‖∞ ≤ ε`|no pixel moves more than ε|0.1|
|`‖w‖₁`|sum of the absolute weights|6|
|`ε‖w‖₁`|the most any allowed δ can move the score|0.6|

Read aloud: **"the score can move by at most ε times the total size of the weights, and FGSM hits that maximum exactly."**

### Step 5: HTB's real example, where high dimensionality is the multiplier

MNIST has **784 pixels**. Suppose every weight is tiny, just 0.01:

```
‖w‖₁ = 784 × 0.01 = 7.84
max score change = ε × ‖w‖₁ = 0.1 × 7.84 = 0.784
```

Now compare a model with the same tiny weights but only 3 pixels: 0.1 × (3 × 0.01) = **0.003**, which is nothing.

**The number of pixels is the multiplier.** Each pixel contributes a tiny, invisible push, but 784 pushes all aligned in the harmful direction add up to a large change. That's the **high dimensionality** reason. The **local linearity** reason is that near a real image, deep networks behave almost like this straight-line toy, so the toy math still applies.

## Part 2: Why FGSM is the best possible single step (constrained linearization)

### Step 6: The real goal is a hard puzzle

What the attacker wants, written formally:

```
max over δ   ℒ(θ, x+δ, y)     s.t.   ‖δ‖∞ ≤ ε
```

|Symbol|Plain English|
|---|---|
|`max over δ`|find the δ that makes this as big as possible|
|`ℒ(θ, x+δ, y)`|the loss (how wrong) on the changed image|
|`s.t.`|"subject to", the rule you must obey|
|`‖δ‖∞ ≤ ε`|no pixel moves more than ε|

It's hard because a real network's loss is a **curved** surface, so there's no clean formula for the best δ.

### Step 7: Replace the curve with a straight line (first-order Taylor)

Take a 1-pixel toy where the loss curve is L(x) = x², starting at x = 1. The slope there is 2.

```
Straight-line guess:   L(1 + δ) ≈ L(1) + slope × δ = 1 + 2δ

δ = 0.1  ->  guess 1 + 2(0.1) = 1.20     true value 1.1² = 1.21   (close!)
δ = 0.5  ->  guess 1 + 2(0.5) = 2.00     true value 1.5² = 2.25   (further off)
```

**Near the starting point, the straight-line guess is very accurate. It only drifts off as you move further.** That's a **first-order Taylor expansion**: approximate a curve by its tangent line. The general form:

```
ℒ(θ, x+δ, y) ≈ ℒ(θ, x, y) + ∇xℒ(θ, x, y)⊤ δ
```

In words: "new loss ≈ old loss + (gradient · δ)." The gradient ∇xℒ is the multi-pixel version of "slope 2."

### Step 8: Drop the part you can't change

The old loss ℒ(θ, x, y) is a fixed number. It doesn't depend on δ, so it can't help you make the new loss bigger. Throw it away, and the puzzle shrinks to:

```
max over δ   g⊤δ      with   g = ∇xℒ(θ, x, y)     and     ‖δ‖∞ ≤ ε
```

This is now exactly the **dot-product-vs-budget** question from Part 1: "with every pixel capped at ε, which δ makes g⊤δ biggest?" We already solved this by hand in Part 1 Step 3.

### Step 9: The answer, and the rule that proves it (Hölder)

The biggest g⊤δ can be is **ε‖g‖₁**, and it's reached by setting each pixel of δ to ε in the **same sign** as that pixel's gradient. In our 3-pixel toy with g = [2, −3, 1] and ε = 0.1:

```
δ* = ε · sign(g) = 0.1 · sign([2, −3, 1]) = 0.1 · [+1, −1, +1] = [+0.1, −0.1, +0.1]

g⊤δ* = 2(0.1) + (−3)(−0.1) + 1(0.1) = 0.2 + 0.3 + 0.1 = 0.6 = ε‖g‖₁  ✓
```

Any other allowed δ gives less (you checked this in Part 1). The formal guarantee is **Hölder's inequality**:

```
|u⊤v| ≤ ‖u‖p · ‖v‖q        where   1/p + 1/q = 1
```

|Symbol|Plain English|
|---|---|
|`u⊤v`|dot product of two lists|
|`p`, `q`|a matched pair of "rulers" (norms) called **conjugate exponents**|
|`1/p + 1/q = 1`|the pairing rule that decides which q goes with a given p|

The pair we need is **p = ∞** and **q = 1**, because 1/∞ + 1/1 = 0 + 1 = 1. (1/∞ is 0 because dividing by a huge number gives almost nothing.) **L∞ and L1 are "dual norms": they're the matched pair.** Plugging them in:

```
|g⊤δ| ≤ ‖δ‖∞ · ‖g‖1
```

Since ‖δ‖∞ is capped at ε, the biggest g⊤δ can be is ε‖g‖₁, **and equality happens exactly when δ lines up sign-by-sign with g**. That's the sign step. So:

```
δ* = ε · sign(g)        and        max g⊤δ = ε‖g‖1
```

Read aloud: **"among all perturbations with no pixel bigger than ε, the one that lines up with the sign of the gradient does the most damage, and FGSM is exactly that perturbation."** This is why FGSM is called the **exact optimizer of the linearized problem**: once you straighten the curve into a plane, the sign step is provably the best move inside the box.

**The geometric picture** (ties to Section 2): maximizing a flat plane over the L∞ **box** always lands you on a **corner** of the box, where every pixel is at ±ε. FGSM lands on that corner.

## Part 3: Targeted vs untargeted

Two goals, one tiny difference in the formula:

||Goal|Formula|Direction|
|---|---|---|
|**Untargeted**|"just be wrong"|`x + ε·sign(∇xℒ(θ,x,y))`|**+**: push loss on the **true** label y **up**|
|**Targeted**|"say _this_ class yₜ"|`x − ε·sign(∇xℒ(θ,x,yₜ))`|**−**: push loss on the **target** label yₜ **down**|

Two things change for the targeted version: the **sign flips** (minus), and the **label** in the gradient is the target yₜ, not the true label. **Making the loss for yₜ smaller means making the model more confident it's yₜ.** The ε budget and the clamping are identical. (Full treatment in Section 9.)

## Part 4: The L2 and L1 versions of the step

The sign step is specific to an L∞ budget. Change the ruler, change the best step. Each one still "lines up with the gradient," but in that ruler's way.

**L2 budget (a ball of radius ε):** instead of taking the sign, you keep the gradient's real shape and just **rescale** it to length ε.

```
δ₂* = ε · g / ‖g‖₂
```

That `g / ‖g‖₂` makes a unit-length arrow pointing the same way as g (dividing a vector by its own length gives length 1), then ε sets its length. Toy with g = [2, −3, 1], ‖g‖₂ = √14 ≈ 3.74, ε = 0.1:

```
δ₂* = 0.1 · [2, −3, 1] / 3.74 = [0.0535, −0.0802, 0.0267]
```

Notice the big gradient (−3) now gets the **biggest** move. L∞'s sign gave every pixel the same move, but L2 gives the strong pixels more.

**L1 budget (a diamond):** the best step puts the **whole** budget on the single pixel with the largest |gradient| (the diamond's corner, from Section 2). In practice a sparse step that puts ±ε on the top few coordinates approximates it.

|Budget|Best step|Looks like|
|---|---|---|
|L∞|`ε·sign(g)`|every pixel ±ε (a faint haze)|
|L2|`ε·g/‖g‖₂`|smooth, strong pixels move most|
|L1|±ε on the biggest-gradient pixel(s)|a few sparse specks|

## Part 5: Backpropagation to inputs (how PyTorch gets the gradient)

The whole attack needs one thing: **g = ∇xℒ**, the slope of the loss for each pixel. Training computes slopes for the **weights**; the attack computes slopes for the **pixels**, with the weights frozen. The four-line recipe (full code in Section 6):

```
set requires_grad=True on x     ->  "track slopes for the pixels"
run a forward pass to get loss  ->  how wrong the model is now
loss.backward()                 ->  fills in the slopes
read x.grad                     ->  that's g, the sensitivity map
```

**`x.grad` is the gradient with respect to the input.** Forgetting `requires_grad=True` makes `x.grad` come back `None`, which is the single most common FGSM bug.

### Where the gradient actually comes from (softmax + cross-entropy)

The model's 10 raw scores are the **logits** `zᵢ`. Softmax turns them into probabilities `pᵢ` that add up to 1. The loss is **cross-entropy**: `ℒ = −log p_y`, the negative log of the probability the model gave the **true** class. (If p_y = 1, loss = −log 1 = 0, perfectly right. If p_y is small, −log of it is large, very wrong.)

The gradient works out to a clean form:

```
∇xℒ = Σ over i  (pᵢ − 𝟙[i=y]) · ∇x zᵢ
```

|Symbol|Plain English|
|---|---|
|`pᵢ`|probability the model assigns to class i|
|`𝟙[i=y]`|1 if i is the true class, 0 otherwise (the indicator from Section 2)|
|`∇x zᵢ`|how logit i moves when you nudge the pixels|
|`pᵢ − 𝟙[i=y]`|a **weight** on each class's contribution|

So each class gets a coefficient `pᵢ − 𝟙[i=y]`:

- **True class y:** coefficient `p_y − 1`, which is negative. It says "lower the true logit."
- **Every other class:** coefficient `pᵢ` (positive). It says "raise that logit."

**The gradient simultaneously pushes the true class down and the rivals up.** That's precisely "make the model wrong."

### Why confidence changes the attack's strength (worked numbers)

|Confidence on true class|Coefficient on true class (p_y − 1)|What it means|
|---|---|---|
|p_y = 0.95 (very sure)|0.95 − 1 = **−0.05**|tiny gradient, model barely reacts, hard to push|
|p_y = 0.20 (unsure)|0.20 − 1 = **−0.80**|large gradient, 16× stronger, easy to push|

**An already-shaky input has a big gradient and is easy to flip; a confident input has a small gradient and resists.** The "non-true" coefficients always sum to `1 − p_y` (0.05 and 0.80 in the two cases), so they mirror the true-class term.

**Why the piecewise-linear note matters:** SimpleCNN is affine layers + ReLU, so within one "region" (a fixed pattern of which ReLUs are on/off), the input->logit map is a straight line and ∇x zᵢ is **constant**. That constant slope is exactly what makes one first-order step work so well locally: the straight-line approximation is actually _exact_ until you cross into a neighboring region.

## Part 6: The two figures

### Figure 1: "FGSM: Fast Gradient Sign Method"
![Figure 1: "FGSM: Fast Gradient Sign Method"](/assets/Pasted%20image%2020261015155243.png)
The top row is the formula made visual on a digit "7":

- **x (Original Input):** the clean "7".
- **ε·sign(∇xℒ) (Perturbation):** a red/blue checkerboard. Red = +ε (brighten), blue = −ε (darken). Every pixel is one or the other, no in-between. That's the sign step turning a varied gradient into uniform ±ε.
- **x_adv (Adversarial Example):** the "7" with that noise added. Still clearly a "7" to you, but now misread by the model. The colorbar runs −0.3 to +0.3, the per-pixel change.

The bottom row shows _why_ the sign step equalizes magnitudes, using **three histograms** (bar charts of "how many pixels have each value"):

- **Raw gradient ∇xℒ:** a bell-shaped spread from about −6 to +4. Varied magnitudes, most near zero.
- **After sign():** only two bars, at −1 and +1. All the variety is gone; only direction remains.
- **After × ε (=0.3):** the same two bars, now at −0.3 and +0.3. **Every pixel ends up at exactly ±ε, which is what the L∞ box corner means.**

### Figure 2: "L∞: Maximum Change Limit"
![Figure 2: "L∞: Maximum Change Limit"](/assets/Pasted%20image%2020261015155322.png)
Three copies of the same shape (a square inside a circle) with ε = 0.02, 0.06, 0.12. As ε grows left to right, the whole image gets grainier, because **every** pixel is being changed by up to ε. The caption's point: **an L∞ budget caps each pixel at ε, so you get many pixels changed by similar small amounts, none exceeding the cap**. At ε = 0.02 it's barely visible; at ε = 0.12 the noise is obvious. That's the imperceptibility-vs-strength tradeoff ε controls.

## Key facts to lock in

- **FGSM is the exact optimizer of the loss after you replace its curved surface with its first-order Taylor (tangent) approximation**; straightening the loss into a plane turns an unsolvable problem into one with a clean answer.
- ==Under an L∞ budget, the best step is δ* = ε·sign(g), giving a loss increase of ε‖g‖₁==, which Hölder's inequality proves because L∞ and L1 are dual norms (1/∞ + 1/1 = 1) and equality holds exactly when δ aligns sign-by-sign with the gradient.
- **Targeted FGSM subtracts the gradient of the target class's loss (`x − ε·sign(∇xℒ(θ,x,yₜ))`)**, flipping the sign and swapping the true label for the target; the ε budget and clamping are unchanged.
- **The best step depends on the budget's norm: L∞ gives ε·sign(g), L2 gives ε·g/‖g‖₂ (rescale), and L1 puts the whole budget on the largest-gradient coordinate(s)**. Only L∞ uses the sign.
- ==The attack needs g = ∇xℒ, the gradient with respect to the input, obtained by setting `requires_grad=True` on x, running a forward pass, calling `loss.backward()`, and reading `x.grad`== with the weights frozen.
- ==With softmax + cross-entropy, ∇xℒ = Σ(pᵢ − 𝟙[i=y])·∇xzᵢ, which lowers the true-class logit and raises the rivals at the same time==; the true-class coefficient is p_y − 1.
- ==A confident input (p_y = 0.95) has a tiny gradient coefficient (−0.05) and resists attack, while an unsure input (p_y = 0.20) has a large one (−0.80) and flips easily==, a 16× difference in push strength.
- **Method point:** **within one ReLU activation region a piecewise-linear network is exactly affine, so the input->logit slope is constant**, which is why a single first-order FGSM step is so effective locally.

