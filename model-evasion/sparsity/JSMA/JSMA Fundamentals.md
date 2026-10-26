# JSMA Fundamentals

## The idea in one breath

JSMA changes **as few pixels as possible** (it minimizes the **L0 norm** - the _count_ of changed pixels) by using the model's own gradients to score every pixel for how much it would help flip the prediction, then changing only the highest-scoring ones - each possibly jumping the whole way from black to white.

> **Heads-up - the pasted saliency formula got mangled.** It came through as `∂F2∂x1=0.6` and `0.6×0.4=0.24`. Cleaned up, those are:
> 
> - `∂F₂/∂x₁ = 0.6` → "how fast the **class-2 score** rises when you nudge **feature 1** up"
> - `Σ_{j≠2} ∂F_j/∂x₁ = −0.4` → "what the **other** classes' scores do, added together, when you nudge feature 1 up"
> - saliency `= 0.6 × |−0.4| = 0.24`

## Part 1 - What "saliency" means (toy first)

Forget images for a second. You have a target class `t` you want to win. For any one pixel, you ask two questions:

- **α (target slope):** if I turn this pixel up, does the target's score go up? (You want **yes → α positive**.)
- **β (competitor slope sum):** if I turn this pixel up, what happens to _all the other classes added together_? (You want them to go **down → β negative**.)

A pixel is only worth using if it does **both**: helps the target **and** hurts the rest. If it passes, its score is "how much it helps" × "how much it hurts". If it fails either sign, its score is **0** - skip it.

Toy numbers:

|Pixel|α (helps target?)|β (hurts others?)|Verdict|Saliency|
|---|---|---|---|---|
|P|+0.6|−0.4|both good|0.6 × 0.4 = **0.24**|
|Q|+0.3|+0.5|helps others more|fails sign|

## Part 2 - The formula

$$S(i,t) = \begin{cases} 0, & \text{if } \alpha_i \le 0 \ \text{ or } \ \beta_i \ge 0 \ \alpha_i \times |\beta_i|, & \text{otherwise} \end{cases}$$

with `α_i = ∂F_t/∂x_i` and `β_i = Σ_{j≠t} ∂F_j/∂x_i`.

|Symbol|Plain English|Toy (pixel P)|Real (MNIST pixel)|
|---|---|---|---|
|`α_i`|target score's slope for pixel i|0.6|∂(score of "8")/∂(pixel 352)|
|`β_i`|summed slope of every other class|−0.4|how "3","0","1"... move together|
|`α_i ≤ 0`|pixel doesn't help target → reject|passes|passes|
|`β_i ≥ 0`|pixel helps competitors → reject|passes|passes|
|`α_i × \|β_i\|`|score when both pass|0.24|the pixel's saliency|

**Read aloud:** "The saliency of pixel i for target t is zero unless raising it both lifts the target and lowers the rest; when it does, the score is the target's rise times the size of the others' fall."

## Part 3 - HTB's real example (the three bar charts)

- **Initial State:** class 0 = 0.6 (winning), class 1 = 0.5, **target class 2 = 0.3** (losing). JSMA must flip this ranking.![JSMA Fundamentals](/assets/14%20JSMA%20Fundamentals-20261020153438133.png)
- **Increase Feature 1:** α = 0.6, β = −0.4 → passes → S = **0.24**. The chart shows the target shoot to 0.9 (**now winning**) while both competitors drop. This is the good pixel.![JSMA Fundamentals](/assets/14%20JSMA%20Fundamentals-20261020153447544.png)
- **Increase Feature 2:** α = 0.3, β = **+0.5** → β is positive, so it _helps_ the competitors even more than the target → **S = 0**. The chart shows class 0 climbing to 0.9 and staying in front. Rejected.![JSMA Fundamentals](/assets/14%20JSMA%20Fundamentals-20261020153503598.png)

So JSMA picks feature 1 and ignores feature 2, even though feature 2 _did_ raise the target a little.
![Part 3 - HTB's real example (the three bar charts)](/assets/Pasted%20image%2020261020153634.png)
## Part 4 - Both directions, and the loop (cycle figure)

Each pixel is checked **both ways** - turning it up _and_ turning it down - and JSMA keeps whichever direction scores higher. For the "down" direction the sign conditions flip. **This is why JSMA can erase a bright pixel (white→black) just as happily as it adds one (black→white).**

The cycle figure is the whole attack:

**Compute Jacobian** (one backward pass per class) → **Build saliency map** (score every pixel) → **Select best feature** → **Modify it by step size θ** → **Update search space** (mask out any pixel that hit 0 or 1) → repeat until the model is fooled or you hit the L0 budget.
![JSMA Fundamentals](/assets/14%20JSMA%20Fundamentals-20261020153558343.png)
Two dials:

- **θ (theta) - step size.** How far you move the chosen pixel each hit. Bigger θ = faster but more visible; a pixel can be selected several times until it saturates at 0 or 1.
- **γ (gamma) - feature budget, as a _fraction_.** The most pixels you're ever allowed to touch is `γ × num_features`. On MNIST (784 pixels), γ = 0.1 means up to ~78 pixels.

## Part 5 - The logits-vs-probabilities trap (exam favourite)

Compute gradients on the **logits** (raw scores), never on the softmax **probabilities**. Because probabilities are forced to sum to 1, their slopes cancel out, which forces `β = −α`. Then the formula collapses to `S = α × |−α| = α²` - the competitor-suppression half vanishes and you've thrown away the thing that makes JSMA JSMA. **Always differentiate the logits.**

## Reading the figures

- **Three bar charts:** a live before/after of Part 3 - target in orange/aquamarine, competitors in red, arrows showing each class moving when you nudge a feature. Feature 1 makes the target win; feature 2 keeps class 0 in front.
- **MNIST "3"→"8":** left is the clean "3", centre is the _saliency map_ (hot yellow = the strategic pixels - fill in the left side and strengthen the middle stroke), right is the result that now reads "8". Note how few pixels are hot.
- **Cycle diagram:** the five-box loop from Part 4.

> **An honest note the course mostly skips:** the classic Papernot JSMA modifies pixels in **pairs** (two at a time) for more leverage per step; the course's "pick one feature" description is the simplified variant you'll implement. Also keep the cost in mind - it's **one backward pass per class per iteration**, which is fine for MNIST's 10 classes but brutal for ImageNet's 1000. And the shape bug the course rightly warns about: flatten channel-height-width and unflatten the _same_ way, or your change silently lands on the wrong pixel and success quietly drops to zero.

## Quick self-check

Two candidate pixels for target `t`, increasing direction:

- **Pixel A:** `∂F_t/∂x = 0.5`, competitor sum `= −0.2`
- **Pixel B:** `∂F_t/∂x = 0.8`, competitor sum `= +0.1`

Which one does JSMA pick, and what's its saliency score? (I'll confirm next message - work it before you scroll.)

## The facts worth locking in for the exam

- **JSMA minimizes L0** - the _count_ of changed pixels, not their size - so it touches very few pixels (20-40 of 784 on MNIST) but each can swing the full 0→1, producing visible dots and strokes rather than smooth noise.
- A pixel's saliency is **zero unless it both raises the target and lowers every competitor combined**; when it qualifies, the score is the target's slope times the magnitude of the competitors' slope (`α × |β|`).
- **Differentiate the logits, not the softmax probabilities**, because probabilities sum to 1 so their slopes cancel (`β = −α`), collapsing saliency to `α²` and deleting the competitor-suppression term.
- **Score each pixel in both directions** (up and down) and keep the higher - this is why JSMA can erase bright pixels as readily as it adds them.
- **θ is the step size and γ is the feature-budget fraction** (max pixels = `γ × num_features`); a bigger θ is faster but more visible, while γ hard-caps how many pixels you may ever touch.
- JSMA's cost is **one backward pass per class per iteration**, cheap on MNIST's 10 classes but prohibitive on ImageNet's 1000.
- **Keep one consistent C-H-W flatten/unflatten order and mask out saturated pixels**; mixing flatten orders moves your change to the wrong pixel with no error and tanks the success rate.
- **Method point:** **JSMA is an L0, white-box, iterative, saliency-guided attack that works best on shallow models** - line it up against the others for the exam: FGSM caps L∞, DeepFool minimizes L2, ElasticNet blends L1+L2, JSMA minimizes L0.

Paste Section 15 when you're ready - that's almost certainly where the first JSMA code (the Jacobian and saliency map in PyTorch) shows up.