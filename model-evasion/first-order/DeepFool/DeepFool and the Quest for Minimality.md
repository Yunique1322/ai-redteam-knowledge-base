# DeepFool And The Quest For Minimality

## The idea in one breath

FGSM starts from a fixed budget ε and asks "how much damage can I do with it?" **DeepFool** flips that question: "what is the **smallest** change that crosses the decision boundary?" It does this by pretending the model is a straight line near the image, calculating the exact shortest step to that line, taking the step, and repeating until the class flips. The size of the final change becomes a **measurement of how robust that image is**, not a number you picked in advance.

> **Heads-up: formulas mangled by the copy-paste**
> 
> |Pasted|Correct reading|
> |---|---|
> |`f(x)=wTx+b`|f(x) = wᵀx + b. The "T" is a **transpose**, which here just means "multiply each pixel by its weight and add them up" (a dot product).|
> |`d=|f(x0)|
> |`r*=−f(x0)||
> |`f(x)≈f(xi)+∇f(xi)T(x−xi)`|f(x) ≈ f(xᵢ) + ∇f(xᵢ)ᵀ(x − xᵢ): today's score plus slope times distance moved.|
> 
> **Symbol clash:** the course uses `f` for two things. In the linear part, f is the single score (positive = one class). In the deep-network part, f is the **true class's** score and g is a **rival class's** score. Their difference, f − g, plays the role of the single score.

---

## Part 1: From a fixed budget to the smallest sufficient change

### Step 1: The two questions

| Aspect | FGSM | DeepFool |
| ------------------ | ------------------------------------------------- | --------------------------------------------------------- |
| Question it asks | "Given a hammer of size ε, where do I strike?" | "What's the smallest hammer that breaks this?" |
| Budget | **You choose ε** up front | **Discovered** per image |
| Ruler it optimizes | L∞ (biggest single pixel change) | L2 (overall straight-line size) |
| Steps | One (or a fixed number for I-FGSM) | As many as needed until the class flips |
| Output tells you | "Did ε = 0.8 work?" (yes/no, plus a success rate) | "This image needed a change of size 0.21" (a measurement) |

### Step 2: Why picking ε is a problem

With FGSM, the result mixes up two things: **how robust the model is** and **which ε you happened to pick**. Two models can both show 70% success at ε = 0.8 and look equally robust, then look completely different at ε = 0.3. The robustness number depends on your setting.

DeepFool removes that setting. For each image it finds the **minimal sufficient perturbation**, and that size is the answer. You've seen this in Section 9: the "smallest ε that flips this image" search, where you tried 0.5, 0.6, 0.8 and so on. DeepFool finds that number directly, without the trial and error, using the L2 ruler.

### Step 3: Why "smallest" matters to an attacker and a defender

Smaller changes are harder to detect, tend to **transfer** better to other models, and give a **tighter robustness bound**. If DeepFool fooled an image with a change of 0.21, you know the true minimum is at most 0.21. That's why the course calls it a **measurement instrument** rather than just a weapon.

---

## Part 2: The linear case, toy first

### Step 1: The toy model (same one as always)

Your cat model from Sections 2 and 3:

- weights **w = [2, −3, 1]**, image **x₀ = [0.5, 0.2, 0.4]**, bias b = 0
- score = 2×0.5 − 3×0.2 + 1×0.4 = 1.0 − 0.6 + 0.4 = **0.8**
- score > 0 means cat, score < 0 means dog, and ==score = 0 is the decision boundary== (perfectly undecided)

Question: what is the **shortest** change to x₀ that lands exactly on score = 0?

### Step 2: How far away is the boundary? (the "d" formula)

The score is 0.8, but that's in "score units", not real distance. How fast does the score change as you move? That depends on how big the weights are. Big weights make a steep hill, so you reach the boundary quickly. Small weights make a gentle hill, so you have to walk further.

The length of the weights (their L2 norm, from Section 2):

- ‖w‖₂² = 2² + (−3)² + 1² = 4 + 9 + 1 = **14**
- ‖w‖₂ = √14 ≈ **3.742**

So the real distance is:

- d = 0.8 ÷ 3.742 ≈ **0.214**

==Distance = how high you are ÷ how steep the hill is.== Think of standing 0.8 m above a river on a slope: a steep bank gets you down in a short walk, a gentle bank takes longer.

### Step 3: Which direction? Straight along w

On a flat (linear) boundary, the shortest path is always **perpendicular to the boundary** (straight down the slope), and that direction is w itself. Since we're on the positive (cat) side and want to go down to 0, we move **against** w, which is where the minus sign comes from.

### Step 4: How far along w? Put it together

Each unit you move along w changes the score by ‖w‖² = 14. We need to drop the score by 0.8, so we move 0.8 ÷ 14 = **0.0571** "units of w":

|Pixel|Weight w|−0.0571 × w|New pixel|Contribution to new score|
|---|---|---|---|---|
|1|2|**−0.114**|0.5 − 0.114 = 0.386|2 × 0.386 = 0.771|
|2|−3|**+0.171**|0.2 + 0.171 = 0.371|−3 × 0.371 = −1.114|
|3|1|**−0.057**|0.4 − 0.057 = 0.343|1 × 0.343 = 0.343|
|Total| - | - | - |0.771 − 1.114 + 0.343 = **0.000** ✓|

Notice the change in each pixel is **proportional to its weight**. Pixel 2 (weight −3) moves 3 times as much as pixel 3 (weight 1). That's the big difference from FGSM, where `sign()` makes every pixel move the same amount.

### Step 5: Compare with FGSM on the same toy

From Section 3, FGSM drops the score by ε × (sum of |weights|) = ε × 6. To reach 0 you need ε = 0.8 ÷ 6 = **0.133**, and every pixel moves exactly 0.133.

|Ruler|DeepFool's change|FGSM's change (ε = 0.133)|Smaller one|
|---|---|---|---|
|L2 (overall size)|**0.214**|0.231|**DeepFool**|
|L∞ (biggest single pixel)|0.171|**0.133**|**FGSM**|

This is the key insight of the whole section: **each attack is minimal under its own ruler**. DeepFool wins on L2, FGSM wins on L∞. "Smallest perturbation" always needs a ruler attached.

### Step 6: The formulas, decoded

**Distance:** d = |f(x₀)| ÷ ‖w‖₂

**Minimal step:** r* = −( f(x₀) ÷ ‖w‖₂² ) × w

|Symbol|Plain English|Toy|Real (HTB)|
|---|---|---|---|
|x₀|the starting image|[0.5, 0.2, 0.4]|one MNIST image, 784 pixels|
|w|the weights; also the direction "uphill"|[2, −3, 1]|the gradient (Part 3)|
|f(x₀)|the score at the start (how high above the boundary)|0.8|true-class score minus rival-class score|
|\|f(x₀)\||the score's size, ignoring the sign|0.8|same|
|‖w‖₂|length of w = how steep the score hill is|√14 ≈ 3.742|length of the gradient|
|‖w‖₂²|that length squared|14|squared gradient length|
|d|true distance to the boundary|0.214|the image's robustness|
|r*|the actual change to add to the image|[−0.114, 0.171, −0.057]|the adversarial perturbation|
|minus sign|move downhill, toward the boundary|from 0.8 down to 0|same|

**Read aloud:** "Take the weight direction, scale it by how high you are divided by how steep it is squared, and walk that far downhill."

**Why ‖w‖ is squared:** one ‖w‖ turns w into a direction of length 1, and the other converts the score into a real distance d. Distance × direction = (0.8 ÷ 3.742) × (w ÷ 3.742) = 0.8 × w ÷ 14. The two 3.742s multiply into 14.

> **An honest note the course skips:** landing **exactly** on score = 0 doesn't flip anything, because the model is perfectly undecided there. Real DeepFool code multiplies the step by a tiny bit more than 1 (an **overshoot**, often 1.02) to push just across. It's the same idea as your Section 1 answer: 0.20 lands exactly on the boundary and 0.21 crosses it.

> **Another honest note:** the course says FGSM moves "perpendicular to these boundaries in the L∞ sense". That's loose wording. **Perpendicular** is an L2 idea, and that's DeepFool's direction. FGSM moves along the **sign** of the gradient, which is the steepest direction when the ruler is L∞, not a perpendicular one.

---

## Part 3: Deep networks, curved boundaries and iterative linearization

### Step 1: The problem

A real network's boundary isn't a flat line. It curves and twists through hundreds of dimensions (784 for MNIST). The one-shot formula above assumes the boundary is flat, so on a curved boundary a single step can fall short.

### Step 2: The fix: pretend it's flat, step, re-check

This is the **foggy mountain** again (your Section 10 hill). You can only see the ground right around you, so:

1. **Linearize**: pretend the score is a straight line here, using the gradient as the slope. This is the first-order **Taylor expansion**, f(x) ≈ f(xᵢ) + slope × distance moved.
2. Use the Part 2 formula to compute the exact step to that pretend-flat boundary.
3. Take the full step.
4. Re-evaluate at the new point. If the class has flipped, stop. If not, go back to 1.

### Step 3: Toy with a curved score (one pixel)

One pixel p, starting at 0, with a curved score **s(p) = (p − 2)² − 1**. Cat while positive. The true boundary is at p = 1.  
The slope at any point is 2 × (p − 2). The 1-pixel version of the Part 2 formula is: move = −score ÷ slope.

|Round|Pixel p|Score|Slope|Move = −score ÷ slope|
|---|---|---|---|---|
|0|0.0000|3.0000|−4.00|**0.7500**|
|1|0.7500|0.5625|−2.50|**0.2250**|
|2|0.9750|0.0506|−2.05|**0.0247**|
|3|0.9997|0.0006|−2.00|**0.0003**|

What to notice:

- Round 0's straight-line guess said "move 0.75 and you're there", but the curve bends, so you land at 0.75 with the score still positive (0.5625). **One linear step fell short.**
- Each new step uses a **fresh slope** from the new position, and the steps shrink quickly: 0.75, 0.225, 0.025, 0.0003.
- The total change is the **sum** of all the steps (≈ 1.0), which matches the course: "the accumulated perturbations from all iterations sum to create the final adversarial perturbation".
- If the boundary had been straight, round 0 would have hit it in one go. **Flat regions take one iteration; curved regions take more**, and the algorithm adapts on its own.

### Step 4: Map it onto a real network

|Toy|Real network|
|---|---|
|score s(p)|f(x) − g(x): true-class score minus a rival class's score|
|slope at p|gradient of (f − g) with respect to the 784 pixels|
|move = −score ÷ slope|r = −(f − g) ÷ ‖∇f − ∇g‖² × (∇f − ∇g), the Part 2 formula with "gradient" in place of w|
|stop when score < 0|stop when the predicted class changes|

The course only introduces f and g here. How DeepFool picks **which** rival class g (it tries the closest one) is very likely the next section's job.

### Step 5: Step size

DeepFool takes the **full** computed step each round. It doesn't use a small α like I-FGSM, because the formula already says how far to go. Too large would trust the straight-line guess too far from where it was measured. Too small would crawl. In practice, near a well-trained boundary, the full step works and DeepFool usually needs only a few iterations.

> **An honest note:** "minimal" on a deep network means **approximately minimal**. Each step is the exact best step for the flat pretend version, not for the real curved boundary, so the final change is a very good upper bound rather than a proven minimum.

---

## Copy-paste cells (mine, not the course's)

These use new `_toy` names so they don't overwrite anything in your challenge notebook.

### Cell 1: DeepFool's one-step on the linear toy

Checks the Part 2 numbers.

```python
import torch
w_toy = torch.tensor([2., -3., 1.]) # toy weights
x_toy = torch.tensor([0.5, 0.2, 0.4]) # toy image
score_toy = (w_toy @ x_toy).item() # 0.8
r_df = -(score_toy / (w_toy @ w_toy)) * w_toy # DeepFool step: -(score / ||w||^2) * w
print("r:", [round(v, 4) for v in r_df.tolist()])
print("new score:", round((w_toy @ (x_toy + r_df)).item(), 4))
print("L2:", round(r_df.norm().item(), 4), "| Linf:", round(r_df.abs().max().item(), 4))
```

**Expected:** `r: [-0.1143, 0.1714, -0.0571]`, new score about `0.0` (it may print `-0.0`), `L2: 0.2138 | Linf: 0.1714`.

### Cell 2: FGSM on the same toy, for comparison

```python
eps_fgsm = score_toy / w_toy.abs().sum().item() # smallest FGSM budget reaching score 0 (0.8 / 6)
r_fgsm = -eps_fgsm * w_toy.sign() # every pixel moves the same amount
print("eps:", round(eps_fgsm, 4))
print("L2:", round(r_fgsm.norm().item(), 4), "| Linf:", round(r_fgsm.abs().max().item(), 4))
```

**Expected:** `eps: 0.1333`, `L2: 0.2309 | Linf: 0.1333`. DeepFool wins on L2 and FGSM wins on L∞.

### Cell 3: Iterative linearization on the curved toy

```python
p = 0.0 # the one pixel
for step in range(4):
    score = (p - 2) ** 2 - 1 # curved "cat score"; boundary where score = 0
    slope = 2 * (p - 2) # gradient of the score at p
    move = -score / slope # the linear formula, in 1 dimension
    print(f"round {step}: pixel={p:.4f} score={score:.4f} move={move:.4f}")
    p += move # take the full step, then re-check
```

**Expected:** four lines matching the Part 3 table: moves 0.7500, 0.2250, 0.0247, 0.0003.

---

## How important is this for the exam?

Honest answer: know the concept and the comparison well (minimal vs fixed budget, L2 vs L∞, iterate until it flips). Deriving the formula by hand is low priority. You'll run DeepFool from code, and the next sections build that code.

## Quick self-check

Same toy weights w = [2, −3, 1], but a new image whose score is **1.2** instead of 0.8. What is DeepFool's perturbation r (the three numbers)?

---

## The facts worth locking in for the exam

- **DeepFool (Moosavi-Dezfooli, Fawzi and Frossard, CVPR 2016) finds the smallest perturbation that changes the class**, instead of maximizing damage within a fixed budget ε as FGSM does.
- Because there is no ε to choose, **DeepFool's perturbation size is a per-image robustness measurement**. This makes it fair for comparing models, unlike FGSM success rates, which depend on the ε you picked.
- **DeepFool minimizes the L2 norm, while FGSM is built for the L∞ norm**. On the toy, DeepFool needed L2 0.214 vs FGSM's 0.231, but FGSM needed L∞ 0.133 vs DeepFool's 0.171, so "minimal" always depends on the ruler.
- For a linear classifier, ==the exact minimal step is r* = −(f(x₀) ÷ ‖w‖₂²) × w==. It moves perpendicular to the boundary, and the distance is d = |f(x₀)| ÷ ‖w‖₂ (score height divided by steepness).
- **DeepFool keeps the gradient's relative sizes instead of taking sign()**. Pixels with more influence move more (a weight of −3 moves three times as much as a weight of 1), while FGSM moves every pixel by the same ε.
- On deep networks, **DeepFool uses iterative linearization**: it treats the model as flat around the current point (first-order Taylor), takes the exact step to that flat boundary, re-evaluates, and repeats until the class flips. The final perturbation is the sum of all the steps.
- For multi-class networks, the single score becomes **f − g, the true class's score minus a rival class's score**, and the gradient of that difference replaces w. The result is approximately minimal, and in practice a small overshoot (×1.02) is needed to actually cross the boundary.
- **Method point:** **to compare robustness fairly, measure the minimal perturbation (DeepFool, L2) rather than the success rate at one hand-picked ε**, and always report which norm a perturbation size was measured in.

