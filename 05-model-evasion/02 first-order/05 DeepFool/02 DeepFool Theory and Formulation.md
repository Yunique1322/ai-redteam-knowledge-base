# DeepFool Theory And Formulation

## The idea in one breath

With 10 classes there are 9 boundaries around every image, one to each rival class. At every iteration, DeepFool measures the distance to **all of them** and steps toward the **closest** one, using the same Section 14 projection formula. A 2% **overshoot** makes sure it actually crosses instead of stopping exactly on the line. Averaging "perturbation size ÷ image size" over a dataset gives **ρ_adv**, a single robustness score for a whole model.

> **Heads-up: formulas mangled by the copy-paste**
> 
> |Pasted|Correct reading|
> |---|---|
> |`k̂(x)=argmaxkfk(x)`|k̂(x) = argmax over k of f_k(x): "the predicted class is whichever class scores highest."|
> |`wk=∇fk(xi)−∇fk̂(xi)(xi)`|w′_k = ∇f_k(xᵢ) − ∇f_k̂(xᵢ). The stray "(xi)(xi)" is a subscript that lost its formatting: the gradient of the **current** class's score.|
> |`fk′=fk(xi)−fk̂(xi)(xi)`|f′_k = f_k(xᵢ) − f_k̂(xᵢ): rival's score minus current class's score (the **score gap**).|
> |`l=argmink≠k̂(xi)|fk′|
> |`ri=|fl′|
> |`ρadv=1|D|
> |`ri=|fl̂′|
> 
> **Symbol clash:** the course writes w_k for the **gradient difference**, but in your toys w means a model's weights. I'll write **w′_k** (w-prime) for the difference so the two don't get mixed up.

---

## Part 1: Multi-class, toy first

### Step 1: A toy with three classes

Two pixels, image **x = [1, 1]**, three classes. Each class has its own weights, and its score is weights · pixels:

|Class|Weights|Score at x = [1, 1]|
|---|---|---|
|A|[3, 1]|3 + 1 = **4** <- highest, so the model predicts A|
|B|[1, 2]|1 + 2 = **3**|
|C|[−4, 6]|−4 + 6 = **2**|

So k̂ = A, the current class. To fool the model, some rival's score has to overtake A's. Each rival has its own boundary, the place where its score equals A's. Which one is closest?

### Step 2: For each rival, how big is the gap? (f′_k)

The gap is the rival's score minus the current score:

- B: 3 − 4 = **−1** (B is 1 point behind)
- C: 2 − 4 = **−2** (C is 2 points behind)

It's tempting to say B is closest because it's only 1 point behind. **That's wrong**, and the next step shows why.

### Step 3: For each rival, how fast can the gap close? (w′_k)

w′_k = rival's weights minus current's weights. This is the direction that **raises the rival and lowers A at the same time**, the same idea as "the gradient of B's score minus A's score".

- B: [1, 2] − [3, 1] = **[−2, 1]**, length √(4 + 1) = √5 ≈ **2.236**
- C: [−4, 6] − [3, 1] = **[−7, 5]**, length √(49 + 25) = √74 ≈ **8.602**

C's direction is much steeper. Each step along it closes the gap far faster.

### Step 4: Distance to each boundary = gap ÷ steepness

This is Section 14's d = |score| ÷ ‖w‖, once per rival:

|Rival|Gap \|f′\||Steepness ‖w′‖₂|Distance|Closest?|
|---|---|---|---|---|
|B|1|2.236|1 ÷ 2.236 = **0.447**||
|C|2|8.602|2 ÷ 8.602 = **0.233**|**yes**|

C is twice as far behind on points but **twice as close in distance**, because its boundary is so steep. That's what argmin of |f′| ÷ ‖w′‖ does: it picks the smallest distance, not the smallest gap.

### Step 5: Take the step toward C

Same formula as Section 14: scale = gap ÷ steepness² = 2 ÷ 74 = **0.0270**.

r = 0.0270 × [−7, 5] = **[−0.189, +0.135]**, so the new image is [0.811, 1.135]:

| Class | New score                          |
| ----- | ---------------------------------- |
| A     | 3 × 0.811 + 1.135 = **3.568**      |
| B     | 0.811 + 2 × 1.135 = 3.081          |
| C     | −4 × 0.811 + 6 × 1.135 = **3.568** |

A and C are **exactly tied**. We're on the boundary, but not across it.

Why no minus sign this time? In Section 14 the score was positive (+0.8) and we had to push it down to 0. Here the gap is negative (−2) and we push it up to 0, so the course uses |f′| with a plus sign and lets w′ supply the direction. Same geometry, different bookkeeping.

### Step 6: The formulas, decoded

|Symbol|Plain English|Toy|Real (MNIST)|
|---|---|---|---|
|k̂(xᵢ)|the class the model currently predicts|A|e.g. digit 7|
|f_k(xᵢ)|score (logit) for class k|A 4, B 3, C 2|10 logits from the model|
|f′_k|rival minus current: the gap to close (negative)|B −1, C −2|9 gaps, one per rival digit|
|w′_k|gradient of the rival's score minus gradient of the current one|B [−2, 1], C [−7, 5]|9 gradients, each with 784 numbers|
|\|f′_k\| ÷ ‖w′_k‖₂|distance to that rival's boundary|B 0.447, C 0.233|9 distances|
|l|the rival with the smallest distance|C|whichever digit's boundary is closest|
|rᵢ|the step this iteration|[−0.189, 0.135]|784-pixel change|
|xᵢ₊₁ = xᵢ + rᵢ|move, then re-check|[0.811, 1.135]|the next iterate|

**Read aloud:** "For every rival class, divide its score gap by how steeply that gap can close. Pick the rival with the smallest result, then step straight toward its boundary by exactly the amount the straight-line picture says is needed. Repeat until the prediction changes."

On a real network the scores aren't straight lines, so this is recomputed every iteration with fresh gradients (Section 14's foggy mountain). **The closest rival can change from one iteration to the next.** That's what "dynamically identifies the nearest boundary at each iteration" means. DeepFool is **untargeted by design**: it never commits to a class in advance.

---

## Part 2: The overshoot

### Step 1: Why it's needed (the toy shows it)

After the exact step, A and C were tied at 3.568. A tie isn't a flip. On a real network, the straight-line picture is only approximate, so the exact step might even land just short.

### Step 2: Multiply by 1.02

|Step taken|New image|A score|C score|Prediction|
|---|---|---|---|---|
|exact (× 1.00)|[0.811, 1.135]|3.568|3.568|tie, not flipped|
|overshoot (× 1.02)|[0.807, 1.138]|3.559|**3.599**|**C, flipped**|

So overshoot 0.02 = "go 2% further than the formula says". It costs a perturbation 2% bigger, but you reliably cross, and with fewer iterations.

This is your Section 1 answer again: 0.20 lands on the boundary and 0.21 crosses it.

> **An honest note the course skips:** the course says the overshoot is applied to each step, (1 + overshoot) × rᵢ. The original DeepFool code applies it once, to the **total** accumulated perturbation at the end, x + (1 + overshoot) × Σ rᵢ, and many libraries copy that. Both conventions exist and both give about 2% extra. When you see the course's code in the next section, check which one it uses.

---

## Part 3: DeepFool vs I-FGSM

|Aspect|I-FGSM|DeepFool|
|---|---|---|
|Uses the gradient's|**sign only** (0.001 and 100 both become +1)|**full magnitudes** (bigger slope means bigger change)|
|Step size|fixed α, chosen by you|computed each iteration from gap ÷ steepness|
|Which class it moves toward|away from the true label (or toward a chosen target)|toward the closest boundary among all classes|
|When it stops|after T iterations|as soon as the class flips|
|Budget|fixed ε (L∞)|none, it discovers the size (L2)|
|Hyperparameters|ε, α, T|max iterations, overshoot|
|Best for|"can I break it within ε?"|"how little does it take to break it?"|

Both iterate and both re-read the gradient each round. The difference is **what they do with it**: I-FGSM flattens it to signs and walks at a fixed pace, while DeepFool keeps the sizes and solves for the exact distance.

---

## Part 4: ρ_adv, one robustness score for a whole model

### Step 1: Toy

From Part 1, the image x = [1, 1] has size ‖x‖₂ = √2 ≈ 1.414, and the perturbation r has size 0.233.

Relative size = 0.233 ÷ 1.414 = **0.164**, so this image falls to a change about **16% of its own size**.

### Step 2: Average over a dataset

Suppose three images needed relative changes of 0.02, 0.05 and 0.08:

ρ_adv = (0.02 + 0.05 + 0.08) ÷ 3 = **0.05**, meaning "on average, a 5% change fools this model."

### Step 3: Formula decoded

ρ_adv = (1 ÷ |D|) × Σ ‖r(x)‖₂ ÷ ‖x‖₂

|Symbol|Plain English|Toy|
|---|---|---|
|D|the set of test images|3 images|
|\|D\||how many images there are|3|
|‖r(x)‖₂|size of DeepFool's perturbation for one image|0.233 (Part 1 image)|
|‖x‖₂|size of that image itself|1.414|
|ratio|perturbation as a fraction of the image|0.164|
|(1 ÷ \|D\|) Σ|average over all images|0.05|

**Read aloud:** "For each image, divide how big DeepFool's change was by how big the image is, then average those fractions."

**Reading it:** ==bigger ρ_adv = more robust==. The course's example: 0.10 ÷ 0.02 = 5, so a model at 0.10 needs perturbations five times larger than one at 0.02.

> **An honest note the course skips:** ‖x‖₂ depends on what space the image is in. A normalized MNIST image has negative background pixels (−0.424), so its length differs from the same image in plain 0 to 1 pixels. ρ_adv is only comparable between models if it's computed **in the same space**, the same lesson as Section 5's "say which space your ε is in".

---

## Part 5: Extensions

### Step 1: L∞ DeepFool

Swap the ruler, and two things change:

- distance becomes |gap| ÷ ‖w′‖₁ (L1 length: sum of absolute values)
- the step direction becomes sign(w′) instead of w′

On the toy:

|Rival|Gap|‖w′‖₁|L∞ distance|
|---|---|---|---|
|B|1|2 + 1 = 3|0.333|
|C|2|7 + 5 = 12|**0.167**|

Step toward C: (2 ÷ 12) × sign([−7, 5]) = **[−0.167, +0.167]**. Every pixel moves the same amount, so this is just **an FGSM step whose ε is computed instead of guessed**. In Section 14, the smallest FGSM ε to reach the boundary was 0.8 ÷ 6, where 6 is exactly the L1 length of [2, −3, 1]. Same formula.

The ruler lesson from Section 14 again:

|Version|Step|L2 size|L∞ size|
|---|---|---|---|
|L2 DeepFool|[−0.189, 0.135]|**0.233** (smaller)|0.189|
|L∞ DeepFool|[−0.167, 0.167]|0.236|**0.167** (smaller)|

### Step 2: Universal adversarial perturbations

A **universal adversarial perturbation** (Moosavi-Dezfooli et al., 2017) is one single noise pattern that fools the model on most images, not just one. You build it by running DeepFool on many training images in turn and adding up the pushes, while keeping the total size capped. Its existence shows that the weakness isn't random per image. The model has **systematic, shared blind directions**, which matters for real attacks because one precomputed pattern can be stamped onto any input.

---

## Copy-paste cells (mine, not the course's)

These use `_toy` names so nothing in your notebook gets overwritten.

### Cell 1: Pick the closest boundary

```python
import torch
x_toy = torch.tensor([1., 1.])
W_toy = {"A": torch.tensor([3., 1.]), "B": torch.tensor([1., 2.]), "C": torch.tensor([-4., 6.])}
scores_toy = {k: (w @ x_toy).item() for k, w in W_toy.items()}
cur = max(scores_toy, key=scores_toy.get)                     # predicted class (argmax)
dists_toy = {}
for k, w in W_toy.items():
    if k == cur:
        continue
    w_diff = w - W_toy[cur]                                   # w'_k: rival grad minus current grad
    gap = scores_toy[k] - scores_toy[cur]                     # f'_k: rival score minus current
    dists_toy[k] = abs(gap) / w_diff.norm().item()            # distance to that boundary
    print(k, "gap", gap, "| w'", w_diff.tolist(), "| distance", round(dists_toy[k], 3))
print("closest:", min(dists_toy, key=dists_toy.get))
```

**Expected:** `B gap -1.0 | w' [-2.0, 1.0] | distance 0.447`, `C gap -2.0 | w' [-7.0, 5.0] | distance 0.232`, then `closest: C`. Your C distance may print 0.232 because of float32 rounding.

### Cell 2: Step with and without overshoot

```python
w_l = W_toy["C"] - W_toy["A"]
gap_l = scores_toy["C"] - scores_toy["A"]
r_toy = abs(gap_l) / (w_l @ w_l) * w_l                        # |f'| / ||w'||^2 * w'
for over in [0.0, 0.02]:
    x_new = x_toy + (1 + over) * r_toy
    print(f"overshoot {over}:", {k: round((w @ x_new).item(), 3) for k, w in W_toy.items()})
print("rho for this image:", round((r_toy.norm() / x_toy.norm()).item(), 3))
```

**Expected:** at `0.0`, A and C are both `3.568`. At `0.02`, A is `3.559` and C is `3.599` (flipped). Then `rho for this image: 0.164`.

---

## How important is this in general?

Medium. You won't derive anything by hand, but you should know three things: DeepFool picks the closest boundary, overshoot makes it actually cross, and ρ_adv is how you report robustness. Those come up when you run the code and write up results.

## Key facts to lock in

- With many classes, **DeepFool computes the distance to every rival class's boundary each iteration and steps toward the closest one**. It is untargeted by design, and the closest rival can change between iterations.
- **The distance to rival k is the score gap divided by the length of the gradient difference**, |f′_k| ÷ ‖w′_k‖₂. A rival that is further behind on points can still be closer if its gradient difference is steep (toy: C was 2 points behind but 0.233 away, while B was 1 point behind but 0.447 away).
- The step is ==rᵢ = (|f′_l| ÷ ‖w′_l‖₂²) × w′_l==, the Section 14 projection formula applied to the chosen rival. It lands exactly on the linearized boundary, which is a tie, not a flip.
- **The overshoot (typically 0.02) scales the step by 1.02 so the image actually crosses the boundary**. It costs a perturbation about 2% larger but removes ties and speeds up convergence. Check whether the code applies it per step or once to the total.
- **I-FGSM throws away gradient magnitudes with sign() and walks a fixed α for T steps, while DeepFool keeps the magnitudes, solves for the exact step size and stops the moment the class flips**, with no ε to choose.
- **ρ_adv is the average over a dataset of ‖r(x)‖₂ ÷ ‖x‖₂, and a larger ρ_adv means a more robust model**. 0.02 means a 2% change fools it on average, and 0.10 is five times more robust. Only compare values computed in the same input space.
- **L∞ DeepFool swaps the L2 length for the L1 length and w′ for sign(w′)**. That makes it an FGSM step whose ε is computed exactly instead of guessed. Each version is smallest under its own ruler.
- **Method point:** **universal adversarial perturbations (one pattern that fools most inputs, built by accumulating DeepFool steps across many images under a size cap) show that adversarial weakness is systematic rather than per-image**. When reporting robustness, give ρ_adv along with the norm and the space it was measured in.
