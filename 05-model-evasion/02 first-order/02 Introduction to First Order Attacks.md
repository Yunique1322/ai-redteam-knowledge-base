# Introduction To First Order Attacks

## The idea in one breath

A model can score 99% on normal test images and still be fooled by an image that has been changed in tiny, invisible ways. **The gradient that teaches the model during training also shows an attacker exactly which pixels to nudge.** This module teaches two ways to use it. **FGSM** picks a change budget first and takes one big step. **DeepFool** searches for the smallest change that flips the answer.

## Part 1: The core reversal (training vs attack)

The section's main claim is that the gradient used to train the model is also used to attack it. We'll see it with the toy model from earlier.

### Step 1: The same toy model

Three pixels, `x = [0.5, 0.2, 0.4]`, and the recipe:

```
score = w1×p1 + w2×p2 + w3×p3 with weights w = [2, −3, 1]
      = 2(0.5) − 3(0.2) + 1(0.4) = 0.8 -> above 0 -> "cat" ✅
```

The model has two kinds of numbers: the **weights** (2, −3, 1), which it learned, and the **pixels** (0.5, 0.2, 0.4), which are the input it's looking at.

### Step 2: Training's question is "how should I change my WEIGHTS?"

During training the pixels are fixed, and the model asks how the score moves if it bumps each **weight**.

- Bump w1 by 0.1 -> score moves by 0.1 × p1 = 0.1 × 0.5 = 0.05. The slope for w1 is **0.5**.
- The slope for w2 is **0.2**, and the slope for w3 is **0.4**.

```
gradient with respect to the weights = [0.5, 0.2, 0.4] <- equals the pixel values
```

Training uses these slopes to adjust the weights so the model becomes **less** wrong.

### Step 3: The attack's question is "how should I change the PIXELS?"

During an attack the weights are frozen, and you ask how the score moves if you bump each **pixel**.

- Bump p1 by 0.1 -> score moves by 0.1 × 2 = 0.2. The slope for p1 is **2**.
- The slope for p2 is **−3**, and the slope for p3 is **1**.

```
gradient with respect to the pixels = [2, −3, 1] <- equals the weights
```

The attack uses these slopes to adjust the pixels so the model becomes **more** wrong.

**It's the same math aimed at a different target.** For each number, the model asks "what happens if I nudge this?" Training nudges the weights to lower the error, and the attack nudges the input to raise it.

### Step 4: The real formulas

Real models use the **loss** J ("how wrong") instead of our raw score, but the logic is the same:

```
Training: θ <- θ − η · ∇θ J(θ, x, y)
Attack: x_adv = x + ε · sign( ∇x J(θ, x, y) )
```

|Symbol|Plain English|In our toy|
|---|---|---|
|`θ` (theta)|the weights|[2, −3, 1]|
|`x`|the input pixels|[0.5, 0.2, 0.4]|
|`y`|the correct label|"cat"|
|`J`|loss = how wrong the model is|rises as the cat score falls|
|`∇θ`|slopes with respect to the **weights**|[0.5, 0.2, 0.4]|
|`∇x`|slopes with respect to the **pixels**|[2, −3, 1]|
|`η` (eta)|training step size (learning rate)|e.g. 0.01|
|`ε` (epsilon)|attack budget per pixel|e.g. 0.15|
|`−` in training|step **downhill** (less wrong)||
|`+` in the attack|step **uphill** (more wrong)||

Read aloud: **"Training moves the weights downhill on the error. The attack freezes the weights and moves the pixels uphill on the error."**

This is why the section says the attack computes "gradients with respect to **inputs** rather than **parameters**." Parameters is just another word for weights.

## Part 2: The decision boundary

The **decision boundary** is the line where the model can't decide. In our toy that's where **score = 0**. Every image above it is "cat" and every image below it is "dog."

Our image sits 0.8 above the line. **An evasion attack is a trip across the decision boundary that's as short as possible.** The two attacks differ in how they plan that trip.

## Part 3: Two approaches, FGSM vs DeepFool

### FGSM: choose the budget first, then take one step

You decide up front: "no pixel moves more than ε." Then you take one step, and the score drops by ε × 6 (from before: 2 + 3 + 1 = 6).

```
ε = 0.10 -> drop 0.6 -> score 0.8 − 0.6 = 0.2 -> still "cat" ❌ attack FAILED
ε = 0.15 -> drop 0.9 -> score 0.8 − 0.9 = −0.1 -> "dog" ✅ attack worked
```

This shows FGSM's trade-off. **FGSM is fast (one gradient, one step), but if your ε is too small it simply fails, and it doesn't tell you how close you got.**

### DeepFool: ask for the smallest change that crosses the line

DeepFool doesn't pick a budget. It asks how far the image is from the boundary and heads straight for it.

**Analogy:** you're standing 0.8 m above sea level on a slope. The steepest way down drops 3.74 m for every 1 m you walk. How far do you need to walk to reach sea level?

```
distance = height ÷ steepness = 0.8 ÷ 3.74 ≈ 0.214
```

Here's where the two numbers come from:

- **Height** is the current score, 0.8: how far above the boundary we are, in score units.
- **Steepness** is the length of the gradient arrow, the L2 norm from last time:

```
‖[2, −3, 1]‖₂ = √(2² + (−3)² + 1²) = √(4 + 9 + 1) = √14 ≈ 3.74
```

So the general rule is:

```
distance to boundary = |score| ÷ ‖gradient‖₂
```

DeepFool walks exactly that distance, straight against the gradient:

```
step = −(score ÷ ‖gradient‖₂²) × gradient
     = −(0.8 ÷ 14) × [2, −3, 1]
     = −0.0571 × [2, −3, 1]
     = [−0.114, +0.171, −0.057]

new pixels = [0.386, 0.371, 0.343]
new score = 2(0.386) − 3(0.371) + 1(0.343) ≈ 0.772 − 1.113 + 0.343 = 0.002 ≈ 0
```

The step lands right on the boundary. DeepFool then adds a tiny extra push (an "overshoot") to get just across.

Compare the two successful attacks with the L2 ruler:

|Attack|Change|L2 size|Biggest single pixel (L∞)|
|---|---|---|---|
|FGSM (ε = 0.15)|[−0.15, +0.15, −0.15]|0.15 × √3 ≈ **0.260**|0.15|
|DeepFool|[−0.114, +0.171, −0.057]|**0.214**|0.171|

**DeepFool's change is smaller overall (L2), but its single biggest pixel move is larger.** It focuses on the pixels that matter most, while FGSM spreads the same amount across every pixel. "Minimal" always depends on which ruler you use, and DeepFool is minimal in **L2**.

### Why DeepFool repeats its step

Our toy is a straight line, so one step lands exactly on the boundary. A real neural network's boundary is **curved**. DeepFool pretends that the area right around the image is flat (the section's "approximates the boundary as a flat surface locally"), takes the shortest step to that pretend-flat boundary, then looks again from the new spot and repeats. After a few rounds it reaches the real, curved boundary.

**Analogy:** walking toward a curved wall in the dark. You feel the wall's angle where you are, step straight toward it, and feel again.

| Aspect | FGSM | DeepFool |
| ---------------- | ------------------------------ | --------------------------------------------- |
| Question it asks | "Can I fool it with budget ε?" | "What's the _smallest_ change that fools it?" |
| Budget | Chosen up front (ε) | Not chosen; it's the result |
| Steps | 1 | Several (it re-checks the slope each time) |
| Norm it uses | L∞ | L2 |
| Main use | Fast baseline attack | Measuring how vulnerable each input is |

## Part 4: White-box vs black-box

- ****White-box****: you have the model, so you compute the exact slopes (PyTorch does it for you).
- ****Black-box****: you can only send inputs and read outputs. There are two options:

**Option 1: estimate the slopes by nudging and measuring** (numerical estimation). Bump pixel 1 by 0.01 and ask the model again:

```
new pixels = [0.51, 0.2, 0.4] -> score = 1.02 − 0.6 + 0.4 = 0.82
slope ≈ (0.82 − 0.80) ÷ 0.01 = 2 <- matches the true slope!
```

**This is the same idea as your GoodWords black-box attack:** `impact = base_prob − predict(base + word)`. You nudge one thing and measure the change. The cost is about one query per pixel, so a 28×28 image needs roughly 784 queries for a single gradient estimate.

**Option 2: transferability.** You train your own similar model (a **surrogate**), attack it in white-box mode, and send the result to the real target. **Adversarial examples often transfer to other models, even ones with different architectures or training procedures**, so this often works.

## Part 5: Mapping to the real example (the "7" becomes a "2")

The section's example is an MNIST-style digit image:

| Toy | Real MNIST |
| --------------------- | ------------------------------------------------------------------------- |
| 3 pixels | 28 × 28 = **784 pixels** |
| 1 score (cat vs dog) | **10 scores**, one per digit 0-9 |
| Boundary: score = 0 | Boundary between "7" and "2": where **score₇ = score₂** |
| Gradient has 3 slopes | Gradient has **784 slopes**, all computed at once |
| ε = 0.15 flips it | Changes so small that the image still clearly looks like a "7" to a human |

The phrase **"the gradient provides that answer for every pixel simultaneously"** is just Step 3 at scale: one backward pass in PyTorch gives all 784 slopes in one go. With 10 classes, DeepFool checks the boundary to each other digit and heads for the nearest one. That's covered in detail in the DeepFool sections.

## Part 6: Why it matters, and the frameworks

- **Accuracy is not robustness.** 99% on normal images says nothing about how the model handles adversarial ones.
    
- **Defenders use these attacks as yardsticks** to test **adversarial training**, **input preprocessing** and **detection systems**. They also use them to find inputs sitting close to the boundary.
    
- **OWASP ML Top 10:** evasion is ****ML01:2023 Input Manipulation Attack****, the top-ranked risk for traditional ML. It's an **inference-time** attack (against the deployed model) that only needs the ability to query. That's different from **training-time** attacks like data poisoning.
    
- **Google SAIF:** **defense in depth** across three stages:
    
    - adversarial training during **development**
    - robustness evaluation before **deployment**
    - input filtering during **operation**
    
    SAIF also calls for red teams that measure **how much perturbation is needed to reach a target attack success rate.**
    

**In practice:** this is a practical engagement with a client-ready report. If you find an evasion issue, ML01:2023 is the category to cite. "Perturbation needed for X% success" is the kind of measurable result a report finding should state.

## Key facts to lock in

- **First-order attacks compute the gradient of the loss with respect to the **input**, not the weights**, and step in the direction that _increases_ loss. Training does the opposite: it changes the weights to _decrease_ loss.
- **FGSM fixes the perturbation budget ε up front and takes a single gradient-sign step**, so it's fast and is the standard baseline. But it simply fails when ε is too small, and it doesn't tell you how far the input was from the boundary.
- **DeepFool looks for the minimal (L2) perturbation that crosses the nearest decision boundary**. It treats the boundary as locally flat, steps straight to it (distance = |score| ÷ ‖gradient‖₂), and repeats because real boundaries are curved. That makes it a per-input measure of robustness.
- **White-box attacks use exact gradients, while black-box attackers either estimate gradients by querying with small nudges or craft examples on a surrogate model and rely on transferability**. Transfer works even across different architectures or training procedures.
- **High test accuracy (98-99%) does not mean adversarial robustness**; a model can classify nearly all normal inputs correctly and still be fooled by tiny changes.
- **OWASP's ML Top 10 lists evasion as ML01:2023 Input Manipulation Attack, the top risk for traditional ML**. It's an inference-time attack that only needs query access, unlike training-time attacks such as data poisoning.
- **Google SAIF defends against evasion with defense in depth: adversarial training in development, robustness evaluation before deployment, and input filtering in operation**, with red teams measuring how much perturbation it takes to reach a target attack success rate.
- **Method point:** **report evasion robustness as a measured number (the perturbation size needed for a given success rate)**, not just as "the model was fooled."