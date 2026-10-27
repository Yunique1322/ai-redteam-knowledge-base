# Section 11: The Adversarial Training Defense

## The idea in one breath
Adversarial training works like a **vaccine**. In every batch, you attack the *current* model with FGSM, then train on the clean and attacked images together, all with their **true** labels. The model and its attacks get stronger together.

The maths behind it is a **min-max** problem:

- the **inner max** finds the worst input within ε of each training image (FGSM approximates this in one step);
- the **outer min** then adjusts the weights to make that worst-case loss small.

Training on a **random ε from 0.1 to 1.0** in each batch (**epsilon spread**) fixes epsilon overfitting. At ε = 1.0, the single-ε model gets **33%** and the spread model **66%**. The cost is about **2-3× the training compute** per epoch.

> **Heads-up:** the PDF flattened the two pseudocode blocks into one line each; I've restored the numbered steps without changing a word. The maths also lost its subscripts:
> - `minθ𝔼(x,y)∼D[L(fθ(x),y)]` means **min over θ of E₍ₓ,ᵧ₎∼D [ L(f_θ(x), y) ]** (normal training).
> - `minθ𝔼(x,y)∼D[maxx′∈ℬϵ(x)L(fθ(x′),y)]` means **min over θ of E₍ₓ,ᵧ₎∼D [ max over x′ ∈ B_ε(x) of L(f_θ(x′), y) ]** (adversarial training).
> - `ℬϵ(x)` is **B_ε(x)**: the "ε-ball", every input whose pixels are each within ε of x.

## Part 1 - The vaccination idea

| Phase in each batch | What happens |
|---|---|
| 1. Attack | Create FGSM examples from the **current** weights: inputs the model would get wrong *right now* |
| 2. Train | Update the weights on clean **and** attacked images, so both get classified correctly |

**Why "co-evolution" matters:**

- **Early in training:** the model is weak, so its own gradients produce weak attacks.
- **Late in training:** the model is strong, so it produces the attacks that matter in deployment.
- **Why that beats a fixed set:** a batch of attacks made once before training would go stale as soon as the model improved. Fresh attacks track the model.

## Part 2 - The training loop

```pseudocode
for each batch (images, labels):
    1. Generate adversarial examples: adv_images = FGSM(model, images, labels, epsilon)
    2. Combine batches: combined = concat(images, adv_images)
    3. Duplicate labels: combined_labels = concat(labels, labels)
    4. Forward pass: outputs = model(combined)
    5. Compute loss: loss = cross_entropy(outputs, combined_labels)
    6. Backward pass and update: loss.backward(), optimizer.step()
```

| Step | What it does |
|---|---|
| 1 `FGSM(model, ...)` | Fresh attacks from the **current** weights, aimed at today's weaknesses |
| 2 `concat(images, adv_images)` | Clean and attacked images in one batch, so 128 becomes 256 |
| 3 `concat(labels, labels)` | **The same true labels twice.** An attacked 3 must still be called "3", not whatever the attack made the model predict |
| 4-6 | Ordinary training on the combined batch. Clean and attacked images count **equally** in the loss |

## Part 3 - The maths: a min-max problem

### Step 1: Toy first (linear cat score, so the inner max can be solved exactly)
For a **linear** model, the worst input within ε is exactly what FGSM gives: move every feature by ε against w. So:

> worst-case score = w·x − ε × (|w₁| + |w₂| + |w₃|) = w·x − ε‖w‖₁

| Weights | Clean score | ‖w‖₁ | Worst at ε = 0.1 | Worst at ε = 0.3 |
|---|---|---|---|---|
| w = [2, −3, 1] (original) | 0.80 | 6 | +0.20 (still a cat) | **−1.00 (flips to dog)** |
| w′ = [2, 0, 1] | 1.40 | 3 | +1.10 | **+0.50 (still a cat)** |

- **Inner max:** "what's the worst within ε?" For a linear model FGSM answers this *exactly*. For a neural network it's only an approximation.
- **Outer min:** "change w so that even the worst case is right." Here that means shrinking the weights the attacker can exploit (‖w‖₁ 6 -> 3) while keeping the clean score positive. Adversarial training does the same thing to a network's boundaries.

### Step 2: The real formulas
Normal training:
**min_θ E₍ₓ,ᵧ₎∼D [ L(f_θ(x), y) ]**

Adversarial training:
**min_θ E₍ₓ,ᵧ₎∼D [ max_{x′ ∈ B_ε(x)} L(f_θ(x′), y) ]**

| Symbol | Plain English | Toy | Real |
|---|---|---|---|
| θ | Model weights | w | LeNet-5 parameters |
| E₍ₓ,ᵧ₎∼D | Average over the training data | one point | average over the MNIST batches |
| L(f_θ(x), y) | Loss on the clean input | depends on w·x | cross-entropy |
| B_ε(x) | Every input with each pixel within ε of x | the box x ± 0.1 | 784-dimensional box, ±0.3 normalized |
| max over x′ ∈ B_ε(x) | The worst input inside that box | x − ε·sign(w), found exactly | FGSM: **one** step approximation |
| min over θ | Choose weights to make the worst case small | w -> w′ | AdamW training steps |

**Read aloud:** "Find the weights that minimise, on average over the data, the largest loss any input inside the epsilon-ball can cause."

### Step 3: Why this creates margins
Making the worst-case loss small across the ε-box means every point in the box must be on the right side of the boundary. Points that used to sit near the boundary force it **at least ε away**. That's the margin from Section 10's figure.

> **An honest note the course skips:**
> - **"At least ε" is a goal, not a guarantee.** It only holds if the inner max is found *exactly* and the training loss actually reaches zero. One FGSM step only *approximates* the worst case on a curved network. A real guarantee needs a **certified defense** (Part 8).
> - **Training only on FGSM has a known failure mode.** It can suddenly "learn" to make its own FGSM gradients unhelpful (**catastrophic overfitting** / **gradient masking**), so FGSM accuracy stays high while I-FGSM/PGD accuracy collapses. The fix in the literature is FGSM from a **random start**, or PGD training. This is also why Section 10 tells you to watch the FGSM - I-FGSM gap.

## Part 4 - Clean vs robust accuracy

| Setting | Clean | Robust | Source |
|---|---|---|---|
| Standard training | ~99% | "~5%" under FGSM | course text |
| Adversarial training, 50/50 clean/attacked (default) | ~98% | ~95% | course text, "might achieve" |
| 70/30 clean/attacked | ~98.5% | ~90% | course text, "might achieve" |

- **Why clean accuracy drops:** some of the model's capacity goes on handling perturbed inputs, leaving less for fine detail on clean ones.
- **The ratio is a dial:** more attacked examples means more robustness. A system that is attacked constantly should lean that way; one that mostly sees clean inputs can lean towards accuracy.

> **An honest note the course skips:**
> - **The course contradicts itself.** Here it says the baseline "collapses to around 5% under FGSM", but Section 9 said **~74%** for this same lab. **I tracked down why in Section 12:** the 5% is what you get when the MNIST data is **not normalized**. That's what Section 12's baseline cell does as written (I measured 3.4%). The 74% is the normalized setup used in Section 13 (I measured 76.0%). In the real pipeline the gain is 74 -> 97.6, about **24 points**, not 90.
> - **The 98 / 95 / 98.5 / 90 figures are illustrative.** They're framed as "might achieve", not measured.

## Part 5 - Epsilon-spread training

```pseudocode
for each batch (images, labels):
    1. Sample random epsilon from spread: batch_epsilon = random_choice([0.1, 0.2, ..., 1.0])
    2. Generate adversarial examples: adv_images = FGSM(model, images, labels, batch_epsilon)
    3. Continue with combined training as before...
```

| Step | What it does |
|---|---|
| 1 `random_choice([0.1 ... 1.0])` | Picks a **new ε for each batch**, uniformly from 10 values |
| 2 `FGSM(..., batch_epsilon)` | Attacks at that strength |
| 3 | The same combined clean + attacked update as Part 2 |

| ε range in training | What the model learns |
|---|---|
| Low (0.1-0.3) | Stay precise under subtle changes |
| High (0.7-1.0) | Survive severe changes |

The course's results table:

| Attack ε | Single-ε training (0.3) | Spread training | Spread's advantage |
|---|---|---|---|
| 0.3 | 98% | 98% | 0 (nothing lost at the training ε) |
| 0.5 | 91% | 95% | +4 |
| 0.7 | 75% | 87% | +12 |
| 1.0 | 33% | 66% | **+33** |

## Part 6 - Training settings

| Setting | Course's advice | Why | What the lab actually uses (Section 13) |
|---|---|---|---|
| Learning rate | Lower than standard: 0.001 vs 0.01 | The combined loss surface is rougher | AdamW with lr = **0.001**. Note the baseline *also* used 0.001, so it's not actually lower than the lab's own baseline |
| Weight decay | Use it (L2 regularization) | Stops over-fitting to specific attack patterns | `weight_decay=1e-4` |
| **Gradient clipping** | Use it | Early attacked batches give huge losses and gradient spikes | `clip_grad_norm_(..., max_norm=1.0)` |
| Epochs | 20-30, against about 10 for standard training | Two things to learn: classification and robustness | default `epochs=25`, but the walkthrough calls it with 10 |
| LR schedule | **Cosine annealing** or step decay | Big steps early, fine-tuning late, settling into a robust minimum | `CosineAnnealingLR` |

## Part 7 - Compute cost

| Work per batch | Standard | Adversarial |
|---|---|---|
| Making the attack | - | 1 forward + 1 backward on B images |
| Training update | 1 forward + 1 backward on B | 1 forward + 1 backward on **2B** |
| Rough total (backward ≈ 2 forward) | 3 units × B | 3B + 6B = **9 units × B, so about 3×** (cell 3) |

- **Time in practice:** MNIST takes 5-10 minutes on a GPU or 20-30 minutes on a CPU. ResNet on CIFAR-10 takes hours instead of about one.
- **Ways to speed it up:**
  - **mixed precision** (FP16), which roughly halves memory and time;
  - **free adversarial training**, which reuses the gradient from the training backward pass to make the attack, saving the extra forward and backward pass.

## Part 8 - Limits and alternatives

| Limit | What it means | Partial fix |
|---|---|---|
| Only as strong as the training attack | FGSM training (an L∞ attack) doesn't cover differently shaped attacks, e.g. **JSMA**, which changes only a few pixels (L0, Sparsity module) | Train against stronger or more varied attacks, at more cost |
| Transfer attacks | The attacker builds their own model, attacks it, and sends those examples to you. They can hit weaknesses you never trained against | **Ensemble adversarial training**: train on attacks made against several models |
| No guarantee | It's tested against known attacks, not proven safe | **Certified defenses**: **randomized smoothing** and **interval bound propagation** *prove* no attack within ε works, at the cost of lower robust accuracy |

## Reading the figure/output

### Figure 1 - Single-ε vs spread training
*(The PDF places this chart at the end of Section 10; it shows this section's table.)*

*(Figure not available in source notes.)*

| Attack ε | Single-ε (red), read off the bars | Spread (green), labelled | What it shows |
|---|---|---|---|
| 0.1-0.3 | about 99 / 98 / 97.5 | 99 / 99 / 98 | Identical up to the training ε |
| 0.4-0.5 | about 95 / 91.5 | 96 / 95 | The gap opens just past 0.3 |
| 0.6-0.8 | about 83 / 75.5 / 63.5 | 92 / 87 / 83 | The single-ε model slides; spread holds above 80% |
| 0.9-1.0 | 46 / 33 (labelled) | 76 / 66 | The single-ε model collapses; spread loses about a third |

**How to read it:**

- **What epsilon overfitting looks like:** the red bars are flat up to the training value and then fall off a cliff.
- **What spread training buys:** the green bars fall gently. The robustness spreads across the whole range at **no cost** at the training ε.
- **Units:** these ε values are normalized, so 1.0 is 78.6/255 on the pixel scale (Section 10).

### Figure 2 - Expected training progression
*(Figure not available in source notes.)*

| Feature | Value | Meaning |
|---|---|---|
| Blue line: clean accuracy | ~99% at the start, a shallow dip to ~97.8% around epochs 15-17, back to ~99% at epoch 25 | Robustness costs little clean accuracy on MNIST, and the cost recovers as the learning rate cools |
| Green line: robust FGSM accuracy | **19%** at epoch 1 -> ~50% at epoch 4 -> ~78% at epoch 10 -> crosses **92% around epoch 21** -> ~93.5% at epoch 25 | Robustness builds slowly. That's why the course says 20-30 epochs |
| Blue dashed line: "Clean Threshold (94%)" | the lab's pass bar | Clean must stay **≥ 94%** |
| Green dashed line: "Robust Threshold (92%)" | the lab's pass bar | Robust FGSM must reach **≥ 92%** (Section 13 says the same) |
| Shaded band | between the robust threshold and the top | The target zone |

> **An honest note on this figure:** its title says "Expected Progression", so it's an illustration, not a logged run. Two consequences:
> - **Don't expect your numbers to match it exactly.** In particular, a robust 19% at epoch 1 is far *below* the baseline's 74% under FGSM.
> - **Don't stop at 10 epochs.** The curve only passes 92% around epoch 21, yet the walkthrough's call in Section 13 uses `epochs=10`. If you're short of 92%, train longer.

## Copy-paste cells (mine, not the course's)
All three stand alone; use any notebook.

### 1) Min-max on the toy: what the outer min is buying
For a linear model, FGSM solves the inner max exactly, so you can see the robust score directly.

```python
import numpy as np
x = np.array([0.5, 0.2, 0.4]) # running toy input, true label "cat" (score > 0)

def worst_case_score(w, eps):
    """Inner max for a LINEAR model, solved exactly: FGSM moves every feature by eps against w."""
    x_adv = x - eps * np.sign(w)
    return w @ x_adv # equals w@x - eps * sum(|w|)

for name, w in {"original w = [2,-3,1]": np.array([2., -3., 1.]),
                "robust w = [2, 0,1]": np.array([2., 0., 1.])}.items():
    print(f"{name} | clean {w @ x:.2f} | |w|_1 = {np.abs(w).sum():.0f} | "
          + " | ".join(f"worst @eps {e}: {worst_case_score(w, e):+.2f}" for e in [0.1, 0.3]))
```

Expected:
```txt
original w = [2,-3,1] | clean 0.80 | |w|_1 = 6 | worst @eps 0.1: +0.20 | worst @eps 0.3: -1.00
robust w = [2, 0,1] | clean 1.40 | |w|_1 = 3 | worst @eps 0.1: +1.10 | worst @eps 0.3: +0.50
```

### 2) The spread table in pixel units and in "digits saved"

```python
single = {0.3: 98, 0.5: 91, 0.7: 75, 1.0: 33} # course's table
spread = {0.3: 98, 0.5: 95, 0.7: 87, 1.0: 66}
for e in single:
    print(f"eps {e:.1f} (= {e * 0.3081 * 255:4.1f}/255 pixel) | single {single[e]}% | spread {spread[e]}% | "
          f"spread gets {10 * (spread[e] - single[e]):3d} more of every 1000 attacked digits right")
```

Expected:
```txt
eps 0.3 (= 23.6/255 pixel) | single 98% | spread 98% | spread gets 0 more of every 1000 attacked digits right
eps 0.5 (= 39.3/255 pixel) | single 91% | spread 95% | spread gets 40 more of every 1000 attacked digits right
eps 0.7 (= 55.0/255 pixel) | single 75% | spread 87% | spread gets 120 more of every 1000 attacked digits right
eps 1.0 (= 78.6/255 pixel) | single 33% | spread 66% | spread gets 330 more of every 1000 attacked digits right
```

### 3) Where "about 3× the cost" comes from

```python
B = 128 # batch size
fwd, bwd = 1, 2 # rule of thumb: a backward pass costs about 2 forward passes
standard = B * (fwd + bwd) # one training pass
adv = B * (fwd + bwd) + 2 * B * (fwd + bwd) # FGSM pass on B images + training pass on 2B (clean + adversarial)
print(f"standard: {standard} units | adversarial: {adv} units | ratio {adv / standard:.1f}x")
```

Expected:
```txt
standard: 384 units | adversarial: 1152 units | ratio 3.0x
```

## Key facts to lock in
- Adversarial training is a **vaccine**: **every batch, attack the current model with FGSM, then train on clean plus attacked images with the true labels** (the batch doubles from 128 to 256).
- The attacks are **fresh every batch from the current weights**, so the defense keeps up as the model improves; a fixed attack set would go stale.
- It solves **min over θ of E[ max over x′ in B_ε(x) of L(f_θ(x′), y) ]**. The inner max is the worst input in the ε-box; **FGSM approximates it in one step** (exactly for linear models). The outer min is ordinary training on that worst case.
- The goal is a **margin of at least ε around the training points**, but FGSM-only training can suffer **catastrophic overfitting / gradient masking**; random-start FGSM or PGD training is the fix.
- There's a **clean vs robust trade-off**, tuned with the clean:attacked ratio (50/50 is the default and favours robustness). The course's "5% -> 95%" figures don't match this lab, whose baseline FGSM accuracy is 74%.
- **Epsilon-spread training samples ε from 0.1 to 1.0 per batch**. At ε = 1.0 it gets **66% against 33%** for single-ε training, with no loss at the training ε.
- Training settings: **AdamW with lr 0.001 and weight decay 1e-4, gradient clipping at 1.0, cosine annealing, 20-30 epochs, about 3× the compute**. The lab's pass bar is **clean ≥ 94% and robust FGSM ≥ 92%**.
- **Method point:** adversarial training **only covers the attack type and ε range it was trained on**. In a report, recommend spread or PGD training, test with **I-FGSM/PGD, several ε values, other attack shapes (e.g. JSMA), and transfer attacks**, and mention **certified defenses** where a guarantee is needed.
