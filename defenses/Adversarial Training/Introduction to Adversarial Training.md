# Section 9: Introduction to Adversarial Training

## The idea in one breath
Guardrails filter what goes into and comes out of a model. That can't work against tiny pixel changes, because an attacked image looks exactly like a clean one. So instead you **harden the model itself**.

The course's ordinary MNIST classifier, trained with no defense, gets **~99%** on clean digits. Under attack at ε = 0.3, it drops to **~74%** with FGSM and **~52%** with I-FGSM. **Adversarial training** fixes this: during training, every batch gets an FGSM-attacked copy, labelled with the **true** digit. The trained model then holds **97.6%** under both attacks.

> **Heads-up:** taken from the PDF, where the text and the chart came through cleanly. One label on the chart is hidden behind the legend. It's the robust model's I-FGSM accuracy, which works out to **97.6%** from the right-hand panel (100 − 2.4).

## Part 1 - From guardrails to hardening the model

### Step 1: Why filtering doesn't work here

| | Guardrails (Sections 2-7) | Adversarial perturbations |
|---|---|---|
| What the attack looks like | Text you can spot ("ignore previous instructions") | Pixel changes too small to see |
| Kind of input | Discrete: words, characters | **Continuous**: any pixel can shift by any tiny amount |
| What you'd have to filter | Known patterns | An endless set of near-copies around *every* real image |
| Where the defense goes | Around the model | **Inside the model** |

Traditional security filters discrete things, like a malformed packet or an odd character. Here the "malicious input" is a legitimate digit nudged by 0.3 per pixel at most, and there's no pattern to block.

### Step 2: The model you'll attack and defend
This is the same picture as your FGSM and I-FGSM work in the Evasion modules:

| Attack at ε = 0.3 | Baseline accuracy (course text) |
|---|---|
| None (clean) | ~99% |
| FGSM (one step) | ~74% |
| I-FGSM (many small steps) | ~52%, stronger because each step re-follows the gradient |

"Not a bug in implementation but a consequence of how neural networks learn decision boundaries": the model's code isn't wrong. The weakness comes from *how* the model was trained.

## Part 2 - Why ordinary training fails

### Step 1: The coverage gap
- Training makes the loss small **on the training images** and nowhere else.
- Real images sit on a thin, low-dimensional "sheet" (**manifold**) inside the huge space of all possible pixel combinations.
- The model never sees points *just off* that sheet, so its decision boundary there is untested and can bend unpredictably.
- FGSM deliberately steps off the sheet in the direction that raises the loss most. A 3 can land in an area the model links with 8. It doesn't look like an 8 to us; the model simply never learned that area.

> **Root cause:** **standard training covers only a tiny fraction of the possible inputs.**

### Step 2: The same picture with the running toy
Cat score = w·x, with w = [2, −3, 1] and x = [0.5, 0.2, 0.4], giving a score of 0.8.

- FGSM moves every pixel by ε = 0.1 in the worst direction (−sign(w)), so the score drops by ε × 6 = 0.6, to **0.2**.
- In a real network, enough steps like that cross the decision boundary, and the label flips.

## Part 3 - The fix: adversarial training

### Step 1: What happens each batch
1. Take a batch of clean training images.
2. Create FGSM versions of them **using the model's current weights**, so the attack keeps up with the model as it learns.
3. Train on **both** the clean and attacked images, all labelled with their **true** digits.
4. Over many training passes, the decision boundaries move outward so the attacked versions stay on the correct side.

The model learns "a 3 pushed in the attacker's direction is still a 3". This was the AI Defense Section 1 cell: `(x_adv, "cat")` added to the batch.

### Step 2: Why not just add random noise?

| | Random noise | FGSM perturbation |
|---|---|---|
| Direction | Any direction, uniformly | The **worst-case** direction (sign of the loss gradient) |
| Effect on the model | Mostly harmless directions, so training effort is wasted | Hits exactly the weakness an attacker would use |
| Result | General smoothing | A defense tailored to the attack |

On the toy with ε = 0.1, FGSM changes the score by −0.6 **every time**. Random ±0.1 noise averages **0.0** and only reaches −0.6 in 1 of the 8 possible sign patterns (cell 1 below).

> **An honest note the course skips:**
> - **"Robust to the attacks we anticipated" is literal.** Training on FGSM at one ε defends best against FGSM-like attacks at that ε. Section 10 shows the robustness carries over imperfectly, and Section 11 tackles the over-fitting to one ε.
> - **The robust model's high scores come from a weaker test than they appear.** In this lab the attack images are made **in advance against the baseline model** and saved to `adv_examples.safetensors` (checked in the later pages of the PDF). So the robust model is tested on attacks crafted for a *different* model. That's a **transfer attack**, which is weaker than attacking the robust model directly (a **white-box** attack). It explains why FGSM and I-FGSM both score exactly 97.6%. More on this in Section 14.

## Reading the figure/output

![Introduction to Adversarial Training](/assets/9%20Introduction%20to%20Adversarial%20Training-20261024204424492.png)

**Left panel, "Model Accuracy":**

| Test | Baseline | Robust | What it shows |
|---|---|---|---|
| Clean | 98.6% | 99.0% | Nothing lost on normal digits; it's even slightly better |
| FGSM ε = 0.3 | 73.8% | 97.6% | +23.8 points |
| I-FGSM ε = 0.3 | 52.0% | 97.6% (label hidden) | +45.6 points. The stronger attack gains nothing over FGSM against the robust model |

**Right panel, "Attacker's Perspective"** (attack success = 100 − accuracy):

| Test | vs baseline | vs robust | Improvement |
|---|---|---|---|
| Clean | 1.4% (no label on the bar) | 1.0% | This is just the normal error rate, not an attack |
| FGSM | 26.2% | 2.4% | **11× fewer** successful attacks |
| I-FGSM | 48.0% | 2.4% | **20× fewer** |

**How to read it:**

- **The right panel is the attacker's view of the left one.** It's the same data flipped, and it's the version to quote in a report: "the attack success rate dropped from 48% to 2.4%".
- **Clean accuracy went *up* (98.6 → 99.0).** That's normal for MNIST, where the extra examples act like mild regularization (protection against over-fitting). On harder data, adversarial training usually *lowers* clean accuracy a little: the trade-off from Section 1.
- **Equal FGSM and I-FGSM scores (97.6%) are the giveaway that this was a transfer test**, as explained in Part 3.

## Copy-paste cells (mine, not the course's)
Both cells stand alone; use any notebook.

### 1) FGSM vs random noise on the cat toy
This checks every ±ε sign pattern and compares it with FGSM.

```python
import itertools, numpy as np
w = np.array([2.0, -3.0, 1.0]); x = np.array([0.5, 0.2, 0.4]); eps = 0.1 # running cat-score toy
fgsm_change = w @ (-eps * np.sign(w)) # FGSM pushes the cat score down
changes = [w @ (eps * np.array(s)) for s in itertools.product([-1, 1], repeat=3)] # all 8 random sign patterns
print(f"clean score {w @ x:.2f}")
print(f"FGSM change: {fgsm_change:+.2f} (always the worst case)")
print("random sign noise changes:", " ".join(f"{c:+.1f}" for c in sorted(changes)))
print(f"random average {np.mean(changes):+.2f} | random hits the worst case in {sum(np.isclose(changes, fgsm_change))}/8 patterns")
```

Expected:
```txt
clean score 0.80
FGSM change: -0.60 (always the worst case)
random sign noise changes: -0.6 -0.4 -0.2 +0.0 +0.0 +0.2 +0.4 +0.6
random average +0.00 | random hits the worst case in 1/8 patterns
```

With 784 MNIST pixels instead of 3, the odds that random noise hits the worst case are 1 in 2⁷⁸⁴, effectively zero.

### 2) The figure's numbers as the attacker sees them

```python
acc = {"Baseline": {"Clean": 98.6, "FGSM": 73.8, "I-FGSM": 52.0},
       "Robust": {"Clean": 99.0, "FGSM": 97.6, "I-FGSM": 97.6}}
for model, row in acc.items():
    print(f"{model:8} | " + " | ".join(f"{k} acc {v:.1f}% -> attack success {100 - v:.1f}%" for k, v in row.items()))
for k in ["FGSM", "I-FGSM"]:
    b, r = 100 - acc["Baseline"][k], 100 - acc["Robust"][k]
    print(f"{k:6}: attacker success {b:.1f}% -> {r:.1f}% ({b / r:.0f}x fewer successful attacks)")
```

Expected:
```txt
Baseline | Clean acc 98.6% -> attack success 1.4% | FGSM acc 73.8% -> attack success 26.2% | I-FGSM acc 52.0% -> attack success 48.0%
Robust | Clean acc 99.0% -> attack success 1.0% | FGSM acc 97.6% -> attack success 2.4% | I-FGSM acc 97.6% -> attack success 2.4%
FGSM : attacker success 26.2% -> 2.4% (11x fewer successful attacks)
I-FGSM: attacker success 48.0% -> 2.4% (20x fewer successful attacks)
```

## Quick self-check
Why does adversarial training use FGSM perturbations instead of random noise of the same size ε? Answer in one sentence, then give the toy numbers: with ε = 0.1, what is FGSM's score change, and what is the average change from random sign noise?

*(Answer key is in COAE-progress.md → Open self-checks.)*

## The facts worth locking in for the exam
- Adversarial training **hardens the model itself**, because pixel-level attacks are **continuous and invisible** and there's nothing for a guardrail to filter.
- The course's baseline MNIST model gets ==~99% clean, ~74% under FGSM and ~52% under I-FGSM at ε = 0.3==. I-FGSM is stronger because it re-follows the gradient at every step.
- Ordinary training fails because of **coverage**: it only reduces the loss on real images, so **the space just off the data manifold is untested**, and that's where FGSM steps.
- The method: **each batch, create FGSM copies with the current weights and train on clean plus attacked images, all with their true labels**.
- It beats random-noise training because **FGSM goes in the worst-case direction**; random noise rarely does, effectively never in 784 dimensions.
- The course's robust model gets **99.0% clean and 97.6% under both attacks**, meaning attack success drops **26.2% → 2.4% (FGSM) and 48.0% → 2.4% (I-FGSM)**.
- It protects best against **the attacks it was trained on, at the ε it was trained with**. Other attacks or sizes need checking separately (Sections 10-11).
- **Method point:** when a defense is claimed, **ask whether the test attacked the defended model directly or reused examples made for another model**. This lab's numbers are **transfer results against baseline-made examples**, so in a report, re-attack the hardened model yourself before calling it robust.
