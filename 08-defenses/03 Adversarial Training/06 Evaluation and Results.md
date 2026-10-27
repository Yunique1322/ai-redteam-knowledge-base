# Section 14: Evaluation and Results

## The idea in one breath
`evaluate_robustness.py` loads your saved LeNet-5 and tests it on the **pre-made** `adv_examples.safetensors`: 500 images × 10 ε values × FGSM and I-FGSM, all crafted against the baseline. It prints, for each ε, the accuracy and the attack success (100 − accuracy). Add `--compare` for a side-by-side with the baseline.

The course's ε-spread model gets **99.0% clean and 97.6% / 97.6% (FGSM / I-FGSM) at ε = 0.3**, against the baseline's 73.8% / 52.0%. At ε = 1.0 the spread model keeps **76.6% / 77.0%**, while the single-ε model drops to 36.8% / 24.8%.

**The key thing the course leaves out:** because the examples were made for a *different* model, this is a **transfer** test. I attacked the robust model directly (**white-box**):

| Attack at ε = 1.0 | Robust model's accuracy |
|---|---|
| FGSM | 78.8% |
| I-FGSM | **2.2%** |
| PGD-20 | **0.0%** |

That's the textbook sign of **gradient masking** (details in Part 6).

> **Heads-up:** the PDF flattened the evaluator's output tables into running text. I've restored the rows and columns without changing any number. The "(16 5 5)"-style mangling doesn't occur in this section.

## Part 1 - Running the evaluator

```shellsession
Ishman1322@htb[/htb]$ python evaluate_robustness.py --model-path robust_model.safetensors
```

```shellsession
Ishman1322@htb[/htb]$ python evaluate_robustness.py --model-path robust_model.safetensors --compare
```

```shellsession
Ishman1322@htb[/htb]$ python evaluate_robustness.py --model-path models/ --compare
```

| Command | What it does |
|---|---|
| `--model-path robust_model.safetensors` | Grades one model on the saved examples |
| `--compare` | Grades the baseline (`baseline_model.safetensors`) first, then yours, and prints the improvement |
| `--model-path models/` | Grades **every** `.safetensors` file in the folder; it finds the baseline and the example file automatically |
| "pre-generated ... by a reference implementation" | The same 500 examples every time, so results don't vary between runs. In the lab, these come from **the baseline** |

## Part 2 - Reading one model's output

Output (restored layout):
```txt
Using device: cuda
Loading adversarial examples from: adv_examples.safetensors
500 test samples
Epsilon spread: [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
Evaluating spread_epsilon_model.safetensors...
======================================================================
Model: spread_epsilon_model.safetensors
======================================================================
Clean Accuracy: 99.0% (495/500)
Epsilon FGSM I-FGSM FGSM Attack I-FGSM Attack
------------------------------------------------------------
0.10 98.4% 98.4% 1.6% 1.6%
0.20 98.0% 97.8% 2.0% 2.2%
0.30 97.6% 97.6% 2.4% 2.4% *
0.40 96.8% 97.0% 3.2% 3.0%
0.50 95.8% 95.6% 4.2% 4.4%
0.60 94.6% 94.6% 5.4% 5.4%
0.70 93.6% 93.0% 6.4% 7.0%
0.80 90.8% 89.0% 9.2% 11.0%
0.90 87.0% 84.2% 13.0% 15.8%
1.00 76.6% 77.0% 23.4% 23.0%
------------------------------------------------------------
* Primary epsilon for summary metrics
Summary (ε=0.3):
  Clean: 99.0%
  FGSM: 97.6% (attack success: 2.4%)
  I-FGSM: 97.6% (attack success: 2.4%)
```

| Column / line | Meaning |
|---|---|
| `Clean Accuracy: 99.0% (495/500)` | Unattacked accuracy on the 500 saved images |
| `FGSM`, `I-FGSM` | **Accuracy**: % of attacked images still classified correctly. Higher is better for the defender |
| `FGSM Attack`, `I-FGSM Attack` | **Attack success** = 100 − accuracy. Lower is better for the defender |
| `*` at 0.30 | The main ε used in the summary |
| Summary | The three headline numbers for a report |

> **An honest note the course skips:**
> - **"Primary ε = 0.3, commonly used in research" mixes up scales.** Research uses 0.3 on the **0-1 pixel** scale; this lab's 0.3 is **normalized**, about 0.09 on the pixel scale (Section 10).
> - **The "attack success" column slightly overstates the attack.** 100 − accuracy also counts images the model got **wrong without any attack**. For the baseline at I-FGSM 0.3, I measured 46.0% that way, against **45.5%** when only images that were correct beforehand are counted (cell 2). The gap is small here, but it's bigger for weaker models.

## Part 3 - The baseline under `--compare`

Output (restored layout):
```txt
Evaluating baseline...
======================================================================
Model: baseline_model.safetensors
======================================================================
Clean Accuracy: 98.6% (493/500)
Epsilon FGSM I-FGSM FGSM Attack I-FGSM Attack
------------------------------------------------------------
0.10 95.2% 94.2% 4.8% 5.8%
0.20 86.6% 80.8% 13.4% 19.2%
0.30 73.8% 52.0% 26.2% 48.0% *
0.40 56.2% 26.8% 43.8% 73.2%
0.50 40.4% 9.0% 59.6% 91.0%
0.60 25.8% 2.4% 74.2% 97.6%
0.70 16.4% 0.6% 83.6% 99.4%
0.80 10.8% 0.4% 89.2% 99.6%
0.90 8.0% 0.2% 92.0% 99.8%
1.00 7.0% 0.0% 93.0% 100.0%
------------------------------------------------------------
Summary (ε=0.3):
  Clean: 98.6%
  FGSM: 73.8% (attack success: 26.2%)
  I-FGSM: 52.0% (attack success: 48.0%)
```

| Pattern | Reading |
|---|---|
| FGSM attack success climbs **gradually**: 4.8 -> 26.2 -> 59.6 -> 93.0% | A single step gets less efficient as ε grows |
| I-FGSM climbs **fast**: 48% at 0.3, **91% at 0.5**, 100% at 1.0 | Iterative attacks follow the curve of the loss, so the undefended model has no answer |
| A 21.8-point FGSM - I-FGSM gap at 0.3 | By the course's rule (more than 10), the baseline is "fragile", as expected |

These are white-box numbers, because the examples were made **against this same baseline**. For the baseline, the test is fair.

## Part 4 - The robust model and the improvement

Output (restored layout; the per-ε table is identical to Part 2's, so it's shortened here):
```txt
Evaluating spread_epsilon_model.safetensors...
======================================================================
Model: spread_epsilon_model.safetensors
======================================================================
Clean Accuracy: 99.0% (495/500)
[... same per-epsilon table as in Part 2 ...]
Summary (ε=0.3):
  Clean: 99.0%
  FGSM: 97.6% (attack success: 2.4%)
  I-FGSM: 97.6% (attack success: 2.4%)
Improvement over baseline:
  FGSM: +23.8% (73.8% -> 97.6%)
  I-FGSM: +45.6% (52.0% -> 97.6%)
```

| Line | Meaning |
|---|---|
| `FGSM: +23.8%` | 97.6 − 73.8, in **percentage points** (not a relative %) |
| `I-FGSM: +45.6%` | 97.6 − 52.0. This is where Section 9's bar chart came from |

## Part 5 - The full comparison summary

Output (restored layout):
```txt
==========================================================================================
COMPARISON SUMMARY - Full Epsilon Spread
==========================================================================================
Clean Accuracy: baseline single_epsilon spread_epsilon
                     98.6% 99.4% 99.0%

FGSM Model Accuracy (defender):
------------------------------------------------------------------------------------------
Epsilon baseline single_epsilon spread_epsilon
------------------------------------------------------------------------------------------
0.10 95.2% 99.0% 98.4%
0.20 86.6% 98.4% 98.0%
0.30 73.8% 97.0% 97.6% *
0.40 56.2% 95.8% 96.8%
0.50 40.4% 93.2% 95.8%
0.60 25.8% 86.4% 94.6%
0.70 16.4% 76.2% 93.6%
0.80 10.8% 65.4% 90.8%
0.90 8.0% 52.4% 87.0%
1.00 7.0% 36.8% 76.6%

FGSM Attack Success (attacker):
------------------------------------------------------------------------------------------
Epsilon baseline single_epsilon spread_epsilon
------------------------------------------------------------------------------------------
0.10 4.8% 1.0% 1.6%
0.20 13.4% 1.6% 2.0%
0.30 26.2% 3.0% 2.4% *
0.40 43.8% 4.2% 3.2%
0.50 59.6% 6.8% 4.2%
0.60 74.2% 13.6% 5.4%
0.70 83.6% 23.8% 6.4%
0.80 89.2% 34.6% 9.2%
0.90 92.0% 47.6% 13.0%
1.00 93.0% 63.2% 23.4%

I-FGSM Model Accuracy (defender):
------------------------------------------------------------------------------------------
Epsilon baseline single_epsilon spread_epsilon
------------------------------------------------------------------------------------------
0.10 94.2% 98.8% 98.4%
0.20 80.8% 98.4% 97.8%
0.30 52.0% 97.2% 97.6% *
0.40 26.8% 95.8% 97.0%
0.50 9.0% 92.2% 95.6%
0.60 2.4% 84.8% 94.6%
0.70 0.6% 71.8% 93.0%
0.80 0.4% 55.6% 89.0%
0.90 0.2% 41.2% 84.2%
1.00 0.0% 24.8% 77.0%

I-FGSM Attack Success (attacker):
------------------------------------------------------------------------------------------
Epsilon baseline single_epsilon spread_epsilon
------------------------------------------------------------------------------------------
0.10 5.8% 1.2% 1.6%
0.20 19.2% 1.6% 2.2%
0.30 48.0% 2.8% 2.4% *
0.40 73.2% 4.2% 3.0%
0.50 91.0% 7.8% 4.4%
0.60 97.6% 15.2% 5.4%
0.70 99.4% 28.2% 7.0%
0.80 99.6% 44.4% 11.0%
0.90 99.8% 58.8% 15.8%
1.00 100.0% 75.2% 23.0%
------------------------------------------------------------------------------------------
* Primary epsilon for summary metrics
Improvement over baseline at ε=0.3:
  single_epsilon_model.safetensors: FGSM +23.2%, I-FGSM +45.2%
  spread_epsilon_model.safetensors: FGSM +23.8%, I-FGSM +45.6%
==========================================================================================
```

| Comparison | Single-ε | Spread | Reading |
|---|---|---|---|
| Clean | **99.4%** | 99.0% | Single-ε is slightly better on clean images: the small price of spread training |
| At ε = 0.3 (FGSM / I-FGSM) | 97.0 / 97.2 | 97.6 / 97.6 | The same at the training ε |
| At ε = 0.6 (I-FGSM attack success) | 15.2% | **5.4%** | Spread stays below 6% (the course's point) |
| At ε = 1.0 (FGSM / I-FGSM accuracy) | 36.8 / 24.8 | **76.6 / 77.0** | Single-ε overfits to its ε; spread degrades gently |

## Part 6 - What the transfer test hides (my measurements)

I attacked my own robust model (Section 13 pipeline, 10 epochs, 98.9% clean) on the **same 500 images**, in two ways:

- the evaluator's way: **transfer**, using examples made for the baseline;
- **white-box**: fresh FGSM, the course's I-FGSM, and PGD (I-FGSM with a random start, 20 steps), all aimed at the robust model itself.

| ε | Transfer FGSM | Transfer I-FGSM | **White-box FGSM** | **White-box I-FGSM** | **White-box PGD-20** |
|---|---|---|---|---|---|
| 0.1 | 98.6% | 98.6% | 97.6% | 97.0% | 97.2% |
| 0.2 | 97.8% | 97.8% | 96.4% | 95.2% | 95.2% |
| **0.3** | **97.0%** | **97.0%** | 95.0% | **91.6%** | **90.4%** |
| 0.4 | 96.4% | 96.4% | 93.2% | 84.2% | 76.4% |
| 0.5 | 95.4% | 95.2% | 90.8% | 75.0% | **43.0%** |
| 0.6 | 92.4% | 92.8% | 89.0% | 61.4% | 17.2% |
| 0.7 | 86.2% | 88.0% | 87.2% | 43.8% | 5.4% |
| 0.8 | 76.8% | 80.6% | 84.6% | 27.2% | 1.8% |
| 0.9 | 63.2% | 74.2% | 82.2% | 10.8% | 0.6% |
| **1.0** | 54.0% | 61.6% | **78.8%** | **2.2%** | **0.0%** |

What it shows:

1. **At the main ε (0.3) the defense is real.** Even PGD aimed at the robust model leaves **90.4%**, against the baseline's 52%. The 3.4-point white-box FGSM - I-FGSM gap is inside the course's "healthy 3-5 points".
2. **Above about 0.4, the model is fooling the attacker rather than resisting.** Its own FGSM gradient is useless (78.8% at 1.0, *higher* than the 54% transfer result), but iterative attacks still drive it to **0%**.
   - This is the signature of **gradient masking** / **catastrophic overfitting**, the FGSM-only training risk from Sections 10-11.
   - The large ε values were only ever trained with **single-step** FGSM.
3. **Iterative attacks transfer *worse* than single-step ones.** That's why the evaluator's I-FGSM column is sometimes *higher* than its FGSM column (0.4: 97.0 vs 96.8; 1.0: 77.0 vs 76.6 in the course, and 61.6 vs 54.0 in mine). I-FGSM examples are tailored to the baseline's exact quirks.
   - **In a transfer test, I-FGSM ≥ FGSM is a red flag**, not proof of strength.
   - **The course's gap rule (I-FGSM should be 3-5 points *lower*) can't be applied to these numbers.**

My 10-epoch model is weaker at high ε than the course's (transfer 54% vs 76.6% at 1.0); the course probably trained longer. The white-box *pattern* is the point, not the exact values.

## Part 7 - Interpreting the patterns (course's rules)

| Course's rule | Use | Check against the course's own numbers |
|---|---|---|
| A "well-trained" model ==degrades gradually and stays above 80% at ε = 1.0== | Target shape | The course's own spread model gets **76.6%** at 1.0, so it **misses** its own bar (cell 1) |
| **A steep drop at one ε** (e.g. 95% at 0.3 -> 60% at 0.4) means **epsilon overfitting** | Fix: spread training | The single-ε model's steepest drops come at higher ε (biggest 15.6 points, 0.9 -> 1.0) |
| **I-FGSM 3-5 points below FGSM is normal**; **more than 10 points means the defense relies on single-step artifacts** | Gap rule | Only meaningful for **white-box** results (Part 6) |
| "Below 85% at ε = 1.0 is acceptable... 80%+ is strong" | Expectation at extreme ε | Mixed wording, read it as: about 80% at 1.0 is a good result; below that is common |

## Part 8 - Improving your results

| Symptom | Likely causes | Fixes the course gives |
|---|---|---|
| **Clean < 95%** | Too little clean learning | Lower the training ε, raise the clean ratio, train more epochs |
| **Robust < 90%** | Training ε too low, too few epochs, or a bug in the loop | Raise the training ε to 0.25-0.3, train more epochs, **check attacks are made fresh every batch** |

```shellsession
Ishman1322@htb[/htb]$ python evaluate_robustness.py --model-path robust_model.safetensors --show-failures
<SNIP>
Misclassified samples (I-FGSM ε=0.3):
  Sample 1842: 3 -> 8 (conf=0.87)
  Sample 2891: 7 -> 1 (conf=0.72)
  Sample 4123: 4 -> 9 (conf=0.65)
  Sample 5567: 9 -> 4 (conf=0.91)
  Sample 7234: 2 -> 7 (conf=0.58)
<SNIP>
```

| Field | Meaning |
|---|---|
| `3 -> 8` | True digit -> predicted digit under attack. 3/8 and 4/9 look alike, so they're common confusions |
| `conf=0.91` | The softmax probability of the *wrong* class. High means the attack pushed the image **deep** into the wrong region, not just across the boundary |

> **An honest note the course skips:**
> - **The "increase training ε to 0.25 or 0.3" advice is for single-ε training.** The lab's spread already goes up to 1.0, so for this code the matching fixes are more epochs, or adding I-FGSM/PGD batches.
> - **The failure-sample numbers are illustrative.** Indices like 1842 and 7234 can't come from a 500-sample example file.

## Reading the figure/output
There are no figures; all four outputs are explained in Parts 2-5. Section 9's bar chart is the ε = 0.3 row of the Part 3-4 tables.

## Copy-paste cells (mine, not the course's)
Use the same notebook as Section 13. Cell 2 needs `adv_examples.safetensors` and `baseline_model.safetensors`; cell 3 also needs `robust_model.safetensors`. They use `LeNet5`, `fgsm_attack`, `i_fgsm_attack`, `MNIST_MEAN`, `MNIST_STD`, `F`, `json`, `torch` and `device`.

### 1) Apply the course's own rules to its comparison tables

```python
eps = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
fgsm = {"baseline": [95.2, 86.6, 73.8, 56.2, 40.4, 25.8, 16.4, 10.8, 8.0, 7.0], # course's comparison summary
        "single_epsilon": [99.0, 98.4, 97.0, 95.8, 93.2, 86.4, 76.2, 65.4, 52.4, 36.8],
        "spread_epsilon": [98.4, 98.0, 97.6, 96.8, 95.8, 94.6, 93.6, 90.8, 87.0, 76.6]}
ifgsm = {"baseline": [94.2, 80.8, 52.0, 26.8, 9.0, 2.4, 0.6, 0.4, 0.2, 0.0],
         "single_epsilon": [98.8, 98.4, 97.2, 95.8, 92.2, 84.8, 71.8, 55.6, 41.2, 24.8],
         "spread_epsilon": [98.4, 97.8, 97.6, 97.0, 95.6, 94.6, 93.0, 89.0, 84.2, 77.0]}
for m in fgsm:
    drops = [fgsm[m][i] - fgsm[m][i + 1] for i in range(len(eps) - 1)]
    i = max(range(len(drops)), key=drops.__getitem__)
    gap03 = fgsm[m][2] - ifgsm[m][2]
    print(f"{m:14} | steepest FGSM drop {drops[i]:4.1f} pts ({eps[i]} -> {eps[i + 1]}) | "
          f"FGSM-I-FGSM gap @0.3 {gap03:4.1f} | @1.0 FGSM {fgsm[m][-1]:4.1f}% -> course's '>80%' bar: {'met' if fgsm[m][-1] > 80 else 'NOT met'}")
```

Expected:
```txt
baseline | steepest FGSM drop 17.6 pts (0.3 -> 0.4) | FGSM-I-FGSM gap @0.3 21.8 | @1.0 FGSM 7.0% -> course's '>80%' bar: NOT met
single_epsilon | steepest FGSM drop 15.6 pts (0.9 -> 1.0) | FGSM-I-FGSM gap @0.3 -0.2 | @1.0 FGSM 36.8% -> course's '>80%' bar: NOT met
spread_epsilon | steepest FGSM drop 10.4 pts (0.9 -> 1.0) | FGSM-I-FGSM gap @0.3 0.0 | @1.0 FGSM 76.6% -> course's '>80%' bar: NOT met
```

### 2) "Attack success" done properly: only count images that were right beforehand

```python
from safetensors.torch import load_file
adv = load_file("adv_examples.safetensors")
X, Y = adv["clean_images"].to(device), adv["clean_labels"].to(device)
base = LeNet5(); base.load_state_dict(load_file("baseline_model.safetensors")); base.to(device).eval()
with torch.no_grad():
    ok = base(X).argmax(1) == Y # images the baseline gets right with NO attack
    fooled = base(adv["ifgsm_eps_0.3"].to(device)).argmax(1) != Y
print(f"evaluator-style 'attack success' (100 - accuracy): {fooled.float().mean().item() * 100:.1f}%")
print(f"true attack success (only originally-correct images): {(fooled & ok).sum().item() / ok.sum().item() * 100:.1f}% "
      f"({(fooled & ok).sum().item()}/{ok.sum().item()})")
```

Expected (my baseline; yours will be close):
```txt
evaluator-style 'attack success' (100 - accuracy): 46.0%
true attack success (only originally-correct images): 45.5% (225/495)
```

### 3) White-box vs transfer: re-attack your robust model yourself
This is the check to put in a report. It reuses the course's attacks and adds PGD (I-FGSM with a random start).

```python
from safetensors import safe_open
from safetensors.torch import load_file

def load_lenet(path):
    m = LeNet5(); m.load_state_dict(load_file(path)); return m.to(device).eval()

adv = load_file("adv_examples.safetensors")
with safe_open("adv_examples.safetensors", "pt") as f:
    meta = f.metadata()
spread = json.loads(meta["epsilon_spread"])
X, Y = adv["clean_images"].to(device), adv["clean_labels"].to(device)

def acc(model, x):
    with torch.no_grad():
        return (model(x).argmax(1) == Y).float().mean().item() * 100

def pgd_attack(model, images, labels, epsilon, steps=20):
    """I-FGSM + random start (= PGD). Not from the course."""
    lo, hi = (0 - MNIST_MEAN) / MNIST_STD, (1 - MNIST_MEAN) / MNIST_STD
    x0 = images.clone().detach()
    x = torch.clamp(x0 + torch.empty_like(x0).uniform_(-epsilon, epsilon), lo, hi)
    alpha = 2.5 * epsilon / steps
    for _ in range(steps):
        x.requires_grad = True
        loss = F.cross_entropy(model(x), labels)
        grad, = torch.autograd.grad(loss, x)
        x = x.detach() + alpha * grad.sign()
        x = torch.clamp(x0 + torch.clamp(x - x0, -epsilon, epsilon), lo, hi)
    return x.detach()

robust = load_lenet("robust_model.safetensors")
print(f"clean (500 saved samples): {acc(robust, X):.1f}%")
print(f"{'eps':>4} | {'transfer FGSM':>13} {'transfer I-FGSM':>15} | {'white-box FGSM':>14} {'white-box I-FGSM':>16} {'white-box PGD-20':>16}")
for e in spread:
    k = f"{e:.1f}"
    wb_f = acc(robust, fgsm_attack(robust, X, Y, e))
    wb_i = acc(robust, i_fgsm_attack(robust, X, Y, e))
    wb_p = acc(robust, pgd_attack(robust, X, Y, e))
    print(f"{e:4.1f} | {acc(robust, adv['fgsm_eps_' + k].to(device)):12.1f}% {acc(robust, adv['ifgsm_eps_' + k].to(device)):14.1f}% | "
          f"{wb_f:13.1f}% {wb_i:15.1f}% {wb_p:15.1f}%")
```

Expected (my 10-epoch model; PGD's random start makes its column vary by about ±1 point):
```txt
clean (500 saved samples): 99.0%
 eps | transfer FGSM transfer I-FGSM | white-box FGSM white-box I-FGSM white-box PGD-20
 0.1 | 98.6% 98.6% | 97.6% 97.0% 97.2%
 0.2 | 97.8% 97.8% | 96.4% 95.2% 95.2%
 0.3 | 97.0% 97.0% | 95.0% 91.6% 90.4%
 0.4 | 96.4% 96.4% | 93.2% 84.2% 76.4%
 0.5 | 95.4% 95.2% | 90.8% 75.0% 43.0%
 0.6 | 92.4% 92.8% | 89.0% 61.4% 17.2%
 0.7 | 86.2% 88.0% | 87.2% 43.8% 5.4%
 0.8 | 76.8% 80.6% | 84.6% 27.2% 1.8%
 0.9 | 63.2% 74.2% | 82.2% 10.8% 0.6%
 1.0 | 54.0% 61.6% | 78.8% 2.2% 0.0%
```

## Key facts to lock in
- `evaluate_robustness.py` **grades a saved LeNet-5 on 500 pre-made images × 10 ε × {FGSM, I-FGSM}**. `--compare` adds the baseline and the improvement; `--show-failures` lists misclassified examples (true -> predicted, with confidence).
- Each row shows ==accuracy (defender) and attack success = 100 − accuracy (attacker)== per ε, with ==ε = 0.3 as the summary row==. 100 − accuracy slightly overstates attacks because it counts images that were wrong to begin with.
- The course's results at ε = 0.3: **baseline 73.8 / 52.0 -> spread model 97.6 / 97.6 (+23.8 / +45.6 points)**; clean 98.6 -> 99.0.
- At ε = 1.0: **spread 76.6 / 77.0 vs single-ε 36.8 / 24.8**. Single-ε gets 99.4% clean against 99.0% for spread. Spread keeps I-FGSM attack success ==below 6% up to ε = 0.6==.
- The course's rules: ==degrade gradually, about 80% or more at ε = 1.0; a steep drop means epsilon overfitting; an I-FGSM gap of 3-5 points is normal, more than 10 is fragile==. Its own spread model misses the 80% bar (76.6%).
- The evaluator's examples come from the **baseline**, so **grading the robust model is a transfer test**. My white-box re-test: ==91.6% under I-FGSM and 90.4% under PGD at ε = 0.3, but 2.2% and 0.0% at ε = 1.0==.
- **White-box FGSM staying high while I-FGSM/PGD collapse**, and **transferred I-FGSM scoring at or above FGSM**, are the signatures of **gradient masking** from FGSM-only training.
- **Method point:** in a report, **quote white-box PGD/I-FGSM results against the defended model at several ε values, not just transferred examples**. Recommend PGD (random-start) adversarial training where large ε matters, and confirm robustness with an attack the model wasn't trained on.
