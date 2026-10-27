# I-FGSM Analysis

## The idea in one breath

This section studies **how to tune** I-FGSM and how it compares with FGSM. It covers:

- **Choosing α and T:** big steps can overshoot; tiny steps may never reach far enough.
- **Random starts:** they turn near-misses into wins.
- **Vocabulary:** I-FGSM, BIM and PGD are close relatives.
- **Targeted I-FGSM:** the Section 9 trick applied every step.

The headline result: at the same budget ε = 0.7, **FGSM flipped 57.8% of correct predictions and I-FGSM flipped 95.3%**. The figures show I-FGSM's wrong answers are also far more confident (99.7% vs FGSM's 51%).

> **Heads-up: flattened formulas in your paste.** Read them like this:
> 
> - `x(t+1)=ΠB∞(x(t)+α⋅sign(∇L))` means **x⁽ᵗ⁺¹⁾ = Π_B∞(x,ε)(x⁽ᵗ⁾ + α·sign(∇ₓℒ))**. The (x, ε) was dropped, but the projection is still around the **original** x.
> - `α=ϵT` means **α = ε / T**
> - `x(t+1)=Πℬ∞(x,ϵ)(x(t)−αsign(∇x(t)ℒ(θ,x(t),yt)))` means **x⁽ᵗ⁺¹⁾ = Π_B∞(x,ε)(x⁽ᵗ⁾ − α·sign(∇ₓℒ(θ, x⁽ᵗ⁾, yₜ)))**
> - As before, (t) is a **step counter**, not a power.

## Part 1: Visualizing I-FGSM

```python
def visualize_ifgsm(model: nn.Module,
                    image: Tensor,
                    label: Tensor,
                    epsilon: float,
                    num_iter: int,
                    targeted: bool = False,
                    target_class: int | None = None) -> None:
    """Wrapper for visualize_attack using I-FGSM.

    Args:
        model: Classifier model
        image: Single image tensor [C,H,W]
        label: True label
        epsilon: Perturbation budget
        num_iter: Number of iterations
        targeted: If True, targeted attack
        target_class: Target class for targeted attacks
    """
    alpha = epsilon / max(num_iter, 1)

    def _make_adv(m, xb, yb):
        y_used = yb if not targeted else torch.full_like(yb, target_class)
        return iterative_fgsm(
            m, xb, y_used,
            epsilon, num_iter, alpha,
            targeted=targeted,
            random_start=True
        )

    mode = "Targeted" if targeted else "Untargeted"
    visualize_attack(
        model, image, label, _make_adv,
        title=f"I-FGSM {mode}",
        targeted=targeted,
        target_class=target_class
    )

# Visualize first sample from test batch
_ = visualize_ifgsm(
    model,
    images[0].detach().cpu(),
    labels[0].detach().cpu(),
    epsilon,
    num_iter,
    targeted=False
)
```

|Code|What it does|
|---|---|
|`alpha = epsilon / max(num_iter, 1)`|the default α = ε/T (0.08 for ε = 0.8, T = 10)|
|`def _make_adv(m, xb, yb):`|the same **closure** pattern as Section 8's FGSM wrapper: an attack function that remembers ε, T and α|
|`torch.full_like(yb, target_class)`|for targeted mode, swaps in the **target** labels|
|`iterative_fgsm(..., random_start=True)`|the Section 11 attack, always with a random start here|
|`visualize_attack(model, image, label, _make_adv, ...)`|the Section 8 four-panel plotter. **This is the payoff of passing the attack as a function: no plotting code had to change**|
|`images[0]`, `labels[0]`, `epsilon`, `num_iter`|first image of your batch, ε = 0.8, T = 10 (from Section 11)|

> **If this errors:** `int | None` needs Python 3.10+; use `Optional[int]` on older Python (same fix as Sections 8 and 11).

> **Note:** unlike the FGSM wrapper, this one has **no check** that `target_class` was given when `targeted=True`. If you forget it, `torch.full_like(yb, None)` raises a `TypeError`.

## Part 2: Reading the two I-FGSM figures

### "I-FGSM Untargeted" (the same "7" as Section 8)

!["I-FGSM Untargeted" (the same "7" as Section 8)](/assets/Pasted%20image%2020261015222953.png)

!["I-FGSM Untargeted" (the same "7" as Section 8)](/assets/Pasted%20image%2020261015222917.png)

|Panel|FGSM (Section 8)|I-FGSM (this section)|
|---|---|---|
|Adversarial prediction|3 at **51.45%**|3 at **99.74%**|
|True class (7) after attack|≈ 43%|≈ 0%|
|Adversarial image|random-looking grey speckle|**streaky, stroke-like patterns** in the background|
|Perturbation panel|almost two-tone (saturated ±ε)|many in-between shades|

- **The probability bars:** clean puts everything on 7 (blue); adversarial puts everything on 3 (red). **One step barely tipped the scale; ten steps made the model _confidently_ wrong.**
- **The perturbation panel** uses the same heat colours as before: **white** = brightened, **orange** = no change, **black/dark red** = darkened, and now lots of **yellow** = small brightening. Those in-between shades are Section 11's point: **I-FGSM pixels don't all sit at ±ε; many stop somewhere inside the box.**
- The streaks in the adversarial image are what the course calls "more structured." The attack is drawing faint stroke-like shapes that push the model toward "3," instead of uniform noise.

### "I-FGSM Targeted" (the same "1" as Section 9)

!["I-FGSM Targeted" (the same "1" as Section 9)](/assets/Pasted%20image%2020261015223057.png)

!["I-FGSM Targeted" (the same "1" as Section 9)](/assets/Pasted%20image%2020261015223031.png)
- **Original:** class 1 at 99.92%. **Adversarial:** class 7 at **99.76%**, with target = 7 (a hit).
- Compare targeted FGSM in Section 9: the same 1 -> 7, but only **43.54%**, with "1" and "2" still close behind.
- **Iterating turns a narrow targeted win into a near-certain one.** To a human, the image is still clearly a "1."

## Part 3: Step size, iterations and random starts

### Toy first: α and T on the hill

The Section 10 hill again: L(x) = 1 − (x − 0.3)², top at x = 0.3, start at 0, budget ε = 0.8. Three settings:

|Setting|Path (x, loss)|Ends at|What went wrong or right|
|---|---|---|---|
|**α = 0.4, T = 2**|0.4 (0.99) -> **0.0** (0.91)|loss 0.91|**Big steps bounce**: step 1 overshoots the top, and step 2 jumps all the way back to the start|
|**α = 0.1, T = 8**|0.1 -> 0.2 -> 0.3 (1.00) -> jitters 0.2 <-> 0.3|loss 0.99-1.00|Good: small enough to settle at the top, and α·T = 0.8 is enough reach|
|**α = 0.01, T = 8**|0.01 -> 0.02 -> ... -> 0.08|loss 0.95|**Too timid**: α·T = 0.08 total travel never reaches the top at 0.3|

I ran these numbers to confirm them. Two lessons:

- **Too large an α overshoots and bounces** (the course's "rapid jumps: 0.3 -> 0.9 -> 1.1").
- **Too small an α·T can't reach far enough**. A practical rule: **α·T ≥ ε**, so the attack can at least reach the edge of its budget.

### The course's illustration

|Setting|Loss path|Feel|
|---|---|---|
|α = 0.1, T = 2|0.3 -> 0.9 -> 1.1|fast, jumpy, may overshoot|
|α = 0.01, T = 20|0.3 -> 0.35 -> 0.41 -> ... -> 1.45|smooth, tracks the curve|

> **An honest note the course skips:** both settings have the **same total reach**, α·T = 0.1 × 2 = 0.01 × 20 = **0.2**. So this is a fair comparison of granularity _for a budget of about 0.2_. With your ε = 0.8, either one would use only a quarter of the budget. When you tune, keep α·T ≥ ε, then trade α against T for smoothness versus speed (each extra step is one more gradient computation).

### Random starts

Adding uniform noise in [−ε, ε] before iterating starts the climb from a different spot. The course's illustrative claim is that this can raise success **from 85% to 92%** at the same budget, by rescuing near-misses where the gradient at the clean image "points nowhere useful."

> **Reminder (Section 11):** your `iterative_fgsm` reseeds with 1337 on every call, so its "random" start is the **same** noise every time. True **multiple restarts** (try several different random starts and keep the best) would need a modified function.

## Part 4: How I-FGSM, BIM and PGD relate

|Name|What it is|Random start?|Restarts?|
|---|---|---|---|
|**BIM** (Kurakin et al., 2016)|iterative sign steps plus projection|traditionally **no**|no|
|**I-FGSM**|the same algorithm as BIM (the names are used interchangeably)|optional|no|
|**PGD** (Madry et al., 2017)|the general "step, then project" attack under a constraint|**yes**|optionally **several**, keep the strongest|

With the sign step, α = ε/T and a random start, your `iterative_fgsm` **is** standard L∞ PGD (one restart). ==Conceptually, PGD = I-FGSM + random initialization (+ optional multiple restarts, keeping the strongest result).== PGD also exists for other norms: for an L2 budget, the step uses the rescaled gradient g/‖g‖₂ instead of the sign (Section 3's L2 variant).

## Part 5: Targeted I-FGSM

### The formula

```
x⁽ᵗ⁺¹⁾ = Π_B∞(x,ε)( x⁽ᵗ⁾ − α · sign( ∇ₓ ℒ(θ, x⁽ᵗ⁾, yₜ) ) )
```

|Symbol|Plain English|
|---|---|
|`yₜ`|the target class (e.g. 7) used in the loss|
|`−`|step **downhill** on the target's loss, toward yₜ|
|`Π_B∞(x,ε)`|snap back into the ε-box around the **original** image each step|

Read aloud: **"Every step, act as if the target were the right answer, step toward it, and snap back inside the budget."** It's targeted FGSM (Section 9) repeated with projection. The course's tips: it often needs more steps or a slightly larger ε, and it benefits from smaller α with more steps, random starts, and **early stopping** once the model predicts yₜ.

### The code

```python
# Find one sample of '1'
one_img, one_lbl = None, None
for xb, yb in test_loader:
    m = (yb == 1)
    if m.any():
        j = m.nonzero(as_tuple=True)[0][0].item()
        one_img = xb[j].to(device)
        one_lbl = yb[j].to(device)
        break

# Try increasing epsilon values until successful
for eps_try in [0.5, 0.8, 1.0]:
    x_adv = iterative_fgsm(
        model,
        one_img.unsqueeze(0),
        torch.tensor(7, device=device).unsqueeze(0), # target label
        epsilon=eps_try,
        num_iter=num_iter,
        alpha=eps_try / max(num_iter, 1),
        targeted=True,
        random_start=True,
    )
    with torch.no_grad():
        pred = model(x_adv).argmax(dim=1).item()
    print(f"epsilon={eps_try:.2f} -> predicted {pred}")

    if pred == 7:
        _ = visualize_ifgsm(
            model,
            one_img.detach().cpu(),
            one_lbl.detach().cpu(),
            eps_try,
            num_iter,
            targeted=True,
            target_class=7,
        )
        break
```

Output:

```txt
epsilon=0.50 -> predicted 1
epsilon=0.80 -> predicted 7
```

|Code|What it does|
|---|---|
|`m = (yb == 1)` / `m.any()`|does this batch contain a "1"?|
|`m.nonzero(as_tuple=True)[0][0].item()`|position of the **first** "1"|
|`xb[j].to(device)`|that image, moved to the model's device|
|`torch.tensor(7, device=device).unsqueeze(0)`|the **target** label as shape [1]. It's the same as `torch.tensor([7], device=device)`, just written differently|
|`alpha=eps_try / max(num_iter, 1)`|keeps α = ε/T for each ε tried|
|`targeted=True`|`step_dir = −1` inside the function|
|`if pred == 7:` -> visualize, `break`|early stopping at the first ε that works|

> **Two honest notes the course skips:**
> 
> 1. **This search doesn't check the "1" is classified correctly**, unlike Section 9's careful version. It takes the first "1" it sees. Usually fine, but in principle you could "succeed" on an image the model already misread.
> 2. **The course's conclusion contradicts its own earlier output.** It says I-FGSM succeeded at 0.8 "compared to FGSM which required epsilon=1.0." But Section 9's output showed FGSM **also** succeeded at 0.8, with the identical printout `0.50 -> 1, 0.80 -> 7`. With such a coarse ε list, this example **doesn't show** an ε advantage. The real advantage is visible in the **confidence** (99.76% vs 43.54%), and probably in a lower true minimum ε. The first cell below measures that properly.

**Notebook note:** this cell **overwrites `x_adv` again** (now a single I-FGSM image) and creates `one_img`, `one_lbl`, `eps_try` and `pred`.

## Part 6: FGSM vs I-FGSM, head to head

```python
# Compare FGSM (one-step) and I-FGSM on the same batch
# Run both attacks with same epsilon
epsilon = 0.7
x_adv_fgsm = fgsm_attack(model, images, labels, epsilon)
x_adv_ifgsm = iterative_fgsm(
    model, images, labels,
    epsilon, num_iter=10,
    random_start=True
)

# Compare success rates
with torch.no_grad():
    fgsm_pred = model(x_adv_fgsm).argmax(dim=1)
    ifgsm_pred = model(x_adv_ifgsm).argmax(dim=1)

orig_correct = clean_pred == labels
fgsm_success = (
    ((fgsm_pred != labels) & orig_correct).float().sum()
    / orig_correct.float().sum().clamp_min(1.0)
)
ifgsm_success = (
    ((ifgsm_pred != labels) & orig_correct).float().sum()
    / orig_correct.float().sum().clamp_min(1.0)
)

print(f"FGSM success rate: {fgsm_success:.1%}")
print(f"I-FGSM success rate: {ifgsm_success:.1%}")
print(f"Improvement: {(ifgsm_success - fgsm_success) / fgsm_success:.1%}")
```

Output:

```txt
FGSM success rate: 57.8%
I-FGSM success rate: 95.3%
Improvement: 64.9%
```

|Code|What it does|
|---|---|
|`epsilon = 0.7`|**a new, smaller budget (≈ 0.216 pixel ≈ 55/255)**, so the comparison is harder than at 0.8|
|`fgsm_attack(...)` / `iterative_fgsm(..., num_iter=10, random_start=True)`|both attacks, same batch, same ε; α defaults to 0.07|
|`orig_correct = clean_pred == labels`|the originally-correct mask (uses `clean_pred` from Section 11, same batch)|
|`fgsm_success`, `ifgsm_success`|flips ÷ originally correct, the Section 6/7 rule|
|`(ifgsm − fgsm) / fgsm`|****relative** improvement**|

### Toy first: absolute vs relative improvement

Say method A succeeds 50% of the time and method B 75%:

```
absolute: 75 − 50 = +25 percentage points
relative: (75 − 50) / 50 = +50% ("B is 1.5× as successful")
```

Now the real numbers:

```
absolute: 95.3 − 57.8 = +37.5 percentage points
relative: 37.5 / 57.8 = 0.649 = +64.9% <- what the code prints
```

**A "64.9% improvement" is relative, not "64.9 percentage points."** In a report, state both: "success rose from 57.8% to 95.3% (+37.5 points, a 64.9% relative increase)." Relative numbers can sound bigger than they are, especially when the starting value is small.

**Notebook note:** this cell sets **`epsilon = 0.7`** (it was 0.8) and **overwrites** `x_adv_fgsm` and `x_adv_ifgsm` with ε = 0.7 versions. Re-run with `epsilon = 0.8` if a later cell expects the old ones.

## Copy-paste cells (mine, not the course's)

These use names in your notebook: `model`, `one_img`, `images`, `labels`, `clean_pred`, `fgsm_attack`, `iterative_fgsm`, `fgsm_success`, `ifgsm_success`.

### Find each attack's true minimum targeted ε

A fair version of the course's 1 -> 7 example: a fine ε scan for both attacks on the same "1."

```python
tgt = torch.tensor([7], device=device)
x1 = one_img.unsqueeze(0)
for name in ["FGSM", "I-FGSM"]:
    found = None
    for e in np.arange(0.30, 1.01, 0.02):
        e = float(e)
        if name == "FGSM":
            xa = fgsm_attack(model, x1, tgt, e, targeted=True)
        else:
            xa = iterative_fgsm(model, x1, tgt, epsilon=e, num_iter=10, targeted=True)
        with torch.no_grad():
            if model(xa).argmax(dim=1).item() == 7:
                found = e
                break
    print(f"{name:7s} min eps for 1->7: {found}")
```

Expected: two lines, and I-FGSM's minimum is most likely **lower** than FGSM's. That's the ε advantage the course's coarse list couldn't show. `None` means no success up to 1.0.

### Sweep α and T at ε = 0.7

Tests the Part 3 lessons on your model: bouncy big steps, timid small steps, and the balanced setting.

```python
ok = (clean_pred == labels)
for a, T in [(0.35, 2), (0.07, 10), (0.035, 20), (0.01, 20)]:
    xa = iterative_fgsm(model, images, labels, epsilon=0.7, num_iter=T, alpha=a)
    with torch.no_grad():
        pa = model(xa).argmax(dim=1)
    rate = ((pa != labels) & ok).float().sum() / ok.float().sum().clamp_min(1.0)
    print(f"alpha={a:<5} T={T:<3} alpha*T={a*T:.2f} success={rate.item():.1%}")
```

Expected: the (0.01, 20) row is clearly the weakest (α·T = 0.2 < ε = 0.7). The (0.07, 10) and (0.035, 20) rows score highest.

### Report both kinds of improvement

Prints the head-to-head result the way you'd write it in a report.

```python
f, i = fgsm_success.item(), ifgsm_success.item()
print(f"FGSM {f:.1%} -> I-FGSM {i:.1%}: +{(i - f) * 100:.1f} points, "
      f"{(i - f) / f:.1%} relative")
```

Expected: `FGSM 57.8% -> I-FGSM 95.3%: +37.5 points, 64.9% relative` (your own numbers may differ slightly).

## Key facts to lock in

- ==At the same L∞ budget ε = 0.7, I-FGSM reached a 95.3% success rate versus 57.8% for FGSM: +37.5 percentage points, a 64.9% relative improvement==. Always say whether an improvement is absolute or relative.
- **Iteration also makes the model confidently wrong: the same "7" became a "3" at 99.74% under I-FGSM versus 51.45% under FGSM, and the targeted 1 -> 7 reached 99.76% versus 43.54%**.
- **Too large an α overshoots and bounces, while too small an α·T can't reach the edge of the budget, so choose α·T ≥ ε and then trade step size against the number of steps**. Each extra iteration costs one more gradient computation.
- **Random starts add uniform noise in [−ε, ε] before iterating, rescuing near-misses where the clean-image gradient is uninformative** (the course's illustrative gain is 85% -> 92%). The HTB implementation reseeds 1337 every call, so its start is always the same noise.
- **BIM is iterative sign steps with projection and no random start; I-FGSM is the same algorithm; PGD (Madry et al., 2017) adds random initialization and optionally multiple restarts, keeping the strongest result**. Your `iterative_fgsm` with `random_start=True` is standard L∞ PGD with one restart.
- ==Targeted I-FGSM is x⁽ᵗ⁺¹⁾ = Π_B∞(x,ε)(x⁽ᵗ⁾ − α·sign(∇ₓℒ(θ, x⁽ᵗ⁾, yₜ))), the same two changes as targeted FGSM applied every step==, and it benefits from smaller α, more steps, random starts and early stopping once the target is predicted.
- **I-FGSM perturbations look more structured (stroke-like patterns) and use in-between values inside [−ε, ε]**, unlike FGSM's uniform ±ε noise, because each step re-aims at the model's most sensitive regions.
- **Method point:** **to compare attacks honestly, use a fine ε scan to find each attack's minimum successful ε on the same correctly classified inputs, and report success, confidence and the L∞ check together**. The course's coarse [0.5, 0.8, 1.0] list showed the same 0.8 for both attacks, hiding I-FGSM's real advantage.
