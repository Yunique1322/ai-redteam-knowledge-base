# Normalization

## The idea in one breath

**Normalization** rescales every pixel with `(x − μ) / σ`, so the data is centred on zero and has a standard spread. The image looks identical to a human, but the numbers the model sees are different. That's why the model trains faster, and it's also the trap for attacks: **your ε, your clamping range and your gradients all live in normalized space**, while humans (and most lab instructions) think in plain [0, 1] pixel space. Get the conversion wrong and your "small" attack is the wrong size, or your image gets silently corrupted.

> **Heads-up: some formulas were flattened when you pasted them.** Read them like this:
> 
> - `xnorm=x−μσ` means **x_norm = (x − μ) / σ**
> - `∂loss∂w=x⋅δ` means **∂loss/∂w = x · δ**. Here δ is the **backprop error signal**, _not_ the perturbation δ from earlier sections. Same letter, different meaning.
> - `−0.42,2.82-0.42, 2.82` and `0,10, 1` are duplicated ranges: **[−0.42, 2.82]** and **[0, 1]**

## Part 1: The toy example, built up step by step

To keep the arithmetic clean, imagine a toy dataset whose pixels have mean **μ = 0.2** and standard deviation **σ = 0.4**. (invented numbers dont try to calculate it)

### Step 1: Normalize three pixels

Subtract the mean, then divide by the spread:

```
black pixel x = 0.0 -> (0.0 − 0.2) / 0.4 = −0.2 / 0.4 = −0.5
mid pixel x = 0.6 -> (0.6 − 0.2) / 0.4 = 0.4 / 0.4 = 1.0
white pixel x = 1.0 -> (1.0 − 0.2) / 0.4 = 0.8 / 0.4 = 2.0
```

- **Subtracting μ** shifts everything so the average pixel lands on 0 (**centering**).
- **Dividing by σ** stretches or squeezes so the typical spread is 1 (**unit variance**).

A normalized value means "**how many standard deviations above or below average**." A white pixel here is 2.0 σ above average.

### Step 2: The valid range changes

Plain pixels live in [0, 1]. After normalizing, the darkest and brightest possible values become:

```
min = (0 − 0.2) / 0.4 = −0.5
max = (1 − 0.2) / 0.4 = 2.0
```

**The valid range is now [−0.5, 2.0], not [0, 1].** Any adversarial image must stay inside this new range.

### Step 3: Undo it (denormalize)

Reverse the two steps: multiply by σ, then add μ back.

```
x = x_norm × σ + μ

x_norm = 1.0 -> 1.0 × 0.4 + 0.2 = 0.6 ✓ (back to the mid pixel)
```

This is what `mnist_denormalize` does, so you can display images for humans.

### Step 4: Converting ε, where the mean cancels out

Suppose an attack changes a normalized pixel from 1.0 to 1.5, so ε_norm = 0.5. What's the change in plain pixels?

```
x_norm = 1.0 -> x = 1.0 × 0.4 + 0.2 = 0.60
x_norm = 1.5 -> x = 1.5 × 0.4 + 0.2 = 0.80

pixel change = 0.80 − 0.60 = 0.20 = 0.5 × 0.4
```

The **+0.2 (the mean) appears in both lines and cancels** when you subtract. Only σ survives:

```
ε_pixel = σ × ε_norm and the other way round: ε_norm = ε_pixel / σ
```

**For converting a _change_ (ε), only σ matters. The mean drops out.** The mean only matters when you convert an actual _pixel value_ (Steps 1 and 3).

### Step 5: The clamping trap

FGSM clamps the adversarial image to the valid range. Clamp to the **wrong** range and you corrupt the image. Watch what happens if you clamp normalized values to [0, 1] by mistake:

```
black pixel: x_norm = −0.5 -> wrongly clamped to 0 -> back in pixels: 0 × 0.4 + 0.2 = 0.2 (black became grey!)
white pixel: x_norm = 2.0 -> wrongly clamped to 1 -> back in pixels: 1 × 0.4 + 0.2 = 0.6 (white became dim!)
```

Every background pixel gets brightened and every stroke gets dimmed, which is a huge change that has nothing to do with ε. **Clamp in the same space the image is in: [−0.5, 2.0] here, not [0, 1].**

### Step 6: The real formulas

```
x_norm = (x − μ) / σ normalize
x = x_norm · σ + μ denormalize
ε_pixel = σ · ε_norm convert a budget (mean cancels)
valid normalized range = [(0 − μ)/σ , (1 − μ)/σ]
```

|Symbol|Plain English|Toy|Real MNIST|
|---|---|---|---|
|`x`|plain pixel in [0, 1]|0.6|any|
|`μ` (mu)|average pixel brightness across the training set|0.2|**0.1307**|
|`σ` (sigma)|how much pixels typically vary from that average|0.4|**0.3081**|
|`x_norm`|pixel in normalized units ("σ's above average")|1.0|any|
|`ε_norm`|attack budget in normalized units|0.5|e.g. 0.8|
|`ε_pixel`|the same budget in plain pixel units|0.2|e.g. 0.246|

Read aloud: **"Normalize by subtracting the mean and dividing by the spread; a budget converts between spaces by multiplying or dividing by σ alone."**

## Part 2: The real MNIST numbers (HTB's example)

### The values

MNIST is mostly black background, so its average brightness is low (**μ = 0.1307**). **σ = 0.3081** comes from all 60,000 training images.

|Pixel|Calculation|Normalized value|
|---|---|---|
|black (0.0)|(0.0 − 0.1307) / 0.3081|**≈ −0.424** (`MNIST_NORM_MIN`)|
|bright stroke (0.8)|(0.8 − 0.1307) / 0.3081|**≈ 2.17**|
|pure white (1.0)|(1.0 − 0.1307) / 0.3081|**≈ 2.821** (`MNIST_NORM_MAX`)|

These two constants, **`MNIST_NORM_MIN ≈ −0.424` and `MNIST_NORM_MAX ≈ 2.821`**, are the clamp bounds you'll see in the attack code later.

The clamping trap on real MNIST: wrongly clamping a normalized black pixel (−0.424) up to 0 turns it into pixel value 0 × 0.3081 + 0.1307 = **0.1307**, about 33 out of 255. The whole background goes grey.

### Converting ε on MNIST

```
ε_pixel = 0.3081 × ε_norm ε_norm = ε_pixel / 0.3081
```

|ε_norm (what the attack code uses)|ε_pixel (plain [0, 1])|% of full brightness|Out of 255|
|---|---|---|---|
|0.1|0.031|3.1%|8|
|0.3|0.092|9.2%|24|
|0.5|0.154|15.4%|39|
|**0.8**|**0.246**|**24.6%**|**63**|
|1.0|0.308|30.8%|79|

The other direction: the classic pixel-space ε = 0.3 becomes 0.3 / 0.3081 ≈ **0.97** in normalized space.

**Whenever you see an ε, ask "pixel space or normalized space?"** ε = 0.8 sounds enormous on a [0, 1] scale (80%!), but in normalized MNIST space it's about 25% of the brightness range. The text rounds that to ≈ 64/255, and the figure says 63, the same thing.

## Part 3: Why normalization helps training

### Step 7: The "all positive inputs" problem, as a toy

For a first-layer weight, the training gradient is **input × error signal** (`∂loss/∂w = x · δ`). Take one neuron with two inputs.

**Unnormalized**: pixels are always positive, say x = [0.5, 0.9]:

```
error signal δ = −0.3 -> gradients = [0.5 × −0.3, 0.9 × −0.3] = [−0.15, −0.27] both negative
error signal δ = +0.3 -> gradients = [+0.15, +0.27] both positive
```

**Both weights are always pushed the same way, so they can only go "both up" or "both down."** To reach a target like "w1 up, w2 down," training has to zig-zag, which is slow.

**Normalized**: inputs can be negative, say x = [−0.4, 1.2]:

```
δ = −0.3 -> gradients = [−0.4 × −0.3, 1.2 × −0.3] = [+0.12, −0.36] opposite signs
```

Now the weights can move in different directions in one step, so training explores freely.

### Step 8: Other benefits the section lists

- **Similar gradient sizes across layers**, so one learning rate (0.001) works everywhere instead of early layers crawling at ~0.001 while later layers jump at ~10.
- **Fewer dead ReLUs**: if every input is positive, weights can get pushed so far negative that a neuron outputs 0 forever.
- **Deep networks need it**: without normalization, gradients can **explode** (grow exponentially through layers) or **vanish** (shrink to 0). A 10-layer network may not train at all.

## Part 4: Why it matters for attacks

- **FGSM uses the gradient with respect to the input.** If the model trains on normalized data, that gradient is in normalized space, so **ε must be given in normalized space too.**
- **The accuracy - vulnerability paradox.** A well-trained model has **sharp, confident decision boundaries**, which a small push can cross. An underfit model has fuzzy boundaries and gradients that "point nowhere useful," so FGSM often fails on it. **A better-trained model can be easier to attack with gradients.**
- **HTB's illustrative numbers:** ε_norm = 0.8 (≈ 0.25 pixel) might reach about 95% attack success on the normalized model, versus about 60% with the same pixel-space change on an unnormalized one. The attack isn't weaker there; the unnormalized model's gradients are just less informative. Treat these as illustrative, not exact.
- **The workflow:** **attack in normalized space (where the model works), then denormalize to [0, 1] for display (where humans look).**

## Part 5: The three figures

### Figure 1: "Training Speed: Unnormalized vs Normalized"
![Figure 1: "Training Speed: Unnormalized vs Normalized"](/assets/Pasted%20image%2020261015161849.png)

|Epoch|Without normalization|With normalization|
|---|---|---|
|1|10% (random guessing)|88%|
|5|35%|97.5%|
|10|65%|98.5%|
|20|85%|(already done)|

The red line crawls upward over 20 epochs. The teal line jumps to the high 80s after one epoch and flattens near 98-99%. The caption "20× faster" compares roughly 1 epoch with normalization to 20 epochs without.

### Figure 2: "Gradient Behavior in Parameter Space"
![Figure 2: "Gradient Behavior in Parameter Space"](/assets/Pasted%20image%2020261015161854.png)
Each panel shows the directions training can push two weights (w1, w2) from the starting point.

- **Unnormalized (red):** a narrow fan pointing only to the right. Every update goes roughly the same way, which is Step 7's "always the same sign" problem.
- **Normalized (teal):** a starburst in every direction. Updates can go any way, which is the "full exploration" in the caption.

### Figure 3: "Epsilon Budget: Normalized vs Pixel Space Impact"
![Figure 3: "Epsilon Budget: Normalized vs Pixel Space Impact"](/assets/Pasted%20image%2020261015161910.png)
- **Top formula:** ε_pixel = σ · ε_norm = 0.3081 · ε_norm.
- **Left bars (purple):** ε in normalized units (0.1 to 1.0), with the pixel equivalent written beside each.
- **Right bars (orange):** the same budgets in pixel units. Every bar is about 0.31× as long.
- **Bottom chart:** each purple - orange pair is the **same** attack described in two units. The note says ε = 0.8 (norm) ≈ 0.25 (pixel) ≈ 64/255 levels.

## Optional lab cells (mine, not the course's)

### Compute the clamp bounds yourself

Run this to confirm the constants the attack code will use.

```python
MU, SIGMA = 0.1307, 0.3081
norm_min = (0.0 - MU) / SIGMA
norm_max = (1.0 - MU) / SIGMA
print(round(norm_min, 3), round(norm_max, 3))
```

Expected: `-0.424 2.821`

### Convert a normalized ε into pixel units

Use this whenever you report an ε to a human.

```python
eps_norm = 0.8
eps_pixel = eps_norm * SIGMA
print(round(eps_pixel, 3), round(eps_pixel * 255))
```

Expected: `0.246 63`

### Check that `mnist_denormalize` brings images back to [0, 1]

This uses the `images` batch from the Section 4 sanity-check cell.

```python
restored = mnist_denormalize(images[:1])
print(round(restored.min().item(), 3), round(restored.max().item(), 3))
```

Expected: close to `0.0` and `1.0`. If it errors, run `print(inspect.getsource(mnist_denormalize))` to see what arguments it expects.

## Key facts to lock in

- ==Normalization computes x_norm = (x − μ)/σ, and for MNIST μ = 0.1307 and σ = 0.3081==, which centres the data near zero with unit variance. Denormalizing reverses it: x = x_norm·σ + μ.
- **After normalization the valid MNIST range is about [−0.424, 2.821], not [0, 1]**, so adversarial images must be clamped to `MNIST_NORM_MIN` and `MNIST_NORM_MAX`. Clamping normalized values to [0, 1] silently brightens the background and dims the digit.
- ==A budget converts between spaces using σ alone: ε_pixel = σ·ε_norm and ε_norm = ε_pixel/σ==, because the mean cancels when you subtract two values. So ε_norm = 0.8 ≈ 0.246 pixel ≈ 63/255, and ε_pixel = 0.3 ≈ 0.97 normalized.
- **Attack code works in normalized space because that's where the model computes its input gradients**, while visualization denormalizes back to [0, 1] for human eyes. ε must be stated in the same space as the gradients and the clamp.
- **Without normalization, all-positive inputs make first-layer weight gradients (x·δ) share the same sign, so updates are correlated and training is slow**: about 85% after 20 epochs, versus 98% after 1 epoch with normalization, and deep networks may fail to converge at all.
- **A well-trained model with sharp, confident decision boundaries can be easier to attack with gradients than an underfit one**, so high clean accuracy and adversarial vulnerability often go together.
- **Method point:** **whenever an ε appears, in a lab, a challenge or a report, write down which space it's in and convert it before running the attack**, then report it in pixel units (and /255) so a human reader understands its real size.