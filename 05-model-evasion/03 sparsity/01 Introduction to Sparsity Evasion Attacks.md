
# Section 1: Introduction to Sparsity Evasion Attacks

## The idea in one breath

Sparsity attacks fool a model by changing **as few pixels as possible**, instead of nudging every pixel a tiny bit like FGSM and DeepFool did. The thing being kept small is the **count** of changed pixels, called the **L0 pseudo-norm**. This section sets up the two attacks the module teaches: **ElasticNet (EAD)**, which gets sparsity indirectly through an L1 penalty, and **JSMA**, which caps the pixel count directly.

> **Heads-up: your paste mangled every formula.** Here is each one read correctly:
> 
> - `||xadv−x||0=|{i∣(xadv)i≠xi}|` -> the **L0 norm** is written ‖x_adv − x‖₀, and it equals the **size of the set of indices i where the adversarial pixel differs from the original**. The little `0` is a subscript on the norm bars. The `|{ ... }|` is "count of elements in this set."
> - `minx′cf(x′)+||x′−x||22+β||x′−x||1` -> this is **minimize over x′** of: `c·f(x′) + ‖x′−x‖₂² + β·‖x′−x‖₁`. The `22` is a superscript 2 on an L2 (so "L2 squared"), and the `1` is a subscript on the last norm (L1). I'll break this down in Part 3.
> - `x̂=(x−μ)/σ` -> normalization: x-hat = (x − μ) / σ, the same shift-and-scale from the FGSM module.
> - `x′∈[0,1]` -> the adversarial image must stay in the valid pixel range 0 to 1.

---

## Part 1 - What "sparse" means, and why L0 is special

### The idea

Every attack you've done measures "how big is the change?" with a **norm** - a ruler. The whole point of this module is a _new ruler_ that measures something the others ignore.

### Toy example (your locked norms toy)

Take a change δ = [0.3, −0.4, 0, 0] applied to a 4-pixel image. The four rulers read it like this:

|Ruler|What it counts|Value for δ = [0.3, −0.4, 0, 0]|
|---|---|---|
|L0|how many pixels changed|2 (the first two)|
|L1|sum of absolute changes|0.3 + 0.4 = 0.7|
|L2|overall size (√ of sum of squares)|√(0.09 + 0.16) = √0.25 = 0.5|
|L∞|biggest single change|0.4|

**L0 is the only ruler that ignores how _big_ each change is - it only asks how _many_.** Double the change to [0.6, −0.8, 0, 0] and L1, L2, L∞ all double, but L0 stays **2**.

### Why it's called a "pseudo-norm"

A real norm obeys a scaling rule: if you double the vector, the norm doubles. L0 breaks that rule (doubling changed nothing), so mathematicians call it a **pseudo-norm** - it looks like a norm and we write it with the ‖·‖ bars, but it's not a true one.

### The real formula

```
‖x_adv − x‖₀ = |{ i | (x_adv)ᵢ ≠ xᵢ }|
```

|Symbol|Plain English|In our toy|
|---|---|---|
|x_adv|the attacked image|[0.3, −0.4, 0, 0] added to x|
|x|the original image|the clean image|
|i|a pixel index (position)|i = 1, 2, 3, 4|
|(x_adv)ᵢ ≠ xᵢ|"this pixel got changed"|true for i=1, i=2|
|{ i \| ... }|the set of pixels where that's true|{1, 2}|
|\| ... \||count of that set|2|

**Read aloud:** "The L0 distance is the number of positions where the adversarial image differs from the original."

---

## Part 2 - From first-order attacks to sparsity

### Why move on from L∞ / L2

FGSM capped L∞ (no single pixel moves more than ε). DeepFool minimized L2 (smallest overall change). Both let **_every_ pixel move** - the change is spread thin across the whole image. That's fine until you hit a real system where the constraint isn't "how much noise" but "how many things can I touch."

### Where "how many" is the real limit

The course gives the motivating cases, and these are worth keeping as vocabulary:

|Situation|Why the count is what's limited|
|---|---|
|Pixels that saturate to bounds|you can only flip a pixel fully to 0 or 1, so you pick a few|
|Tokens in text|you change words **one at a time**, not "all words by 3%"|
|Bits in a binary|you can flip a handful of bits, not add smooth noise everywhere|
|Evading anomaly detectors|defenders watch **global noise level**; a few big edits keep global noise low and **slip under a detector tuned to L2/L∞**|

That last row is the security punchline: **a sparse attack can beat a defense that only measures overall noise**, because 5 changed pixels barely move the global noise statistic.

---

## Part 3 - The two paths to sparsity

This is the map for the whole module. Two very different ways to get "few pixels changed."

### Path A - ElasticNet (EAD): get sparsity _indirectly_

EAD doesn't cap the pixel count. It adds a penalty that **pressures** most changes down to exactly zero. The objective:

```
minimize over x′ : c·f(x′) + ‖x′−x‖₂² + β·‖x′−x‖₁ subject to x′ ∈ [0,1]
```

|Term|Plain English|Job|
|---|---|---|
|c·f(x′)|the "make it misclassify" loss, scaled by c|**push** the image over the boundary|
|‖x′−x‖₂²|L2 squared - overall size of the change|keep edits **smooth / bounded** (no giant spikes)|
|β·‖x′−x‖₁|L1 - sum of absolute changes, scaled by β|**this is the sparsity knob** - L1 drives small changes to exactly 0|
|c|balances success vs. compactness|big c = fool it harder; small c = stay smaller|
|β|strength of the sparsity push|big β = fewer changed pixels|
|x′ ∈ [0,1]|stay a valid image|box constraint|

**Read aloud:** "Find the image x′ that fools the model (c·f), while keeping the change both small overall (L2²) and sparse (β·L1), and staying a valid image."

The key fact - and the reason L1 appears everywhere sparsity is wanted: **an L1 penalty naturally zeroes out small coordinates, so it's a _continuous stand-in_ for the discrete L0 goal.** You can't smoothly optimize "count of changes" (L0 is not differentiable), but you _can_ optimize L1, and L1 happens to produce mostly-zero results. EAD = "L1 for sparsity + L2 for smoothness," hence **elastic net** (the statistics name for mixing L1 and L2). The result is **a few larger edits rather than many tiny ones**. FISTA (Sections 3+) is just the solver that handles the non-smooth L1 term.

### Path B - JSMA: get sparsity _directly_

JSMA (Jacobian-based Saliency Map Attack) sets a hard **L0 budget** and spends it carefully: each iteration it changes **one or two pixels**, chosen by a **saliency map** - a score, built from the input Jacobian, that ranks pixels by how much they'd _raise the target class while suppressing competitors_. No penalty, no balancing act - just "pick the most useful pixel, change it, repeat until the budget runs out or it flips."

### The contrast to hold onto

|Aspect|ElasticNet (EAD)|JSMA|
|---|---|---|
|How sparsity happens|indirectly, via L1 penalty|directly, via explicit L0 budget|
|Pixels changed per step|all, but most pushed toward 0|1 or 2, deliberately|
|Main machinery|FISTA (proximal gradient + momentum)|saliency map from the Jacobian|
|Feels like|DeepFool with a sparsity penalty|a greedy pixel-picker|
|Both are|**targeted** (aim for a chosen class)|**targeted**|

---

## Part 4 - Threat model and budgets

The course is precise about the attacker's assumptions; keep these terms.

|Setting|What the attacker can do|How they pick pixels|
|---|---|---|
|**White-box**|compute real gradients through the model|read derivatives directly|
|**Black-box**|only query the model (no gradients)|estimate importance by queries, or **transfer** a sparse pattern from a surrogate|

Both are **inference-time** attacks (attacking the trained model's predictions, not its training) - same category as FGSM and DeepFool. The **primary budget is L0**, sometimes with **auxiliary L2 or L∞ limits** to keep individual edits bounded and valid. And the box-constraint detail you already know: images stay in [0,1] after each update, and if the model normalizes inside (x̂ = (x−μ)/σ), the **gradient flows back through the normalization by the chain rule**, so you can keep reasoning in pixel space - exactly the point that bit you in the DeepFool challenge today.

## Key facts to lock in

- ==The L0 pseudo-norm counts how many coordinates changed, ‖x_adv−x‖₀ = |{i : (x_adv)ᵢ ≠ xᵢ}|==, and it ignores the size of each change entirely - doubling the perturbation leaves L0 unchanged, which is why it's a _pseudo_-norm, not a true one.
- **Sparsity attacks constrain _how many_ features change, not _how much_ each changes**, which is the opposite focus from FGSM (caps L∞) and DeepFool (minimizes L2), both of which let every pixel move a little.
- **An L1 penalty is the continuous stand-in for the discrete L0 goal**, because L1 naturally pushes small coordinates to exactly zero while still being optimizable, whereas L0 itself is non-differentiable and can't be minimized by gradient methods.
- **ElasticNet (EAD) minimizes c·f(x′) + ‖x′−x‖₂² + β·‖x′−x‖₁**, where c trades success against compactness, the L2² term keeps edits smooth, and β controls sparsity through the L1 term - producing a few larger edits instead of many tiny ones.
- **JSMA enforces an explicit L0 budget by changing one or two pixels per iteration**, chosen by a saliency map derived from the input Jacobian that scores pixels for raising the target class while suppressing competitors.
- **Both EAD and JSMA are targeted, inference-time attacks**, available white-box (real gradients) or black-box (query-estimated importance, or transfer from a surrogate), with L0 as the primary budget and optional L2/L∞ limits to keep edits valid.
- **Sparse edits can evade detectors tuned to global noise statistics**, because a handful of large changes barely moves the overall L2/L∞ noise level a defender might be watching - this is the core security relevance of sparsity.
- **Method point:** when you meet a sparsity challenge, first identify which budget the server enforces - **an explicit "max pixels changed" cap points to JSMA; a "total/smooth change" budget points to EAD** - because the budget type, not the dataset, decides which attack you reach for.

