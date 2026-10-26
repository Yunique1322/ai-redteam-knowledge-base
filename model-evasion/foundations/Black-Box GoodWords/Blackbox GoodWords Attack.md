# Blackbox GoodWords Attack

**The idea in one breath:** in the white-box attack you _read_ the model's word probabilities and cherry-picked good words. **Black-box means you can only submit a message and see the spam score back - no parameters, no architecture, no training data.** So instead of looking up the best words, you have to **discover** them by probing, and because each probe (query) costs you, you need a smart strategy to find good words fast. That strategy is a **multi-armed bandit**.

## Modelling the classifier as an unknown function

**Treat the classifier as a black-box function `f: X → [0,1]`** - you feed it a message (from `X`, the space of all messages) and it returns a spam probability between 0 and 1. You can't see _how_ it computes that, only the output.

Your job: **find a transformation `T` that lowers `f(T(x))` for spam `x` while keeping the message's meaning.** Formally, the goal is:

```
minimize over W ⊆ V, |W| ≤ k : f(x ⊕ W)
```

In plain words: **pick a small set of words `W` (at most `k` of them) from a candidate vocabulary `V`, such that appending them to message `x` drives the spam score as low as possible.** The `⊕` symbol means **append the words** (bag-of-words addition) - _not_ set union, just "glue them onto the message." `k` is your word budget (you don't want to add 500 words).

## Measuring a word's value without gradients - finite differences

White-box attacks can use gradients (the model's internal slopes). Black-box can't. So you measure a word's effect the only way you can - **by before-and-after comparison, called a **finite difference**:**

```
r_w(x) = f(x) − f(x ⊕ {w})
```

This is the ****reward** for adding word `w` to message `x`: the original spam score minus the score after appending the word.** Concretely: if "thanks" drops spam probability from 0.95 → 0.80, the reward is **0.15** (a good word). If it only drops to 0.94, the reward is **0.01** (useless). ==Positive reward = the word helped.== For a whole set: `r_W(x) = f(x) − f(x ⊕ W)`. These rewards are the signal that drives everything else.

## The exploration vs exploitation dilemma

Here's the tension that defines the whole section. With a limited query budget, every query you spend testing a _new, unproven_ word (**exploration**) is a query you didn't spend using a _known-good_ word to actually evade (**exploitation**).

- ****Too much exploitation**** → you lean only on words you already know work, and **miss better words you never tried.**
- ****Too much exploration**** → you **waste queries testing duds** instead of winning.

You need a principled way to balance the two. That's the bandit framework.

## The multi-armed bandit framework

**Picture a casino of slot machines ("one-armed bandits"), each with an unknown payout rate.** You have limited pulls and want max total reward - but you don't know which machine is best. **The multi-armed bandit problem: how do you allocate pulls to maximize reward while learning which machine is good?**

The mapping to the attack:

- ==Each candidate word = one bandit "arm" with unknown effectiveness.==
- ==Testing a word = pulling that arm== → you observe its reward (spam-score drop) but **spend one query from your budget.**

So "which word do I test next?" _is_ the bandit problem. And there's a well-known near-optimal answer: UCB.

## Upper Confidence Bound (UCB) - the strategy

The intuition HTB gives is restaurants: **you want to eat where the food's good (exploitation), but you should occasionally try new places (exploration) because your favourite might not actually be the best.** UCB scores each option by combining _how good it was_ with _a bonus for how little you've tried it._ A place visited once gets a big "uncertainty bonus" (one visit isn't reliable); a place visited 50 times has a trustworthy average, so its score rests mostly on the food.

The formula assigns each word a score:

```
UCB(w) = r̄_w + c · √( ln(t) / n_w )
         (exploitation) (exploration bonus)
```

Breaking down every piece:

- ==`r̄_w` = the word's **average observed reward** so far== - words that historically lowered spam score most get a higher base.
- ==`n_w` = how many times you've tested word `w`.== It's in the **denominator**, so **less-tested words get a _bigger_ bonus** (more uncertainty = more worth exploring).
- ==`t` = total queries so far==, inside a `ln` (log) so the **bonus grows only _slowly_** over time.
- ==`c` = the exploration constant (≈ 2.0)== - the knob that tunes how much you favour exploration.
- **The `√` (square root) acts as a "confidence interval"** around your estimate - wide for rarely-tested words, narrow for well-tested ones.

The **key behaviour: the exploration bonus _shrinks_ as `n_w` grows**, so the more you test a word, the more its score depends on its real performance rather than hype.

### The worked example (this is the exam's favourite UCB illustration)

After `t = 200` queries, three words:

|word|tested `n_w`|avg reward `r̄_w`|UCB score|
|---|---|---|---|
|`thanks`|50×|0.15|0.15 + 0.47 = **0.62**|
|`appreciate`|5×|0.12|0.12 + 1.48 = **1.60**|
|`wonderful`|0×| - |0 + ∞ = **∞**|

Two things to take away:

- **An **untested** word has an infinite bonus → UCB always tries untested words first.**
- **The counterintuitive one: `appreciate` wins over `thanks` _despite a lower observed average_ (0.12 < 0.15)**, because it's only been tried 5 times - its huge uncertainty bonus (1.48) says "we can't trust 0.12 yet; maybe its true value is 0.20, give it a chance." **UCB quantifies uncertainty so you don't prematurely lock onto a word that merely _looks_ best early.**

## Theoretical guarantees (why UCB is principled, not just a hack)

**UCB gives **logarithmic regret**: `O(log T)`.** "Regret" = the cumulative gap between your choices and always having picked the single best word. **Growing as `log T` instead of linearly with `T` is _sublinear_** - meaning even after thousands of queries, your total loss from exploration mistakes stays tiny (bounded by the log of the query count). **It also converges: as each word's bonus `√(ln t / n_w)` shrinks toward zero, the exploitation term takes over and selections concentrate on the genuinely best words.**

The caveat worth noting: **those guarantees (UCB1) formally hold for i.i.d., stationary, bounded rewards.** Here the reward of a word varies by message, which adds noise, so **in practice UCB is a strong _heuristic_ for spending your query budget rather than a hard guarantee.**

## Implementation strategy (the three phases)

How this gets built, conceptually:

1. ****Vocabulary construction (offline, zero queries).**** Assemble a candidate word pool from **external sources - common English dictionaries, legitimate message corpora, domain vocab - and filter out obvious spam words.** This is free: it never touches the target model, so you can build it unconstrained.
2. ****Adaptive scoring.**** As you query, **track each word's impact with an **exponential moving average (EMA)**** - a running average that weights recent results more, so it adapts if the target's behaviour shifts. **Every query updates your beliefs** about which words pay off.
3. ****Combination discovery.**** Testing words one at a time only reveals ****first-order** (individual) effects.** But **word _pairs_ can be **super-additive** (synergistic)** - e.g. "meeting" alone −10%, "tomorrow" alone −8%, but **together −25%**, because the pair reads as coherent legitimate scheduling, a pattern the model learned. So a combinatorial search on top of individual scores finds these synergies.

## The facts worth locking in for the exam

- ==Black-box = query-only access: you see `f(x) → [0,1]` spam scores but no parameters, architecture, or training data==, which turns the white-box lookup into a _search/exploration_ problem under a query budget.
- ==Without gradients, you measure a word's value by finite difference: `r_w(x) = f(x) − f(x ⊕ {w})`== (original score minus score-after-appending); positive reward = effective word.
- **The optimization is `minimize f(x ⊕ W)` over word sets `W ⊆ V` with `|W| ≤ k`**, where `⊕` is appending (bag-of-words addition), not set union.
- **Word selection is a multi-armed bandit: each word is an "arm" with unknown payout, each test "pulls" it and spends a query**, creating the **exploration (try new words) vs exploitation (reuse known-good words) trade-off.**
- **UCB scores each word as `r̄_w + c·√(ln t / n_w)`** = average reward + an uncertainty bonus that **is infinite for untested words (always tried first) and shrinks as a word is tested more**; `c ≈ 2` tunes the balance. It can pick a lower-average word if that word is under-explored.
- **UCB gives sublinear, logarithmic regret `O(log T)` and converges on the best words**, formally under i.i.d./stationary/bounded rewards - in practice a strong heuristic despite message-dependent noise.
- **Build the candidate vocabulary offline (zero queries) from dictionaries/legit corpora**; track impact with an **exponential moving average**; and search **word combinations for super-additive synergy** beyond single-word (first-order) effects.
