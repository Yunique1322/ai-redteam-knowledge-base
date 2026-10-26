# Section 17: Supervised Fine-Tuning for Safety

## The idea in one breath
This is the theory behind the defense. **Supervised fine-tuning (SFT)** shows the model attack→refusal pairs and nudges its weights to make refusals more likely for attack-like inputs. To do it cheaply, you use **LoRA** (Low-Rank Adaptation): freeze the giant weight matrices and train a tiny low-rank add-on, about **1-1.4% of the parameters**. The training set mixes jailbreak refusals, priming defenses and benign conversations (the course uses 138 safety : 86 benign). It **generalises** to unseen attacks because the model learns the *features* of manipulation, not the exact wordings. The pass bar is **≥ 90% attack refusal and ≥ 85% benign helpfulness**.

> **Heads-up:** the PDF mangled the maths subscripts. Decoded:
> - Loss: `ℒ=−∑i=1nlogP(xi|x1,...,xi−1;θ)` means **ℒ = − Σ (i = 1 to n) log P(xᵢ | x₁,...,x₍ᵢ₋₁₎; θ)**.
> - LoRA: `W′=W+BA` with `B∈ℝd×r`, `A∈ℝr×k`, `r≪min(d,k)` means **W′ = W + B·A**, where B is d×r, A is r×k, and the rank r is far smaller than d or k.

## Part 1 - How SFT works

### Step 1: The objective
A language model predicts the next token from the ones before it. Training compares each prediction with the true next token and updates the weights.

> **ℒ = − Σᵢ log P(xᵢ | x₁,...,x₍ᵢ₋₁₎; θ)**

| Symbol | Plain English |
|---|---|
| x₁...xₙ | The tokens of one training example |
| P(xᵢ | ...; θ) | The model's probability for the correct next token, given the context |
| log, then − and Σ | Take the log, sum over positions, negate. **Lower loss = higher probability on the correct tokens** |

For safety tuning, the examples are **attack prompt → refusal**. The model learns to put high probability on refusal tokens when the context looks like an attack.

### Step 2: Why this generalises (the key idea)
- Networks don't memorise examples as items; they learn **distributed features**.
- Training on roleplay jailbreaks teaches features like "you are now...", "pretend to be...", "without restrictions...", not the exact strings.
- Those features fire for **novel** jailbreaks with the same structure, so the model refuses attacks it never saw.
- Studies back this up: models trained on some jailbreak categories refuse **held-out** categories.
- Not perfect: a sufficiently novel attack can still evade. But the attacker now has to beat learned detection across several feature dimensions, not just find any jailbreak.

## Part 2 - LoRA: cheap fine-tuning

### Step 1: The trick
Full fine-tuning updates billions of weights. LoRA notes that a fine-tuning update usually lives in a **low-dimensional subspace**, so instead of changing W directly it adds a low-rank piece:

> **W′ = W + B·A**, with B (d×r), A (r×k), rank **r = 16** ≪ d, k.

- W stays **frozen**; only A and B get gradients.
- Far fewer trainable numbers, so it fits on a 16GB (or smaller) consumer GPU.

### Step 2: Toy first (why low rank is small)
Take a 1000×1000 weight matrix: that's 1,000,000 numbers. A rank-16 update B(1000×16) + A(16×1000) is 16,000 + 16,000 = **32,000** numbers, about **3%**. The same ratio, scaled up, is where the "train ~1% of the model" figure comes from.

### Step 3: Why quality survives
You're not teaching a new skill, only **adjusting the decision boundary of an existing one** (safety refusal). That genuinely fits in a low-rank subspace, so the small add-on is enough.

> **An honest note the course skips:** this section says LoRA cuts "roughly 1.2 billion to approximately 8 million" trainable parameters. The actual run in Section 19 reports **11,272,192 trainable (1.43%) of 785,713,152**. The "1.2B / 8M" line is a rough illustration with different numbers from the model actually used (a ~786M-parameter Llama-3.2-1B). Go by Section 19's measured figures.

## Part 3 - Where the adapters attach
LoRA is added to specific layers:

| Target modules | Role |
|---|---|
| `q_proj`, `k_proj`, `v_proj`, `o_proj` | The attention query/key/value/output projections |
| `gate_proj`, `up_proj`, `down_proj` | The feed-forward layers |

Attention layers control **how the model weighs different parts of the input context**. A jailbreak mixes manipulation framing with a harmful request; adapting the attention projections is what lets the model treat the framing as a *refusal signal* rather than an instruction. The feed-forward layers then produce the refusal-appropriate representation.

## Part 4 - Training-data composition

| Category | What it teaches | Count (course) |
|---|---|---|
| Jailbreak refusals | Refuse roleplay / authority / hypothetical / encoding attacks | 96 |
| Priming defenses | Stop mid-reply when harmful content is in the assistant prefix | 42 |
| Benign conversations | Stay helpful on legitimate queries, including adjacent topics | 86 |

- **Total 224**: 138 safety (61.6%) + 86 benign (38.4%).
- Safety should **slightly** outnumber benign, not swamp it. Determined empirically: too few safety examples and the model stays vulnerable; too many and it over-refuses.

## Part 5 - Avoiding over-refusal
Causes of over-refusal, and the course's fixes:

| Cause | Fix used here |
|---|---|
| Safety examples vastly outnumber benign | Keep benign ≈ 38% (86 of 224) |
| Refusals written to decline whole topics | Refuse the **specific** harmful request, not the topic |
| Benign set avoids sensitive-adjacent topics | Include benign security/chemistry examples to show they have legitimate uses |

Evaluation measures **both** safety and helpfulness. A model at 100% refusal but 50% helpfulness has **failed**. The thresholds: **≥ 90% attack refusal, ≥ 85% benign helpfulness**.

## Reading the figure/output
No figures in this section.

## Copy-paste cells (mine, not the course's)
This is theory; the code begins in Section 18. One tiny illustrative cell on LoRA size:

```python
d = k = 1000; r = 16
full = d * k
lora = d * r + r * k
print(f"full update: {full:,} params | LoRA rank {r}: {lora:,} params ({100 * lora / full:.1f}%)")
```

Expected: `full update: 1,000,000 params | LoRA rank 16: 32,000 params (3.2%)`.

## Quick self-check
The course trains on 96 roleplay/authority/hypothetical/encoding jailbreaks, yet claims the model will refuse *novel* jailbreaks it never saw.
- **(a)** What property of neural-network learning makes that plausible?
- **(b)** What's the one-line limit on that claim?

*(Answer key is in COAE-progress.md → Open self-checks.)*

## The facts worth locking in for the exam
- SFT minimises ==ℒ = − Σ log P(xᵢ | earlier tokens)== over attack→refusal (and benign→helpful) examples, raising the probability of refusal tokens in attack-like context.
- It generalises because networks learn **distributed features of manipulation (roleplay/authority phrasing), not exact strings**, so held-out attacks still trigger refusal.
- ==LoRA freezes W and trains a low-rank add-on W′ = W + B·A (rank 16)==, training ~1.4% of the parameters (Section 19: 11.27M of 786M) and fitting on a consumer GPU.
- Adapters target **attention projections (q/k/v/o) plus feed-forward (gate/up/down)**, because attention decides how adversarial framing is weighed.
- The dataset is **138 safety (96 jailbreak + 42 priming) : 86 benign ≈ 60/40**, a slight safety bias set empirically.
- **Over-refusal** is avoided by balancing benign examples, refusing specific requests not whole topics, and covering sensitive-adjacent benign topics.
- **Method point:** the pass bar is **both ≥ 90% refusal and ≥ 85% helpfulness**; in a report, never quote a refusal rate without the paired helpfulness rate, or you hide over-refusal.
