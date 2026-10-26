# Section 15: Introduction to LLM Adversarial Tuning

## The idea in one breath
The last part hardened an *image* model by training on attacked images. This part does the same thing for an *LLM*: instead of filtering attacks at the edge (guardrails), you train the model itself to recognise manipulation and refuse. That's **adversarial tuning**. It's a second layer of defense in depth: guardrails catch the obvious cases, a tuned model catches the ones that slip past the filter.

The two attacks defended against are **jailbreaks** (prompts that talk the model out of its safety rules) and **priming attacks** (harmful text injected into the model's *own* reply slot so it just continues). The lab's target is to cut priming-attack success from about 98% to under 10% while keeping the model helpful.

> **Heads-up:** no maths, code or figures in this section. Note on wording: these notes describe attacks only at the level needed to build the defense, and redact the specific attack strings from the course.

## Part 1 - Why both layers

| Layer | Where | Catches | Misses |
|---|---|---|---|
| Guardrails (Sections 2-7) | Around the model, at inference | Known patterns | Novel attacks that don't match a pattern |
| Adversarial tuning | Inside the model's weights, at training | Manipulation it has learned the *shape* of | Sufficiently new attacks |

They complement each other: a prompt has to evade the filter **and** fool a model that has been taught to refuse.

## Part 2 - The safety-alignment gap
- Commercial LLMs already get safety training: supervised fine-tuning, then **RLHF** (reinforcement learning from human feedback) and constitutional methods. They refuse obvious harmful requests.
- The gap is between **training-time** safety and **inference-time** pressure. Safety training covers the obvious cases; attackers deliberately work *outside* that distribution, combining tricks and iterating on the model's replies.
- The training set can't cover every variation, so novel attacks get through.

## Part 3 - Adversarial tuning as the fix
- Instead of hoping general safety transfers, you **show the model attack patterns paired with correct refusals**, and train on them. This mirrors image adversarial training: learn from adversarial inputs to generalise better.
- The training set mixes **attack prompts → refusals** and **benign queries → helpful answers**.
- The balance matters. Train only on attacks and you get **over-refusal**: the model starts declining harmless chemistry or security questions. Effective tuning keeps helpfulness on legitimate queries while improving robustness.

## Part 4 - The two attacks (threat model)

| Attack | Idea (high level) | Why it's hard |
|---|---|---|
| **Jailbreak** | Dress a disallowed request up as roleplay, a claimed authority, a hypothetical, or an encoding, so instruction-following overrides safety | The model still reaches its safety decision point and *can* refuse |
| **Priming** | Put the start of a harmful reply in the **assistant** slot, as if the model had already begun, then ask it to continue | It **skips** the safety decision: the reply has apparently already started |

Priming is the more dangerous one, and is the main target of this module's defense. Section 16 explains why it works and how the defense interrupts it.

## Reading the figure/output
No figures in this section.

## Copy-paste cells (mine, not the course's)
No code in this section. The lab's code starts in Section 18.

## Quick self-check
A client already runs an input guardrail that blocks known jailbreak phrases. Give one reason they should *also* adversarially tune the model, and one reason they shouldn't drop the guardrail once they have.

*(Answer key is in COAE-progress.md → Open self-checks.)*

## The facts worth locking in for the exam
- Adversarial tuning **hardens the LLM itself to refuse attacks**, as a training-time layer alongside inference-time guardrails: **defense in depth**.
- The gap exists because **safety training covers the expected cases while attackers operate outside that distribution**, so general alignment doesn't transfer to every novel attack.
- The method is **supervised fine-tuning on attack→refusal pairs mixed with benign→helpful pairs**, which mirrors image adversarial training.
- The main risk is **over-refusal**: too many safety examples and the model declines legitimate requests, so the benign/attack balance must be tuned.
- The two attacks are **jailbreaks (talk the model into complying) and priming (inject a harmful reply prefix so the model continues)**; priming is worse because it **bypasses the safety decision point**.
- **Method point:** in a report, present adversarial tuning as **one layer, not a complete fix**, and always pair the safety metric with a **helpfulness metric** so a "100% refusal" model that refuses everything is caught.
