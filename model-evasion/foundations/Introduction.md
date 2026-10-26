# Introduction

==**Evasion** = manipulating an input at **inference time** so a trained model produces a wrong output.== It leaves training and parameters untouched, and succeeds by sending a crafted input through the normal interface so **a single example crosses the model's learned **decision boundary**.**

## Training-time vs inference-time attacks

- ****Training-time attacks** change _what the model learns_.** They need access to the **dataset or pipeline** and tend to shift behavior **globally**:
    - ****Data poisoning**** - inject or alter training samples so the model learns biased patterns.
    - ****Label manipulation**** - tamper with annotations so ground truth no longer matches reality.
    - ****Trojan attacks**** - implant hidden triggers that activate a chosen behavior when the attacker's pattern appears.
- ****Inference-time manipulation (evasion)** changes only _what the model sees when predicting_** - no change to training or parameters, just the one input.

## Transferability

**An adversarial example crafted against a **surrogate model** can often fool a different **production model** - especially when their architectures or training data are similar.** This is what enables offline preparation and ****black-box** attacks, where the defender exposes only an API, not the model internals** (vs **white-box**, where you can see the model).

## Traditional ML vs LLM evasion

Same core idea - **modify only the input seen at prediction time - but the leverage comes from different places:**

- ****Traditional ML** targets fixed feature representations, so small controlled edits push an example across the decision boundary:**
    - Spam filter using a ****bag-of-words** model: add benign tokens → term frequencies change → the **posterior probability** used for classification shifts**, while the message still reads the same to a human.
    - Static malware detector: **rearrange sections or perturb imports → byte patterns and crafted signatures change without changing program intent.**
- ****LLMs** produce open-ended text and follow instructions, so **prompt injection** is the central pattern:** place adversarial directives in the prompt/context so the model **prioritizes them over the original **system** guidance**. The twist: **the output is itself an action surface** (code, a database query, renderable markup), and **tool routing / API calling can turn that output into follow-on actions.**

The split to remember: **traditional models expose structured features + a label threshold (edit the statistics); LLMs expose conversational state + instruction-following (edit the decision process with placed text).**

## This module - the GoodWords attack

**Manipulates probabilistic spam filters by inserting carefully chosen benign tokens.** It works because ****Naive Bayes** assumes **conditional independence** between features**, so adding ham-associated tokens **shifts the posterior probabilities enough to force a misclassification**, without looking suspicious. It also draws the **white-box vs black-box line** and the **trade-off between subtle perturbation and reliable success.**

## Exam keywords

Evasion = inference-time, params untouched, single input across the **decision boundary**
· training-time = **data poisoning / label manipulation / trojan** (need dataset access, global shift) · **transferability** via a **surrogate model** with similar architecture/data → **black-box** vs **white-box** · **bag-of-words** → term frequencies → **posterior** · **prompt injection** overrides **system** guidance, output = action surface · **GoodWords** exploits **Naive Bayes conditional independence**.
