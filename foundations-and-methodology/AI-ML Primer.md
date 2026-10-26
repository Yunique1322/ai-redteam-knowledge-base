# AI/ML Primer for Red Teamers

Just enough machine learning to attack it with intent. This is deliberately short. The goal is to give you the model intuition you need to reason about where an AI system breaks: its decision boundary, its gradients, what it memorizes, and how it follows instructions. For the hands-on side see [Applied ML in Infosec](applied-ml-in-infosec/), and for the attacker's frameworks see [Red Team Methodology](red-team-methodology/).

## Why the fundamentals earn their place

Every attack in this knowledge base leans on one specific fact about how a model learns or infers. Evasion attacks exploit the shape of the decision boundary and the gradient of the loss. Poisoning exploits the fact that a model learns whatever the data teaches it. Privacy attacks exploit the fact that a trained model quietly memorizes parts of its training set. Prompt injection and output attacks exploit how a generative model follows instructions and how its output gets consumed downstream. So the concepts below are grouped by the handle each one gives you.

## The vocabulary worth knowing

The **features** are the input variables the model sees, and they are where you inject a perturbation, whether that is pixels, tokens, or packet statistics. The **label** is the ground truth, and it is what you flip in poisoning or force in a targeted attack. The **parameters** or weights are what the model learns during training; you steal them in a model-extraction attack and tamper with them in a backdoor. The **loss function** measures how wrong a prediction is, and attacks either maximize it (evasion) or shape it (poisoning). The **gradient** is the direction that changes the loss fastest, and it is the engine behind the first-order evasion attacks like FGSM, DeepFool, and JSMA. The **decision boundary** is the surface separating one predicted class from another, and evasion is really just the art of pushing an input across it with as little change as possible. Finally, **overfitting**, or memorization, is when a model learns its training data too specifically, and that is exactly what makes membership inference and data-extraction attacks possible.

## The learning paradigms

**Supervised learning** trains on labeled examples to predict a label. Linear and logistic regression draw linear boundaries and are cheap to attack and handy as substitute models. Decision trees split on one feature at a time, which makes them readable but brittle. Naive Bayes is the classic target of the GoodWords spam-evasion attack. Support vector machines maximize the margin between classes, and that margin geometry is exactly what a minimal-perturbation attack exploits.

**Unsupervised learning** finds structure without labels. K-means groups points by distance to a centroid, PCA reduces dimensionality and doubles as a way to spot outliers, and anomaly detection flags the odd ones out, which is the backbone of a lot of ML-based security tooling and therefore a defense you may need to evade.

**Deep learning** stacks layers of neurons that learn features in a hierarchy. A perceptron is a single weighted sum through a non-linearity; stack them and train by backpropagating gradients and you have a neural network. Convolutional networks (CNNs) handle images and are the usual target for adversarial-image work. Recurrent networks model sequences and give you the intuition that leads into language models.

**Generative AI** produces new content. Large language models are next-token predictors trained on huge text corpora, and they are the target of prompt injection, jailbreaks, and output-handling attacks. Diffusion models generate images by iteratively removing noise.

## The lifecycle is the attack surface

```
data collection -> preprocessing -> training -> validation -> deployment -> inference
      ^                  ^              ^                          ^            ^
  poisoning          feature        backdoor /                model theft   evasion /
  (bad data)         tampering      trojan insert             tampering     prompt injection
```

Any technique in this KB maps to the stage it abuses. Training-time attacks let you influence what the model learns: label flipping, clean-label poisoning, and trojan backdoors, all in [Data Poisoning](../data-poisoning/). Inference-time attacks leave the model fixed and craft malicious input instead: FGSM, I-FGSM, DeepFool, ElasticNet, JSMA, and single-pixel in [Model Evasion](../model-evasion/), and prompt injection and jailbreaks in [Prompt Injection](../prompt-injection/). Stealing the model or its secrets is membership inference and model extraction, split across [Privacy](../privacy/) and [Application & System](../application-system/). And everything downstream of the model, where its output becomes someone else's input, lives in [Output Attacks](../output-attacks/).

## Norms, or "how much did I change the input?"

Evasion attacks try to keep the perturbation small, and "small" is measured with a norm. The L0 norm counts how many features you changed, so it is about sparsity and is what JSMA and single-pixel attacks minimize. The L2 norm is the ordinary Euclidean magnitude of the change, which DeepFool and ElasticNet go after. The Linf norm is the largest change to any single feature, which is what FGSM and I-FGSM bound. The norm you pick defines what "minimal" or "imperceptible" even means, and that in turn decides which attack family applies. There is more on this in [Model Evasion](../model-evasion/).

## The metrics both sides use

Accuracy is the fraction the model gets right, and it is misleading on imbalanced data. Precision and recall trade off false positives against false negatives and are what you tune a detector around. F1 is their harmonic mean. For an attacker the number that matters is the attack success rate: the fraction of adversarial inputs that actually achieve the goal, which for a targeted attack means landing on a specific wrong class rather than just any wrong answer. The applied treatment is in [Metrics for Evaluating a Model](applied-ml-in-infosec/Metrics%20for%20Evaluating%20a%20Model.md).
