# Model Evasion

Inference-time attacks on a model you cannot change. The model is fixed; your job is to craft an input it misclassifies while keeping the change small, often small enough to be imperceptible. This domain pulls together three bodies of work, ordered from the gentlest intuition up to the precise, surgical attacks.

[Foundations](foundations/) is the GoodWords attack on a spam classifier, in both white-box and black-box forms. It is the easiest way in: you just add benign-looking features until the classifier flips, with no gradients involved. The black-box version has to discover which features help by querying the model, which is good practice for the harder attacks.

[First-Order](first-order/) is gradient-based evasion of neural networks. [FGSM](first-order/FGSM/) is the Fast Gradient Sign Method, taking one step along the sign of the loss gradient, plus its iterative form I-FGSM, which takes many small steps and is far stronger at the same budget, and the targeted variant that aims at a chosen class. [DeepFool](first-order/DeepFool/) asks a different question, finding the smallest change that reaches the nearest decision boundary.

[Sparsity](sparsity/) is about changing as few features as possible. [ElasticNet](sparsity/ElasticNet/) uses a FISTA optimizer with a mix of L1 and L2 to concentrate the change on a handful of pixels. [JSMA](sparsity/JSMA/) is the Jacobian-based Saliency Map Attack, including the single-pixel and the stronger pairwise-saliency variants.

The thread running through all of it is the norm you minimize, because that is what decides which attack applies. FGSM and I-FGSM bound the largest single-feature change, the Linf norm. DeepFool and ElasticNet go after the overall magnitude, the L2 norm. JSMA and single-pixel minimize the count of features touched, the L0 norm. The [AI/ML Primer](../foundations-and-methodology/AI-ML%20Primer.md) has the background on norms, and the defensive counterpart, training a model to shrug these off, is [Adversarial Training](../defenses/Adversarial%20Training/).
