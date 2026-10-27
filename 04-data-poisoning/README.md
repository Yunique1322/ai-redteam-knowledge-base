# Data Poisoning

Attacking the training data so the model learns what you want it to: lower accuracy, a specific misclassification, or a hidden backdoor that only fires on a trigger. These are training-time attacks, so you are influencing the model before it is ever deployed.

[Introduction to AI Data](01%20Introduction%20to%20AI%20Data.md) and [AI Data Attacks](02%20AI%20Data%20Attacks.md) set up the data attack surface and the threat model.

[Label Flipping](03%20Label%20Flipping%20Attacks/) is the simplest poison: change the labels on some training samples. It starts from a logistic-regression baseline, does untargeted flipping to drag accuracy down, and then targeted flipping aimed at forcing one chosen class, with the evaluation for each.

[Feature Attacks](04%20Feature%20Attacks/) is clean-label poisoning, which is the harder-to-spot version: the poisoned samples keep labels that look correct, so nothing obviously wrong shows up in review. It covers picking a target, crafting the attack in feature space, and measuring it against a one-vs-rest baseline.

[Trojan Attack](05%20Trojan%20Attack/) plants a backdoor in a CNN. A small trigger pattern forces a target prediction while the model behaves normally on everything else, which is what makes it so hard to catch. It runs through the architecture, the data prep, the attack components, and the evaluation.

[Pickles and Steganography](06%20Pickles%20and%20Steganography/) is the supply-chain angle: malicious pickle payloads that execute when a serialized model is loaded, and tensor steganography that hides data or code inside the weights themselves.

Poisoning maps to the Data component of the methodology, and it is dangerous for three reasons. It survives into every later use of the model, a backdoor can be nearly invisible under normal evaluation, and the pickle and steganography vectors quietly turn "just grab a pretrained model" into a code-execution and data-hiding problem. The defenses are in [Defenses](../08-defenses/), and the privacy-preserving training methods are in [Privacy](../07-privacy/).
