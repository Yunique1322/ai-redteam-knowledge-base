# Privacy

Pulling training data back out of a trained model, and the defenses built specifically to stop it. A deployed model is a lossy record of what it learned, privacy attacks exploit that, and privacy defenses work by bounding how much any single training example is allowed to influence the result.

[Privacy Threats and Attack Fundamentals](Privacy%20Threats%20and%20Attack%20Fundamentals.md) sets up the threat model: what privacy even means for a model, and what an attacker is actually trying to recover.

[Shadow Model Attack](Shadow%20Model%20Attack/) is membership inference, deciding whether a particular record was in the training set. You train shadow models that imitate the target, label their behavior as member or non-member, and train a classifier on the difference. The whole thing feeds on the overfitting gap, the fact that a model tends to be more confident on data it was trained on.

[DP-SGD](DP-SGD/) is the first defense, Differentially Private SGD by way of Opacus. During training it clips each example's gradient and adds calibrated noise, so the finished model is almost the same whether or not any one person was in the data. The notes walk the pipeline and then measure the privacy-versus-accuracy trade-off, because there always is one.

[Private Aggregation of Teacher Ensembles (PATE)](Private%20Aggregation%20of%20Teacher%20Ensembles/) is the second. You train many teacher models on disjoint slices of the private data, have them vote on labels for separate public data with noise added to the vote counts, and train a student only on those noisy labels. The student never touches the private data, so once trained it can be queried forever at no extra privacy cost.

Membership inference is the canonical privacy attack and a stepping stone toward stronger data-extraction, and it maps to the Data component of the methodology alongside [Data Poisoning](../data-poisoning/). I keep the two defenses here, right next to the attack they answer, because they are privacy-specific; the general robustness defenses are in [Defenses](../defenses/). The recurring tension is worth saying out loud: more privacy almost always costs some accuracy.
