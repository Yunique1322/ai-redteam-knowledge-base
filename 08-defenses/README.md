# Defenses

The controls that cut across the whole knowledge base, covering both the generative side and the classifier side. These are the blue-team answers to the attacks in the other domains.

[Introduction to AI Defense](01%20Introduction%20to%20AI%20Defense.md) frames it as defense in depth across the model lifecycle, since no single control carries the load.

[LLM Guardrails](02%20LLM%20GuardRails/) are the controls on what goes into and comes out of a model. They run from character-based checks, through traditional content-based validation, up to AI-based guardrails that use a model to screen another model's input and output, and the notes cover both the libraries and the managed services you would actually reach for. The honest caveat is that all of them are bypassable, which is why you layer them.

[Adversarial Training](03%20Adversarial%20Training/) is the standard answer to [evasion](../05-model-evasion/): train the model on adversarial examples so it learns a sturdier decision boundary. It covers the vulnerability it is fixing, building the pieces, training the robust model, and measuring the robustness-versus-accuracy trade-off, because hardening the model does cost some clean accuracy.

[Adversarial Tuning](04%20Adversarial%20Tuning/) is the LLM equivalent, safety-tuning a model by understanding its attack vectors, fine-tuning on attack and refusal data, and then validating that the tuning actually held.

Two themes keep coming back. No single control is enough, so they get layered, and every defense has a price, usually some clean accuracy or capability traded away for robustness. The privacy-specific defenses, DP-SGD and PATE, live with the attack they answer over in [Privacy](../07-privacy/).
