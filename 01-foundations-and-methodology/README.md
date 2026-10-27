# Foundations & Methodology

The groundwork for everything else here: enough ML to reason about where a system breaks, the applied examples that show how security models actually get built, and the frameworks that organize how you approach an AI target.

Start with the [AI/ML Primer](01%20AI-ML%20Primer.md) if the theory is rusty. It is a short, attacker-focused tour of the learning paradigms, the model lifecycle as an attack surface, the norms (L0, L2, Linf), and the metrics both sides care about.

[Applied ML in Infosec](02%20applied-ml-in-infosec/) is the hands-on half: preprocessing and transforming data, picking metrics, and three worked classifiers for spam, network anomaly detection, and malware. This is the "what you're attacking" material, and it is worth building one of these yourself before you try to break one.

[Red Team Methodology](03%20red-team-methodology/) is the framing. It covers the ML OWASP Top 10, the LLM OWASP Top 10, Google's SAIF, and the attack-surface map that splits an AI system into Model, Data, Application, and System components. The rest of this knowledge base is organized along exactly those lines:

- Data attacks: [Data Poisoning](../04-data-poisoning/) and [Privacy](../07-privacy/)
- Model attacks: [Model Evasion](../05-model-evasion/)
- Application attacks: [Prompt Injection](../02-prompt-injection/) and [Output Attacks](../03-output-attacks/)
- System attacks: [Application & System](../06-application-system/)
- Defense, which cuts across all of them: [Defenses](../08-defenses/)
