# Foundations & Methodology

The groundwork for everything else here: enough ML to reason about where a system breaks, the applied examples that show how security models actually get built, and the frameworks that organize how you approach an AI target.

Start with the [AI/ML Primer](AI-ML%20Primer.md) if the theory is rusty. It is a short, attacker-focused tour of the learning paradigms, the model lifecycle as an attack surface, the norms (L0, L2, Linf), and the metrics both sides care about.

[Applied ML in Infosec](applied-ml-in-infosec/) is the hands-on half: preprocessing and transforming data, picking metrics, and three worked classifiers for spam, network anomaly detection, and malware. This is the "what you're attacking" material, and it is worth building one of these yourself before you try to break one.

[Red Team Methodology](red-team-methodology/) is the framing. It covers the ML OWASP Top 10, the LLM OWASP Top 10, Google's SAIF, and the attack-surface map that splits an AI system into Model, Data, Application, and System components. The rest of this knowledge base is organized along exactly those lines:

- Data attacks: [Data Poisoning](../data-poisoning/) and [Privacy](../privacy/)
- Model attacks: [Model Evasion](../model-evasion/)
- Application attacks: [Prompt Injection](../prompt-injection/) and [Output Attacks](../output-attacks/)
- System attacks: [Application & System](../application-system/)
- Defense, which cuts across all of them: [Defenses](../defenses/)
