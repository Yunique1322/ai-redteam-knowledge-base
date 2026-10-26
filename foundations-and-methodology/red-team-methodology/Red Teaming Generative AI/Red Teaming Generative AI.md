# Red Teaming Generative AI

**Mindset:** Generative AI moves fast → misconfigurations are common. Approach it **black-box** (probe → observe → adapt), and be creative since there's no fixed checklist.

**Key technique - host your own copy:** If the target uses an open-source model, download and run it yourself. Lets you develop attacks freely (no rate limits, no logs, no alerting the target), then deliver the polished payload to the real system. Attacks often transfer since it's the same model. (Same transferability idea as your ResNet malware CNN.)

**Data dependence:** Systems that _learn from user queries_ are both poisonable (feed bad data via normal use) and a recon goldmine (their collected data aids further attacks). Examine data-handling.

**The four components (= your module structure, ≈ SAIF areas):**

|Component|Covers|Attacks|
|---|---|---|
|**Model**|The model + input/output|Prompt injection, evasion, insecure output|
|**Data**|Training + inference data|Poisoning, data theft|
|**Application**|The app the AI sits in|**Traditional web vulns (XSS, SQLi)** ← your OSCP skills|
|**System**|Hardware, OS, config, deployment|DoS/resource exhaustion, rate-limit gaps|

**Two things to remember:**

- **Application component = classic web pentesting** around the AI. Your existing skills are a direct asset, and the exam chains AI attacks + web attacks.
- **Each component has its own TTPs** - hence the four separate deep-dive sections. Your OSCP methodology (systematic, component-by-component, chained) transfers; only the techniques are new.

**Exam takeaway:** For any AI target, sweep all four components (probe Model, examine Data, pentest Application, check System) - that's how you find the chained vulnerabilities COAE tests.