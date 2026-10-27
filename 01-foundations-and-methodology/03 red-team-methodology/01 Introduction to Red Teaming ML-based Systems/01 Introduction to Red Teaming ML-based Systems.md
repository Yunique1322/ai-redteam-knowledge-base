# Introduction To Red Teaming ML-based Systems

![Introduction To Red Teaming ML-based Systems](/assets/Pasted%20image%2020260831143349.png)
- **Vulnerability Assessment (base)** - automated scanning for _known_ vulnerabilities. Tools like Nessus/OpenVAS scan and produce a list of findings with remediation advice. **It does NOT exploit anything** - just identifies and catalogs. Broad but shallow.
- **Penetration Testing (middle)** - the focused, time-boxed exploitation you know from CPTS/OSCP. Scoped to specific systems/apps, uses tools + manual technique, actually _exploits_ vulnerabilities to prove they're real and show impact. Deeper, but narrow scope.
- **Red Teaming (top)** - the advanced, adversarial simulation. Mimics real attackers' full TTPs (tactics, techniques, procedures), tests _everything_ - technical vulnerabilities, people (social engineering, phishing), processes, physical security. Emphasizes **stealth and persistence** (evading the defensive "blue team"), pursues specific objectives (steal data, reach critical systems), and runs for **weeks to months**. Broadest scope, most realistic, most sophisticated.

**ML-based systems are best assessed via red teaming**, for specific reasons:

1. **Time.** Many ML attacks (crafting adversarial examples, extracting a model, poisoning data) require _more time than a typical pentest allows_. This is literally why a real engagement is **7 days**, not a few hours - advanced AI attacks take time to develop. (Recall this from the exam-format discussion - the long window trades intensity for depth.)
2. **Interconnected components.** ML systems are made of many interacting parts (the model, the data pipeline, the application wrapping it, the serving infrastructure). **Vulnerabilities often arise at the _interaction points_ between components** - not in any single piece. A narrow pentest scope might exclude a component or an interaction, leaving vulnerabilities undetectable. Red teaming's holistic scope catches these.

This connects directly to what you learned about a real engagement: it's not just "prompt-inject the chatbot" - it's chaining attacks across the whole system (web app + AI + infrastructure). That holistic, cross-component thinking _is_ red teaming, and it's why red teaming demands both traditional web skills and AI-specific skills.

**Why ML systems are uniquely vulnerable** (the genuinely new part)

This is the core insight of the section. Traditional software fails in traditional ways (buffer overflows, injection, misconfig). ML systems have an _entirely additional_ attack surface because they rely on three things normal software doesn't:

1. **Large datasets** - the model is shaped by its training data. This creates the **data poisoning** surface (corrupt the data -> corrupt the model) you've been noticing since the preprocessing sections. Normal software doesn't "learn from data," so it doesn't have this weakness.
2. **Statistical inference** - the model makes probabilistic guesses, not deterministic decisions. This creates the **evasion** surface (craft input that lands on the wrong side of a statistical boundary) - everything you saw with spam word-stuffing and the IDS blind spots. A model that decides by statistics can be fooled by inputs engineered against those statistics.
3. **Complex model architectures** - the models are intricate (millions of parameters, opaque decision-making). This creates surfaces like **model extraction** (stealing the model) and **adversarial examples** (the gradient attacks), plus the sheer complexity means "much room for security issues to arise."