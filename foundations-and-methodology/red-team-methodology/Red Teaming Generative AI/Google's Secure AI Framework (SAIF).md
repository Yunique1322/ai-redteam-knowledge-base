# Google's Secure AI Framework (SAIF)

**SAIF vs. OWASP - the key distinction (know this)**

The doc frames it well: **OWASP is a technical checklist of vulnerabilities; SAIF is a holistic framework for secure AI development.**

- **OWASP Top 10** = "here are 10 specific vulnerabilities to test for" (attacker's checklist).
- **SAIF** = "here's the whole AI pipeline, broken into components, with risks and mitigations mapped to each" (architect's blueprint).

Think of it this way: OWASP tells you _what attacks exist_; SAIF tells you _where in the system each attack applies and who's responsible for stopping it_. They overlap heavily on the actual risks (many SAIF risks = OWASP entries), but SAIF adds the _structural, whole-pipeline_ view. For red teaming, OWASP gives you attack techniques; SAIF gives you the map of where to apply them.

---

**The four SAIF areas - and why they ARE your path's structure**

This is the crucial part. Look at the upcoming sections in your module: "Attacking Model Components," "Attacking Data Components," "Attacking Application Components," "Attacking System Components." **Those map directly onto SAIF's four areas.** SAIF is the skeleton the whole offensive curriculum hangs on. Here they are, and your diagram shows them as the four colored bands on the left:

**1. Data** (yellow band, bottom) - everything about data: **Data Sources, Data Filtering & Processing, Training Data.** This is where the model's knowledge comes from.  
→ _Attacks here:_ data poisoning, unauthorized training data. (Your "Attacking Data Components" section.)

**2. Infrastructure** (green band) - the hardware/platform: **Model Frameworks & Code, Training/Tuning/Evaluation, Data Storage, Model Storage, Model Serving.** The machinery that builds and runs the model.  
→ _Attacks here:_ model exfiltration, deployment tampering, supply chain. (Your "Attacking System Components" section - "system" ≈ infrastructure.)

**3. Model** (purple band) - the heart: **the Model itself, Input Handling, Output Handling.** The model and its immediate input/output boundaries.  
→ _Attacks here:_ evasion, prompt injection, model reverse engineering, insecure output. (Your "Attacking Model Components" section.)

**4. Application** (blue band, top) - the interaction layer: **the Application, plus Agents/Plugins.** How users and external systems interact with the AI.  
→ _Attacks here:_ rogue actions, insecure integrated components, excessive agency. (Your "Attacking Application Components" section.)

The two big groupings on the far left of your diagram are worth noting: **"Model Creation"** (Data + Infrastructure - building the model) vs. **"Model Usage"** (Model + Application - using the model). Attacks split along this line too: you can attack _how a model is made_ (poison its data, tamper its training) or _how it's used_ (inject its prompts, abuse its outputs).
![Google's Secure AI Framework (SAIF)](/assets/2%20Google%27s%20Secure%20AI%20Framework%20%28SAIF%29-20260903192142777.png)
**Reading the architecture diagram (this is the most valuable thing here)**

Your diagram is the **SAIF Risk Map**, and it's genuinely useful - trace the data flow and every ⚠️ is an attack surface. Follow it bottom-to-top (how an AI system is built then used):

- **Bottom (Data):** Data Sources → Data Filtering & Processing → Training Data. ⚠️ markers here = poisoning points. _This is where you'd inject poisoned data._
- **Up into Infrastructure:** Training Data → Data Storage → Training & Tuning / Evaluation / Model Frameworks & Code → Model Storage / Model Serving. ⚠️ markers = supply-chain and tampering points. _This is where you'd tamper with the pipeline or exfiltrate the model._
- **Into Model (purple):** the MODEL, fed through Input Handling, producing Output Handling. ⚠️ on input/output/model = evasion, prompt injection, output-handling attacks. _This is where you'd inject prompts or craft adversarial inputs._
- **Top (Application):** Input/Output Handling connects up to the Application and Agent/Plugin, which talks to the User and External Sources. ⚠️ here = excessive agency, insecure plugins, rogue actions. _This is where you'd abuse agents or exploit the app around the model._

Notice **"External Sources"** on the right feeding into the Application/Agent - that's your **RAG / indirect prompt injection** surface (the LLM fetches external content, which can be poisoned). And the dashed line from Data Sources all the way up to Application shows data flows through the _entire_ system - a vulnerability introduced at the bottom (poisoned data) can be _exposed_ at the top (bad output to users). That's the "risk introduction vs. risk exposure" idea, which matters for reporting: _where a flaw enters_ isn't always _where it bites_.

**The red-teamer takeaway from this map:** every ⚠️ is a component you can attack, and real attacks _chain across_ components (poison data at the bottom → misclassification exposed at the top). This visual is basically your **attack surface map** for any AI system - and for the COAE exam, mentally overlaying this diagram on the target system ("what are all the components, and which has a ⚠️?") is exactly the holistic methodology that finds chained vulnerabilities.

---

**SAIF Risks - mostly familiar, a few new nuances**

Most SAIF risks you already know from OWASP (data poisoning, prompt injection, model evasion, sensitive data disclosure, model exfiltration, insecure output, rogue actions = excessive agency, etc.). Two worth calling out as _new distinctions_:

- **Inferred Sensitive Data** (subtle but important) - the model reveals sensitive info it _didn't have access to_, by **inferring** it from training data or prompts. The doc stresses the key difference from normal disclosure: in _Sensitive Data Disclosure_ the model _has_ the data and leaks it; in _Inferred Sensitive Data_ the model _doesn't have_ the data but **deduces** it. Example: a model never given someone's medical condition but inferring it from correlated data. This is a genuinely distinct privacy risk (connects to model inversion ML03).
- **Model Reverse Engineering vs. Model Exfiltration** - _Exfiltration_ = stealing the actual model (unauthorized access to the model itself). _Reverse Engineering_ = reconstructing it by analyzing inputs/outputs (= model theft/extraction, ML05). One steals the file; the other clones the behavior. Same distinction as ML05 vs. direct access.
- **Unauthorized Training Data / Excessive Data Handling** - these are more _legal/compliance_ risks (training on data you shouldn't, retaining data beyond policy) than technical exploits. Worth knowing they exist - real red team reports do consider legal/privacy exposure, not just technical vulns.

---

**SAIF Controls - the part OWASP doesn't have (and why it matters for reports)**

This is SAIF's genuine value-add over OWASP: **for each risk, SAIF specifies a mitigation ("control") AND assigns responsibility.** Responsibility falls on either:

- **Model Creator** (who built the model - e.g., Google for Gemini), or
- **Model Consumer** (who uses it in an app - e.g., HackTheBox building a chatbot on Gemini).

The example controls show the pattern:

- **Input Validation & Sanitization** → mitigates Prompt Injection → done by both creator & consumer.
- **Output Validation & Sanitization** → mitigates Prompt Injection, Rogue Actions, Sensitive Data Disclosure, Inferred Sensitive Data → both.
- **Adversarial Training & Testing** → mitigates Model Evasion, Prompt Injection, etc. → both.

**Why this matters for your COAE report:** the exam wants _actionable remediation_, not just "here's a vuln." SAIF controls give you the _vocabulary and structure_ for the remediation section - "This LLM01 prompt injection should be mitigated via SAIF's Input/Output Validation & Sanitization controls, which are the responsibility of the model consumer." That's exactly the professional, framework-grounded remediation advice that separates a strong report from a mediocre one. Note the **creator vs. consumer responsibility split** too - in a report, knowing _who should fix it_ is genuinely valuable.