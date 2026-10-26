# Attacking The Data Component

Second component: attacking the **Data** (training data + inference data). Two attack types, both familiar. Brief since it's mostly review.

**Risks:**

**1. Improper training data** - biased or unrepresentative data → low-quality, discriminatory, or harmful output. Not an attack per se, but a weakness (and a poisoning target).

**2. Data Poisoning** - corrupt the _training data_ (vs. model poisoning which corrupts _weights directly_). Your diagram shows it: clean dataset → good model (green) vs. one poisoned entry (the skull) → corrupted model (red). Same impact as model poisoning: misleading/biased/harmful output.
![Attacking The Data Component](/assets/Pasted%20image%2020260903194448.png)
- **Backdoor attack** (the key term) - embed a specific **trigger** in the poisoned data so the model behaves normally _except_ when it sees that trigger input, then produces the attacker's desired malicious output. Stealthy - looks fine until the trigger fires.

**3. Data Leaks / Theft** - steal the training or inference data. Consequences: PII leaks (GDPR/legal), stolen curated datasets (took years to build = valuable IP), and it _enables further attacks_ - with the dataset, an attacker can reverse-engineer the model or craft precise adversarial inputs. (Note: securing training data matters as much as securing the model.)

**TTPs - how these attacks actually happen:**

_Poisoning:_ the hard requirement is **knowing what data the model trains on AND being able to inject into it** - often impossible if data is well-sourced/sanitized. But one important exception:

- **Federated learning** - multiple parties contribute to training; a malicious participant injects **poisoned updates** during their legitimate participation, skewing the global model without suspicion. (Worth knowing - it's the realistic poisoning scenario.)

_Data theft:_ mostly **traditional attacks** - your OSCP/CPTS skillset applies directly:

- Poorly configured cloud storage, weak encryption (at rest/in transit), insecure data pipelines, vulnerable APIs.
- **Supply chain** - compromise a third-party data vendor to access the dataset _before_ it reaches the target.
- **Insider threat** - employees with legitimate access exfiltrate data (via phishing/social engineering of their creds, or deliberately). Hard to detect because the access is authorized.

---

**Exam relevance (quick):**

- **Backdoor attack** (trigger-based poisoning) is the key new term - model works normally until the trigger appears.
- **Data theft is largely traditional security** - cloud misconfig, weak crypto, exposed APIs, insider threats. Your existing pentest skills find these; don't assume data attacks require exotic ML techniques.
- Maps to OWASP: Data Poisoning = ML02/LLM04, backdoors = ML02, data theft = supply chain (ML06) + traditional vulns.

The through-line: attacking data is often _traditional infrastructure attacking_ (find the exposed storage, the weak API, the insider) more than ML-specific technique - which plays to your strengths. Next is **Attacking Application Components** - which is even _more_ your territory (classic web vulns around the AI). Send it when ready.