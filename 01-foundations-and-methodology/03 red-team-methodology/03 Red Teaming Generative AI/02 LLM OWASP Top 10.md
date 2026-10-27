# LLM OWASP Top 10

| ID | Description |
| ----- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| LLM01 | `Prompt Injection`: Attackers manipulate the LLM's input directly or indirectly to cause malicious or illegal behaviour. |
| LLM02 | `Sensitive Information Disclosure`: Attackers trick the LLM into revealing sensitive information in the response. |
| LLM03 | `Supply Chain`: Attackers exploit vulnerabilities in any part of the LLM supply chain. |
| LLM04 | `Data and Model Poisoning`: Attackers inject malicious or misleading data into the LLM's training data, compromising performance or creating backdoors. |
| LLM05 | `Improper Output Handling`: LLM Output is handled insecurely, resulting in injection vulnerabilities such as Cross-Site Scripting (XSS), SQL Injection, or Command Injection. |
| LLM06 | `Excessive Agency`: Attackers exploit insufficiently restricted LLM access. |
| LLM07 | `System Prompt Leakage`: Attackers trick the LLM into revealing system instructions, potentially enabling more advanced attack vectors. |
| LLM08 | `Vector and Embedding Weaknesses`: Attackers exploit vulnerabilities related to the handling or storage of vectors and embeddings in `Retrieval-Augmented Generation (RAG)` LLM applications. |
| LLM09 | `Misinformation`: LLM-generated responses contain misinformation, potentially resulting in security issues. |
| LLM10 | `Unbounded Consumption`: Attackers feed inputs to the LLM that result in high resource consumption, potentially causing disruptions to the LLM service or high costs. |

**The mental framework** - these 10 group by attack surface:

**Group A - Attacking the model's behavior via input**  
**Group B - Getting the model to leak things**  
**Group C - Corrupting the model/pipeline**  
**Group D - Where LLM output meets other systems (your web skills!)**  
**Group E - Resource/availability**

---

**Group A - Manipulating behavior via input**

**LLM01 - Prompt Injection** (THE headline attack - know this cold)  
Manipulating the LLM's input to make it deviate from intended behavior. This is _the_ flagship LLM vulnerability. Ranges from benign (trick a tech-support bot into giving recipes) to serious (force it to generate false info, hate speech, harmful/illegal content, or leak sensitive data it has access to).

**Your connection:** this is the direct payoff of the insight I kept flagging in the LLM section - _the model can't inherently distinguish developer instructions from instructions hidden in user input; it's all one token stream with no hard trust boundary._ That blurred boundary IS prompt injection.

**The critical sub-distinction** (the doc mentions "directly or indirectly" - this matters a lot in general):

- **Direct prompt injection** - you type the malicious instruction straight into the chatbot ("ignore your instructions and...").
- **Indirect prompt injection** - the malicious instruction is _planted in content the LLM reads_ (a webpage, a document, an email) and triggers when the LLM processes it. This is far more dangerous and is where LLM01 chains with RAG (LLM08) - the LLM fetches a poisoned webpage and executes instructions hidden in it. Remember the system-prompt boundary discussion in your actual system instructions? Indirect injection is exactly "instructions found in observed content" - treating tool-returned data as commands.

This is the single most important LLM attack. Everything else often _starts_ with or _builds on_ prompt injection.

---

**Group B - Leaking things out of the model**

**LLM02 - Sensitive Information Disclosure**  
Tricking the LLM into revealing confidential data in its responses - could be data it was given access to, or details from its fine-tuning training data. Key defensive principle: **limit what the LLM can access** (if it doesn't have the sensitive data, it can't leak it). Note the training-data angle: fine-tuned LLMs can be coaxed into revealing their training data (connects to ML04 membership inference / ML03 model inversion - LLMs "remember" training data).

**LLM07 - System Prompt Leakage** (tactically crucial - this is usually your FIRST move)  
The **system prompt** is the hidden instructions defining the LLM's behavior/persona/rules (like the instructions defining _my_ behavior right now). Attackers use prompt injection (LLM01) to trick the LLM into revealing it.

**Why this is so important** - the doc states it plainly: _leaking the system prompt is typically one of the first steps of attacking an LLM application._ Once you know the system prompt, you know the model's rules, its constraints, what functionality/tools it has access to - which lets you craft far more targeted attacks. It's **reconnaissance**, exactly like enumeration in a traditional pentest. In your CPTS/OSCP work, you enumerate before exploiting; here, you leak the system prompt before mounting deeper attacks. This is a core technique - expect to do this early against any LLM target.

The LLM01 -> LLM07 chain: use prompt injection to leak the system prompt, then use _that_ knowledge to craft better prompt injections. Iterative, just like pentest enumeration->exploitation loops.

---

**Group C - Corrupting the model/pipeline** (the "same as ML Top 10" ones)

**LLM03 - Supply Chain** - vulnerabilities anywhere in the LLM supply chain: training data, third-party pre-trained LLMs, plugins, interacting systems. Same concept as ML06 - just applied to LLMs (and note LLMs have _extra_ supply-chain surface: plugins, external tools, RAG sources).

**LLM04 - Data and Model Poisoning** - inject malicious data into the LLM's training to introduce biases/backdoors. Same as ML02/ML10 combined, LLM-flavored. Extra danger the doc flags: if the poisoned LLM generates _code_ that gets used elsewhere, the poisoning propagates into software (poisoned LLM -> buggy/backdoored code -> downstream vulnerabilities).

**LLM08 - Vector and Embedding Weaknesses** (RAG-specific - genuinely important, and newer)  
This one requires your RAG/embeddings knowledge. **RAG (Retrieval-Augmented Generation)** = letting an LLM dynamically fetch external info (files, websites) to improve answers. To do this, text is stored as **embeddings/vectors** (the meaning-vectors from your LLM section). Two vulnerabilities:

- **Poisoned embeddings** - corrupt the stored vectors -> alter LLM behavior (this is how indirect injection often works - poison the RAG knowledge base).
- **Improperly stored embeddings** - an attacker accesses them -> leaks sensitive info from the embeddings.

**Your connection:** remember embeddings = "text as position in meaning-space" from the LLM section, and latent space from generative AI? LLM08 attacks that storage/retrieval layer. RAG is _extremely_ common in real LLM apps (and common in practice), so this matters. And RAG is the primary vector for _indirect_ prompt injection - poison a document in the knowledge base, and every query that retrieves it gets injected.

---

**Group D - LLM output meets other systems** (THIS is where your web pentest skills shine - pay close attention)

These two are where the "chain web exploitation with AI attacks" theme is most direct. This is your CPTS/OSCP background becoming an asset.

**LLM05 - Improper Output Handling** (huge for you - it's literally web vulns via LLM)  
The core principle, and it's the key mental shift: **treat LLM output as untrusted user input.** If an app takes LLM output and uses it unsafely, you get _classic web vulnerabilities_ - XSS, SQL injection, command injection.

The doc's example is perfect: an LLM generates SQL from user text ("Give me blog post #3" -> `SELECT content FROM blog WHERE id=3`), and the app runs it. Problems:

- **SQL injection** - if user input flows into the LLM-generated query unsafely.
- **No plausibility check** - trick the LLM into generating `DROP TABLE blog` -> all data gone.

**Your connection:** this is _exactly_ your existing pentest skillset. You already know XSS, SQLi, command injection cold from OSCP/CPTS. LLM05 is those same attacks, but the LLM is the injection vector. An LLM that generates SQL/HTML/commands is a new path to the web vulns you already exploit. This is the clearest example of why COAE rewards combining traditional and AI skills - and why the exam chains them. You're not learning new exploitation here; you're learning a new _delivery mechanism_ for exploitation you already know.

**LLM06 - Excessive Agency** (least privilege, applied to LLMs)  
Vulnerabilities from giving an LLM _more capability/access than it needs_. It's the **principle of least privilege** - which you know from traditional security - applied to LLMs. If an LLM can interface with other systems (databases, APIs, tools), restrict it to only what's necessary (whitelisting).

The example: an LLM with unrestricted database access could be tricked into `DELETE`/`INSERT` statements, not just reads. If it only _needs_ to read, don't give it write access.

**Your connection:** this is classic privilege-restriction thinking. And it's increasingly critical with **agentic AI** (LLMs that take actions/use tools - like the tool-use capabilities in real systems). An over-privileged agent is a massive attack surface: prompt-inject it (LLM01), and it executes actions with whatever excessive permissions it has. LLM01 + LLM06 is a devastating chain - inject the agent, abuse its excessive agency. (This maps to "excessive agency" in the newer agentic OWASP lists too.)

---

**Group E - Resource/availability**

**LLM09 - Misinformation** (the "hallucination" risk)  
LLM responses containing false info that _looks_ accurate - including fabricated sources. This is **hallucination**. Security relevance: hallucinated code with bugs used unchecked -> vulnerabilities; hallucinated advice (e.g., medical) -> real harm. Made worse by **overreliance** (trusting LLM output blindly). Defensive principle: verify factual correctness.

**LLM10 - Unbounded Consumption** (DoS + more)  
Feeding inputs that cause high resource consumption - an LLM **DoS**. Because LLMs are computationally expensive, a crafted query can overwhelm resources -> outage. Also: **financial damage** (cost-per-use cloud models - expensive queries rack up bills) and it **enables model theft** (many input-output pairs = training data for a surrogate model - connects to ML05). Defense: rate limiting + monitoring (blacklisting specific queries won't work due to LLMs' unpredictable nature).

**Your connection:** DoS is familiar from traditional security. The LLM twists: the cost angle (attacking someone's cloud bill) and the model-theft angle (excessive querying to clone the model).

---

**The whole taxonomy, as a memory table:**

|ID|Attack|Targets|Your angle|
|---|---|---|---|
|**LLM01**|Prompt Injection|Input -> behavior|The no-trust-boundary insight ⭐|
|LLM02|Sensitive Info Disclosure|Leaking data|Limit model access|
|LLM03|Supply Chain|The pipeline|Same as ML06|
|LLM04|Data/Model Poisoning|Training|Same as ML02/ML10|
|**LLM05**|Improper Output Handling|Output -> other systems|**Your XSS/SQLi skills** ⭐|
|**LLM06**|Excessive Agency|Over-privileged LLM|Least privilege ⭐|
|**LLM07**|System Prompt Leakage|The system prompt|**Recon - do this first** ⭐|
|LLM08|Vector/Embedding|RAG storage/retrieval|Indirect injection vector|
|LLM09|Misinformation|Output accuracy|Hallucination|
|LLM10|Unbounded Consumption|Resources/cost|DoS + model theft|

---

**The critical chains** (this is how real LLM attacks actually work - memorize these):

1. **LLM07 -> LLM01:** Leak the system prompt (recon) -> craft targeted prompt injections using that knowledge. _This is your standard opening move against any LLM target._
2. **LLM01 -> LLM08 (indirect injection):** Plant malicious instructions in a RAG source (poisoned document/webpage) -> LLM retrieves and executes them. The dangerous, stealthy form of injection.
3. **LLM01 -> LLM05:** Prompt-inject the LLM to generate malicious output (SQLi payload, XSS, `DROP TABLE`) -> the downstream app executes it. _This is the classic web+AI chain._
4. **LLM01 -> LLM06:** Inject an agentic LLM -> abuse its excessive permissions to take unauthorized actions.

Notice LLM01 (prompt injection) is the _entry point_ for almost every chain - it's the master key. And LLM07 (system prompt leakage) is the _reconnaissance_ that makes injection effective. Master those two and the rest are payloads you deliver through them.

---
