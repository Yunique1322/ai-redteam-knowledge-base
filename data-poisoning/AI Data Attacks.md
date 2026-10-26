# AI Data Attacks

## Three attack families (know the boundaries)

- ****Data attacks** (this whole module): corrupt the training data or the stored model artifact. Earliest phase of the lifecycle.**
- **Evasion attacks:** manipulate inputs at inference time to fool an already-deployed model.
- **Privacy attacks:** extract sensitive information the model memorized.

The line that separates them: **data attacks work before the model ever serves a prediction, so by the time it does, the damage is already baked in.**

## Attack per pipeline stage
![AI Data Attacks](/assets/2%20AI%20Data%20Attacks-20260917225056838.png)
- **Collection → data poisoning** through the trusted input channel (no break-in, you just use the API everyone else uses). Sub-types: **label flipping**, **feature attacks**, and **backdoor-trigger** keywords. E-commerce: fake reviews carrying trigger keywords. Healthcare: altered DICOM metadata or clinical notes that mislabel samples.![AI Data Attacks](/assets/2%20AI%20Data%20Attacks-20260917225108633.png)
- **Storage → two doors.** (a) Steal or tamper with stored datasets, which **bypasses whatever validation ran at ingestion (poisoning through a different door).** (b) Model files carry **executable trust**, so an attacker with write access plants a **trojan** (normal on clean inputs, attacker-chosen output on triggers) or goes further with **model steganography**: **hiding code in the file so `pickle.load()` executes it. The outcome there is code execution, not wrong predictions.** A poisoned `.pkl` in the Flask server is a foothold on serving infra; a poisoned `.pt` in the clinical system reaches the hospital network.![Attack per pipeline stage](/assets/Pasted%20image%2020260917225115.png)
- **Processing → attack the code, not the data.** **You corrupt data that was clean on arrival, so the raw data upstream still passes any audit and the corruption is much harder to find.** Compromising the Spark sentiment job flips labels without touching a single review.![AI Data Attacks](/assets/2%20AI%20Data%20Attacks-20260917225152784.png)
- **Modeling → inherited, not introduced.** The training job faithfully bakes in whatever is upstream. No detection, no resistance.![Attack per pipeline stage](/assets/Pasted%20image%2020260917225158.png)
- **Deployment → intercept the model file in transit** between storage and production when there is no hash, signature, or integrity check. Same trojan/stego outcome, **but the access requirement drops: you only need the deployment path (a misconfigured CI/CD pipeline, an unauthenticated model-pull endpoint, or a MITM position), not storage write access.**![Attack per pipeline stage](/assets/Pasted%20image%2020260917225206.png)
- **Retraining → online poisoning, the most effective entry point,** precisely because **the pipeline is structurally built to trust new data.** Normal-looking submissions accumulate slowly across retraining cycles, and monitoring has to separate three near-identical things: genuine distribution drift, random noise, and adversarial manipulation. The system was designed to adapt to the first and **has no built-in mechanism to reject the third.**![Attack per pipeline stage](/assets/Pasted%20image%2020260917225211.png)

## Impact range

Mild (subtle bias, degraded predictions) → severe (full model compromise) → worst (embedded trojan or deserialization RCE that **breaks out of the model and into the surrounding infrastructure**).

## Framework mapping, and a numbering trap

**Heads up, this module was written against the older 2023 OWASP list, so its numbers differ from the 2025 mapping I gave you at the start of the last module.** The course says:

- Data poisoning → **OWASP LLM03: Training Data Poisoning** (2023 numbering)
- Everything else → **OWASP LLM05: Supply Chain Vulnerabilities** (2023 numbering)

In the current **2025** list, those same concepts are **LLM04: Data and Model Poisoning** and **LLM03: Supply Chain**. Same ideas, renumbered. **Answer with whichever list the question is phrased in, but know both mappings so a number swap doesn't catch you out.**

SAIF (Google's Secure AI Framework) covers it lifecycle-style: Secure Design, Data, Secure Supply Chain, Security Testing, Secure Deployment, and Secure Monitoring & Response. The one worth remembering: **online poisoning maps to Secure Monitoring & Response, because catching it requires continuous statistical anomaly detection over time, not a one-time deployment gate.**![Framework mapping, and a numbering trap](/assets/Pasted%20image%2020260917225224.png)

## Exam anchors

The three families and their lifecycle phase; the stage-to-attack map; the **trojan (wrong outputs) vs steganography/pickle (RCE)** split; online poisoning as a "feature not a bug" stealth attack; and the old-vs-new OWASP numbering.


![AI Data Attacks](/assets/2%20AI%20Data%20Attacks-20260918152758745.png)
