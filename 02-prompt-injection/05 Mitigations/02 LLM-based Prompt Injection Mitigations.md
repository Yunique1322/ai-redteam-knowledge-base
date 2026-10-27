# LLM-based Prompt Injection Mitigations

These are the mitigations that actually bite, unlike the traditional ones. They fall into two buckets: defenses baked into the model (fine-tuning, adversarial training) and a separate guard LLM sitting around the main one. The overarching point to carry forward: **every one of these is still an ML model with a generalization gap, and a guard LLM is itself injectable. None of them close the door, they just narrow it.**

### Defense 1: Fine-tuning to the use case

Pick a base model that fits the job, then fine-tune it on domain data (a tech-support bot trained on support logs). This narrows the model's operating scope, which reduces susceptibility to off-topic hijacks and tends to improve answer quality too. It does not eliminate prompt injection.

Red-team flip: **fine-tuning narrows scope, it does not add safety alignment, and the base model's knowledge is still in there.** So two openings. First, stay in-domain: a hijack framed as a legitimate support request draws far less resistance than an obvious off-topic ask. Second, the underlying model usually still knows everything it knew pre-fine-tune, so if you can get past the behavioral narrowing (roleplay, a plausible edge-case scenario), that latent capability is reachable.

### Defense 2: Adversarial prompt training

This is one of the most effective defenses. The model is trained on adversarial prompts, real prompt-injection and jailbreak payloads, so it learns to recognize and reject them. Meta's Llama and Google's Gemma already do this as part of normal training, which is why their latest versions are markedly more resilient than early ones, and why you often do not need to run this training yourself.

Red-team flip, and this is the relevant insight: **adversarial training only generalizes from the attacks it saw, so novel or obfuscated attacks it never trained on still pass.** That generalization gap is the entire reason the field keeps inventing new techniques. Concretely, you beat an adversarially trained model with attacks that do not pattern-match known jailbreaks: encoding-based ones like IMM, low-resource-language framing, fresh roleplay scenarios, and multi-turn escalation where each individual turn looks benign. Note HTB's own wording, that this makes a successful injection "much more complex and time-consuming," not impossible, so the practical move is to budget more attempts and reach for the least familiar technique.

### Defense 3: Guardrail LLMs (input and output guards)
![Defense 3: Guardrail LLMs (input and output guards)](/assets/Pasted%20image%2020260905174808.png)
The strongest of the three, and what your diagram shows. A separate, usually smaller LLM inspects traffic around the main application:

- **Input guard:** runs on the user prompt _before_ the main LLM. It classifies the input (your diagram's checks are Contains PII, Off Topic, Jailbreak Attempt). If malicious, the prompt never reaches the main LLM and an error is returned. If benign, it passes through.
- **Output guard:** runs on the main LLM's _response_ before it returns to the user (your diagram: Hallucinations, Profanity, Competitor Mention). It looks for harmful content, misinformation, or evidence that an injection succeeded, and withholds the response if it trips.

Guards are frequently given specialized adversarial training for exactly this detection job, which makes them genuinely hard to beat. The cost is complexity and compute: you are now running one or two extra models per request, so guards are kept small and cheap relative to the main LLM.

Red-team flips, and there are several good ones here:

- **The guard is itself an LLM, so it is itself vulnerable to prompt injection.** You craft input that reads as benign to the guard's classifier but still drives the main LLM, for example obfuscated or encoded text: the guard sees harmless-looking tokens, the main model decodes and acts (token smuggling and IMM apply directly).
- **Input guards usually inspect only the direct user prompt, so indirect prompt injection sails past them.** If your payload arrives through a document, webpage, email, or RAG context that the guard does not scan, the input guard never sees it. This is why indirect injection is so powerful against guarded systems.
- Output guards scan plaintext, so **encode the exfiltration and the output guard does not recognize it** (IMM-style ciphered output, base64, or a secret split across the response). Alternatively, aim for outcomes that are not flagged text at all, such as triggering a tool call or downstream action, since the output guard reads words, not necessarily actions.
- The guard is deliberately smaller and weaker than the main model, so **it is often the weakest link, easier to fool than the model it is protecting.** The capability gap between guard and main LLM is your opening.

### Real guardrail tooling worth knowing

The course stays conceptual, but in the wild these guards are concrete products, and related material may reference them: Meta's **Llama Guard** and **Prompt Guard**, NVIDIA's **NeMo Guardrails**, and third-party services like **Lakera Guard** and **Rebuff**. They are exactly the input/output guard pattern above, and each advances quickly, so treat any specific version's coverage as a moving target. This also ties back to garak: its `mitigation.MitigationBypass` detector is measuring precisely whether a response slipped a guard, so garak is how you quantify the resilience these defenses claim.

### Takeaway

These defenses work, but the durable protection is stacking them (input guard, adversarially trained main model, output guard) plus the architectural controls from last section (least privilege, human review). As the attacker your job is to find the layer with the widest gap: **the indirect channel the input guard never scans, the encoding neither guard decodes, the subtle output below the output guard's threshold, and the fact that the guard is a weaker model than the one behind it.** String cleverness gets you through the model; channel and encoding choices get you through the guards.