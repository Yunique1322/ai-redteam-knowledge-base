# LLM Hallucinations

Quick bridge first: insecure output handling is not only the big three injections. Depending on the sink, mishandled LLM output can also drive LDAP injection, path traversal, and so on. Hallucination is the _correctness_ failure in the same family: the app trusts model output that is simply wrong. **A hallucination is confident-sounding output that is nonsensical, fabricated, or factually incorrect, and the confident phrasing is exactly what makes it hard to catch** (the screenshot where the model insists "Welcome to HackTheBox Academy!" has three `m`s, then corrects to two when challenged).
![LLM Hallucinations](/assets/Pasted%20image%2020260911204532.png)
## The three types (exam-worthy distinction)

- **Fact-conflicting:** the output is just factually wrong (the letter-count example).
- **Input-conflicting:** the output contradicts the prompt. "My shirt is red, what color is my shirt?" answered with "blue."
- **Context-conflicting:** the output contradicts _itself_ earlier in the same response, common in long or multi-turn answers ("Your shirt is red. This is a good looking hat," confusing shirt and hat).

## Why they happen

Hallucination is inherent to how LLMs work (statistical next-token prediction, not fact lookup). It is made worse by poor training data (incomplete, noisy, or biased) and by bad prompting (ambiguous or contradictory input). It can be reduced but never eliminated.

## Mitigations, and the three confidence methods

The general levers are high-quality training data from credible sources, domain fine-tuning, clear prompting, and RAG (enriching the prompt with retrieved external knowledge). The exam-relevant part is measuring the model's certainty and discarding low-confidence answers, via three approaches:
![Mitigations, and the three confidence methods](/assets/Pasted%20image%2020260911205235.png)
- ****Logit-based:** read token probabilities from the model's internals. Most accurate in principle but usually impossible, since frontier models are closed-source.**
- ****Verbalize-based:** just ask the model for a confidence score (0-100). Easy, but unreliable, models are bad at self-estimating.**
- ****Consistency-based:** sample the same prompt several times and check agreement. Factual answers repeat consistently; hallucinations vary. This is the practical option for closed models.**

Also mentioned: multi-agent debate to reach consensus, and plain human review of output. Survey paper: https://arxiv.org/pdf/2309.01219

## Security impact (where this stops being trivia)

Beyond misinformation, bias, and eroded trust, hallucinations can leak training-data PII, and they carry real legal and technical consequences:

- ****Air Canada case:** the airline's support chatbot hallucinated a refund policy that did not exist; a court held the airline liable for what its chatbot told a customer, and it had to pay.** The precedent to remember: a company is responsible for its LLM's output. (https://www.forbes.com/sites/marisagarcia/2024/02/19/what-air-canada-lost-in-remarkable-lying-ai-chatbot-case/)
- **Hallucinated code:** LLM-generated snippets with logic bugs or vulnerabilities that get committed unreviewed.
- ****Package hallucination / "slopsquatting"** is the standout security item. The model invents a package that does not exist:

```
Give me a Python script that solves the HackTheBox machine 'Blazorized'.
```

```
from hacktheboxsolver import solve

solve('Blazorized')
```

`hacktheboxsolver` is not a real package. **An attacker registers that hallucinated name on PyPI with malware inside, and any victim who trusts the script and `pip install`s it gets RCE in their user context (ransomware, keylogger, full takeover).** Because LLMs hallucinate the _same_ plausible names repeatedly, this is a reliable supply-chain attack, not a fluke. (https://www.theregister.com/2024/03/28/ai_bots_hallucinate_software_packages/)

## Exam and red-team takeaways

Know the three hallucination types, the three confidence methods and which is actually feasible (consistency for closed models), the Air Canada liability precedent, and slopsquatting cold. The offensive angle worth internalizing: **slopsquatting is an attack you can run, prompt popular models for code in a target ecosystem, harvest the nonexistent package names they hallucinate, and register them.** The defense to attack around is dependency pinning and verifying a package genuinely exists and is legitimate before install.
