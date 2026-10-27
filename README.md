# AI Red Team Knowledge Base

My working reference for attacking AI and machine-learning systems. It covers the attack surface, the techniques, the actual code, and the defenses, and it is laid out the way you would reach for things during an engagement rather than in the order a course happens to teach them.

Every page keeps its commands and code exactly as I ran them, with the screenshots inline. Images live in one place, [`/assets/`](assets/), and are linked so they render correctly on GitHub.

> ## [Master Cheatsheet](CHEATSHEET.md)
> Every attack here boiled down to a short recipe: what you are trying to do, how to set it up, the call or parameters that matter, and how you know it worked. It is in attack-surface order and each entry links to the full note. Start here.

## How I use this

I drive it from the [cheatsheet](CHEATSHEET.md) and follow a link into a domain when I need the full story. The domains follow the attack surface the methodology lays out, Data then Model then Application then System, with the fundamentals up front and the defenses, which cut across everything, at the end.

## Domains

| Domain | What's in it |
|--------|-------------|
| [Foundations & Methodology](01-foundations-and-methodology/) | An attacker's ML primer, the applied infosec classifiers (spam, network anomaly, malware), and the frameworks: the ML and LLM OWASP Top 10, Google's SAIF, and the Model/Data/Application/System map. |
| [Prompt Injection](02-prompt-injection/) | Direct and indirect injection, jailbreaks, the tooling, and the mitigations. Attacking the model through what you feed it. |
| [Output Attacks](03-output-attacks/) | Insecure output handling: XSS, SQLi, code injection, function-calling abuse, exfiltration, and the risk from hallucinations. Attacking what the model's output then does. |
| [Data Poisoning](04-data-poisoning/) | Training-time attacks: label flipping, clean-label poisoning, trojan backdoors, and malicious pickles and tensor steganography. |
| [Model Evasion](05-model-evasion/) | Adversarial examples at inference time: the GoodWords foundations, the first-order attacks (FGSM, I-FGSM, DeepFool), and the sparse ones (ElasticNet, JSMA, single-pixel). |
| [Application & System](06-application-system/) | Everything around the model: reverse engineering, denial of service, insecure components, rogue agent actions, deployment tampering, and MCP security. |
| [Privacy](07-privacy/) | Membership inference with shadow models, and the two defenses that answer it, DP-SGD and PATE. |
| [Defenses](08-defenses/) | The cross-cutting controls: LLM guardrails, adversarial training, and safety tuning. |

## How it's organized

The original course's module order is gone. A technique lives where you would actually reach for it, grouped by the part of the system it hits, and the numeric prefixes on the folders and notes just set the reading order. The spine is the model lifecycle: poisoning hits training, evasion and prompt injection hit inference, privacy attacks mine the trained model, the application and system attacks go after everything around it, and the defenses span the lot. Code, commands, and model implementations are kept exactly as I used them; only the prose around them has been tidied for reference.

## Acknowledgements

I built this from what I learned doing hands-on AI red-teaming training, and I want to credit the platform that taught me most of it:

- [HTB Academy](https://academy.hackthebox.com/), which is where the bulk of this material comes from.

None of this is copy-pasted. Every page is written in my own words from my notes, my own practice, and my own research. HTB is credited as what I learned from, not as the source of any text here.

Turning my raw Obsidian notes into this structured, methodology-first knowledge base was done with [Claude Code](https://claude.com/claude-code): organizing, restructuring, and editing my own notes into a form worth publishing.
