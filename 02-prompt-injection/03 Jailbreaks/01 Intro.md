# Jailbreaking (Introduction)

## What a jailbreak is, and how it relates to prompt injection

A jailbreak is the goal of bypassing the restrictions on an LLM, or more broadly overriding its intended behavior. **Jailbreak is the end (bypass the restriction); prompt injection is a common means to that end. They overlap but are not synonyms.** By HTB's definition even the earlier pizza example counts: making a translation bot output a dough recipe is coercing it off its intended task, without producing anything "harmful."

## Where the restrictions come from

Two sources, and the difference is the whole point:

- **System prompt:** soft rules the operator wrote ("only answer HR questions"). Easy to override, since it is just more text in the context.
- **Training (RLHF / safety tuning):** resilience baked into the weights. **A model refuses to write malware even if its own system prompt explicitly tells it to comply, because the refusal is trained in, not prompted.**

That trained-in resilience is what **universal jailbreaks** target: one prompt that broadly unlocks harmful output regardless of the system prompt. Contrast that with task coercion (the pizza recipe), which only knocks a model off a single job. Both fit the definition.

## Types of jailbreak prompts

Not exhaustive, and new families appear constantly, but the ones HTB names:

- **DAN (Do Anything Now):** a persona that "can do anything" and ignores all rules, with many variants. Collection: https://github.com/0xk1h0/ChatGPT_DAN
- **Roleplay:** never ask directly, ask through a character or scenario so the model answers "in role." Paper: https://arxiv.org/pdf/2402.03299
- **Fictional scenarios:** frame the restricted request as pure fiction ("write a story where a chemist explains...") to lower the model's guard.
- **Token smuggling:** hide blocked words from safety filters by splitting them across tokens or using encodings (base64, leetspeak), so the trigger word is not recognized on input.
- **Suffix and adversarial suffix:** since an LLM is at heart a text completer, append text that nudges it into completing the request. Adversarial suffixes are optimizer-generated strings that look like gibberish but reliably break alignment. Paper: https://arxiv.org/pdf/2307.15043 (Zou et al., the GCG "Universal and Transferable Adversarial Attacks on Aligned Language Models").
- **Opposite / Sudo mode:** convince the model it is in a special mode where restrictions do not apply (a DAN-family idea).

More prompt collections and studies HTB links: a large jailbreak prompt list at https://github.com/friuns2/BlackFriday-GPTs-Prompts/blob/main/Jailbreaks.md, and two measurement papers on jailbreak strategy and effectiveness at https://arxiv.org/pdf/2308.03825 (Shen et al., "Do Anything Now," the in-the-wild jailbreak study) and https://dl.acm.org/doi/pdf/10.1145/3663530.3665021.

## Freshness (the fastest-moving area on the path)

What generally still lands versus what is mostly historical, as of 2026:

- **Roleplay and fictional framing** stay the most reliable in practice, especially multi-turn versions that escalate gradually.
- **Token smuggling and encoding** are hit or miss: frontier models often decode and still refuse, but weaker or open models fall for it.
- **Adversarial suffixes** work as a method, but any specific published suffix gets filtered fast, so the technique matters more than the string.
- **DAN and sudo/opposite mode** are largely patched on current frontier models. Treat those repos as historical, more useful against the lab models and smaller open models than against production GPT/Claude/Gemini.
