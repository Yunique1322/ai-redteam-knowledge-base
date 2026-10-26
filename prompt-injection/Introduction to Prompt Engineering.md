# Introduction To Prompt Engineering

**Prompt engineering = crafting input to steer the output.** Since the prompt is the LLM's only input, it's your only lever. Three best practices (which double as attack principles):

- **Clarity** - be specific and unambiguous.
- **Context & constraints** - provide detail, examples, format requirements.
- **Experimentation** - subtle changes shift the output; iterate and keep what works.

**Two things worth internalizing for the attack mindset:**

1. **LLMs are non-deterministic** - the same prompt can give different responses each time. For attacking, this means: a payload that fails once might work on retry, and a working payload might not be 100% reliable. Persistence and repetition matter.
2. **Subtle phrasing changes matter enormously** - this is the whole game in injection/jailbreaking. You'll iterate on wording, tone, and framing to find what slips past defenses. The "experiment and refine" loop IS the attack methodology.

**Scope for this module** (what you'll attack):

- **LLM01 Prompt Injection** - manipulating input to make the LLM misbehave.
- **LLM02 Sensitive Information Disclosure** - tricking it into leaking sensitive info (here, via prompt manipulation).
- Same two under SAIF (Prompt Injection, Sensitive Data Disclosure).

---

The key takeaway: prompt engineering and prompt injection are the same skill pointed in different directions - one steers the LLM toward _your legitimate goal_, the other toward _the LLM's forbidden behavior_. The iterate-and-refine loop is central to both