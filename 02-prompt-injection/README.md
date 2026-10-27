# Prompt Injection

Getting a model to ignore its instructions and follow yours instead. This is the signature attack on the application layer of a generative system, and it runs the whole range from a single crafted prompt to instructions you smuggle in through data the model later reads.

Before the attacks, [Introduction to Prompt Engineering](01%20Introduction%20to%20Prompt%20Engineering.md) covers how prompts steer a model in the first place, which is the groundwork for subverting them.

The core techniques are in [Prompt Injection](02%20Prompt%20Injection/). You start with reconnaissance to fingerprint the system prompt and whatever guardrails are in place, then move to direct injection, where the malicious instruction is in your own input, and indirect injection, where you plant the instruction in something the model ingests later: a web page, a file, an email, or the output of a tool.

[Jailbreaks](03%20Jailbreaks/) is the next step, bypassing the model's safety alignment to pull out behavior it is supposed to refuse. [Tools of the Trade](04%20Tools%20of%20The%20trade/) covers the tooling for automating discovery and exploitation, and [Mitigations](05%20Mitigations/) looks at both the traditional defenses (filtering, delimiting, privilege separation) and the LLM-based ones, along with why each only holds part of the way.

It is the top entry on the LLM OWASP Top 10 for a good reason: any time untrusted text reaches the model, directly or through a tool or RAG pipeline, it can carry instructions. It also chains straight into [Output Attacks](../03-output-attacks/), which is what the manipulated output then does, and into [Application & System](../06-application-system/), where it starts abusing connected tools and MCP servers.
