# Output Attacks

What happens after the model produces text. Its output is almost always fed into something else, a browser, a database, a shell, a function call, and if that output is trusted without checking, the model turns into an injection vector. That is the whole idea behind insecure output handling.

[Introduction to Insecure Output Handling](01%20Introduction%20to%20Insecure%20Output%20Handling/) sets up the core problem of treating model output as trusted. [Insecure Output Handling](02%20Insecure%20Output%20Handling/) then walks the concrete sinks. XSS is model output rendered into a page without escaping. SQL injection is output interpolated straight into a query. Code injection is output handed to an interpreter or `eval`. Function calling is output driving a tool call with arguments you control. Exfiltration is coaxing the output into leaking data, classically through a markdown image URL pointed at a host you own. Hallucinations are a risk in their own right, since a confidently invented package name can set up a dependency-confusion attack. The mitigations section covers the fix, which is to treat output as untrusted input and encode, validate, and least-privilege it at the point it gets consumed.

[Abuse Attacks](03%20Abuse%20Attacks/) covers abusing the model's raw capabilities at scale.

This is really the downstream half of [Prompt Injection](../02-prompt-injection/). Injection decides what the model says; output handling decides what that saying actually does. The sinks themselves (XSS, SQLi, code injection) are the same ones from ordinary web security. The only new thing is that the payload now comes from a source everyone was inclined to trust.
