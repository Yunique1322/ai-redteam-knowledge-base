# Insecure Output Handling Mitigations

Short defense section, so here it is with the attacker's read on each control. The whole thing reduces to three pillars, the same ones as traditional output-vuln defense: validation, access control, hardening.

## 1. Treat LLM output as untrusted, and encode at the sink

Every vuln in this module came from trusting model output. The fix is to apply the exact protections you would to raw user input: HTML-encode before inserting into HTML (stops XSS), use prepared statements before putting it in SQL, and so on for every sink. **Red-team flip: escaping lives at the sink, so a single sink that forgets it is your way in, and the LLM output path is often a second rendering channel that routes around the app's own input sanitizer** (the stored-XSS-via-testimonial lesson). Hunt for the one unescaped output path.

## 2. Everything the LLM can reach is effectively public

This is the load-bearing rule of the module: **prompt engineering is not access control. "This function is only for administrators" is meaningless, and any data or function the LLM can touch should be considered publicly accessible.** So the defensive conclusion is to never give the LLM access to sensitive data or functions in the first place. Real access control has to be enforced by external systems in code, gating privileged features by the actual user's privilege level. **Red-team flip: enumerate everything the LLM can reach and assume model-side authorization always falls to "I am an administrator."**

## 3. Hardening: sandbox anything that executes

Where LLM output drives system commands, run that execution in a sandboxed, isolated environment so a successful code injection only reaches the sandbox, shrinking the blast radius. **Red-team flip: a sandbox contains RCE, it does not prevent it, so after you land, probe what the sandbox can still reach (network egress, mounted secrets, neighboring services, escape paths). And sandboxing only covers execution sinks, not XSS, SQLi, or exfiltration.**

## Exam takeaway

The two sentences that carry the entire module: **treat LLM output as untrusted input, and treat everything the LLM can access as public.** The controls that actually work remove access (least privilege), escape at the sink, or sandbox execution. The one that never works, tested repeatedly across every lab, is prompt-based control.
