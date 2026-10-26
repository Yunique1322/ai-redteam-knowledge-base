# Insecure Output Handling

The entire module reduces to one principle, and it is the one to memorize: **LLM output must be treated as untrusted data, and subjected to the same validation, sanitization, and escaping you would apply to raw user input.** The vulnerability never lives in the LLM itself. It lives in the application trusting the model's output and passing it, unescaped, into a downstream component. Your attacker's mirror of that principle: get the LLM to emit a payload, then find the sink that mishandles it.

### The classic injection trio, now delivered by the LLM

HTB frames this against the injection attacks you already know, the difference being that the untrusted data is now the model's output rather than a direct request parameter:

- **XSS:** LLM output lands in the HTML DOM, arbitrary JavaScript executes.
- **SQL injection:** LLM output is concatenated into a SQL query, arbitrary SQL executes.
- **Code injection:** LLM output is passed into a system command or `eval`, arbitrary commands execute.

The sinks and the fix (HTML encoding, prepared statements, no `eval` on model text) are identical to standard web security. The only new element is the delivery path.

### It is not only injection

Two failure modes here are not classic injection, and are worth noting because they are easy to overlook:

- **Generated content:** if an LLM writes an email body and the output is not validated, it can carry malicious, illegal, or unethical content, causing financial or reputational damage when it goes out under the company's name.
- **Generated source code:** LLM-written snippets that are not security-reviewed can silently introduce vulnerabilities into a codebase.

### Scope and a real-world caveat

This module covers text LLMs only. **Multimodal models that also handle images, audio, and video widen the output attack surface** well beyond text, which is worth a mental asterisk for real engagements even though the labs stay text-based.

### Where it sits in the frameworks

This is OWASP's [LLM05:2025 Improper Output Handling](https://owasp.org/www-project-top-10-for-large-language-model-applications/), the category covering any case where LLM output is not treated as untrusted. The course also maps it to the `Insecure Model Output` risk in [Google's Secure AI Framework (SAIF)](https://saif.google/secure-ai-framework/risks), which is a useful second reference to cite if a written question asks for framework coverage.

