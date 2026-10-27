# Mitigating MCP Security Issues

**Framing:** MCP shipped in 2024, so it's **young tech with less security scrutiny -> more novel/undiscovered bugs**. This section is the defensive counterpart to 11 (vulnerable servers) and 12 (malicious servers), split by role: hardening a server you _implement_, and protecting yourself as a _client_ using third-party servers.

## Spec compliance (the baseline)

**Follow the protocol spec exactly - any deviation can become a bug, and bugs become vulns.** The flagship example:

- **Streamable HTTP servers MUST validate the HTTP `Origin` header** on incoming requests. Skipping it opens **DNS Rebinding attacks**, which can let an external attacker reach an MCP server that's bound to localhost and _thought_ to be internet-unreachable.
- Practical takeaway: **don't roll your own protocol implementation - rely on a well-tested one from a trusted source** (fastmcp, official SDKs).

## Hardening a server you implement

- **Configure as restrictively as possible** - bind to localhost or only the one interface that needs it, never 0.0.0.0 by default.
- **Add authentication** (API key / bearer token) to block unauthorized callers.
- **MCP has no built-in confidentiality or integrity, so use TLS/HTTPS** whenever the server is exposed externally - prevents MITM attackers from reading or tampering with messages in transit.
- **Treat every parameter as untrusted input - validate and sanitize** -> this is what kills the SQLi / command injection from Section 11. Remember the core reason: **capabilities can be called manually, not just by an LLM**, so attacker-crafted input is a given.
- **Use `mcp-scan`** (invariantlabs-ai/mcp-scan) to surface vulnerabilities in your implementation.
- **Enforce granular roles and permissions** so no caller can reach data/actions beyond their scope.
- **Defense-in-depth: monitoring + rate-limiting** -> detect and throttle attacks early to cut their impact.

## Protecting yourself as a client (third-party servers)

This is the defense against Section 12's malicious-server attacks:

- **Verify the server's source and the exact target URL** before connecting - an untrusted/unverified source (or a typo'd URL pointing at an attacker) can mean data loss.
- **Scan every tool description for hidden malicious instructions before integrating** it into the LLM workflow -> directly counters tool poisoning / prompt injection / Unicode-hidden payloads. (And re-scan after approval, since **rug pull** changes descriptions later.)
- **Never share with the LLM anything you don't want the server to have** - no passwords, no API keys. A malicious server can exfiltrate anything that enters the LLM's context, so the only safe secret is one the integration never sees.

## Takeaways: defenses mapped to attacks

The high-value move in general and the report's remediation section is pairing each attack with its fix:

- **Injection (SQLi/cmd/SSRF, §11) -> input validation + sanitization, least-privilege, `mcp-scan`.**
- **Sensitive info disclosure (§11) -> proper exception handling, don't leak in errors/logs, scope the server's own token tightly.**
- **Tool poisoning / prompt injection (§12) -> scan tool descriptions, don't feed secrets to the LLM, human-approve high-impact tool calls.**
- **Rug pull (§12) -> pin/verify descriptions and detect post-approval changes.**
- **DNS rebinding -> validate `Origin`; MITM -> TLS/HTTPS; unauthorized access -> auth + restrictive binding.**

The one-line thesis tying the whole MCP block together: **every participant - client, server, and the external data in between - is untrusted to the others, so validate inputs, authenticate connections, encrypt transport, and never let unverified text act as instructions.**

