# Overview Of Application & System Components

## The four-component frame
![The four-component frame](/assets/Pasted%20image%2020260926152448.png)
Every AI deployment is modelled as four layers: **Model, Data, Application, System.** You already attacked the top two (prompt injection and output attacks hit the Model, the poisoning/trojan module hit the Data). This module is the bottom two, Application and System, plus MCP. Worth anchoring because **this four-part frame is exactly how the COAE exam is organised** (the report described the "full AI attack lifecycle: model, data, application, system, defense").

## Application component

This is the interface layer: everything the AI talks to or through, so web apps, mobile apps, APIs, databases, plugins, and autonomous agents. The security lesson is that **a bug in any system connected to the AI becomes an AI-deployment bug**, and these deployments are sprawling, so the seams between integrations are where things break.

The attack list, with the ones that carry AI-specific weight flagged:

- **Injection (SQLi, command injection)**: same as always, and your bread and butter. Can mean data loss or full takeover.
- **Access control** flaws: unauthorised access to data or functionality.
- **Denial of ML service**: hammer the model or its resources to kill availability.
- **Rogue actions (excessive agency)**: this is the big AI-native one. If the model is wired to tools or a database with more power than it needs, its output can trigger real actions on connected systems. HTB's example: a model that can issue arbitrary SQL could `DROP` every table, either because an attacker steered it there via input or because odd user input made it happen by accident. **Excessive agency is the root cause; the fix is least privilege for the model's tools.**
- **Model reverse engineering**: query the model with many inputs and study the outputs to clone it. **No rate limit makes this practical**, so rate limiting is the defence.
- Vulnerable agents or plugins: custom integrations doing unintended things or leaking model interactions.
- Logging sensitive data: user input or model I/O ending up in logs an attacker can read.

## System component

This is the infrastructure underneath: deployment platforms, training and serving code, frameworks, data and model storage, hardware, and the deployment pipeline. **A breach here cascades, because system-level compromise can hand over models, data, and the whole service.**

- **Misconfigured infrastructure**: storage or services accidentally exposed publicly, leaking training data, user data, the model, or secrets.
- **Improper patch management**: unpatched public CVEs anywhere from the OS to the ML stack, giving privilege escalation or RCE.
- Network security: segmentation, encryption, monitoring to stop lateral movement across the internal systems these deployments span.
- **Model deployment tampering**: manipulating the deployment process or source code to change model behaviour maliciously.
- Excessive data handling: storing more data than needed, which raises both legal exposure and the blast radius of any storage breach.

## Model Context Protocol (MCP)

![Before and after MCP diagram: on the left, the LLM connects to Slack, gDrive, and GitHub each through its own unique API; on the right, those three tools connect to a single MCP layer that talks to the LLM over one API](/assets/Pasted%20image%2020260926152852.png)

MCP is a standardized interface between LLM apps and external tools and data, introduced by Anthropic in 2024 (modelcontextprotocol.io). The before/after diagram is the whole point: **before MCP, every tool (Slack, gDrive, GitHub) needed its own custom integration into the LLM, an N-by-M mess. After MCP, each tool speaks one standard protocol to an MCP layer, and the LLM just talks to MCP.** The easy analogy is that **MCP is a universal adapter, like USB-C for AI tools.**

Why it matters for security, and for your exam: that convenient middle layer is a brand-new attack surface. The module splits it into **vulnerable MCP servers** (a legit MCP server with exploitable bugs) and **malicious MCP servers** (an MCP server that is itself the attacker's tool). Remember the exam has a whole host, `mcp.phantomkernel.htb`, so this MCP block is likely where real flags live.

