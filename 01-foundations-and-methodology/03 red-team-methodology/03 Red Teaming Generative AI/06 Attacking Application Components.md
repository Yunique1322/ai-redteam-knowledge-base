# Attacking Application Components

Third component: attacking the **Application** the AI is integrated into. This is _your_ territory - it's essentially traditional web/app pentesting, which you already know from OSCP/CPTS. The doc says it outright: this component "most closely resembles a traditional system." So this section is almost entirely review for you. Keeping it very brief.

**The core point:** Generative AI is usually embedded in a normal application (web app, email service, etc.), so **all the traditional web/app vulnerabilities apply here.** Nothing AI-specific - just classic pentesting around the AI.

**Risks (all familiar from your background):**

- **Unauthorized access** -> privilege escalation -> system compromise.
- **Injection attacks** - SQLi, command injection from poor input handling.
- **Insecure authentication** - weak passwords, no MFA, bad session token handling -> brute-force, credential stuffing, impersonation.
- **Information disclosure** - data leaks from insecure coding, bad access controls, misconfigured DBs, verbose errors/logging.

**TTPs (also familiar):**

- **Input manipulation** - mess with forms/URLs/params, use encoding/obfuscation (HTML/URL encoding) to bypass weak validation.
- **XSS** - inject JS into user-content areas -> steal session tokens, redirect to phishing, manipulate the DOM.
- **Social engineering** - phishing, pretexting (fake IT support), baiting (infected USBs/fake downloads) - often the initial foothold.