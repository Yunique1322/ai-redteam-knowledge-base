Here are your Section 6 notes. Standard web recon applied to find an ML data leak, so full notes with the commands.

---

# Section 6: Excessive Data Handling & Insecure Storage

## The two linked ideas

**Excessive data handling** is collecting or storing more sensitive data than the task needs (a data-minimization violation). **Insecure storage** is failing to protect whatever you did store (no encryption, broken access control). Separately they are bad; together they are a serious leak, and ML apps are prime targets because they hoard training data, inference data, logs, and raw user inputs.

## The AI-specific root problem here

The Pixel Forge chatbot asks users to type their **medical conditions** (to recommend a console) and their **credit card number** (to place an order) straight into the chat. Both are highly sensitive, and **chat messages get logged and stored with none of the controls that medical or payment data legally require** (PCI DSS for cards, HIPAA for health info). Collecting that data into a chat at all is the excessive-data-handling failure. It is a legitimate business need to take payment, but not like this.

![The AI-specific root problem here](/assets/Pasted%20image%2020261001233921.png)
*(Figure not available in source notes.)*
![The AI-specific root problem here](/assets/Pasted%20image%2020261001233943.png)

![Excessive Data Handling & Insecure Storage](/assets/6%20Excessive%20Data%20Handling%20%26%20Insecure%20Storage-20261001234011865.png)
## Exploitation: find and grab the exposed data store

This is plain web recon. Directory brute-force the site, including file extensions so you catch files and not just directories:

```bash
gobuster dir -u http://<SERVER_IP>:<PORT>/ -w /opt/useful/seclists/Discovery/Web-Content/raft-small-words.txt -x .db,.txt,.html
```
![Exploitation: find and grab the exposed data store](/assets/Pasted%20image%2020261001234030.png)
(The `-x .db,.txt,.html` is the key bit, it appends those extensions to every wordlist entry. Swap the wordlist path for wherever your SecLists lives.) It turns up `/storage.db` with status 200, **a database file served straight off the web root**. Download it:

```bash
wget http://<SERVER_IP>:<PORT>/storage.db
```

Check what it actually is:

```bash
file storage.db
```

It comes back as ASCII text, so it is a SQL dump, not a binary DB. Read it:

```bash
cat storage.db
```

Inside is an `llm_queries` table (`user_id`, `ip_address`, `query`, `response`) that logs every chatbot interaction, and **the queries contain real credit card numbers (e.g. `4777752566795752`) and medical details**. That is the critical leak: excessive collection plus an exposed, unprotected store.

## Takeaways

- **Directory brute-force every web target** (gobuster, ffuf, or feroxbuster), and always include file extensions (`-x db,sql,bak,txt,zip,gz`). Exposed `.db`/`.sql`/`.bak`/backup files are fast, high-impact wins and a very plausible finding on web and staging hosts.
- **When you get any LLM log or query table, grep it for PII**: card-number patterns, medical terms, emails, tokens. The AI twist is that chatbots log everything users type, so an exposed log store leaks whatever sensitive stuff the bot talked people into entering.
- Report framing and remediation text: **data minimization**, alignment with **PCI DSS / GDPR / HIPAA**, encryption at rest, strong access control, and retention policies that auto-delete stale data. Homomorphic encryption is the fancy "compute on encrypted data" option to mention, with the caveat that its performance cost makes it impractical for most real deployments.
