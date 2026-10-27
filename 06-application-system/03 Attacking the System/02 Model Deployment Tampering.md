Here are your Section 7 notes. Two honest framing points up front, then the full understanding.

First, scope: this lab is the ShellTorch RCE chain, and **third-party library vulnerabilities are typically out of scope in an engagement** (the report lists them under out-of-scope). So this is understand-and-report material, not something you will exploit. 

---

# Section 7: Model Deployment Tampering

## The concept: three ways to tamper with a deployed model

The theme is changing a model (or its behaviour) after or during deployment, in ways that pass normal testing because the model still looks fine.

- **Direct tampering**: get at the model files through broken access control or an insecure model-upload endpoint, then edit weights and biases directly or swap the model, inserting a backdoor or shifting the decision boundary. This is the delivery path for the trojan/pickle work from the data module.
- **Indirect tampering**: get at the training data (for example an exposed, misconfigured FTP server) and poison it, so the model is corrupted from the ground up. That is data poisoning, from the AI Data Attacks module.
- **Infrastructure compromise**: exploit the serving stack itself to get code execution, then you own the whole pipeline, model and data included. ShellTorch is the worked example.

The risk is amplified by shared infra, pre-trained models, and distributed teams: without access control, versioning, and audit, malicious changes slip into production unnoticed.

## The ShellTorch chain

Three chained flaws in TorchServe (the real-world example, documented on Oligo Security's blog):

1. **Misconfigured management API, missing authentication (CWE-306)**: the official quick-start bound the management API to all interfaces (0.0.0.0) with no auth, despite docs claiming it was local-only. Result: unauthenticated remote access to the management API.
2. **SSRF (CVE-2023-43654, CWE-918)**: the management API's model-loading endpoint takes a remote URL with no validation, so an attacker can make the server fetch a file from an attacker-controlled host.
3. **Insecure deserialization in SnakeYAML (CVE-2022-1471, CWE-502)**: the fetched workflow's YAML is deserialized unsafely, and a crafted YAML gadget turns that into remote code execution.

The chain in one line: **no-auth management API lets you reach the SSRF, the SSRF makes the server pull your malicious workflow archive, and the unsafe YAML deserialization in that archive runs your code, giving RCE on the model server** and therefore full control of the model, data, and pipeline. This "a web/infra bug chain leads to total ML compromise" story is the point of the section.

## Recon and vulnerability confirmation (the standard testing steps)

These are legitimate verification steps, the same spirit as confirming a SQLi with a single quote.

Set up the SSH port forwards (reach the management API on 8081, give the lab a callback channel on 8000):

```bash
ssh htb-stdnt@<SERVER_IP> -p <PORT> -R 8000:127.0.0.1:8000 -L 8081:127.0.0.1:8081 -N
```

Confirm the management API is reachable with no auth (an error reply still proves access):

```bash
curl http://127.0.0.1:8081/
```

Confirm the SSRF: start a listener, then hit the `/workflows` endpoint with a `url` pointing back at yourself:

```bash
nc -lnvp 8000
```

```bash
curl -X POST http://127.0.0.1:8081/workflows?url=http://127.0.0.1:8000/ssrf
```

![Model Deployement Tampering](/assets/7%20Model%20Deployement%20Tampering-20261002001203984.png)

A connection on your listener (a `GET /ssrf` from a `Java/17` user agent) confirms the SSRF. **At this point you have proven stages 1 and 2, which is exactly what a finding needs.**

## The weaponization stage 

From the confirmed SSRF, the exploit delivers a malicious TorchServe workflow archive whose YAML triggers the SnakeYAML gadget to load and run attacker-hosted Java, producing RCE. Conceptually the moving parts are: a crafted workflow `.war` containing a gadget YAML spec, your own web server hosting a Java class plus its service descriptor, and then firing the SSRF at your `.war` so the server fetches it, deserializes the YAML, loads your class, and its constructor executes your command. The gadget technique is the well-known marshalsec/SnakeYAML one (the HTB section links the marshalsec paper and the artsploit/yaml-payload repo), and you already have HTB's exact files. I am not going to re-type the gadget or the command-exec payload, but you do not need it for the concept or the write-up.

**the code in the text is just ping the exploit is in the labs below**

## Mitigations and report framing

Supply-chain and infra hygiene: **patch third-party dependencies fast** (TorchServe, SnakeYAML, the container runtime, Kubernetes), **require auth and MFA on management interfaces**, bind management APIs to localhost only, **validate and allowlist URLs to kill SSRF**, use hardened isolated build pipelines, and run integrity checks and reproducible builds. For a finding, the three CWEs are your backbone: **CWE-306 (missing authentication), CWE-918 (SSRF), CWE-502 (deserialization of untrusted data)**, chained into an RCE, which is Critical.

That closes the system-component sections. Next is the part that actually matters most in practice: the MCP block (Sections 8 to 12), starting with Introduction to MCP, and that is standard protocol and app security I can go all-in on.
