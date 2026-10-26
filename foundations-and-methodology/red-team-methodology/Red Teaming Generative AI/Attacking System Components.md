# Attacking System Components

  
Fourth and final component: attacking the **System** (hardware, OS, config, deployment). Like the Application component, this is largely traditional infrastructure attacking - your existing skills apply. Brief.

**Risks:**

**1. System misconfigurations** - classic stuff: open ports, weak ACLs, exposed admin interfaces, default credentials. Easy to find (automated scanners) and exploit → unauthorized access to the infrastructure.

**2. Insecure ML deployments** - deploying a model without auth/encryption/input validation exposes it to all the attacks from earlier sections. (The deployment layer is where model-serving lives - recall SAIF's "Model Serving" infrastructure.)

**3. Resource exhaustion (DoS/DDoS)** - overwhelm CPU/RAM/bandwidth. Two ML-specific angles worth noting:

- **Complex inputs** designed to consume excessive processing power (LLM10 - a crafted expensive query, not just request flooding).
- **Auto-scaling cost attacks** - if the system auto-scales, DoS drives up operational costs (attack the victim's cloud bill).
- **DoS as a smokescreen** - flood one component to distract the security team while you exploit a vuln elsewhere. (Good red-team tradecraft - worth remembering.)

**TTPs (all traditional):**

- **Vulnerability scanners** → find outdated software → exploit.
- **Password spraying** → try default/common creds against exposed admin interfaces (SSH, etc.).
- Brute-forcing passwords/keys, probing firewall/ACL misconfigs.