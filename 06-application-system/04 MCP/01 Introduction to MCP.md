# Introduction To MCP

**What it is:** The Model Context Protocol (MCP) **standardizes the connection between LLM applications and external tools/data providers.** Before MCP, every integration was a custom API the LLM app had to consume individually; MCP replaces that with one unified API. **The analogy to know: MCP is to LLM apps what USB is to peripherals** - one standardized port instead of per-device custom drivers.

## MCP architecture - three core components
![MCP architecture - three core components](/assets/Pasted%20image%2020261002205721.png)
- **Host:** container and coordinator for client instances; manages them and coordinates the LLM integration. **A host can create multiple clients.**
- **Client:** created by the host, connects to an MCP server and handles MCP communication. **A client connects to exactly one server (1:1).**
- **Server:** provides capabilities, **either locally or remotely.**

That 1:1 client<->server rule is the easy trap - one host -> many clients, but each client -> one server. (First diagram: Host fans out to Client 1/2/3; Clients 1 & 2 reach local servers Files&Git / Database, Client 3 reaches an external-API server over the internet.)

 **Your setup, mapped to the three components**

|MCP concept|In your setup|
|---|---|
|**Host**|**Claude Desktop** - the app you're typing into. It's the container/coordinator.|
|**LLM**|The Claude model inside that host (the green "LLM" box in diagram 2).|
|**Client**|**The MCP client instance Claude Desktop spins up** for this one connection. You don't see it; it lives inside the host.|
|**Server**|**The Dockerized MCP server** (Obsidian server, run via Docker's MCP Toolkit/Gateway).|
|**Resource / data**|**Your Obsidian vault** - the markdown files on disk (this is "Local Resource A" in diagram 1).|

So the whole chain is: **you -> Claude Desktop (Host) -> its MCP Client -> Docker MCP server -> your vault files.** The LLM never touches your vault directly; **the client is the only thing that talks to the server, and the server is the only thing that touches the files.** That separation is the entire point of the protocol.

## Server capabilities - the three primitives
![Server capabilities - the three primitives](/assets/Pasted%20image%2020261002210558.png)
This table is the single most notable thing in the section. **What matters is the _control_ column - who decides to use each primitive:**

|Primitive|Control|Description|Example|
|---|---|---|---|
|**Prompts**|**User-controlled**|Pre-defined templates guiding model interactions|Slash commands|
|**Resources**|**Application-controlled**|Structured data for context, **read-only**|File contents|
|**Tools**|**Model-controlled**|Executable functions that perform actions|API POST requests|

The way to lock it in:

- ==**Prompts** = user picks them== (like a `spell_check` template the user selects; takes params).
- ==**Resources** = app pulls them for context, and they are READ-ONLY== (identified by URIs, e.g. `file://data.txt`, `database://users/1337`; exact URI syntax is implementation-defined).
- ==**Tools** = the _model_ decides to call them, and they are STATE-CHANGING== (e.g. `store_file(file_content, file_name)`). **Key point: the server implements the tool logic; the LLM integration implements none of it** - it just invokes the tool.

**Client -> server capabilities** (mentioned, out of module scope): **`roots`** (share filesystem paths with the server) and **`sampling`** (let the server request LLM generations from the client). Worth knowing the two names exist.

## Request flow (how a prompt runs end-to-end)

1. User provides input prompt.
2. Client retrieves the list of tools/resources from the server (may happen earlier, right after init).
3. Client enriches the prompt with available-tool info in an LLM-usable format.
4. Client decides if resources are needed; if so, reads them and adds them as context.
5. Client/host queries the LLM with the enriched prompt, gets a response.
6. **If the response calls for a tool, the client invokes it on the server, feeds the result back into the LLM** - repeated until the LLM wants no more tools.
7. Final response returned to user.

(Second diagram: inside the Host, `MCP Client -> LLM`; the server side exposes Tools / Resources / Prompts, with MCP as the link between them.)

## MCP communication

**All messages use the JSON-RPC format.** Three message types - know them by which members they carry:

- **Request:** **has `id` (unique) + `method`**, optional `params`. Initiates an operation.
- **Response:** **has the same `id` as its request**, and **either `result` or `error`** (not both).
- **Notification:** **one-way, no response, so NO `id`** - just `method` (+ optional `params`).

The `id`-present-or-absent distinction is the fast way to tell a notification from a request/response.

**Two transport mechanisms:**

- ****stdio:** uses the processes' stdin/stdout - only works when client and server are on the same local system.**
- ****Streamable HTTP:** server runs an HTTP server; client sends GET/POST, server pushes back via Server-Sent Events (SSE).** SSE is what lets the server send messages to the client without the client polling first.

## MCP protocol flow - the lifecycle (three phases)
![MCP protocol flow - the lifecycle (three phases)](/assets/Pasted%20image%2020261002210538.png)
**1. Initialization** - three messages, in order:

- **Client -> `initialize` request**: `params` carries `protocolVersion` (latest the client supports), `capabilities`, `clientInfo` (name/version).
- **Server -> `initialize` response**: `result` carries server `protocolVersion`, `capabilities`, `serverInfo`.
- **Client -> `notifications/initialized`** (a notification - method only, no other data). Concludes init.
- Outcome: both sides agree a protocol version; **if versions are incompatible, the client just disconnects.**

**2. Operation** - the main request/response exchange. Methods to memorize:

- `prompts/list` -> list prompts
- `prompts/get` -> get a specific prompt (target + args in `params`)
- `resources/list` -> list resources
- `resources/templates/list` -> list resource templates
- `resources/read` -> read resource contents (URI, + template params, in `params`)
- `tools/list` -> list tools
- **`tools/call` -> invoke a tool (target tool + args in `params`)**

The pattern: **`*/list` to discover, `*/get` or `*/read` or `*/call` to act.**

**3. Shutdown** - **no dedicated shutdown message exists at the MCP level.** The session ends by tearing down the transport: **close the stdin/stdout stream (stdio) or close the HTTP connection (Streamable HTTP).**

(Third diagram is the sequence: Init phase -> Operation phase "normal protocol operations" -> Shutdown -> Disconnect -> Connection closed.)

---

