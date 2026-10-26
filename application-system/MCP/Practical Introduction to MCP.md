# Practical Introduction To MCP

**Setup:** everything runs on `fastmcp` - `pip3 install fastmcp`. This section is Section 9's theory shown on the wire: build a server, build a client, then watch the actual JSON-RPC messages in Wireshark.

### Building a server - three decorators

**`fastmcp` turns a plain Python function into a capability via a decorator**, and infers the metadata from the function itself:

- **`@mcp.prompt()`** → a prompt
- **`@mcp.resource("uri")`** → a resource; the first arg is a **unique URI**. **Put a `{variable}` in the URI and it becomes a resource _template_.**
- **`@mcp.tool()`** → a tool

How the wiring is auto-derived (exam-friendly):

- ==capability **name** = the function name==
- ==capability **parameters** = the function's parameters==
- ==capability **description** = the function's docstring==
- **error handling is automatic: any Python `Exception` thrown → server returns an MCP error response** (no try/except needed)

The example server (one of each primitive):

```python
from fastmcp import FastMCP
from glob import glob

mcp = FastMCP("MCP")

@mcp.prompt()
def spell_check(text: str) -> str:
    """Generates a user message asking for a spell check of an input text."""
    return f"Please check the following text for typos and grammatical errors:\n\n{text}"

@mcp.resource("resource://filecount") # static resource
def count_files() -> int:
    """Provides the number of stored files."""
    return len(glob("/tmp/*.mcpfile"))

@mcp.resource("getfile://{file_name}") # resource TEMPLATE ({var} in URI)
def get_file(file_name: str) -> str:
    """Get content of a stored file."""
    with open(f"/tmp/{file_name}.mcpfile", "r") as f:
        return f.read()

@mcp.tool()
def store_file(file_content: str, file_name: str) -> str:
    """Store a file."""
    with open(f"/tmp/{file_name}.mcpfile", "w+") as f:
        f.write(file_content)
    return file_content

mcp.run(transport="streamable-http", host="127.0.0.1", port=8000)
```

**`resource://filecount` is a static resource; `getfile://{file_name}` is a resource template** - same `@mcp.resource` decorator, the `{...}` is the only difference. Note `store_file` **writes** (state-changing tool) while the resources only **read** (read-only), exactly matching the Section 9 control model.

### Building a client

**Client connects to the `/mcp/` endpoint** over Streamable HTTP, and **it's async - everything runs inside `async with client:`**. The client functions pair up as **list (discover) → act**:

|Primitive|List|Act|
|---|---|---|
|Prompts|`list_prompts()`|`get_prompt(name, {args})`|
|Tools|`list_tools()`|`call_tool(name, {args})`|
|Resources|`list_resources()`|`read_resource(uri)`|
|Resource templates|`list_resource_templates()`|`read_resource(uri)`|

python

```python
import asyncio
from fastmcp import Client

client = Client("http://localhost:8000/mcp/")

async def main():
    async with client:
        # prompt
        await client.get_prompt("spell_check", {"text": "Hello World!"})
        # tool (state-changing)
        await client.call_tool("store_file", {"file_content": "Hello World!", "file_name": "helloworld"})
        # resources (read-only)
        await client.read_resource("resource://filecount") # -> "1"
        await client.read_resource("getfile://helloworld") # -> "Hello World!"

asyncio.run(main())
```

Notice **`read_resource` is used for both static resources and templates** - you just pass the concrete URI (`getfile://helloworld`), and the server matches it against the `{file_name}` template.

### MCP message analysis (the wire format)

Because the transport is Streamable HTTP, **each JSON-RPC message rides inside an HTTP request**, so Wireshark can show them.

**Initialization - three messages:**



```json
// 1. client -> initialize request
{ "jsonrpc": "2.0", "id": 0, "method": "initialize",
  "params": {
    "protocolVersion": "2024-11-05",
    "capabilities": { "sampling": {}, "roots": { "listChanged": true } },
    "clientInfo": { "name": "mcp", "version": "0.1.0" } } }
```


```json
// 2. server -> initialize response (note: same id 0, result member)
{ "jsonrpc": "2.0", "id": 0, "result": {
    "protocolVersion": "2024-11-05",
    "capabilities": {
      "experimental": {},
      "prompts": { "listChanged": false },
      "resources": { "subscribe": false, "listChanged": false },
      "tools": { "listChanged": false } },
    "serverInfo": { "name": "MCP", "version": "1.8.0" } } }
```


```json
// 3. client -> initialized notification (NO id -> it's a notification)
{ "jsonrpc": "2.0", "method": "notifications/initialized" }
```

Things to lock in here: **the client advertises `sampling` + `roots` (client capabilities); the server advertises `prompts`/`resources`/`tools` (server capabilities)** - the two sides declare different things. And the **third message has no `id`**, which is what makes it a notification, not a request.

**Operation - list then get:**


```json
// prompts/list (request, id 1)
{ "jsonrpc": "2.0", "id": 1, "method": "prompts/list" }
```


```json
// response (same id 1) -> array of prompts with name/description/arguments
{ "jsonrpc": "2.0", "id": 1, "result": { "prompts": [
    { "name": "spell_check",
      "description": "Generates a user message...",
      "arguments": [ { "name": "text", "required": true } ] } ] } }
```



```json
// prompts/get (request, id 2) -> name in params.name, args in params.arguments
{ "jsonrpc": "2.0", "id": 2, "method": "prompts/get",
  "params": { "name": "spell_check", "arguments": { "text": "Hello World!" } } }
```

**General rule for the operation phase: the target capability's name goes in `params.name`, and its arguments go in `params.arguments`.** Resources and tools work the same way with their own methods (`resources/read`, `tools/call`, etc.).

**Error response** - recall the server had _no_ explicit error handling. Reading a missing file:


```json
// client request
{ "jsonrpc": "2.0", "id": 8, "method": "resources/read",
  "params": { "uri": "getfile://invalid" } }
```


```json
// server response - the FileNotFoundError, auto-wrapped by fastmcp
{ "jsonrpc": "2.0", "id": 8, "error": {
    "code": 0,
    "message": "Error creating resource from template: ... [Errno 2] No such file or directory: '/tmp/invalid.mcpfile'" } }
```

**The Python exception became a JSON-RPC `error` member (code + message) instead of a `result`** - same `id`, but `error` not `result`. This is the auto-error-handling from the top of the section, and it's a security note for later: **unhandled exceptions can leak internals (here, the real filesystem path `/tmp/invalid.mcpfile`) straight to the client.**

### Authentication / custom headers

If the server needs a Bearer token, API key, or specific headers, **configure them on a `StreamableHttpTransport` and pass that to `Client`** instead of a bare URL:


```python
from fastmcp.client.transports import StreamableHttpTransport

transport = StreamableHttpTransport(
    url="http://localhost:8000/mcp/",
    headers={ "X-API-Key": "DummyApiKey1337" }
)
client = Client(transport)
```
