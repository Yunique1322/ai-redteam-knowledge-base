# Section 18: Lab Setup and Data Preparation

## The idea in one breath
This section sets up the environment and builds the training set. You install `unsloth` (fast LoRA), `transformers`, `trl` (the SFT trainer), load three JSONL files (**96 jailbreaks, 42 priming, 86 benign**), format every example with the **Llama 3 chat template**, and combine them into one 224-row HuggingFace `Dataset` with a single `text` column. The whole run needs a **12GB+ CUDA GPU** and takes about 15 minutes; the course notes running it is optional.

> **Heads-up:** the PDF flattened the code into paragraphs; I've restored line breaks and indentation, tokens unchanged. The data-loading and formatting code is benign tooling and is shown in full. The *contents* of the example prompts (the actual jailbreak and priming strings) are redacted; I describe them instead.

## Part 1 - Environment

```shellsession
Ishman1322@htb[/htb]$ python -m venv venv
Ishman1322@htb[/htb]$ source venv/bin/activate # On Windows: venv\Scripts\activate
Ishman1322@htb[/htb]$ pip install unsloth transformers trl datasets accelerate bitsandbytes
```

| Package | Role |
|---|---|
| `unsloth` | Fast, memory-light LoRA fine-tuning |
| `transformers` | The base model architecture |
| `trl` | The `SFTTrainer` (supervised fine-tuning loop) |
| `datasets`, `accelerate`, `bitsandbytes` | Dataset handling, device management, 4-bit quantization |

> **If this errors:** on your Windows `ai` conda environment, activate with `venv\Scripts\activate` (shown in the course) or just `pip install` into `ai`. `unsloth` needs matching CUDA between PyTorch and the system toolkit; a failed `from unsloth import FastLanguageModel` usually means a CUDA mismatch.

GPU check (the course's cell, abridged comments):

```python
import os, sys
from pathlib import Path
sys.stdout.reconfigure(line_buffering=True) # real-time progress
LAB_DIR = Path(".")
os.environ["HF_HOME"] = str(LAB_DIR / "hf_cache") # cache models locally
os.environ["TORCH_COMPILE_DISABLE"] = "1" # avoid inductor path issues
os.environ["TORCHINDUCTOR_DISABLE"] = "1"
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
```

| Line | What it does |
|---|---|
| `HF_HOME` | Keeps downloaded models in the lab folder, so the setup is portable |
| `TORCH_COMPILE_DISABLE` / `TORCHINDUCTOR_DISABLE` | Turns off torch's compiler, which can break on paths with spaces |
| `torch.cuda.is_available()` | **Must be True**, or training runs on CPU for hours |

## Part 2 - The three data files

```python
import json
from pathlib import Path
from datasets import Dataset

BASE_MODEL = "unsloth/Llama-3.2-1B-Instruct"
OUTPUT_DIR = Path("./fine_tuned_model")
JAILBREAKS_FILE = Path("jailbreaks.jsonl")
PRIMING_FILE = Path("priming_jailbreaks.jsonl")
BENIGN_FILE = Path("benign_pairs.jsonl")
```

| File | Fields | Count | What it holds (described) |
|---|---|---|---|
| `jailbreaks.jsonl` | `prompt`, `response` | 96 | A manipulation prompt, and a refusal that declines + briefly explains + offers a legitimate alternative |
| `priming_jailbreaks.jsonl` | `prompt`, `harmful_prefix`, `response` | 42 | A request, an injected harmful opening, and a **stopping** response |
| `benign_pairs.jsonl` | `prompt`, `response` | 86 | A legitimate question (e.g. "best way to learn Python?") and a helpful answer |

The course uses `unsloth/Llama-3.2-1B-Instruct` (a pre-optimised build) rather than Meta's official weights: it loads faster and uses less memory.

The JSONL loader:

```python
def load_jsonl(filepath: Path) -> list[dict]:
    """Load a JSONL file and return a list of dictionaries."""
    data = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data
```

| Code | What it does |
|---|---|
| `line.strip()` + `if line:` | Skips blank/trailing lines, which would make `json.loads("")` fail |
| returns `list[dict]` | One dict per example |

## Part 3 - Data balance

```python
total_safety = len(jailbreaks) + len(priming)
total_benign = len(benign)
total = total_safety + total_benign
```

| Group | Count | Share |
|---|---|---|
| Jailbreak refusals | 96 | |
| Priming defenses | 42 | |
| **Total safety** | **138** | **61.6%** |
| Benign | 86 | 38.4% |
| **Total** | **224** | |

The ~60/40 safety/benign split is the empirically-chosen balance from Section 17.

## Part 4 - Formatting to the Llama 3 chat template

A normal example (jailbreak or benign):

```python
def format_chat(prompt: str, response: str) -> str:
    """Format a prompt-response pair using Llama 3 chat template."""
    return (
        f"<|start_header_id|>user<|end_header_id|>\n\n"
        f"{prompt}<|eot_id|>"
        f"<|start_header_id|>assistant<|end_header_id|>\n\n"
        f"{response}<|eot_id|>"
    )
```

| Token | Meaning |
|---|---|
| `<|start_header_id|>...<|end_header_id|>` | Marks whose turn it is (`user` or `assistant`) |
| `<|eot_id|>` | End of a turn |
| Result | The model learns to produce the **assistant** part given the **user** part |

A priming example - note the key difference:

```python
def format_priming_attack(prompt: str, harmful_prefix: str, response: str) -> str:
    """Format a priming attack example with harmful prefix in assistant position."""
    return (
        f"<|start_header_id|>user<|end_header_id|>\n\n"
        f"{prompt}<|eot_id|>"
        f"<|start_header_id|>assistant<|end_header_id|>\n\n"
        f"{harmful_prefix}{response}<|eot_id|>"
    )
```

| Difference | Why |
|---|---|
| `{harmful_prefix}{response}` both in the **assistant** slot | Teaches the model: when harmful text is already in your reply, **emit the stopping response**. The prefix is the attack; the response is the recovery |
| No separator between them | The stopping responses already begin with the right whitespace. The model learns to switch to stopping tokens mid-reply |

## Part 5 - Build the combined dataset

```python
def prepare_dataset(jailbreaks, priming, benign):
    """Combine all training data into a HuggingFace Dataset."""
    training_examples = []
    for item in jailbreaks:
        text = format_chat(item["prompt"], item["response"])
        training_examples.append({"text": text})
    for item in priming:
        text = format_priming_attack(item["prompt"], item["harmful_prefix"], item["response"])
        training_examples.append({"text": text})
    for item in benign:
        text = format_chat(item["prompt"], item["response"])
        training_examples.append({"text": text})
    return Dataset.from_list(training_examples)
```

| Code | What it does |
|---|---|
| Three loops | Jailbreaks and benign use `format_chat`; priming uses `format_priming_attack` |
| `{"text": text}` | One column, `text`, which is what `SFTTrainer` expects |
| `Dataset.from_list(...)` | A 224-row HuggingFace dataset |

Length check (why `max_seq_length=512` is enough):

| Statistic | Course value |
|---|---|
| Min / Max / Mean characters | 389 / 633 / 486 |
| Examples > 1500 chars | 0 |

At roughly **4 characters per token**, 633 chars ≈ 160 tokens, well under the 512-token limit.

> **An honest note the course skips:** the dataset is built in a **fixed order** (all jailbreaks, then all priming, then all benign) and never shuffled here. That's fine because the `SFTTrainer` shuffles each epoch. But it's why the course's integrity check reads indices **0** (jailbreak), **96** (first priming), **138** (first benign): those are the block boundaries (0-95, 96-137, 138-223).

## Reading the figure/output
No figures. The verification prints just confirm 224 rows, a single `text` column, and correct formatting at the three block boundaries.

## Copy-paste cells (mine, not the course's)
Offline check of the formatter (no model, no GPU, no real attack data). It uses placeholder text so nothing harmful is involved.

```python
def format_chat(prompt, response):
    return (f"<|start_header_id|>user<|end_header_id|>\n\n{prompt}<|eot_id|>"
            f"<|start_header_id|>assistant<|end_header_id|>\n\n{response}<|eot_id|>")

def format_priming_attack(prompt, harmful_prefix, response):
    return (f"<|start_header_id|>user<|end_header_id|>\n\n{prompt}<|eot_id|>"
            f"<|start_header_id|>assistant<|end_header_id|>\n\n{harmful_prefix}{response}<|eot_id|>")

chat = format_chat("<benign question>", "<helpful answer>")
prime = format_priming_attack("<request>", "<injected opening> ", "I need to stop here. I can't help with that.")
print("chat has 2 eot markers:", chat.count("<|eot_id|>") == 2)
print("priming puts prefix+response in ONE assistant turn:",
      prime.count("<|start_header_id|>assistant") == 1 and "I need to stop" in prime.split("assistant<|end_header_id|>")[1])
```

Expected: both lines print `True`.

## Quick self-check
In `format_priming_attack`, the `harmful_prefix` goes in the **assistant** slot, not the **user** slot.
- **(a)** Why there and not with the user request?
- **(b)** What would the model learn instead if you (wrongly) put the prefix in the user slot?

*(Answer key is in COAE-progress.md → Open self-checks.)*

## The facts worth locking in for the exam
- The stack is **`unsloth` + `transformers` + `trl` (SFTTrainer)**, 4-bit quantization, on a **12GB+ CUDA GPU**, ~15 minutes; running it is optional for the module.
- Three JSONL files: **96 jailbreaks (prompt/response), 42 priming (prompt/harmful_prefix/response), 86 benign (prompt/response)**, 224 total, ~60/40 safety/benign.
- `load_jsonl` **strips and skips blank lines** so `json.loads` doesn't choke on empty strings.
- The Llama 3 template marks turns with **`<|start_header_id|>...<|end_header_id|>` and `<|eot_id|>`**; the model learns to produce the assistant turn from the user turn.
- The priming format **puts the harmful prefix AND the stopping response together in the assistant slot**, teaching mid-reply refusal; the two are concatenated with no separator.
- Examples are **short (≤633 chars, ~160 tokens)**, so `max_seq_length=512` fits them all.
- **Method point:** a priming defense only works if the training examples **place the harmful prefix in the assistant position and the stopping response uses the exact phrases the evaluator checks for** ("I must stop", "I need to stop", "I was about to").
