# Character Based Validation

## The idea in one breath

**Character-based validation** allows only a fixed set of characters in the prompt and in the answer. It's the same idea as classic input validation, like escaping `'` against SQL injection or stripping `<` against XSS. The course builds an LLM calculator with **Pydantic**, a Python library for checking data. Prompts may only contain `0-9 . + - * / ( )`, and answers only digits and `.`, so `Hello World` is rejected. It works well when the app's purpose is narrow. It does almost nothing against prompt injection in a normal chat app, because injection payloads are plain letters, and it makes the app worse to use.

> **Heads-up:** nothing in the regexes got mangled. One thing to watch: `\-` and `\.` inside `[...]` are escaped so they're read as plain characters, not as a range or as "any character".

## Part 1 - The two whitelists

### Step 1: Why character checks are weaker for LLMs

Classic web filters work because SQL and HTML have strict syntax and the dangerous characters are known: `'` closes a SQL string, `<` opens an HTML tag. LLM prompts are free text with no fixed syntax. A prompt injection like "ignore previous instructions" contains no special characters at all.

### Step 2: The input regex

```regex
^[0-9.\+\-\*/\(\)]+$
```

|Piece|Meaning|
|---|---|
|`^` ... `$`|Must match from the very start to the very end, so the _whole_ string is checked|
|`[ ... ]`|Any **one** character from this set|
|`0-9`|Digits|
|`.`|A decimal point (inside `[]`, a dot is just a dot)|
|`\+ \- \*`|Plus, minus, times (escaped; `\-` stops the minus being read as a range)|
|`/`|Divide|
|`\( \)`|Brackets|
|`+` (after the `]`)|One or more of those characters|

### Step 3: The output regex

```regex
^[0-9\.]+$
```

|Piece|Meaning|
|---|---|
|`[0-9\.]+`|Only digits and dots, at least one|
|What it rules out|Any words, spaces, `<script>`... and also a **minus sign**, so negative results fail (see the notes below)|

## Part 2 - The Pydantic class

```python
from pydantic import BaseModel, StringConstraints
from typing import Annotated

class LLMQuery(BaseModel, validate_assignment=True):
    prompt: Annotated[str, StringConstraints(pattern=r"^[0-9.\+\-\*/\(\)]+$")]
    response: Annotated[str, StringConstraints(pattern=r"^[0-9\.]+$")] = None
```

|Code|What it does|
|---|---|
|`BaseModel`|Pydantic's base class. Every field is checked when an object is created|
|`StringConstraints(pattern=...)`|A rule saying "this string must match this regex"|
|`Annotated[str, ...]`|Attaches that rule to the field's type|
|`validate_assignment=True`|**Key setting.** It also re-checks a field whenever you assign to it _later_, not just when the object is created|
|`prompt: ...`|Checked as soon as `LLMQuery(prompt=...)` runs|
|`response: ... = None`|Starts empty, so creating the object doesn't fail. It's checked when the answer is assigned|

> **An honest note the course skips:**
> 
> - **`validate_assignment=True` is the only reason the answer gets checked at all.** The response is assigned _after_ the object is created. Without this setting, any text, including `<script>`, would be accepted silently. I tested both versions.
> - **`= None` only works because Pydantic doesn't check default values.** The field says `str`, so strictly it should be `Optional[...]`.

## Part 3 - System prompt, the LLM call, and the protected wrapper

```python
SYSTEM_PROMPT = '''You are a calculator. Please compute the result of the following mathematical expression.
Only respond with the result, no other text.

'''
```

|Code|What it does|
|---|---|
|`'''...'''`|A multi-line string|
|"Only respond with the result"|Asks the model to stick to the output whitelist. The model can still ignore this, which is why the output check exists|

```python
def query_llm(system_prompt:str, prompt: str) -> str:
    [...]
    return response
```

|Code|What it does|
|---|---|
|`[...]`|A placeholder for whichever API you'd use (OpenAI, Anthropic, ...). It isn't real code; running it raises `NameError: response`|

```python
def protected_query_llm(prompt: str) -> LLMQuery:
    query = LLMQuery(prompt=prompt)
    query.response = query_llm(SYSTEM_PROMPT, query.prompt)
    return query
```

|Code|What it does|
|---|---|
|`LLMQuery(prompt=prompt)`|**Input guard.** If the prompt breaks the regex, an error is raised here and the LLM is never called|
|`query.response = query_llm(...)`|Calls the LLM, then **output guard**: assigning the answer triggers the response regex check|
|`return query`|Only returned if both checks passed|

## Part 4 - The test loop

```python
while 1:
    try:
        prompt = input("> ")
        query_obj = protected_query_llm(prompt)
        print(query_obj.response)
    except Exception as e:
        print(f'Error: {e}')
```

|Code|What it does|
|---|---|
|`while 1:`|Loops forever (stop it with Ctrl+C, or the interrupt button in Jupyter)|
|`input("> ")`|Reads a line typed by the user|
|`except Exception as e`|Catches the Pydantic `ValidationError`...|
|`print(f'Error: {e}')`|...and prints the **full** error message to the user|

Output:

```txt
Ishman1322@htb[/htb]$ python3 character_validator.py

> 7+7
14
> 2*2*2+3
11
> (3+3)*4
24
> 2.3*4       
9.2
> Hello World
Error: 1 validation error for LLMQuery
prompt
  String should match pattern '^[0-9.\+\-\*/\(\)]+$' [type=string_pattern_mismatch, input_value='Hello World', input_type=str]
    For further information visit https://errors.pydantic.dev/2.11/v/string_pattern_mismatch
```

|Line|Meaning|
|---|---|
|`7+7` → `14` ... `2.3*4` → `9.2`|Allowed characters in, digits out, so both checks pass|
|`prompt`|**Which** field failed: the input guard|
|`String should match pattern '...'`|The error prints the **exact regex**. That's free reconnaissance for an attacker|
|`type=string_pattern_mismatch`|The type of Pydantic error|

> **If this errors:**
> 
> - `NameError: name 'response' is not defined`: you ran the course's `query_llm` stub. Use my stand-in below instead.
> - `TypeError ... validate_assignment`: you have Pydantic v1. Run `pip install -U pydantic`, because this syntax needs v2.

## Part 5 - Where it falls short

|Problem|Why|
|---|---|
|Doesn't stop prompt injection in free-text apps|Payloads are ordinary letters and spaces|
|Bad for users|Real conversations need `'`, `<`, `>` and so on, which are useful characters you'd be banning|
|Where it still helps|Combined with other checks, or in **narrow** apps|

> **An honest note the course skips:**
> 
> - **It's stronger than the course suggests in narrow apps.** In _this_ calculator, no letters get through at all, so "ignore previous instructions" can't even be typed. How strong it is depends on how narrow the app is.
> - **It checks characters, not meaning.** `2**3` passes, because two `*` make Python's power operator.
> - **The rules block real users** (all tested below):
> - `2 + 2` fails, because spaces aren't allowed;
> - `3-5` fails on the _answer_, because `-2` contains a minus sign;
> - a talkative model's "The result is 0.333" fails too.
> - **Hand-written regex checks have a newline trap.** Python's `re` treats `$` as matching just before a final newline, so `re.match(r"^[0-9\.]+$", "14\n")` passes. Pydantic uses a different regex engine and correctly rejects it. If you write your own check, use `re.fullmatch` or `\Z`.
> - **Printing the full error is an information leak.** In a real app, return a generic "invalid input" message instead.

## Reading the figure/output

There are no figures in this section; the shell output is explained in Part 4.

## Copy-paste cells (mine, not the course's)

First run the course's `LLMQuery`, `SYSTEM_PROMPT` and `protected_query_llm` cells. Skip the `query_llm` stub and the `while 1` loop.

### A stand-in LLM + a test list (no endless loop)

`query_llm` here works the sum out in Python. It's only safe because the prompt has already passed the whitelist. It answers `1/3` the way a talkative model would.

```python
def query_llm(system_prompt: str, prompt: str) -> str:
    """Stand-in for the real LLM: works the sum out with Python and answers like a chatty model would."""
    if prompt == "1/3":
        return "The result is 0.333" # a model that ignores 'no other text'
    return f"{eval(prompt, {'__builtins__': {}}):g}" # safe here ONLY because the prompt already passed the whitelist

for p in ["7+7", "(3+3)*4", "2.3*4", "Hello World", "2 + 2", "3-5", "1/3", "7+7\n"]:
    try:
        q = protected_query_llm(p)
        print(f"{p!r:15} -> {q.response}")
    except Exception as e:
        err = e.errors()[0]
        print(f"{p!r:15} -> BLOCKED on {err['loc'][0]:8} (input_value={err['input']!r})")
```

Expected:

```txt
'7+7' -> 14
'(3+3)*4' -> 24
'2.3*4' -> 9.2
'Hello World' -> BLOCKED on prompt (input_value='Hello World')
'2 + 2' -> BLOCKED on prompt (input_value='2 + 2')
'3-5' -> BLOCKED on response (input_value='-2')
'1/3' -> BLOCKED on response (input_value='The result is 0.333')
'7+7\n' -> BLOCKED on prompt (input_value='7+7\n')
```

### Proof that `validate_assignment=True` is what checks the answer

This builds the same class without that setting, then assigns a bad answer to both versions.

```python
class LLMQueryNoAssign(BaseModel): # same fields, but WITHOUT validate_assignment=True
    prompt: Annotated[str, StringConstraints(pattern=r"^[0-9.\+\-\*/\(\)]+$")]
    response: Annotated[str, StringConstraints(pattern=r"^[0-9\.]+$")] = None

q = LLMQueryNoAssign(prompt="1/3")
q.response = "The result is 0.333 <script>alert(1)</script>" # assignment is NOT checked
print("without validate_assignment:", q.response)
try:
    q2 = LLMQuery(prompt="1/3"); q2.response = "The result is 0.333 <script>alert(1)</script>"
except Exception as e:
    print("with validate_assignment :", e.errors()[0]["type"], "on", e.errors()[0]["loc"][0])

import re
print(bool(re.match(r"^[0-9\.]+$", "14\n")), "<- Python re: '$' also matches before a final newline")
```

Expected:

```txt
without validate_assignment: The result is 0.333 <script>alert(1)</script>
with validate_assignment : string_pattern_mismatch on response
True <- Python re: '$' also matches before a final newline
```

## Quick self-check

Using the course's calculator, for each prompt say whether the **input** check blocks it, the **output** check blocks it, or it passes both (and why):

- (a) `12 / 4`
- (b) `10-15`
- (c) `2**3`

## The facts worth locking in for the exam

- Character-based validation **allows only an approved set of characters in the prompt and/or the answer**, like classic escaping of `'` against SQL injection and stripping of `<` against XSS.
- The course's calculator uses **`^[0-9.\+\-\*/\(\)]+$` for the prompt and `^[0-9\.]+$` for the answer**, and `^...$` makes the whole string match.
- In Pydantic, the ==prompt is checked when the object is created, and the answer is only checked because of `validate_assignment=True`==. Without it, any answer is accepted silently.
- If the prompt fails the check, **an error is raised before the LLM is called**, so no model call happens and no compute is spent.
- It **doesn't stop prompt injection in free-text apps**, because payloads are plain letters. It is strong only in **narrow apps where letters aren't needed at all**.
- It **checks characters, not meaning** (`2**3` passes), and it **blocks real users**: spaces, negative answers and talkative answers all fail.
- Showing the raw validation error **reveals the regex to an attacker**; real apps should return a generic error.
- **Method point:** when you meet a character filter, **work out the allowed set from its error messages, then use only allowed characters, or move the attack to a part of the app that isn't filtered**. In the report, recommend it **only alongside content-based or AI-based checks**.

I've logged this section and the self-check answer in your progress file.