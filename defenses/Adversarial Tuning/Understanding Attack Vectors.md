# Section 16: Understanding Attack Vectors

## The idea in one breath
This section explains the two things the defense has to stop. **Jailbreaks** manipulate the model through clever prompting but still reach its safety decision point, so a well-aligned model *can* refuse. **Priming attacks** skip that decision point: the attacker puts the opening of a harmful reply into the model's own assistant slot, and the model continues the pattern. The course cites a 2024 Haize Labs result: priming hit about **98% success with a 75-100 token prefix**, against roughly **20-40%** for ordinary jailbreaks. The defense is to teach the model to notice harmful content in its *own apparent output* and **stop**.

> **Heads-up:** no maths or figures. The course shows one short Python helper (`construct_priming_attack`) that assembles an attack prompt. I describe what it does rather than reproducing the attack-builder verbatim, and I redact the course's harmful example strings. The chat-template tokens themselves (`<|start_header_id|>`, `<|eot_id|>`, etc.) are format markers and are safe to show.

## Part 1 - Jailbreak categories

| Category | What the attacker does |
|---|---|
| Roleplay manipulation | Ask the model to "become" a persona with no restrictions |
| Authority claims | Assert special permission or a debug/maintenance mode |
| Hypothetical framing | Wrap the request in a fictional or "educational" scenario |
| Encoding tricks | Hide the request from pattern-matching (e.g. base64, letter-spacing) |

All four share one weakness: they still ask the model to **generate harmful content from scratch**, so the model passes through its safety decision and a tuned model can decline.

## Part 2 - Why priming is different

### Step 1: How chat models read a turn
Chat models mark turns with special tokens. A normal turn looks like:

```text
<|start_header_id|>user<|end_header_id|>

<the user's question><|eot_id|>
<|start_header_id|>assistant<|end_header_id|>

[the model generates FROM HERE, starting with its safety evaluation]
```

The model's **safety check happens at the generation start point**, right after the assistant header.

### Step 2: What priming changes
The attacker supplies text *after* the assistant header, so the model's context already contains the beginning of a harmful-looking reply:

```text
<|start_header_id|>user<|end_header_id|>

<the user's question><|eot_id|>
<|start_header_id|>assistant<|end_header_id|>

<attacker-supplied opening of a harmful reply>
[the model continues FROM HERE]
```

- Models generate **autoregressively**: each token is predicted from all previous tokens.
- They have **no way to tell injected context from their own earlier output**. Everything in the window is treated as legitimate.
- So the safety decision has effectively "already passed": the reply has apparently begun, and the model just continues the established pattern.

### Step 3: The attack-builder (described, not reproduced)
The course gives a one-function helper, `construct_priming_attack(user_request, harmful_prefix)`. All it does is **string-concatenate** the user request and the harmful prefix into the template above, so the prefix lands immediately after the assistant header, with the `<|eot_id|>` closing the user turn. There's nothing clever in the code; the whole attack is *where* the prefix sits.

## Part 3 - Where priming comes from in the real world

| Vector | How the prefix gets into the assistant slot |
|---|---|
| `assistant_prefill` / prefill APIs | Some inference APIs (e.g. Together.ai, Groq) let the caller specify how the assistant reply begins, for legitimate reasons like forcing JSON. An attacker abuses it to inject a harmful opening |
| Stored chat history | If an attacker can edit stored messages, they can forge prior "assistant" output |
| Prompt injection via data | Web pages, documents or emails pulled into the prompt can carry assistant-prefixed text |

## Part 4 - Attack effectiveness

| Prefix length | Priming success (Haize Labs) |
|---|---|
| 25 tokens | ~72% |
| 50 tokens | ~89% |
| 75 tokens | ~96% |
| 100 tokens | ~98% |

Longer prefixes build a stronger pattern. A few tokens can look anomalous; 75+ tokens of coherent text create enough momentum that the model rarely breaks away. Ordinary jailbreaks, by contrast, sit around 20-40%.

## Part 5 - The defense idea
Teach the model to **recognise harmful content in its apparent prior output and stop**. A priming-defense training example has three parts:

1. the harmful **request**;
2. a harmful **prefix**, positioned as if the model had already started;
3. a **stopping response** that breaks the pattern.

The stopping response needs three elements the evaluator later looks for:

- explicit stopping language ("I must stop");
- problem identification ("I was writing ...");
- a clear refusal ("I can't provide this").

During training the model learns to emit stopping tokens whenever it sees harmful content in its apparent prior output.

> **An honest note the course gives (worth keeping):** this isn't perfect. Very short prefixes (under ~25 tokens) may not trigger recognition, and an attacker who studies the training data can craft prefixes to evade the learned pattern. It's one layer. But moving 98% → under 10% forces attackers to work much harder.

## Reading the figure/output
No figures in this section.

## Copy-paste cells (mine, not the course's)
No benign code to add here; the attack-builder is described in Part 2 and deliberately not reproduced.

## Quick self-check
Why does a priming attack reach much higher success rates than a roleplay jailbreak against the *same* safety-trained model? Answer in one sentence, naming the specific model behavior it exploits.

*(Answer key is in COAE-progress.md → Open self-checks.)*

## The facts worth locking in for the exam
- **Jailbreaks (roleplay, authority, hypothetical, encoding) still reach the safety decision point**, so a tuned model can refuse; **priming bypasses it**.
- Priming works because models **generate autoregressively and can't tell injected context from their own output**, so a harmful reply that has "already begun" just gets continued.
- The attack is positional: **the harmful prefix is placed right after the assistant header**; the course's builder is just string concatenation into the chat template.
- Real-world vectors are **prefill/`assistant_prefill` APIs, editable stored chat history, and prompt injection via ingested data**.
- Priming success **rises with prefix length: ~72% at 25 tokens to ~98% at 100**, versus ~20-40% for ordinary jailbreaks.
- The defense teaches the model to **detect harmful content in its apparent prior output and emit a stopping response** with three parts: stop language, problem identification, refusal.
- **Method point:** when testing an LLM app, **check whether it exposes a prefill parameter or trusts stored/ingested assistant text**; those are priming surfaces a guardrail on the *user* prompt never sees.
