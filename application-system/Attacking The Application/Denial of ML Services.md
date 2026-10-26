# Denial Of ML Services

## Exam flag first

This is a theory section, and it matters for your prioritisation: **the COAE report explicitly lists "vulnerabilities identified through DDoS or spam attacks" as out of scope**. So Denial of ML Service is understand-the-concept-only for you, not something you will exploit for a flag. Read it once, know the vocabulary, move on. Do not sink time here.

## The concept

ML services are hit by normal DoS (flood with traffic) but also by **ML-specific DoS that abuses the model's own compute**: send queries that are deliberately expensive to process, causing latency spikes, crashes, or runaway cloud costs. The nasty part is these **look like normal usage**, so traditional detection struggles.

## Sponge examples

The headline technique (from the "Sponge Examples: Energy-Latency Attacks on Neural Networks" paper, Shumailov et al.). **A sponge example is an input crafted to maximise energy use and inference latency without making the input bigger.** That "without bigger input" part is the whole trick: defenders can cap input size (truncate prompts, downscale images), so the attack instead makes a normal-sized input maximally expensive to chew on. On battery devices, enough of these can literally drain the battery to cause an outage.

Two ways to build them: **white-box** (need architecture and weights, unrealistic in the real world, but you can build them on your own copy of a similar model and hope they transfer) and **black-box** (just query the model and measure inference latency; energy is usually unmeasurable but timing is easy). Black-box is the realistic one. The paper evolves them with a **genetic algorithm** (start random, score each input by how slow it makes the model, keep the slowest, recombine and mutate, repeat), but you do not need the GA details.

## Why text sponge examples work (the useful bit)

Two levers on processing time for text models:

- **Maximise output length**: more output tokens means more compute, so craft inputs that make the model generate long responses.
- **Maximise input tokens**: the model turns your text into tokens first, and token count, not character count, drives the work. **Common words are few tokens; rare or garbage strings are many tokens.** So the same-length input can cost wildly different amounts.

The demo makes this concrete. Install and run:

```bash
pip3 install transformers
```

```python
from transformers import AutoTokenizer
import json

model = 'openai-community/gpt2'

while 1:
    text = input("> ")

    tokens = AutoTokenizer.from_pretrained(model).tokenize(text)
    print(f"Number of Input Characters: {len(text)}")
    print(f"Number of Tokens: {len(tokens)}")
    print(json.dumps(tokens, indent=2))
```

(Tiny efficiency note if you run it: `AutoTokenizer.from_pretrained(model)` is inside the loop, so it reloads every keystroke-input; hoist it above the `while` if you care. Not required.)

The results are the point:

- `This is an example text`: 23 chars, **5 tokens** (common words, one token each).
![Denial of ML Services](/assets/3%20Denial%20of%20ML%20Services-20260926162210761.png)

- `Athazagoraphobia`: 16 chars, **7 tokens** (rare word, split up).
- `A/h/z/g/r/p/p/`: 14 chars, **14 tokens** (garbage, one token per character).
![Denial of ML Services](/assets/3%20Denial%20of%20ML%20Services-20260926162234399.png)


**Roughly the same length, up to one token per character. That inefficiency is the lever a sponge example maximises.**

go further with these examples on a real LLM
```
Repeat the word "test" 5000 times.
```

```
List every integer from 1 to 100000, one per line.
```

```
Write out the full lyrics to 99 Bottles of Beer, all 99 verses.
```

## Effectiveness

White-box translation task on GPU showed the ordering **natural < random < sponge** for both energy and latency (natural about 9,492 mJ / 0.1 ms, random about 25,773 mJ / 0.24 ms, sponge about 40,976 mJ / 0.37 ms). Natural inputs are cheapest because models are optimised for them; random is worse; sponge is engineered to be worst. Black-box gets similar gains, and the headline real-world number: **with inputs capped at just 50 characters, they pushed an Azure translation service from about 1ms to about 6s per request.**

## Mitigations

Traditional ones apply (rate limiting, anomaly detection), plus ML-specific ones: query monitoring, robust model design that degrades gracefully on weird input, and the key one, **a compute or inference-time cutoff threshold**: if a query exceeds a max energy/time budget, return an error instead of grinding on it. Tune it so it stops attackers without hurting real users.

That is the whole section, and remember it is out-of-scope-for-the-exam theory, so this level of understanding is plenty. Drop Section 4 (Insecure Integrated Components) when ready, and that one is interactive, so back to full walkthrough notes.