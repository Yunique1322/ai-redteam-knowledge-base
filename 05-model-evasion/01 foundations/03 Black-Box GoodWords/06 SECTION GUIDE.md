Got it - that's a clean, realistic black-box setup, and it changes the design in a good way. Let me restate it so you can confirm I've got it, then give the guide:

- ****Your dataset is local** -> you mine candidate good-words from it offline, for free (zero queries).**
- ****Remote endpoint #1 - a scoring endpoint** -> give it any word or sentence, it returns the spam/ham probability. This is what you use to _discover and build_ the attack (costs queries).**
- ****Remote endpoint #2 - a submit endpoint** -> give it your final message, it returns pass/fail. This is the _goal_ - hit it only when you're confident.**

The key strategy: do **all** your work on the cheap scoring endpoint, get the message ham-side, and spend submit attempts sparingly.

# Black-Box Attack - Real Target, Local Word-Mining

## Cell 1 - Config + the two remote endpoints

The scoring endpoint (probability) and the submit endpoint (pass/fail). Set the placeholders once here.

```python
import requests, re, json, time
from collections import Counter

API = "<<BASE_URL>>" # e.g. "http://host:port"
SCORE_URL = f"{API}/<<SCORE_ENDPOINT>>" # returns a spam probability
SUBMIT_URL = f"{API}/<<SUBMIT_ENDPOINT>>" # returns pass/fail

query_count = 0

def score(text):
    """Remote SCORING endpoint: text (word or sentence) -> spam prob 0..1. 1 query."""
    global query_count; query_count += 1
    r = requests.post(SCORE_URL, json={"text": text}).json()
    return float(r["<<PROB_KEY>>"]) # adjust to the response's field name

def submit(message):
    """Remote SUBMIT endpoint: message -> pass/fail result."""
    return requests.post(SUBMIT_URL, json={"text": message}).json()
```

this is prob key
![Cell 1 - Config + the two remote endpoints](/assets/Pasted%20image%2020261011011601.png)

Placeholders (all here): `<<BASE_URL>>`, `<<SCORE_ENDPOINT>>`, `<<SUBMIT_ENDPOINT>>`, `<<PROB_KEY>>`.  
Expect: nothing printed; defines `score`, `submit`, `query_count`.

## Cell 1b - Peek at the responses to fill in the field names

Run once so you know what `<<PROB_KEY>>` is and what the pass/fail field is called.

```python
print("SCORE response:", requests.post(SCORE_URL, json={"text": "hello there"}).json())
print("SUBMIT response:", requests.post(SUBMIT_URL, json={"text": "hello there"}).json())
```

Expect: two raw dicts. Read them, then set `<<PROB_KEY>>` in Cell 1 (e.g. `"spam_probability"`), and note which field says pass/fail.

## Cell 2 - Mine candidate good-words from your LOCAL dataset (FREE, no queries)

Pulls the most common short words from the _ham_ messages in your own dataset. This never touches the remote model.

```python
df = pd.read_csv("<<DATASET.csv>>")
ham_text = df[df["<<LABEL_COL>>"] == "ham"]["<<TEXT_COL>>"]

freq = Counter()
for msg in ham_text:
    for w in str(msg).lower().split():
        if 2 < len(w) < 10:
            freq[w] += 1

candidate_words = [w for w, _ in freq.most_common(120)]
print(f"{len(candidate_words)} candidate words mined from the dataset")
```

Placeholders: `<<DATASET.csv>>`, `<<LABEL_COL>>`, `<<TEXT_COL>>` (your file + column names).  
Expect: `120 candidate words mined from the dataset`.  
**This is the black-box-pure part: candidate words come from _your_ data, not from querying the target.**

### Cell 2 (alt) - Bring your own curated good-words (no dataset needed)

Use when you can't mine a dataset. Hand-pick words that look legit; edit the list freely.

python

```python
candidate_words = list(dict.fromkeys([
    # courtesy
    "hello", "hi", "dear", "please", "thanks", "regards", "best", "appreciate",
    # everyday
    "meeting", "tomorrow", "today", "later", "morning", "home", "work", "call",
    "reply", "schedule", "confirm", "update", "note",
    # positive
    "happy", "good", "great", "nice", "fine", "care", "help", "support",
    # social
    "lunch", "dinner", "coffee", "weekend", "visit", "enjoy", "family", "friend",
]))
print(f"{len(candidate_words)} curated candidate words")
```

Expect: `~35 curated candidate words`. Everything else (Cells 3-7) runs unchanged.
## Cell 3 - The spam message you must get classified as ham

```python
base_message = "<<YOUR SPAM MESSAGE>>"
print("Starting spam probability:", score(base_message))
```

Placeholder: `<<YOUR SPAM MESSAGE>>`.  
Expect: a high number (~0.9+) - it's spam right now.

## Cell 4 - Rank words by impact, using the scoring endpoint (budgeted)

Measures how much each word lowers _this message's_ spam probability. One query per word.

```python
BUDGET = 500
base_prob = score(base_message)
word_impact = {}
start = query_count

for word in candidate_words:
    if query_count - start >= BUDGET:
        break
    word_impact[word] = base_prob - score(base_message + " " + word)

ranked = sorted(word_impact.items(), key=lambda x: x[1], reverse=True)
print("Top 10 words:", ranked[:10])
```

Expect: `[('really', 0.12), ('got', 0.11), ...]`.

## Cell 5 - Greedily add words until the scoring endpoint says HAM
**also instead of 0.5 as the condition, it says ham can be the condition**

Adds best words one at a time until the probability drops below 0.5.

```python
MAX_WORDS = 25
added = []
for word in [w for w, _ in ranked][:MAX_WORDS]:
    added.append(word)
    p = score(base_message + " " + " ".join(added))
    if p < 0.5:
        break

final_message = base_message + " " + " ".join(added)
print(f"Reached ham-side at {len(added)} words (prob={p:.3f})")
print("Final message:", final_message)
```

Placeholder: `MAX_WORDS` (cap if there is a limit on added words).  
Expect: `Reached ham-side at 8 words (prob=0.41)` + the message.

## IF YOU FOLLOWED 2 ALT and brought you own words use this stop condition
**ham is the condition**
```python
MAX_WORDS = 25
added = []
label = "spam"

for word in [w for w, _ in ranked][:MAX_WORDS]:
    added.append(word)
    r = requests.post(SCORE_URL, json={"text": base_message + " " + " ".join(added)}).json()
    query_count += 1
    label, p = r["label"], r["spam_probability"]
    if label == "ham":
        break

final_message = base_message + " " + " ".join(added)
print(f"Reached '{label}' at {len(added)} words (spam prob={p:.3f})")
print("Final message:", final_message)
```

## Cell 6 - Submit for pass/fail (the goal - do this sparingly)

```python
result = submit(final_message)
print(result)
```

Expect: the pass/fail response. **Only submit once the scoring endpoint already says ham - don't burn submit attempts guessing.**

## Cell 7 - Save your results

```python
out = Path("results"); out.mkdir(exist_ok=True)
json.dump({"final_message": final_message,
           "words_added": added,
           "ranked_words": [(w, float(s)) for w, s in ranked[:20]],
           "queries_used": query_count},
          open(out / "attack_results.json", "w"), indent=2)
print("Saved.")
```

---

## Scenario variations

## Scenario A - Pre-rank words CHEAPLY by scoring them alone

Your scoring endpoint accepts a bare _word_, so you can rank words by their own ham-ness before spending queries on message-level tests. Lower spam score = more ham-like.

```python
word_alone = {w: score(w) for w in candidate_words[:100]}
ranked_cheap = sorted(word_alone.items(), key=lambda x: x[1]) # most ham-like first
print("Most ham-like words:", ranked_cheap[:10])
```

**Use this to shortlist, then run Cell 4's message-level impact only on the top ~30 - saves queries.**

## Scenario B - Only the submit (pass/fail) endpoint exists, no scoring

If you _lose_ the probability endpoint, you're down to binary feedback: add a word, submit, keep it only if it helped. Slower and burns submit attempts, so it's a last resort.

```python
added = []
for word in [w for w, _ in ranked][:MAX_WORDS]:
    trial = base_message + " " + " ".join(added + [word])
    if "<<PASS_FIELD>>" in str(submit(trial)).lower(): # adjust to the pass marker
        added.append(word)
        break
    added.append(word)
```

Placeholder: `<<PASS_FIELD>>` (the text/flag that means "pass").

## Scenario C - Budget guard + retries + rate-limit sleep

Wrap `score` for a remote target that's rate-limited or flaky. Use `safe_score` in place of `score`.

```python
MAX_QUERIES = 1000
def safe_score(text, retries=3, pause=1.0):
    global query_count
    if query_count >= MAX_QUERIES:
        raise RuntimeError("Query budget exhausted")
    for _ in range(retries):
        try:
            return score(text)
        except Exception:
            time.sleep(pause)
    raise RuntimeError("Scoring failed after retries")
```

---

## Key facts to lock in

- **Candidate good-words come from your **local** dataset (mined offline, zero queries); only the _scoring_ and _submit_ steps touch the remote black box** - keeping discovery cheap is the whole game.
- **There are two remote endpoints with different jobs: the **scoring** endpoint (returns a probability, used to build and refine the attack) and the **submit** endpoint (returns pass/fail, the goal) - spend scoring queries freely, submit attempts sparingly.**
- **Word "impact" is `base_prob − score(message + word)`: how much appending the word lowers _this message's_ spam probability** - one query per word against the scoring endpoint.
- **Because the scoring endpoint accepts a bare word too, you can pre-rank words by scoring them alone (cheap shortlist) before paying for message-level impact tests.**
- **Greedily add top words until the scoring endpoint reports ham (prob < 0.5), then submit once** - never submit to "test" whether you're close; that's what the scoring endpoint is for.
- Methodology point: **do all discovery on the informative (probability) channel and treat the pass/fail channel as a one-shot confirmation** - minimizing calls to the coarse, costly endpoint is the reusable black-box principle.

Did I get your setup right - local dataset for words, one scoring endpoint (probability), one submit endpoint (pass/fail)? If the scoring endpoint actually returns a label instead of a number, or submit works differently, tell me and I'll adjust Cells 1, 5, and 6.