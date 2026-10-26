# Spam Filter Implementation



**The idea in one breath:** before you can _attack_ a spam filter, you need a real one to attack. This section builds that **victim** - a working **Naive Bayes** spam classifier trained on 5,000+ real SMS messages. No attack yet; it's the standard ML pipeline (**get data → clean it → turn words into numbers → train → check it works**).

## 1. Library installation

```bash
# Install the AI Library (or update it)
pip install --upgrade git+https://github.com/PandaSt0rm/htb-ai-library
```

HTB's own helper library (theme colours + model save/load). Run once.

## 2. Setup and dependencies

HTB gives the imports as **three separate groups** - I'm keeping them separate because each runs fine on its own.

**Group 1 - standard library + numpy + the reproducibility seeds:**

```python
import json
import pickle
import random
import re
import urllib.request
import zipfile
from pathlib import Path
import numpy as np

# Reproducibility
random.seed(1337)
np.random.seed(1337)
```

**Group 2 - data-science / ML libraries:**

```python
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score
```

**Group 3 - HTB utilities (colours + model helpers):**

```python
from htb_ai_library import (
    AZURE,
    HACKER_GREY,
    HTB_GREEN,
    MALWARE_RED,
    NODE_BLACK,
    NUGGET_YELLOW,
    WHITE,
    AQUAMARINE,
    load_model,
    save_model,
)
```

What matters:

- ****Random seeds** make randomness repeatable.** ML steps shuffle/split data using a random generator that starts at a **seed**; fix it → identical results every run → you can trust an attack result isn't luck. (`1337` = "leet".)
- ****`pathlib.Path`** builds paths that work on Windows _and_ Linux** - `Path("data") / "file.txt"` → `data\file.txt` on Windows, `data/file.txt` on Linux.

## 3. Loading the SMS Spam dataset (with caching)

Dataset = the ****SMS Spam Collection** (UCI), ~5,574 labeled messages** (`spam`/`ham`). First, set up the directory and path:

```python
print("\n[*] Loading SMS Spam Dataset...")

data_dir = Path("data")
data_dir.mkdir(exist_ok=True)
dataset_path = data_dir / "sms_spam.csv"
```

==`mkdir(exist_ok=True)` creates the folder if missing and doesn't error if it already exists== - makes the code **idempotent** (safe to re-run).

Next is the cache-check `if`/`else`. **HTB split this one statement across two blocks - the second block is the _continuation of the `else` branch_ (indented inside it) and can't run alone, so these two belong together as one block:**

```python
if dataset_path.exists():
    print(f"[+] Using cached dataset: {dataset_path}")
    df = pd.read_csv(dataset_path)
else:
    print("[*] Downloading from UCI repository...")
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip"
    zip_path = data_dir / "sms_spam.zip"

    urllib.request.urlretrieve(url, zip_path)

    with zipfile.ZipFile(zip_path, "r") as zf:
        with zf.open("SMSSpamCollection") as f:
            lines = [line.decode("utf-8").strip() for line in f]

    # Parse tab-separated format
    data = []
    for line in lines:
        parts = line.split('\t')
        if len(parts) == 2:
            data.append({"label": parts[0].lower(), "message": parts[1]})

    df = pd.DataFrame(data)
    df.to_csv(dataset_path, index=False)
    zip_path.unlink()
    print(f"[+] Dataset saved to {dataset_path}")
```

- **Purely a speed trick** - if a local copy exists, read it (milliseconds); otherwise download once (2-3s) and cache a CSV for next time.
- The file is read **straight from the zip in memory**, splitting each line on the tab (`\t`) into `label` and `message`.
- ==`if len(parts) == 2` skips malformed lines== so you only keep clean pairs. The result is a pandas **`DataFrame`** (a table), cached as CSV; the zip is deleted (`unlink()`).

## 4. Understanding the dataset

```python
print(f"[+] Loaded {len(df)} messages")
print(f" Spam: {sum(df['label'] == 'spam')}")
print(f" Ham: {sum(df['label'] == 'ham')}")
```

**Output:**
![Spam Filter Implementation](/assets/3%20Spam%20Filter%20Implementation-20261010162732865.png)

****747 spam (13.4%) vs 4,825 ham (86.6%) - class imbalance.**** Why it's an attack enabler, not trivia: **a model that sees mostly ham becomes biased toward predicting ham**, and the GoodWords attack pushes messages _toward ham_ - the direction the model already leans. The imbalance quietly helps the attacker (and it's realistic - real inboxes are mostly legit).

```python
print("\n[*] Sample messages:")
print("\nSPAM samples:")
for msg in df[df['label'] == 'spam']['message'].head(3):
    print(f" - {msg[:80]}...")
print("\nHAM samples:")
for msg in df[df['label'] == 'ham']['message'].head(3):
    print(f" - {msg[:80]}...")
```

**Output:**
![Spam Filter Implementation](/assets/3%20Spam%20Filter%20Implementation-20261010162809007.png)

Spam screams **`Free`, `WINNER!!`, `£900`**; ham is casual chat. **These patterns become the classifier's features.**

## 5. Text preprocessing - the key twist

Raw text gets normalized, but the **twist specific to spam: they deliberately KEEP the spam signals** (`!!!`, `£`, `$`, numbers, CAPS) instead of stripping them - **because those symbols _are_ the features that mark spam.**

Extra imports for the cleaners:

```python
import html as html_module
import unicodedata
```

### `minimal_clean` - light touch (preserves spam indicators)

```python
def minimal_clean(text):
    """
    Minimal cleaning that preserves spam indicators.

    Parameters: text (str) raw SMS message
    Returns: str cleaned text with entities decoded, unicode normalized, and
             whitespace collapsed while keeping informative symbols.
    """
    # Decode HTML entities (e.g., &amp; -> &)
    text = html_module.unescape(text)

    # Normalize unicode characters
    text = unicodedata.normalize('NFKC', text)

    # Clean up excessive whitespace
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\n+', ' ', text)
    text = re.sub(r'\r+', ' ', text)

    return text.strip()
```

- **`html.unescape`** turns entities back into characters (`&amp;` → `&`) - handles web-scraped text.
- **`unicodedata.normalize('NFKC', ...)`** folds odd unicode to a standard form (ligature "ﬁ" → "fi"), so identical content isn't split by encoding quirks.
- The three `re.sub(...)` calls **collapse whitespace/newlines/returns** (`re` = regex; `\s+` = "one or more whitespace chars").

### `clean_text` - final cleaning for vectorization

```python
def clean_text(text):
    """
    Final cleaning for vectorization.

    Converts to lowercase and removes only problematic characters so that
    informative symbols remain available to the vectorizer.

    Parameters:
        text (str): Preprocessed message from `minimal_clean`.

    Returns:
        str: Normalized, whitespace‑collapsed text ready for tokenization.
    """
    text = text.lower()
    # Keep numbers, currency symbols, punctuation - they're spam features!
    # Only remove truly problematic characters
    text = re.sub(r'[^\w\s£$€¥!?.,;:\'\"-]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()
```

- **Lowercases** so `FREE` and `free` are the _same_ word.
- The regex `[^\w\s£$€¥!?.,;:\'\"-]` means ****"anything NOT in this set → replace with a space"**** (`^` inside `[...]` = "not"). **Keeps word chars, whitespace, currency, punctuation; removes only the weird stuff** - the kept symbols are intentional spam features.

### Apply both stages

```python
print("[*] Applying minimal text cleaning (preserving spam indicators)...")
df['preprocessed'] = df['message'].apply(minimal_clean)

# Apply final cleaning for vectorization
df['clean_message'] = df['preprocessed'].apply(clean_text)
```

**`.apply(func)` runs that function over every row of the column.** Two versions kept: **`preprocessed` (analysis)** and **`clean_message` (model)**.

### Inspect what survived

```python
print("\n[*] Sample spam messages with preserved features:")
spam_samples = df[df['label'] == 'spam'].sample(3, random_state=42)
for idx, row in spam_samples.iterrows():
    msg = row['preprocessed'][:100] + "..." if len(row['preprocessed']) > 100 else row['preprocessed']
    print(f" - {msg}")
```

**Output:**

![Spam Filter Implementation](/assets/3%20Spam%20Filter%20Implementation-20261010163108547.png)

Preprocessing preserved **punctuation, currency ("750 Pound prize"), and CAPS ("FREE NOW!", "MATCHED")** - the important features.

## 6. Removing duplicates and empty messages

```python
# Remove only exact duplicates
original_size = len(df)
df = df.drop_duplicates(subset=['label', 'clean_message'])
print(f"\n[+] Removed {original_size - len(df)} duplicates")

# Remove empty messages
before_empty = len(df)
df = df[df['clean_message'].str.len() > 0]
print(f"[+] Removed {before_empty - len(df)} empty messages")
```

**Output:**

```text
[+] Removed 419 duplicates
[+] Removed 0 empty messages

[+] Final dataset: 5153 messages
    Spam: 638 (12.4%)
    Ham: 4515 (87.6%)
```

- **`drop_duplicates` removes only _exact_ duplicates** (same label AND cleaned text) - 419 gone. The higher spam duplicate rate reveals spam's **repetitive nature** (a trait the attack exploits).
- **Final: 5,153 messages, 12.4% spam.**

## 7. Creating train and test sets

Foundational ML rule: **never judge a model on the data it learned from - it could just memorize.** So split: **train on one chunk, test on messages it's never seen.**

```python
X = df['clean_message'].values
y = df['label'].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\n[+] Data split:")
print(f" Training: {len(X_train)} messages")
print(f" Testing: {len(X_test)} messages")
```

**Output:**

![Spam Filter Implementation](/assets/3%20Spam%20Filter%20Implementation-20261010163332717.png)

- ==`test_size=0.2` → 80% train / 20% test== (4,122 vs 1,031).
- ==`stratify=y` keeps the 12.4% spam ratio in _both_ halves== - without it a random split could starve the test set of spam and make metrics meaningless.
- **`.values` pulls plain NumPy arrays** out of the pandas table.

## 8. Training the Naive Bayes classifier

**HTB split this one `if`/`else` across three blocks** (set-up+`if`, then `else:`, then the rest of the `else`). The `else:` and its continuation can't run without the `if`, **so these three are really one block** - merged here:

```python
print("\n[*] Training Naive Bayes classifier...")

model_dir = Path("models")
model_dir.mkdir(exist_ok=True)
model_path = model_dir / "spam_classifier.pkl"

if model_path.exists():
    print(f"[+] Loading saved model from {model_path}")
    with open(model_path, 'rb') as f:
        saved_data = pickle.load(f)
        vectorizer = saved_data['vectorizer']
        classifier = saved_data['classifier']

    # Transform data using existing vocabulary
    X_train_vec = vectorizer.transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
else:
    # Configure vectorizer to capture spam patterns
    vectorizer = CountVectorizer(
        max_features=3000,
        token_pattern=r'\b\w+\b|[£$€¥]+|\d+|!!+|\?\?+|\.\.+',
        lowercase=True,
        stop_words='english'
    )
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    classifier = MultinomialNB()
    classifier.fit(X_train_vec, y_train)

    # Save model for reproducibility
    with open(model_path, 'wb') as f:
        pickle.dump({'vectorizer': vectorizer, 'classifier': classifier}, f)
    print(f"[+] Model saved to {model_path}")
```

Two big concepts packed in here:

**Vectorization - turning words into numbers.** ****`CountVectorizer` converts each message into a row of word counts** - this is **bag-of-words** from Section 2 made real.** It builds a **vocabulary** of all words, then each message becomes "how many times each vocab word appears." Word order is discarded.

- **`token_pattern`** (regex, `|` = "or") defines a token: words, currency, numbers, and **repeated punctuation (`!!+`, `??+`, `..+`)** - so "!!!" becomes a learnable spam feature.
- ==`max_features=3000`== keeps only the top 3,000 tokens → less **overfitting**.
- ==`stop_words='english'` drops useless words ("the", "is")== - literally Section 2's point: the low-goodness-score overlap words with no discriminative power.

**The `fit`/`transform` rule (the thing that'll bite you in the attack):**

- ==**`fit_transform` (train)** = learn the vocabulary AND convert.==
- ==**`transform` (test / saved model)** = convert using the SAME vocabulary== - never re-learn from new data; unknown words are ignored.

**Training + saving.** ==**`MultinomialNB`** = Multinomial Naive Bayes, the Section 2 classifier.== **`.fit()` is where it learns `P(word|spam)` and `P(word|ham)`** by counting word occurrences per class - **those learned frequencies are exactly what the GoodWords attack exploits.** ****`pickle`** freezes the trained objects to disk** so you don't retrain; critically, **save the vectorizer _with_ the model** because the numbers only make sense in that vocabulary.

## 9. Evaluating model performance

```python
# Calculate accuracy scores
train_acc = classifier.score(X_train_vec, y_train)
test_acc = classifier.score(X_test_vec, y_test)
print(f"[+] Training accuracy: {train_acc:.4f}")
print(f"[+] Testing accuracy: {test_acc:.4f}")

# Get detailed predictions
y_pred = classifier.predict(X_test_vec)
print("\n[*] Classification Report:")
print(classification_report(y_test, y_pred))
```

**Output:**

![Get detailed predictions](/assets/Pasted%20image%2020261010163456.png)

- ==`.score()` = **accuracy** = fraction classified correctly.== Train 99.22%, test 98.64%.
- **The tiny gap (0.58%) matters**: a big train-vs-test gap would mean ****overfitting** (memorizing, not learning)**. Small gap = it genuinely learned to spot spam.
- ==**Precision** = "when it says spam, how often is it right?"== (spam 95% → 5% false alarms). ==**Recall** = "of all actual spam, how much did it catch?"== (spam 94% → misses 6%).
- Ham is 99%/99% (easy, tons of examples); **spam is lower because of class imbalance**.
- ****Why this matters for the attack:** you now have a _strong_ target (98.6%)** - sneaking spam past this is meaningful, not beating up a toy.

---
### The facts worth locking in for the exam

- **The target is a `MultinomialNB` classifier over bag-of-words counts**, and `.fit()` learns `P(word|spam)` and `P(word|ham)` just by counting how often each word appears in each class - **those learned probabilities are exactly what every evasion attack manipulates.**
- **Vectorize training data with `fit_transform` (learn the vocabulary + convert), but test or attacked data with `transform` (reuse that same vocabulary).** Any word not in the training vocabulary is silently ignored - so an attacker's "good words" only count if they already exist in it.
- **The ham-majority class imbalance (~87% ham) biases the model toward predicting ham**, which is the exact direction the GoodWords attack pushes - the dataset's skew works in the attacker's favour.
- **Preprocessing deliberately keeps the spam signals** (`!`, `£`/`$`, numbers, CAPS) instead of stripping them, because those symbols are discriminative features; ==`stop_words='english'` removes the non-discriminative overlap words== ("the", "is") that appear equally in spam and ham.
- **A small train-vs-test accuracy gap means the model generalised, i.e. no overfitting** (~99% train / ~98.6% test here). A strong baseline matters - **beating a 98.6% filter is a genuine evasion result, not an artifact of a weak target.**
- ==Precision = "when it flags spam, how often is it right" (95%); recall = "of all actual spam, how much did it catch" (94%).== Spam scores lower than ham only because there are fewer spam examples to learn from.
