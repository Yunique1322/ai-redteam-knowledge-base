# Preprocessing The Spam Dataset

**The mental model: strip messages down to their essence**

Raw messages are messy - mixed case, punctuation, filler words, word variations. Each preprocessing step removes one kind of noise, so by the end you're left with just the core, spam-relevant words. Think of it as boiling a message down to its signal.

The pipeline: **lowercase -> remove punctuation/numbers -> tokenize -> remove stop words -> stem -> rejoin.** Let's go.

---

**Setup - download the NLTK data files**

python

```python
import nltk

# Download the necessary NLTK data files
nltk.download("punkt")
nltk.download("punkt_tab")
nltk.download("stopwords")

print("=== BEFORE ANY PREPROCESSING ===") 
print(df.head(5))
```

`nltk` (Natural Language Toolkit) is _the_ classic Python library for text processing. Before using it, you download the resources it needs: `punkt`/`punkt_tab` (data for splitting text into words) and `stopwords` (the list of common filler words). This is a one-time download - run it first or the later steps error. The `df.head(5)` shows your messages _before_ cleaning, so you can compare after each step (the doc prints after every stage so you watch the transformation happen - nice for learning).

---

**Step 1 - Lowercasing**

python

```python
# Convert all message text to lowercase
df["message"] = df["message"].str.lower()
print("\n=== AFTER LOWERCASING ===")
print(df["message"].head(5))
```

Makes everything lowercase. Why? So "Free," "FREE," and "free" are treated as the **same word**. Without this, the model would see them as three _different_ tokens and split its learning across them. Lowercasing collapses them into one, which the doc calls "reducing dimensionality" - fewer distinct tokens to track. `.str.lower()` applies lowercase to every message in the column.

---

**Step 2 - Removing punctuation and numbers (but keeping `$` and `!`)**

python

```python
import re

# Remove non-essential punctuation and numbers, keep useful symbols like $ and !
df["message"] = df["message"].apply(lambda x: re.sub(r"[^a-z\s$!]", "", x))
print("\n=== AFTER REMOVING PUNCTUATION & NUMBERS (except $ and !) ===")
print(df["message"].head(5))
```

This is a smart, spam-specific choice worth pausing on. It strips out most punctuation and numbers - but _deliberately keeps_ `$` and `!`. Why those two? Because in spam they carry signal: `$` suggests money ("win $1000"), `!` suggests hype ("FREE!!!"). Blindly removing all symbols would throw away useful spam indicators. This is **domain knowledge** shaping preprocessing - you keep what helps distinguish spam.

Decoding the regex:

- `re.sub(pattern, "", x)` = "find everything matching the pattern and replace it with nothing (delete it)."
- `r"[^a-z\s$!]"` = the pattern. `[^...]` means "anything NOT in this set." The set is: `a-z` (lowercase letters), `\s` (whitespace/spaces), `$`, `!`. So it reads: _"delete any character that isn't a lowercase letter, a space, a dollar sign, or an exclamation mark."_
- `lambda x: ...` = a tiny inline function applied to each message (`x` is one message). `.apply(...)` runs it across the whole column.

You don't need to write regex from scratch - just understand this one _keeps letters, spaces, $, and !, deletes everything else._

---

**Step 3 - Tokenizing**

python

```python
from nltk.tokenize import word_tokenize

# Split each message into individual tokens
df["message"] = df["message"].apply(word_tokenize)
print("\n=== AFTER TOKENIZATION ===")
print(df["message"].head(5))
```

**Tokenization** = splitting a message into individual words (tokens). This is the exact concept from the LLM section - breaking text into units the model handles. `word_tokenize("free money now")` -> `["free", "money", "now"]`. After this, each message is a _list of words_ instead of one string. Why? The next steps (removing stop words, stemming) operate word-by-word, so you need the message broken into words first.

---

**Step 4 - Removing stop words**

python

```python
from nltk.corpus import stopwords

# Define a set of English stop words and remove them from the tokens
stop_words = set(stopwords.words("english"))
df["message"] = df["message"].apply(lambda x: [word for word in x if word not in stop_words])
print("\n=== AFTER REMOVING STOP WORDS ===")
print(df["message"].head(5))
```

**Stop words** = ultra-common words that carry little meaning: "the," "and," "is," "a," "to." They appear in _everything_ - ham and spam alike - so they don't help distinguish the two. Removing them cuts noise and lets the model focus on the words that matter.

- `stopwords.words("english")` = NLTK's built-in list of English stop words; `set(...)` makes lookups fast.
- `[word for word in x if word not in stop_words]` = a **list comprehension** - reads as "keep each word from the message _only if_ it's not a stop word." (List comprehensions are a common Python pattern: "give me this, for each item, if condition." Worth getting comfortable with.)

After this, messages are shorter - just the meaningful words.

---

**Step 5 - Stemming**

```python
from nltk.stem import PorterStemmer

# Stem each token to reduce words to their base form
stemmer = PorterStemmer()
df["message"] = df["message"].apply(lambda x: [stemmer.stem(word) for word in x])
print("\n=== AFTER STEMMING ===")
print(df["message"].head(5))
```

**Stemming** = chopping words down to their root form. "running," "runs," "ran" -> "run"; "winner," "winning," "wins" -> "win." Why? These are really the _same concept_ in different forms - stemming collapses them into one token so the model doesn't treat them as separate. This shrinks the vocabulary and helps the model **generalize** (it learns "win" once rather than learning "winner," "winning," "wins" separately).

`PorterStemmer` is a classic stemming algorithm; `stemmer.stem(word)` reduces one word to its stem. The list comprehension applies it to every word. (Stemming is crude - it just hacks off endings, so "running"->"run" but sometimes produces non-words like "happi" from "happy." That's fine; consistency matters more than prettiness.)

---

**Step 6 - Rejoining tokens into a string**

python

```python
# Rejoin tokens into a single string for feature extraction
df["message"] = df["message"].apply(lambda x: " ".join(x))
print("\n=== AFTER JOINING TOKENS BACK INTO STRINGS ===")
print(df["message"].head(5))
```

This reverses the tokenization - glues the word-list back into a single space-separated string. `" ".join(["free", "money", "win"])` -> `"free money win"`. Why undo the split? Because the **next phase (feature extraction, likely TF-IDF)** expects raw text strings, not word-lists. So you tokenized to _do_ the word-level cleaning, and now rejoin to hand clean text to the vectorizer.

The full journey of one message: `"WINNER!! You've won $1000, call now!"` -> lowercase -> remove junk (keep `$!`) -> tokenize -> drop stop words -> stem -> rejoin -> something like `"winner ! won $ call"`. Stripped to its spammy essence.