# Black-Box Core Components


**The idea in one breath:** you're building the pieces the black-box attack needs _before_ any clever searching - **a candidate word list mined from legitimate text (built offline, zero queries), and a query-budget manager that splits your limited queries across phases.** This is the plumbing; the actual bandit-style discovery comes next section.

## Attack simulation setup

```python
print("\n[*] Simulating black-box attack scenario...")
print("[*] Budget: 1000 queries")

# Simulate limited query access
query_budget = 1000
queries_used = 0
query_log = []
```

**The whole premise of black-box is a limited query budget** - real hosted APIs rate-limit or charge per prediction. ==`query_budget = 1000`== is your allowance, **`queries_used`** counts consumption so you don't overrun, and `query_log` is a placeholder for per-query data (unused here). This is the bandit's "limited pulls" made concrete.

## Building the candidate vocabulary

The plan: **mine common words from ham messages, filter them, then add a small curated list** - all offline, spending no queries against the target.

### Step 1 - extract word frequencies from ham

```python
def extract_ham_word_freq(X_train, y_train, sample_size=500):
    """
    Compute token frequencies from a sample of ham messages.

    Parameters
    ----------
    X_train : array-like of str
        Cleaned training messages.
    y_train : array-like of str
        Labels aligned with X_train ('ham' or 'spam').
    sample_size : int, default 500
        Number of ham messages to analyze.

    Returns
    -------
    dict[str, int]
        Mapping of word -> frequency within sampled ham messages.
    """
    ham_msgs = X_train[y_train == 'ham']
    limit = min(sample_size, len(ham_msgs))
    freq = {}
    for msg in ham_msgs[:limit]:
        for w in str(msg).split():
            if 2 < len(w) < 10: # keep typical conversational tokens
                freq[w] = freq.get(w, 0) + 1
    return freq

wf_example = extract_ham_word_freq(X_train, y_train, sample_size=500)
print("[*] Example: extract_ham_word_freq")
print(f" Ham messages sampled: {min(500, sum(y_train == 'ham'))}")
print(f" Unique tokens found: {len(wf_example)}")
top5 = sorted(wf_example.items(), key=lambda x: (-x[1], x[0]))[:5]
for w, c in top5:
    print(f" {w}: {c}")
```

**Output:**

![Black-Box Core Components](/assets/7%20Black-Box%20Core%20Components-20261010181727171.png)

- ==`X_train[y_train == 'ham']` boolean-filters to just the ham messages==; `[:limit]` caps it to the sample size.
- **`freq.get(w, 0) + 1` is the standard dict-counting idiom** - "current count (0 if the word's new) plus one."
- **The length filter `2 < len(w) < 10` keeps 3-9-character words** - short enough to drop stopword-like junk ("a", "I", "to") but it excludes long artifacts (URLs, phone numbers).
- **Mirrors how a real attacker gathers legit language** (public forums, leaked corpora) to imitate ham.
- Worth flagging: the top words come out as `you, and, the, have, but` - which are **common but _not_ discriminative** (they appear plenty in spam too). **Frequency alone is a weak proxy for "good word"** - which is exactly why the next section _tests_ each candidate against the model instead of trusting frequency.

### Step 2 - keep the high-frequency ones

```python
def select_high_frequency_words(word_freq, max_words=100, min_freq=5):
    """
    Select the most frequent ham words above a minimum frequency.

    Parameters
    ----------
    word_freq : dict[str, int]
        Token frequency table for sampled ham messages.
    max_words : int, default 100
        Maximum number of words to return.
    min_freq : int, default 5
        Minimum frequency a word must meet to be considered.

    Returns
    -------
    list[str]
        Top words sorted by decreasing frequency then lexicographically.
    """
    sorted_by_freq = sorted(word_freq.items(), key=lambda x: (-x[1], x[0]))
    top = [w for w, c in sorted_by_freq if c > min_freq][:max_words]
    return top

top_words_example = select_high_frequency_words(wf_example, max_words=100, min_freq=5)
print("[*] Example: select_high_frequency_words")
print(f" Selected top words: {len(top_words_example)} (min_freq=5)")
print(" First 10:", ", ".join(top_words_example[:10]))
```

**Output:**

![Step 2 - keep the high-frequency ones](/assets/Pasted%20image%2020261010181805.png)

- **The sort key `(-x[1], x[0])` is a two-tier sort: by frequency descending (the negative flips the order), then by the word alphabetically as a tie-break.** That tie-break makes the ordering **deterministic/reproducible** - without it, words with equal frequency could come out in different orders across runs or Python versions.
- **`[:max_words]` caps at 100 to fit the budget** - testing all 2,076 tokens would blow past 1,000 queries; 100 candidates fit comfortably.

### Step 3 - merge in curated conversational terms

```python
def merge_with_curated(top_words, additional_candidates=None):
    """
    Merge data-driven top words with curated conversational candidates.

    Parameters
    ----------
    top_words : list[str]
        High-frequency ham words from the previous step.
    additional_candidates : list[str] | None
        Optional curated list to include regardless of frequency.

    Returns
    -------
    list[str]
        Deduplicated merged list (lexicographically ordered).
    """
    if additional_candidates is None:
        additional_candidates = [
            "ok", "cos", "ill", "thats", "later", "said", "ask", "didnt",
            "dont", "doing", "going", "come", "home", "tomorrow", "today", "sorry",
            "thanks", "yeah", "yes", "sure", "see", "tell", "know", "think",
        ]
    merged = set(top_words) | set(additional_candidates)
    return sorted(merged)

merged_example = merge_with_curated(top_words_example)
added = sorted(set(merged_example) - set(top_words_example))
print("[*] Example: merge_with_curated")
print(f" Merged size: {len(merged_example)} | Added curated: {len(added)}")
print(" Sample added terms:", ", ".join(added[:5]))
```

**Output:**

![Black-Box Core Components](/assets/7%20Black-Box%20Core%20Components-20261010181841633.png)

- **`set(top_words) | set(additional_candidates)` is a set _union_ - the `|` operator merges and **deduplicates** in one step** (a set can't hold duplicates).
- **The curated list hand-adds conversational words (contractions like `dont`, temporal terms like `tomorrow`) that spam rarely uses** - a hedge against sampling bias when the ham sample is small and you can't see the model's vocabulary.
- `set(merged) - set(top_words)` is **set _difference_**, used here just to show which curated terms were genuinely new (12 of them).

### Step 4 - assemble it into one reusable builder

```python
def build_candidate_vocabulary(
    X_train,
    y_train,
    sample_size=500,
    max_words=100,
    min_freq=5,
    additional_candidates=None,
):
    """
    Build a candidate vocabulary for black-box discovery from ham messages.

    Parameters
    ----------
    X_train : array-like of str
        Cleaned training messages.
    y_train : array-like of str
        Labels aligned with X_train ('ham' or 'spam').
    sample_size : int, default 500
        Number of ham messages to analyze.
    max_words : int, default 100
        Maximum number of top frequent ham words to keep before merging extras.
    min_freq : int, default 5
        Minimum frequency threshold for inclusion from the ham corpus.
    additional_candidates : list[str] | None
        Optional curated conversational terms to include.

    Returns
    -------
    list[str]
        Deduplicated candidate words ordered by decreasing ham frequency,
        then lexicographically for stable ties.
    """
    word_freq = extract_ham_word_freq(X_train, y_train, sample_size=sample_size)
    top_words = select_high_frequency_words(word_freq, max_words=max_words, min_freq=min_freq)
    merged = merge_with_curated(top_words, additional_candidates=additional_candidates)

    # Stable final ordering driven by ham frequency, then lexical for ties
    def sort_key(w):
        return (-word_freq.get(w, 0), w)

    return sorted(merged, key=sort_key)

cv_example = build_candidate_vocabulary(X_train, y_train)
print("[*] Example: build_candidate_vocabulary")
print(f" Candidates: {len(cv_example)}")
print(" First 10:", ", ".join(cv_example[:10]))
```

**Output:**

![Black-Box Core Components](/assets/7%20Black-Box%20Core%20Components-20261010181949030.png)

**This just chains the three previous functions** (extract → select → merge) and re-sorts the final merged set by ham frequency with the same lexicographic tie-break. **A single reproducible helper so the discovery algorithm later uses the exact same list.** (Note `sort_key` uses `word_freq.get(w, 0)` so curated words not seen in the sample get frequency 0 and sort to the end.)

### Build the actual list

```python
# Build candidate vocabulary for discovery
candidate_words = build_candidate_vocabulary(X_train, y_train)
print(f"[+] Testing {len(candidate_words)} candidate words extracted from ham messages")
```

**Output:**

![Black-Box Core Components](/assets/7%20Black-Box%20Core%20Components-20261010182027854.png)

**`candidate_words` (111 of them) is the arsenal the discovery phase will query.**

## Query management framework

### Budget allocation

```python
def estimate_budget_allocation(total_budget):
    """
    Estimate allocation across exploration, exploitation, and combination.

    Parameters
    ----------
    total_budget : int
        Total query budget available for discovery.

    Returns
    -------
    dict
        Mapping phase -> integer number of queries that sums to `total_budget`.
    """
    explore = int(0.4 * total_budget)
    exploit = int(0.4 * total_budget)
    combine = total_budget - explore - exploit # absorb rounding
    return {
        'exploration': explore,
        'exploitation': exploit,
        'combination': combine,
    }

# Quick demo for budget allocation
allocation = estimate_budget_allocation(query_budget)
print("\n[*] Budget allocation:")
for phase, budget in allocation.items():
    print(f" {phase:12}: {budget:4d} queries")
print(f" Total: {sum(allocation.values())} / {query_budget}")
```

**Output:**

![Black-Box Core Components](/assets/7%20Black-Box%20Core%20Components-20261010182111086.png)

- **A **40-40-20 split**: 40% exploration (test new words), 40% exploitation (lean on known-good words), 20% combination (test word combos).** These are the bandit's exploration/exploitation phases from §6, plus the combination-synergy search.
- ==`combine = total_budget - explore - exploit` absorbs any rounding== so the three always sum back to exactly `total_budget` (important - you can't accidentally allocate 1001 queries).

### Prepare the discovery phase

```python
# Discovery phase - test word effectiveness
word_scores = {}
test_spam_samples = spam_test_messages[:50] # More test messages

# Test in batches to be more efficient
print(f"[*] Discovery phase: testing {len(candidate_words)} candidates...")

# Randomly sample candidates and messages for better coverage
np.random.shuffle(candidate_words)
np.random.shuffle(test_spam_samples)
```

- ==`word_scores = {}` will hold each word's measured effectiveness== (the reward estimates that feed UCB).
- **50 spam messages as the test set** - enough to average out variance (5 would be too noisy).
- **`np.random.shuffle` randomizes the order of both candidates and messages** so that **if you hit the budget limit mid-run, you haven't systematically tested only the frequency-ranked words** and never reached the curated ones. It removes ordering bias.

One framing note from the text: ==**"discovery" = exploration + combination (60%)**==; the 40-40-20 can be re-tuned - more exploitation when you just need fast evasion, more exploration when queries are plentiful.

## The facts worth locking in for the exam

- **The black-box candidate vocabulary is built **offline from ham text** (mine word frequencies → keep high-frequency ones → merge a curated conversational list), spending **zero queries** against the target** - matching the §6 "offline vocabulary construction" phase.
- **Frequency is only a weak proxy for a good word** - mining ham by frequency surfaces common-but-non-discriminative words ("you, and, the"), which is why the next phase _empirically tests_ candidates against the model rather than trusting frequency, and why a curated list is merged in to hedge sampling bias.
- **Deterministic ordering matters for reproducibility**: the sort key `(-frequency, word)` sorts by frequency descending with a lexicographic tie-break, so equal-frequency words always order the same way across runs.
- **The query budget is split 40-40-20 across exploration / exploitation / combination, always summing exactly to the total** (the last phase absorbs rounding); "discovery" = exploration + combination = 60%, and the split is tunable to the situation.
- **Candidates and test messages are shuffled before discovery so that hitting the budget early doesn't systematically skip part of the vocabulary** - it removes ordering bias.
- **Key black-box constraint throughout: every test of a word against the model costs a query**, so the whole framework exists to spend a finite budget wisely - the economic pressure that white-box attacks never face.
