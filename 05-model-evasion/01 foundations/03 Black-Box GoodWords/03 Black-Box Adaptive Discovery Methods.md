# Black-Box Adaptive Discovery Methods

## The idea in one breath

Last section handed you a pile of candidate words and a fixed query budget. If you tested every word the same number of times, you'd waste most of your budget re-confirming duds. **This section makes the search _smart_: treat it like a gambler with limited coins at a row of slot machines - mostly play the words that have already paid off, but every so often gamble a coin on an untested one so you don't overlook a hidden jackpot.** Then it goes one step further and hunts for word _pairs and triplets_ that work better together than they do apart.

Two ideas drive the whole section:

1. **Adaptive selection** - the ε-greedy algorithm, which decides _which word to test next_ based on what you've learned so far.
2. **Synergy search** - combinatorial testing, which finds combinations that are **super-additive** (the whole beats the sum of the parts).

---

## Part 1 - Initializing the scorer

```python
def initialize_adaptive_scorer():
    """Initialize adaptive scoring data structures"""
    return {
        'word_scores': {}, # Maps word -> effectiveness score
        'word_counts': {}, # Maps word -> number of times tested
        'exploration_rate': 0.2 # 20% exploration for discovery phase
    }
```

This just hands back a dictionary with three slots - think of it as the attacker's **notebook** that gets updated as the attack runs. The two inner dictionaries (`{}` means "empty dictionary, to be filled later") track two _different_ things, and keeping them separate is the point:

- `word_scores` - **how good** each word is. If appending "thanks" drops spam probability by 0.15 on average, then `word_scores['thanks'] = 0.15`.
- `word_counts` - **how reliable** that score is, i.e. how many times you've tested the word. **A word tested once might have gotten lucky; a word tested 50 times has a trustworthy average.** Score tells you _how good_; count tells you _how sure_.

The `exploration_rate` of **0.2** is the ε ("epsilon") in epsilon-greedy - it means 20% of your picks will be gambles on new/rare words, and the other 80% will exploit proven winners.

---

## Part 2 - Epsilon-greedy word selection

> **Reassembly note:** the source split this one function across two code blocks - the `if` (exploration) branch in one block and the `else:` (exploitation) branch in the next. A bare `else:` can't run on its own, so I've put them back together into the single real function below. Nothing else changed.

```python
def epsilon_greedy_select(scorer, available_words):
    """Select word using epsilon-greedy strategy

    Parameters:
        scorer (dict): Adaptive scorer state
        available_words (list): Candidate words to choose from

    Returns:
        str: Selected word for testing
    """
    import random

    if random.random() < scorer['exploration_rate']:
        # Exploration: try untested or rarely tested words
        untested = [w for w in available_words if w not in scorer['word_counts']]
        if untested:
            return random.choice(untested)
        else:
            # Choose least tested word
            return min(available_words,
                      key=lambda w: scorer['word_counts'].get(w, 0))
    else:
        # Exploitation: choose best performing word
        return max(available_words,
                  key=lambda w: scorer['word_scores'].get(w, 0))
```

### The concept: the multi-armed bandit

This is a textbook solution to the **exploration - exploitation tradeoff**, which comes from **multi-armed bandit** theory. The name: a slot machine is a "one-armed bandit"; imagine a whole row of them, each paying out at an unknown rate, and you have limited coins. Every pull forces a choice - ****exploit** (pull the arm that's paid best so far) or **explore** (try an arm you barely know, in case it's secretly better)**. Too much exploiting and you might ride a mediocre machine forever; too much exploring and you waste coins on losers. Your words are the arms; your query budget is the coins.

### How the code makes that choice

`random.random()` produces a random decimal between 0.0 and 1.0 - think of it as spinning a dial. `if random.random() < 0.2` is true roughly 20% of the time, so:

**The 20% exploration branch.** First it builds `untested` - a list of words not yet in `word_counts` (the list comprehension `[w for w in available_words if w not in scorer['word_counts']]` reads as "every word `w` that we haven't tested yet"). If any untested words exist, `random.choice(untested)` picks one at random - every unknown word gets a fair shot. If everything's been tried at least once, it falls to:

```python
return min(available_words, key=lambda w: scorer['word_counts'].get(w, 0))
```

`min(..., key=...)` means "give me the word with the smallest value of this measurement." The measurement is `scorer['word_counts'].get(w, 0)` - the word's test count. (`.get(w, 0)` is a safe dictionary lookup: "return this word's count, or **0** if it isn't in the dictionary yet," which avoids a crash.) So it picks the **least-tested** word - if "meeting" was tried 3 times and "tomorrow" 15, it revisits "meeting." **Exploration isn't just random flailing; it deliberately shores up your shakiest knowledge.**

**The 80% exploitation branch.** `max(..., key=lambda w: scorer['word_scores'].get(w, 0))` is the mirror image - "give me the word with the _highest_ effectiveness score" (untested words default to 0, so they lose here). If "got" scores 0.22 and "like" scores 0.12, it grabs "got." **This is what makes most of your queries actually _advance the attack_ instead of just gathering trivia.**

(The `lambda w: ...` is just a throwaway one-line function meaning "for a given word `w`, look up this number" - it tells `min`/`max` _what_ to compare the words on.)

### The regret aside - vocabulary worth knowing

The course then warns that fixed-ε is simple but not optimal, using some formal terms. In plain English:

**Regret** = how much worse you did than a magic oracle that picked the perfect word every single time from the start. It's the gap between your actual total payoff and the ideal one. Lower regret = smarter search.

- ****Linear regret** (bad): regret keeps growing in proportion to the number of queries `T`.** A _fixed_ 20% exploration rate means that even late in the attack - when you already know the winners - you still throw away 1-in-5 queries on random gambles. Those late wasted gambles pile up forever, so your regret grows in a straight line with `T`.
- ****Logarithmic regret, written `O(log T)`** (good): regret grows _much_ slower than `T`** - like a curve that flattens out. `log T` means "as you do 10× more queries, the cost only creeps up by a constant step." You make fewer and fewer mistakes over time.

Getting that `O(log T)` guarantee needs something fancier than fixed-ε:

- **UCB (Upper Confidence Bound)** - an algorithm that's "optimistic about uncertainty": it tries words that are _either_ high-scoring _or_ under-tested, and provably achieves log regret - but only **under i.i.d., bounded-reward assumptions** (i.i.d. = each test's result is drawn _independently_ from the _same_ distribution; bounded = rewards stay within a fixed range). Decode `O(...)`: it's "big-O," a notation for _how fast something grows_, not an exact count.
- **Annealed (decaying) ε** - start with lots of exploration, then shrink ε as you learn, so late queries stop wasting coins.

The section deliberately sticks with fixed-ε "for clarity" and files UCB / decaying-ε under "advanced options." **You don't need to implement them - you need to be able to _say why fixed-ε is good enough in practice but not theoretically optimal._**

---

## Part 3 - Score updates with an exponential moving average

```python
def update_word_score(scorer, word, impact, alpha=0.3):
    """Update word score using exponential moving average

    Parameters:
        scorer (dict): Adaptive scorer state
        word (str): Word being scored
        impact (float): Observed reduction in spam probability
        alpha (float): Learning rate
    """
    if word not in scorer['word_scores']:
        scorer['word_scores'][word] = impact
        scorer['word_counts'][word] = 1
    else:
        # Exponential moving average
        old_score = scorer['word_scores'][word]
        scorer['word_scores'][word] = (1 - alpha) * old_score + alpha * impact
        scorer['word_counts'][word] += 1
```

### The concept: a belief that updates gently

After each test you learn a new `impact` (how much that word dropped spam probability _this time_). You don't want to overwrite your whole opinion based on one noisy measurement, but you don't want to ignore new evidence either. The fix is an ****exponential moving average (EMA)** - a running average that leans on history but nudges toward each new result.** Think of it as a **reputation**: one good day doesn't instantly make someone trustworthy, but it moves the needle a little.

### The math, decoded

The first time you see a word, there's nothing to average against, so its score just _becomes_ the observed impact, and its count starts at 1.

Every time after that, the key line is:

```python
new_score = (1 - alpha) * old_score + alpha * impact
```

`alpha` (the **learning rate**) is 0.3. Read the formula as a weighted blend: ==keep **70%** of what you already believed (`1 - alpha = 0.7`), mix in **30%** of the new observation (`alpha = 0.3`).==

Worked example from the course: "thats" currently scores 0.10, and a fresh test shows impact 0.15.

```
new_score = (0.7 × 0.10) + (0.3 × 0.15)
          = 0.070 + 0.045
          = 0.115
```

The score inches from 0.10 toward 0.15 - it moved, but didn't lurch. **A big `alpha` would make the score jumpy and overreact to flukes; a tiny `alpha` would make it stubborn and slow to notice a word going stale. 0.3 is a middle setting.** (It's called _exponential_ because each older observation's influence keeps getting multiplied by 0.7 again and again, fading away exponentially - recent tests matter most, ancient ones almost vanish.) And `word_counts[word] += 1` just ticks the reliability counter up by one.

---

## Part 4 - Combinatorial search for synergies

> **Reassembly note:** the source split this single function across _four_ consecutive code blocks (setup, then singles, then pairs, then triplets) - none of the later pieces can run on their own, since they're all indented continuations of the same function body ending in `return combination_scores`. I've reassembled them into the one real function below.

```python
def discover_word_combinations(message, test_words, max_size=3):
    """Discover effective word combinations through systematic search

    Parameters:
        message (str): Target spam message
        test_words (list): Promising words to test
        max_size (int): Maximum combination size

    Returns:
        dict: Mapping of word combinations to effectiveness scores
    """
    from itertools import combinations

    combination_scores = {}
    message_vec = vectorizer.transform([message])
    message_score = classifier.predict_proba(message_vec)[0][1]

    # Test individual words first
    for word in test_words[:20]:
        test_message = message + " " + word
        test_vec = vectorizer.transform([test_message])
        score = classifier.predict_proba(test_vec)[0][1]
        impact = message_score - score
        combination_scores[(word,)] = impact

    # Test pairs for synergy
    if max_size >= 2:
        for word1, word2 in combinations(test_words[:15], 2):
            test_message = message + " " + word1 + " " + word2
            test_vec = vectorizer.transform([test_message])
            score = classifier.predict_proba(test_vec)[0][1]

            # Calculate synergy
            individual_impact = combination_scores.get((word1,), 0) + combination_scores.get((word2,), 0)
            actual_impact = message_score - score
            synergy = actual_impact - individual_impact

            if synergy > 0: # Positive synergy detected
                combination_scores[(word1, word2)] = actual_impact

    # Test triplets for top pairs
    if max_size >= 3:
        top_pairs = sorted(
            [(k, v) for k, v in combination_scores.items() if len(k) == 2],
            key=lambda x: x[1], reverse=True
        )[:5]

        for pair, pair_score in top_pairs:
            for word in test_words[:10]:
                if word not in pair:
                    triplet = tuple(sorted(pair + (word,)))
                    test_message = message + " " + " ".join(triplet)
                    test_vec = vectorizer.transform([test_message])
                    score = classifier.predict_proba(test_vec)[0][1]
                    combination_scores[triplet] = message_score - score

    return combination_scores
```

### The concept: 1 + 1 can equal 2.2

Single words each nudge the classifier a bit. But some words _reinforce each other_ - "meeting" and "tomorrow" together scream "legitimate scheduling message" far louder than either does alone. **That extra boost beyond the sum of the individual effects is **synergy** (a super-additive effect), and this function hunts for it systematically.** Note it leans on the global `vectorizer` and `classifier` from your prerequisite script - `predict_proba(...)[0][1]` pulls the **spam** probability (class index 1; index 0 is ham), exactly as in earlier sections.

### Setup and baseline

`from itertools import combinations` loads Python's tool for generating every k-sized subset of a list - e.g. `combinations(['a','b','c'], 2)` yields `('a','b'), ('a','c'), ('b','c')`. Then `message_score` records the spam probability of the _untouched_ message (typically 0.90-0.99 for real spam) - your starting line that everything is measured against.

### Step 1: individual words (the baseline for synergy)

It first tests the top 20 words one at a time, storing each result as `combination_scores[(word,)] = impact`. **That `(word,)` - with the trailing comma - is a **single-element tuple**, not just parentheses.** It's deliberate: by storing singles, pairs, and triplets all as tuples, the dictionary holds every combination in one uniform format you can compare later. `impact = message_score - score` is just "how far did spam probability fall" - bigger is better.

### Step 2: pairs, and the synergy calculation

`combinations(test_words[:15], 2)` walks through every pair from the top 15 words - that's **105 pairs** (written "15 choose 2"). For each pair it measures the real combined impact, then checks whether the pair _outperformed expectations_:

```python
individual_impact = combination_scores.get((word1,), 0) + combination_scores.get((word2,), 0)
actual_impact = message_score - score
synergy = actual_impact - individual_impact
```

Read it as: ==**synergy = what the pair actually did − what you'd expect if the two words just added up independently.**== Worked example from the course: "got" alone drops spam by 0.22, "like" alone by 0.12, so independence predicts 0.34 together. If the pair actually delivers 0.38, then `synergy = 0.38 − 0.34 = 0.04` - a positive surprise, so the pair is worth keeping (`if synergy > 0`). Only genuinely cooperative pairs get stored.

### Step 3: triplets, but only the promising ones

Here's where combinatorics would bite you. The course notes that 20 words generate 20 singles, **190 pairs, and 1140 triplets** if you tested _everything_ - a combinatorial explosion that would blow your query budget instantly. **So the triplet search is deliberately greedy: it only builds on the **5 best pairs**, and for each tries adding only the **top 10** single words** - at most 50 triplet tests instead of over a thousand.

Mechanically: `top_pairs = sorted([...], key=lambda x: x[1], reverse=True)[:5]` filters `combination_scores` down to the two-word entries (`if len(k) == 2`), sorts them by score highest-first (`reverse=True`), and keeps the top 5 (`[:5]`). Then for each strong pair it tries each promising word, with two guards:

- `if word not in pair` - skips adding a word the pair already contains (no "got + like + got").
- `triplet = tuple(sorted(pair + (word,)))` - **sorts the three words before using them as the key** so that ("meeting","tomorrow","thanks") and ("thanks","meeting","tomorrow") land in the _same_ dictionary slot instead of being stored twice.

The payoff the course promises: triplets like ("meeting", "tomorrow", "thanks") can cut spam probability by 40-50% - far more than their three individual effects summed, because together they paint a complete, convincingly legitimate scene.

### The through-line of the whole function

Notice the escalating discipline on _how many_ candidates each stage considers: **20** singles -> top **15** for pairs -> top **5** pairs × top **10** words for triplets. **Every stage narrows to the current winners before spending queries on the next, more expensive stage - that funnel is exactly how you search a combinatorial space on a tight query budget without going bankrupt.**

---

### Key facts to lock in

- **Epsilon-greedy spends a finite query budget by choosing exploration vs exploitation on a fixed probability ε** - with probability ε (0.2) it tries a new or rarely-tested word, and the other 80% of the time it reuses the current best-scoring word, so most queries advance the attack while a steady trickle still hunts for hidden winners.
- **The scorer deliberately tracks two different things - `word_scores` (how effective a word is) and `word_counts` (how many times it's been tested)** - because a word tested once could have gotten lucky, while one tested 50 times has a trustworthy average; score says how good, count says how sure, and exploration revisits the _least-tested_ word rather than a purely random one.
- ==Word scores update with an exponential moving average: `new = (1−α)·old + α·impact`, keeping 70% of the old belief and 30% of the new evidence at α=0.3== - a high α overreacts to flukes while a low α is slow to adapt, so α is the responsiveness-versus-stability knob, and a word's very first observation just becomes its score outright.
- **Fixed ε gives no logarithmic-regret guarantee and incurs linear regret in the worst case**, because the exploration rate never shrinks and so keeps wasting queries on random gambles even after the winners are known; regret is the gap between your payoff and an always-perfect picker's, and reaching `O(log T)` regret needs UCB (under i.i.d., bounded-reward assumptions) or a decaying/annealed ε.
- ==Synergy = actual combined impact − the sum of the words' individual impacts, and a positive value means the combination is super-additive== - semantically related words that paint one legitimate scene (meeting + tomorrow + thanks) reinforce each other and can cut spam probability 40-50%, far beyond what they do separately.
- **The combination search beats combinatorial explosion with a narrowing funnel - 20 words tested individually -> top 15 tested as pairs -> top 5 pairs × top 10 words as triplets** - because testing every triplet of 20 words (1,140 of them) would exhaust the budget outright, so each stage only extends the previous stage's proven winners.
- Methodology point reusable in practice: **store every combination as a sorted tuple (`tuple(sorted(...))`) and each single word as a one-element tuple `(word,)`** so that different orderings of the same words collapse to one dictionary key and all combination sizes share a uniform, comparable format - a clean way to dedupe and rank the results of any combinatorial search.