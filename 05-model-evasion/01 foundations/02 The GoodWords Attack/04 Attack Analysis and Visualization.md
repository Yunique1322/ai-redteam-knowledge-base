# Attack Analysis And Visualization

**The idea in one breath:** the last section proved the attack _works_; this one dissects _why_ - **how much each individual word actually contributes**, and **how each message's spam-probability slides across the 0.5 boundary as you pile on words.** The punchline: there's no magic word; the attack wins by **accumulation**.

## Word impact analysis - measuring each word alone

Set up a sample and a results list:

```python
print("\n[*] Analyzing individual word impact...")

# Use 50 spam messages as a representative sample
sample_spam = spam_test_messages[:50]
word_impacts = []

print(f"[+] Testing {len(top_good_words[:20])} words on {len(sample_spam)} spam messages")
```

**Take 50 spam messages as a representative sample** (enough for a stable average, cheap to compute), and test the top 20 good words one at a time.

The measurement loop - HTB split this one `for` loop across two blocks (the second block is the indented tail of the same loop), so here it is as the single loop it actually is:

```python
for word, _, _, _ in top_good_words[:20]:
    total_impact = 0

    for message in sample_spam:
        # Calculate original spam probability
        vec_orig = vectorizer.transform([message])
        prob_orig = classifier.predict_proba(vec_orig)[0][1] # spam prob

        # Calculate probability after adding the word
        vec_aug = vectorizer.transform([message + " " + word])
        prob_aug = classifier.predict_proba(vec_aug)[0][1]

        # Measure the probability reduction
        impact = prob_orig - prob_aug
        total_impact += impact

    # Calculate average impact across all messages
    avg_impact = (total_impact / len(sample_spam)) * 100
    word_impacts.append((word, avg_impact))

    # Show progress for first few words
    if len(word_impacts) <= 3:
        print(f" Word '{word}': {avg_impact:.2f}% reduction")
```

The logic, plainly:

- **For each word, add _only that one word_ to each message and measure how much the spam probability drops.**
- ==`predict_proba(...)[0][1]` = the spam probability== - `[0]` picks the first (only) message's result, `[1]` picks the spam class (alphabetical: 0=ham, 1=spam). `prob_orig` is before adding the word (~0.95+ for real spam - the model is confident), `prob_aug` is after.
- ==`impact = prob_orig - prob_aug` is the drop in _percentage points_, not percent==: 0.95 -> 0.85 is a **10-point** drop, not a 10% relative drop (that would only reach 0.855). Sum over all 50 messages, divide, ×100 to average.

Sort to rank:

```python
word_impacts.sort(key=lambda x: x[1], reverse=True)
```

**Output:**

```text
[*] Analyzing individual word impact...
[+] Testing 20 words on 50 spam messages
  Word 'lor': 3.98% reduction
  Word 'ü': 3.93% reduction
  Word '...': 3.89% reduction
Most Effective Word: 'lor' (reduces spam prob by 4.0%)
```

![Attack Analysis and Visualization](/assets/5%20Attack%20Analysis%20and%20Visualization-20261010170610854.png)
Here's the conceptually important bit - the **gap between goodness score and practical impact.** `lor` topped the goodness ranking at **50.22** (50× more common in ham), yet on its own it only drops spam probability by **~7.2 points**. Why the mismatch? **Goodness measures a word's ham-likeness _in isolation_; practical impact measures its ability to move an _already spam-saturated_ message toward ham.** A word can be strongly ham-associated and still be far too weak alone to overcome a pile of strong spam signals like "FREE", "WINNER", "£".

## Word impact bar chart

```python
# Plot word impacts
fig, ax = plt.subplots(figsize=(10, 8), facecolor=NODE_BLACK)

words = [w for w, _ in word_impacts]
impacts = [i for _, i in word_impacts]
colors = [HTB_GREEN if i > 15 else NUGGET_YELLOW if i > 10 else HACKER_GREY for i in impacts]

bars = ax.barh(range(len(words)), impacts, color=colors, edgecolor='white', linewidth=0.5)
```

- **List comprehensions split the `(word, impact)` tuples into two parallel lists** for plotting.
- **The `colors` line is a conditional comprehension: green if impact > 15, yellow if > 10, else grey.** Since every word here is in the 3-4 range, **they all come out grey** - which is itself the finding.
- ==`ax.barh(...)` = horizontal bars== (better for readable word labels than vertical).

```python
ax.set_yticks(range(len(words)))
ax.set_yticklabels(words)
ax.set_xlabel('Average Spam Probability Reduction (%)', fontsize=12, color=HTB_GREEN)
ax.set_title('Individual Word Impact on Spam Detection', fontsize=14, color=HTB_GREEN, pad=20)
ax.grid(axis='x', alpha=0.2)
ax.set_facecolor(NODE_BLACK)

for spine in ax.spines.values():
    spine.set_color(HACKER_GREY)
ax.tick_params(colors=HACKER_GREY)

plt.tight_layout()
plt.savefig(output_dir / "word_impact.png", dpi=150, facecolor=NODE_BLACK)
plt.close()
print(f"[+] Plot saved to {output_dir / 'word_impact.png'}")
```
![Plot word impacts](/assets/Pasted%20image%2020261010171216.png)
This is the first chart you pasted. **Every word sits in a narrow ~3-4% band - no "silver bullet" word.** That's not a weakness in word selection; it's **the mathematical reality of attacking Naive Bayes**: training piled up a lot of probability mass on strong spam words, and no single good word can outweigh all of it. **Hence the 15-20-word threshold - the attack works by accumulating many small, consistent shifts**, not by one dominant word.

## Probability shift analysis - per-message, as words pile on

Set up the grouped-bar experiment:

```python
print("\n[*] Visualizing probability shifts...")

# Sample messages for detailed analysis
sample_messages = spam_test_messages[:8]
test_word_counts = [0, 5, 10, 20, 30]

fig, ax = plt.subplots(figsize=(14, 6), facecolor=NODE_BLACK)

x = np.arange(len(sample_messages))
width = 0.15
colors_list = [MALWARE_RED, NUGGET_YELLOW, AZURE, HTB_GREEN, AQUAMARINE]
```

**8 messages, tested at 0/5/10/20/30 words** (baseline -> initial -> threshold -> saturation -> beyond). `x = np.arange(8)` gives base x-positions 0-7; `width = 0.15` sizes each bar so five fit per message; the colour list runs red->aquamarine across the five word-counts.

The build loop - HTB split this one `for` loop across **four** blocks (the inner loop, the bar creation, and the checkmark loop are all indented inside the outer `for`), so here it is reassembled as the one loop it is:

```python
for i, num_words in enumerate(test_word_counts):
    # Select the appropriate number of good words
    selected = [w for w, _, _, _ in top_good_words[:num_words]]
    probs = []

    for msg in sample_messages:
        # Augment message with selected words
        if num_words > 0:
            aug_msg = msg + " " + " ".join(selected)
        else:
            aug_msg = msg

        # Calculate spam probability
        vec = vectorizer.transform([aug_msg])
        spam_prob = classifier.predict_proba(vec)[0][1]
        probs.append(spam_prob)

    # Create grouped bars with distinct colors
    bars = ax.bar(x + i*width, probs, width,
                   label=f'{num_words} words',
                   color=colors_list[i], alpha=0.8)

    # Mark successful evasions
    for j, (bar, prob) in enumerate(zip(bars, probs)):
        if prob < 0.5:
            ax.text(bar.get_x() + bar.get_width()/2, prob + 0.02,
                   '✓', ha='center', va='bottom', color=HTB_GREEN, fontweight='bold')
```

- **`enumerate(test_word_counts)` gives both the index `i` (0-4) and the value `num_words`** - you need `i` to place each group of bars and pick its colour.
- Inner loop: **augment each of the 8 messages with the selected words, compute its spam probability, collect into `probs`.**
- **`ax.bar(x + i*width, ...)` draws that word-count's 8 bars, shifted right by `i*width`** so the five groups sit side-by-side per message (e.g. message 3's bars land at 3.00, 3.15, 3.30, 3.45, 3.60).
- **The checkmark loop: `zip(bars, probs)` pairs each bar with its probability**, and drops a green ✓ above any bar below 0.5 (an evasion). `bar.get_x() + bar.get_width()/2` centres it; `prob + 0.02` nudges it just above the bar.

Finish the chart:

```python
ax.axhline(y=0.5, color='white', linestyle='--', alpha=0.5, label='Decision boundary')
ax.set_xlabel('Message Index', fontsize=12, color=HTB_GREEN)
ax.set_ylabel('Spam Probability', fontsize=12, color=HTB_GREEN)
ax.set_title('Probability Shift with Increasing Good Words', fontsize=14, color=HTB_GREEN, pad=20)
ax.set_xticks(x + width * 2)
ax.set_xticklabels([f'M{i+1}' for i in range(len(sample_messages))])
ax.legend(loc='upper right')
ax.grid(axis='y', alpha=0.2)
ax.set_facecolor(NODE_BLACK)
```

- ==`axhline(y=0.5)` draws the decision boundary== - above = classified spam, below = classified ham.
- `set_xticks(x + width*2)` centres each message's label under the _middle_ of its 5-bar group (the 3rd bar).

```python
for spine in ax.spines.values():
    spine.set_color(HACKER_GREY)
ax.tick_params(colors=HACKER_GREY)

plt.tight_layout()
plt.savefig(output_dir / "probability_shift.png", dpi=150, facecolor=NODE_BLACK)
plt.close()
print(f"[+] Plot saved to {output_dir / 'probability_shift.png'}")
```

Reading the second chart (a good illustration):

- ****M7 and M8 start _below_ 0.5 at 0 words** - they're natural false negatives** the classifier already misreads as ham, so the attack is unnecessary for them.
- **The other six need varying word counts to cross 0.5: M5 flips at 5 words; M2/M3/M4 at 10; M1 and M6 hold near 1.0 until 20 words force them under.**
- **The required word count tracks baseline spam strength** - strongly-spammy messages (M1, M6) need more good words to overwhelm their accumulated spam evidence, weakly-spammy ones (M5) need few. **Same mechanism as the sigmoid: each word adds a fixed `log P(w|ham)` chunk; the ✓ appears once the accumulated ham evidence tips the message past the boundary.**

## Key facts to lock in

- **Each good word's _individual_ impact is small and uniform (~2-4 percentage-point drop in spam probability); there is no "silver bullet" word**, because training accumulates heavy probability mass on strong spam features that no single word can outweigh.
- **A word's goodness score (ham-likeness in isolation) is NOT the same as its practical impact (ability to move an already-spammy message)** - `lor` scored 50 on goodness but only ~4% in impact. High goodness ≠ high standalone effect.
- **The attack therefore succeeds by _accumulation_ of many small, additive log-probability shifts**, which is exactly why reliable evasion needs ~15-20 words rather than one or two.
- **Impact is measured in percentage _points_ (absolute), not relative percent** - 0.95->0.85 is a 10-point shift; and spam probability is read as `predict_proba(...)[0][1]` (index 1 = spam class).
- **The number of good words required per message scales with that message's baseline spam strength** - weak spam crosses the 0.5 boundary with ~5 words, strong spam needs ~20 - and messages already below 0.5 are pre-existing false negatives the attack doesn't need to touch.
- **Methodology point reusable in practice: isolate a feature's effect by perturbing one thing at a time and averaging the probability shift over a sample of inputs** - that's how you separate heavy-hitter features from filler.

