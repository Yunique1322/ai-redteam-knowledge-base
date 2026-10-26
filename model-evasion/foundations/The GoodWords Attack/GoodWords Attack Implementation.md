# GoodWords Attack Implementation



**The idea in one breath:** you built the victim classifier last section; now you **read its learned word probabilities directly (white-box), rank words by how "ham-like" they are, and staple the top ones onto spam until the filter flips.** Then you measure how many words it takes.

## Extracting GoodWords (white-box)

==White-box = you can see inside the model.== Instead of guessing good words, you read the exact probabilities it learned:

```python
print("\n[*] Extracting GoodWords from model...")

# Get feature names and probabilities
feature_names = vectorizer.get_feature_names_out()
ham_log_probs = classifier.feature_log_prob_[0] # Ham class
spam_log_probs = classifier.feature_log_prob_[1] # Spam class
```

- **`vectorizer.get_feature_names_out()` gives the 3,000-word vocabulary** (the words, in feature order).
- **`classifier.feature_log_prob_` is MultinomialNB's learned `log P(word | class)`** - a table with one row per class. **`[0]` is ham, `[1]` is spam** (sklearn sorts class names alphabetically, and "ham" < "spam"). These are the `P(w|ham)` and `P(w|spam)` numbers from Section 2's math - you're reading them straight out of the model.
- They're stored in **log space** for the underflow reason you already know (multiplying 3,000 tiny probabilities would collapse to zero near 10⁻³⁰⁸; logs turn that into safe addition).

## Calculating goodness scores

```python
# Calculate goodness scores
goodness_scores = []
for i, word in enumerate(feature_names):
    ham_prob = np.exp(ham_log_probs[i])
    spam_prob = np.exp(spam_log_probs[i])
    goodness = ham_prob / (spam_prob + 1e-10)
    goodness_scores.append((word, goodness, ham_prob, spam_prob))
```

- **`np.exp(log_prob)` undoes the log** - it converts a stored log-probability back into an actual probability (e.g. `np.exp(-5.3) ≈ 0.005`). You need the real probabilities to take their ratio.
- ==`goodness = ham_prob / (spam_prob + 1e-10)` is the goodness score `S(w)` from Section 2== - "how many times more common is this word in ham than spam." **The `1e-10` (epsilon) stops a divide-by-zero** when a word never appeared in spam; without it a ham-only word gives `0.003 / 0.000` (undefined), with it you get a huge-but-finite score like ~300,000,000.
- Each entry stores **`(word, goodness, ham_prob, spam_prob)`** - keeping the raw probabilities too, because a goodness of 50 from `0.5/0.01` touches far more messages than a 50 from `0.005/0.0001`, and you may want to know which.

## Selecting the top good words

```python
# Sort by goodness
goodness_scores.sort(key=lambda x: x[1], reverse=True)
top_good_words = goodness_scores[:100]

print(f"[+] Top 10 GoodWords (most 'ham-like'):")
for word, score, hp, sp in top_good_words[:10]:
    print(f" {word:15} | goodness: {score:8.2f} | ham_p: {hp:.4f} | spam_p: {sp:.4f}")
```
**Output:**

![Sort by goodness](/assets/Pasted%20image%2020261010164813.png)

- `sort(key=lambda x: x[1], reverse=True)` **sorts by the goodness value (index 1), highest first** (`lambda x: x[1]` just says "sort on the second item of each tuple"). **Keep the top 100 as your attack arsenal.**
- Worth noticing: the top words are **SMS-specific conversational tokens - "lor" (Singlish slang), "...", "da", "later", "lol"** - **not** the generic "meeting/thanks" from the theory section. **The real good words are extracted empirically from *this* model's data**, which is exactly why white-box extraction beats guessing.

## Setting up the attack experiment

Grab only the spam from the test set (you're testing whether spam evades):

```python
print("\n[*] Testing GoodWords attack...")

# Extract only spam messages for testing
spam_test_messages = X_test[y_test == 'spam']
print(f"[+] Testing on {len(spam_test_messages)} spam messages")
```
**Output:**
```text
[*] Testing GoodWords attack...
[+] Testing on 128 spam messages
```
==`X_test[y_test == 'spam']` is boolean filtering== - keep only the test messages whose label is spam (128 of them).

Define how many good words to try at each step:

```python
# Define test points from baseline (0) to saturation (40)
word_counts = [0, 5, 10, 15, 20, 25, 30, 35, 40]
attack_results = []

print(f"[*] Testing word counts: {word_counts}")
```
**Test 0 words (baseline), then in steps of 5 up to 40.** Zero establishes the model's natural miss-rate; 40 is past the point where more words stop helping (saturation). Steps of 5 are enough resolution to see the curve's shape without wasting runs.

Here HTB shows the loop skeleton first - just the word-selection part, printing the 5 words it would use (this block is a preview, it doesn't run the attack yet):

```python
for num_words in word_counts:
    # Select the top N good words for this iteration
    selected_words = [w for w, _, _, _ in top_good_words[:num_words]]

    # Show which words we're using (first iteration only for clarity)
    if num_words == 5:
        print(f" Using words: {', '.join(selected_words)}")
```
**`[w for w, _, _, _ in top_good_words[:num_words]]` pulls just the word out of each tuple** (the `_` throw away goodness/ham_p/spam_p) for the top `num_words` entries. **This is greedy selection - always add the highest-goodness words first.** Not provably optimal (word combos could have synergy), but testing every 20-of-100 combination would be ~5.4×10²⁰ runs; greedy finishes in seconds.

## The augmentation helper

```python
def augment_message(message, words_to_add):
    """Append good words to a message"""
    if len(words_to_add) > 0:
        return message + " " + " ".join(words_to_add)
    return message

# Test augmentation on one example using the top 5 words
sample_spam = spam_test_messages[0]
sample_augmented = augment_message(
    sample_spam,
    [w for w, _, _, _ in top_good_words[:5]]
)
print(f"\nOriginal: {sample_spam[:50]}...")
print(f"Augmented: {sample_augmented[:80]}...")
```
**The whole attack, in one function: glue the good words onto the end of the message with spaces.** The spam body is untouched; the good words just ride along after it (`" ".join(words_to_add)` turns the word list into one space-separated string).

## The full attack loop

This is the complete loop (the one that actually runs the attack and produces the numbers):

```python
for num_words in word_counts:
    # Select the top N good words for this iteration
    selected_words = [w for w, _, _, _ in top_good_words[:num_words]]

    # Count how many spam messages evade after augmentation
    evaded = 0
    for message in spam_test_messages:
        # Augment the message
        augmented = augment_message(message, selected_words)

        # Transform and predict
        vec = vectorizer.transform([augmented])
        prob = classifier.predict_proba(vec)[0]

        # Check evasion: ham probability > spam probability
        if prob[0] > prob[1]:
            evaded += 1

    # Record results for this configuration
    evasion_rate = (evaded / len(spam_test_messages)) * 100
    attack_results.append({
        'num_words': num_words,
        'evasion_rate': evasion_rate,
        'evaded': evaded,
        'total': len(spam_test_messages)
    })

    print(f" Words: {num_words:2d} | Evasion: {evasion_rate:6.2f}% ({evaded}/{len(spam_test_messages)})")
```

Walk the inner logic:
- **`vectorizer.transform([augmented])` - note `transform`, not `fit_transform`**: the augmented message is converted using the *trained* vocabulary (the Section 3 rule in action). This is why your good words must already be in the vocab to count.
- **`classifier.predict_proba(vec)[0]` returns `[P(ham), P(spam)]`** for the message (`[0]` because you passed one message in a list and get a list of results back). Order is alphabetical again → index 0 = ham.
- ==`if prob[0] > prob[1]` = "did ham win?" = evasion.== Count it.
- ==`evasion_rate` = evaded / 128 × 100.==

The key conceptual point: **the decision boundary sits at 0.5, and only *which side wins* matters, not by how much.** A message at `[0.501, 0.499]` evades exactly as well as one at `[0.99, 0.01]`. **So the attacker's target is just to nudge the probability across 0.5 - not to drive it to an extreme.** That's what makes this cheap.

```python
# Convert to DataFrame for easy plotting
results_df = pd.DataFrame(attack_results)
```
**Output (from the loop above):**
```text
[*] Testing GoodWords attack...
[+] Testing on 128 spam messages
  Words: 0 | Evasion: 6.25% (8/128)
  Words: 5 | Evasion: 41.41% (53/128)
  Words: 10 | Evasion: 74.22% (95/128)
  Words: 15 | Evasion: 96.09% (123/128)
  Words: 20 | Evasion: 100.00% (128/128)
  Words: 25 | Evasion: 100.00% (128/128)
  Words: 30 | Evasion: 100.00% (128/128)
  Words: 35 | Evasion: 100.00% (128/128)
  Words: 40 | Evasion: 100.00% (128/128)
```
**The collapse: 6% → 41% (5 words) → 74% (10) → 96% (15) → 100% (20).** The baseline 6.25% (8/128) is just the model's natural false-negative rate. **By 20 words, every single spam evades.** `pd.DataFrame(attack_results)` turns the list of result dicts into a table for plotting.

## Visualizing the effectiveness curve

Three blocks build one chart (kept separate as HTB has them - each operates on the shared `ax`, run them in order). **Plot setup:**

```python
plt.style.use('dark_background')
fig, ax = plt.subplots(figsize=(12, 6), facecolor=NODE_BLACK)

ax.plot(results_df['num_words'], results_df['evasion_rate'],
        marker='o', markersize=8, linewidth=2.5,
        color=HTB_GREEN, markeredgecolor='white', markeredgewidth=1)

ax.fill_between(results_df['num_words'], 0, results_df['evasion_rate'],
                alpha=0.3, color=HTB_GREEN)
```
==`ax.plot(...)` draws the line of evasion-rate vs words; `fill_between(..., alpha=0.3)` shades under it== (30% opacity). `NODE_BLACK`, `HTB_GREEN` etc. are the HTB colour constants you imported.

**Threshold lines + peak annotation:**

```python
# Add threshold lines
ax.axhline(y=50, color=NUGGET_YELLOW, linestyle='--', alpha=0.7, label='50% threshold')
ax.axhline(y=90, color=AZURE, linestyle='--', alpha=0.7, label='90% threshold')

# Highlight maximum
max_idx = results_df['evasion_rate'].idxmax()
max_rate = results_df.loc[max_idx, 'evasion_rate']
max_words = results_df.loc[max_idx, 'num_words']
ax.scatter(max_words, max_rate, s=200, color=MALWARE_RED, zorder=5)
ax.annotate(f'Peak: {max_rate:.1f}%\n@ {max_words} words',
           xy=(max_words, max_rate), xytext=(max_words+5, max_rate-10),
           color='white', fontsize=10,
           arrowprops=dict(arrowstyle='->', color=MALWARE_RED, lw=1.5))
```
- **`axhline` draws the horizontal dashed reference lines at 50% and 90%** - the "more often fooled than not" line and the "near-total evasion" line.
- `idxmax()` finds the row with the highest evasion; `scatter(..., zorder=5)` drops a red dot on the peak (`zorder=5` keeps it on top of everything); `annotate` adds the "Peak: 100%" label with an arrow.

**Final styling + save:**

```python
ax.set_xlabel('Number of Good Words Added', fontsize=12, color=HTB_GREEN)
ax.set_ylabel('Evasion Rate (%)', fontsize=12, color=HTB_GREEN)
ax.set_title('GoodWords Attack Effectiveness', fontsize=14, color=HTB_GREEN, pad=20)
ax.grid(True, alpha=0.2)
ax.set_facecolor(NODE_BLACK)
ax.legend()

for spine in ax.spines.values():
    spine.set_color(HACKER_GREY)
ax.tick_params(colors=HACKER_GREY)

plt.tight_layout()
output_dir = Path("attachments")
output_dir.mkdir(exist_ok=True)
plt.savefig(output_dir / "attack_effectiveness.png", dpi=150, facecolor=NODE_BLACK)
plt.close()
print(f"\n[+] Plot saved to {output_dir / 'attack_effectiveness.png'}")
```

```
*(Figure not available in source notes.)*
```

![GoodWords Attack Implementation](/assets/4%20GoodWords%20Attack%20Implementation-20261010165656130.png)

Labels/title/grid styling, then **`savefig` writes the PNG to an `attachments/` folder** (created with `mkdir(exist_ok=True)`), and `plt.close()` frees the figure.

**Why the curve is an S (sigmoid)** - this is the exam-relevant bit your chart shows. In log space each good word adds a *fixed chunk* to the ham score:

> log P(ham | message) = log P(ham) + Σ log P(w | ham)

That accumulation is **linear in log space** - straight-line growth. But the classifier reports a probability between 0 and 1, and **converting a log-score back to a 0-1 probability (via the softmax/sigmoid) bends that straight line into an S-curve.** Hence the three phases visible in the plot: a **slow start** (0-5 words, still ~6-41%), a **steep collapse** (the 5→15 word zone where it rockets 41%→96% - this is the decision boundary being crossed for message after message), and a **flat saturation** (15→20 words to 100%, after which extra words do nothing because everything already evades).

## The facts worth locking in for the exam

- **In a white-box GoodWords attack you read the model's learned probabilities directly from `classifier.feature_log_prob_`** (`[0]`=ham, `[1]`=spam, because classes are sorted alphabetically), and rank words by the **goodness score `P(w|ham) / (P(w|spam) + ε)`** - the epsilon exists purely to avoid divide-by-zero for words never seen in spam.
- **The good words are extracted *empirically* from the target's own data, not guessed** - here they came out as SMS slang ("lor", "...", "da", "lol"), which is why white-box beats a generic word list.
- **The attack never changes the spam body - it only *appends* high-goodness words**, and vectorizes with `transform` (reusing the trained vocabulary), so a good word only helps if it's already in the vocabulary.
- **Evasion is decided solely at the 0.5 boundary - `P(ham) > P(spam)` - and magnitude is irrelevant**, so the attacker only needs to nudge the probability just past 0.5, not maximize it. That's why so few words are needed.
- **Effectiveness follows a sigmoid: ~6% at 0 words (natural false-negative rate), ~41% at 5, ~74% at 10, ~96% at 15, and 100% by 20**, then flat - matching the ~15-30-words / >90% figure from the theory, and demonstrating catastrophic (not gradual) failure.
- **The root cause is unchanged: additive log-probabilities plus no context modeling** - the bag-of-words classifier literally cannot notice the semantic mismatch between spam content and tacked-on legit words, so quantity of good words overwhelms the spam signal.
