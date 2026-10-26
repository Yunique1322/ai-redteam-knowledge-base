# Black-Box Attack Implementation


## The idea in one breath

The white-box attack had X-ray vision - it read `feature_log_prob_` directly. Now we go blind: no probability tables, no parameters, just a query interface. **This section proves the three-phase discovery algorithm still works - black-box reaches roughly 85-95% of white-box evasion despite never seeing the model's internals**, and it finds _different but equally effective_ words along the way.

## Step 1 - Build the vocabulary and show the budget

```python
print("\n[*] Using three-phase discovery algorithm...")

# Build candidate vocabulary
candidate_words = build_candidate_vocabulary(X_train, y_train)
print(f"[+] Built vocabulary of {len(candidate_words)} candidate words")
```

```python
# Show budget allocation
allocation = estimate_budget_allocation(query_budget)
print(f"\n[*] Budget allocation:")
for phase, budget in allocation.items():
    print(f" {phase:12}: {budget:4d} queries")
```

This rebuilds the Section 7 candidate pool (ham-mined frequent words + curated conversational terms like "thanks", "tomorrow", "gonna") - **~112 words we _guessed_ might be ham-like, which we now have to test empirically because we can't peek at the model**. Then it prints the 40-40-20 split of the 1,000-query budget.

**Output:**

```
[*] Using three-phase discovery algorithm...
[+] Built vocabulary of 112 candidate words

[*] Budget allocation:
    exploration : 400 queries
    exploitation: 400 queries
    combination : 200 queries
```

## Step 2 - Run the three-phase discovery

```python
# Run three-phase discovery
discovered_words, combination_scores, total_queries = three_phase_discovery(
    spam_test_messages[:50],
    candidate_words,
    budget=query_budget
)

print(f"\n[+] Discovery complete. Total queries used: {total_queries}/{query_budget}")
print(f"[+] Top 10 discovered words:")
for word, score in discovered_words[:10]:
    print(f" {word:10} | impact: {score:.3f}")
```

This calls the Section 9 function on the first 50 test spam messages and unpacks its three return values. **Output:**

```
[*] Phase 1: Exploration (budget: 400 queries)
  [P1 100/400] tested_words=10 | top3=need:0.166, say:0.005, thanks:0.000
  [P1 200/400] tested_words=17 | top3=happy:0.219, sleep:0.175, say:0.005
  [P1 300/400] tested_words=30 | top3=got:0.005, sleep:0.005, say:0.005
[+] Exploration complete. Queries: 400, Words tested: 40
  Top5 after exploration:
    happy | score: 0.024
    sleep | score: 0.001
    need | score: 0.001
    got | score: 0.001
    say | score: 0.001

[*] Phase 2: Exploitation (budget: 400 queries)
  [P2 mid 200/400] top3=say:0.181, yeah:0.106, later:0.096
[+] Exploitation complete. Total queries: 800

[*] Phase 3: Combination search (budget: 200 queries)
  [P3 mid ~130/200] combos_tested=65 | best=happy:0.000
  [P3 mid ~200/200] combos_tested=130 | best=later + going:0.000
[+] Combination search complete. Total queries: 1000

[+] Discovery complete. Total queries used: 1000/1000
[+] Top discovered words (sample):
    really | impact: 0.121
    got | impact: 0.117
    eat | impact: 0.115
    sleep | impact: 0.078
    later | impact: 0.068
```
![Black-Box Attack Implementation](/assets/10%20Black-Box%20Attack%20Implementation-20261010191945321.png)
### The key result: why black-box "think" (15.2%) beats white-box "lor" (7.19%)

This is the conceptual heart of the section. Black-box's top word, "really", reduces spam probability by **12.1 points** - _more_ than white-box's top word "lor" managed (**4.0 points**). How can the blind attack find a _stronger_ word? **Because the two attacks measure two different things:**

- **White-box goodness** ranks words by the ratio $\dfrac{P(w \mid ham)}{P(w \mid spam)}$. In plain English: `P(w|ham)` = "among all words in legit messages, how often is it this one," and the ratio = **how many times more _ham-leaning_ the word is, measured purely from training statistics, in isolation**. "lor" wins here because it's almost exclusively a ham word.
- **Black-box impact** measures the **_empirical_ probability shift - the actual drop in spam probability when you append the word to real spam and feed it to the classifier**.

These can disagree, and that disagreement is the exam point: **a word can be extremely ham-leaning (huge ratio) yet barely move a message's score (small impact), while a word with only moderate ham-lean can shift the score a lot** through its log-probability _weight_ or how it reinforces the message's other features. "lor" is theoretically the most ham-y word but adding it to a message does little; "really" is less lopsided but packs a bigger practical punch when appended. **High goodness ≠ high standalone impact.**

## Step 3 - Display combinations

```python
if combination_scores:
    print(f"\n[+] Top 5 word combinations:")
    top_combos = sorted(combination_scores.items(), key=lambda x: x[1], reverse=True)[:5]
    for combo, score in top_combos:
        combo_str = ', '.join(combo)
        print(f" {combo_str:30} | synergy: {score:.3f}")

# Update queries_used for compatibility
queries_used = total_queries
```

Prints the top synergistic combinations if Phase 3 found any. ==The last line, `queries_used = total_queries`, is important - and it's where a bug hides (next step).==

## Step 4 - Test the discovered words on fresh messages

> **Reassembly note:** the course split this one evaluation loop across three blocks - the outer `for num_words` setup, the inner `for msg` measurement, and the rate-recording tail. The last two can't run on their own (they're continuations of the outer loop body), so here it is as the single real loop.

```python
# Test discovered words
blackbox_results = []
test_counts = [0, 5, 10, 15, 20, 25, 30] # Test with more words

for num_words in test_counts:
    if queries_used >= query_budget:
        break

    selected = [w for w, _ in discovered_words[:num_words]]
    evaded = 0
    tested = 0

    # Test on a different subset of spam messages
    eval_messages = spam_test_messages[30:50] # Different messages from discovery

    for msg in eval_messages:
        if queries_used >= query_budget:
            break

        aug = msg if num_words == 0 else msg + " " + " ".join(selected)
        vec = vectorizer.transform([aug])
        prob = classifier.predict_proba(vec)[0][1] # spam prob
        queries_used += 1

        if prob < 0.5: # evasion threshold
            evaded += 1
        tested += 1

    if tested > 0:
        rate = (evaded / tested) * 100
        blackbox_results.append({'num_words': num_words, 'evasion_rate': rate})
        print(f" Words: {num_words:2d} | Evasion: {rate:6.2f}% | Queries total: {queries_used}")

print(f"\n[+] Black-box attack complete. Total queries: {queries_used}/{query_budget}")
```

What it's doing: for each word-count (0, 5, 10, ... 30), it augments each of 20 held-out messages (indices 30-50) with that many top discovered words, and counts how many now score **below 0.5** (`prob < 0.5` = evaded). **Each eval message costs only **one** query here, not two - there's no baseline to subtract, because "evaded" is an absolute threshold test (is it under 0.5 or not).** The rate is `evaded / tested × 100`.

### ⚠️ The bug worth flagging in a report

As written, **this evaluation loop runs zero tests.** Trace the counter: Step 3 set `queries_used = total_queries = 1000`, and `query_budget = 1000`. The very first line of the loop is `if queries_used >= query_budget: break` → `1000 >= 1000` is True → **it breaks immediately, `blackbox_results` stays empty, and nothing prints.** The teaching text and the comparison chart assume the eval ran, but the reused counter is already maxed out from discovery.

The fix is to give evaluation its _own_ counter (verification arguably shouldn't count against the discovery budget at all):

```python
eval_queries = 0
for num_words in test_counts:
    selected = [w for w, _ in discovered_words[:num_words]]
    evaded = 0
    tested = 0
    eval_messages = spam_test_messages[30:50]

    for msg in eval_messages:
        aug = msg if num_words == 0 else msg + " " + " ".join(selected)
        prob = classifier.predict_proba(vectorizer.transform([aug]))[0][1]
        eval_queries += 1
        if prob < 0.5:
            evaded += 1
        tested += 1

    rate = (evaded / tested) * 100
    blackbox_results.append({'num_words': num_words, 'evasion_rate': rate})
    print(f" Words: {num_words:2d} | Evasion: {rate:6.2f}% | Eval queries: {eval_queries}")
```

One more methodological note: the eval uses messages 30-50, which are _inside_ discovery's 0-50 range - so it's only _partially_ held out, not a clean disjoint test set. For a defensible report you'd want fully separate messages.

With a working counter, the expected progression (from the course) is **0 words ≈ 5-10% evasion, 10 words ≈ 70-75%, 15 words ≈ 90-95%, 20+ words ≈ 95-100%.**
![⚠️ The bug worth flagging in a report](/assets/Pasted%20image%2020261010192333.png)

## Step 5 - Compare with white-box

```python
print("\n" + "="*60)
print("ATTACK SUMMARY")
print("="*60)
print(f"Model Accuracy: {test_acc:.2%}")
print(f"Best White-box Evasion: {results_df['evasion_rate'].max():.1f}% @ {results_df.loc[results_df['evasion_rate'].idxmax(), 'num_words']} words")
if blackbox_results:
    bb_max = max(r['evasion_rate'] for r in blackbox_results)
    print(f"Best Black-box Evasion: {bb_max:.1f}% (with {queries_used} queries)")
print(f"Most Effective Word: '{word_impacts[0][0]}' (reduces spam prob by {word_impacts[0][1]:.1f}%)")
```

This ties the two attacks together, pulling `results_df` and `word_impacts` from the white-box section. Note `if blackbox_results:` guards the black-box line - so if the eval loop didn't run (the bug above), **this summary silently omits the black-box result** rather than erroring.

## The chart you pasted

The screenshot is this comparison plotted: **x-axis = number of good words added, y-axis = evasion rate (%)**, green solid = white-box (full access), blue dashed = black-box (labelled 840 queries). What it shows:

- **Both start low at 0 words** - ~7% white-box, ~15% black-box - these are spam messages the filter _already_ misclassifies (baseline false negatives), nothing to do with the attack yet.
- The curves **cross around 5 words**, then **white-box climbs steeper and saturates at a full 100% by ~20 words**, while **black-box lags and plateaus around 90%**.
- The gap between the two curves at any word-count is exactly the "cost of being blind" - and it's small. **Black-box tracking at ~90% of a 100% white-box ceiling is the visual proof of the "85-95% of white-box effectiveness" claim.**

(Minor note: the chart's legend says "840 queries" while this section's code uses a 1,000 budget - the figure is illustrative of a run that left some budget unspent, consistent with the eval-counter issue above.)

## Step 6 - Save results

```python
# Save results
results = {
    'white_box': attack_results,
    'black_box': blackbox_results,
    'top_good_words': [(w, float(s)) for w, s, _, _ in top_good_words[:20]],
    'word_impacts': word_impacts[:10]
}

with open(output_dir / "results.json", 'w') as f:
    json.dump(results, f, indent=2)

print(f"\n[+] Results saved to {output_dir / 'results.json'}")
print("[+] Attack demonstration complete!")
```

Dumps everything to JSON. **The one subtlety: `float(s)` converts NumPy floats to plain Python floats** - `json.dump` can't serialize NumPy's number types, so that cast prevents a `TypeError`.

## Why the attack works - the defender takeaway

The section's closing analysis is the real exam payload. **Obscuring model internals ("security through obscurity") gives limited protection, because an attacker with only query access can _empirically discover_ alternative attack vectors through systematic testing.** And the deeper reason: **the vulnerability is **architectural, not informational** - Naive Bayes combines features additively in log space, so piling on ham-leaning words always drags the total toward "ham" whether or not the attacker knows the exact probabilities.** Black-box just takes more queries to reach the same destination. The attack surface is also _broad_ - white-box's "lor" and black-box's "really" are different words that both work, so there are many viable good-word sets, not one.

## The facts worth locking in for the exam

- **Black-box discovery achieves roughly 85-95% of white-box evasion despite zero access to model internals**, proving query-only attackers are nearly as dangerous as full-knowledge ones - the gap you see on the comparison chart (black-box plateauing ~90% vs white-box 100%) is small.
    
- ==White-box "goodness" = the ratio P(w|ham)/P(w|spam) (theoretical ham-lean from training stats), while black-box "impact" = the measured drop in spam probability when the word is appended to real spam - and these two metrics can disagree.== That's why black-box's "really" (12.1% impact) outperformed white-box's "lor" (4.0%): high goodness ≠ high standalone impact.
    
- **Evasion is an absolute threshold test - a message "evades" when `predict_proba(...)[0][1] < 0.5` - so each evaluation costs only ONE query (no baseline needed), unlike the two-query impact tests during discovery.**
    
- **The attack works because the vulnerability is architectural, not informational: Naive Bayes sums feature log-probabilities, so adding ham-leaning words always pulls the total toward "ham" regardless of whether the attacker knows the exact parameters** - which is why hiding model internals ("security through obscurity") offers limited defense.
    
- **The attack surface is broad, not narrow - white-box and black-box selected entirely different winning words yet both reached high evasion**, meaning many distinct good-word sets can defeat the filter.
    
- Critical-reading facts about this implementation: **the evaluation loop reuses `queries_used` after it was set to the full 1,000-query total, so as written it breaks immediately and runs no tests** (fix: give evaluation its own counter); and the eval messages (30-50) sit inside discovery's range (0-50), so the "held-out" set only partially is.
    
- Methodology point reusable on the exam: **when reporting an evasion attack, plot evasion rate vs. number of injected words for both threat models on one axis** - the baseline at 0 words shows pre-existing false negatives, the crossover and plateau show each attack's efficiency, and the vertical gap quantifies the real-world cost of losing white-box access.
    

Paste Section 11 when ready.