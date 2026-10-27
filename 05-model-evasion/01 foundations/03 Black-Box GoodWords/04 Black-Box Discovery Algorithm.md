# Black-Box Discovery Algorithm

This section assembles everything into one function, `three_phase_discovery`, that runs the complete black-box attack on a fixed budget. It spends its 1,000 queries in three deliberate stages - ****explore** widely to find promising words, **exploit** to measure the best ones precisely, then **combine** to find pairs/triplets that beat their parts** - with the 40-40-20 split deciding how many queries each stage gets.

## The information-gain math - why three phases

The design is justified with one formula from information theory:

$$I(Q) = H(W) - H(W \mid Q)$$

Translating every symbol:

- **`H`** = **entropy** = ==**"how much you don't know."** High entropy = total fog (any outcome equally likely); low entropy = you're fairly sure.==
- **`W`** = word effectiveness (what you're uncertain about). **`Q`** = a query (one test).
- **`H(W)`** = your uncertainty **before** the query. **`H(W | Q)`** = uncertainty **after** seeing the result (the `|Q` = "given the query outcome").
- **`I(Q)`** = **information gain** = the drop in uncertainty the query bought you.

So in plain English: ==**information gain = how foggy you were before − how foggy you are after.** A query is worth exactly the amount of fog it clears.==

The analogy: 100 unknown words = maximum fog. Test one, see it drops spam probability by 0.15 - now _that_ word's fog collapses while 99 stay unknown; overall fog dips. Since queries are scarce, you spend each where it clears the most fog - which is what the three phases target:

- ****Exploration** cuts uncertainty across the _whole_ vocabulary** (wide net).
- ****Exploitation** cuts the _remaining_ uncertainty about the best words** (measure how well, precisely).
- ****Combination** cuts uncertainty you can't reach any other way** (synergies single tests can't predict).

The course calls optimizing this under a fixed budget the **budgeted information maximization problem**; the 40-40-20 phased split is a practical _approximation_ to its ideal solution - breadth (test many) vs depth (measure the best accurately).

## The complete function - paste this into one cell

The course builds this across ~9 blocks, and it teaches the Phase-1 loop body as four separate fragments (select -> measure -> update -> milestone) before giving the assembled loop. **It's all **one function**, so those pieces can't run on their own - here it is reassembled into the single real block you actually run** (I also fixed one copy-paste artifact: the Phase-1 header `print` must use `\n`, not a literal line break, or it's a syntax error).

```python
def three_phase_discovery(spam_messages, candidate_words, budget=1000):
    """Three-phase discovery: exploration, exploitation, combination

    Parameters:
        spam_messages (list): Target spam messages
        candidate_words (list): Vocabulary to test
        budget (int): Total query budget

    Returns:
        tuple: (discovered_words, combination_scores, queries_used)
    """
    scorer = initialize_adaptive_scorer()
    queries_used = 0

    # Allocate budgets using 40-40-20 split strategy
    allocation = estimate_budget_allocation(budget)
    exploration_budget = allocation['exploration']
    exploitation_budget = allocation['exploitation']
    combination_budget = allocation['combination']

    # Phase 1: Broad exploration (allocated budget)
    print(f"\n[*] Phase 1: Exploration (budget: {exploration_budget} queries)")

    p1_marks = {
        max(1, int(0.25 * exploration_budget)),
        max(1, int(0.50 * exploration_budget)),
        max(1, int(0.75 * exploration_budget)),
    }
    p1_reported = set()

    while queries_used < exploration_budget and len(candidate_words) > 0:
        # Select inputs
        test_message = random.choice(spam_messages)
        word = epsilon_greedy_select(scorer, candidate_words)

        # Measure impact with two queries
        vec_orig = vectorizer.transform([test_message])
        prob_orig = classifier.predict_proba(vec_orig)[0][1]
        vec_aug = vectorizer.transform([test_message + " " + word])
        prob_aug = classifier.predict_proba(vec_aug)[0][1]
        impact = prob_orig - prob_aug

        # Update score and account for budget
        update_word_score(scorer, word, impact)
        queries_used += 2

        # Milestone report
        if queries_used in p1_marks and queries_used not in p1_reported:
            top3 = sorted(scorer['word_scores'].items(), key=lambda x: x[1], reverse=True)[:3]
            print(
                f" [P1 {queries_used}/{exploration_budget}] "
                f"tested_words={len(scorer['word_scores'])} | "
                f"top3=" + ", ".join(f"{w}:{s:.3f}" for w, s in top3)
            )
            p1_reported.add(queries_used)

    print(f"[+] Exploration complete. Queries: {queries_used}, Words tested: {len(scorer['word_scores'])}")
    top5 = sorted(scorer['word_scores'].items(), key=lambda x: x[1], reverse=True)[:5]
    if top5:
        print(" Top5 after exploration:")
        for w, s in top5:
            print(f" {w:12} | score: {s:.3f}")

    # Phase 2: Focused exploitation
    scorer['exploration_rate'] = 0.1 # Reduce exploration

    # Get top words for exploitation
    top_words = sorted(scorer['word_scores'].items(), key=lambda x: x[1], reverse=True)[:30]
    top_word_list = [w for w, _ in top_words]

    print(f"\n[*] Phase 2: Exploitation (budget: {exploitation_budget} queries)")
    initial_queries = queries_used
    p2_mid = initial_queries + max(1, exploitation_budget // 2)

    while queries_used < initial_queries + exploitation_budget and len(top_word_list) > 0:
        test_message = random.choice(spam_messages[:20]) # Focus on fewer messages
        word = random.choice(top_word_list[:15]) # Focus on best words

        vec_orig = vectorizer.transform([test_message])
        prob_orig = classifier.predict_proba(vec_orig)[0][1]

        vec_aug = vectorizer.transform([test_message + " " + word])
        prob_aug = classifier.predict_proba(vec_aug)[0][1]

        impact = prob_orig - prob_aug
        update_word_score(scorer, word, impact)
        queries_used += 2

    print(f"[+] Exploitation complete. Total queries: {queries_used}")

    # Phase 3: Combination discovery (allocated budget)
    remaining_combo = combination_budget
    print(f"\n[*] Phase 3: Combination search (budget: {remaining_combo} queries)")

    best_combinations = {}
    combos_tested = 0

    if remaining_combo > 50: # Need minimum queries for combinations
        for i in range(min(3, len(spam_messages))):
            if queries_used >= budget or remaining_combo <= 0:
                break

            test_msg = spam_messages[i]
            combos = discover_word_combinations(test_msg, top_word_list[:20], max_size=3)

            # Track best combinations across messages
            for combo, score in combos.items():
                if combo not in best_combinations or score > best_combinations[combo]:
                    best_combinations[combo] = score

            # Account for queries (~2 per combination) while respecting the budget
            to_add = min(remaining_combo, len(combos) * 2)
            queries_used += to_add
            remaining_combo -= to_add
            combos_tested += len(combos)

            # Midpoint snapshot
            if combination_budget > 0 and remaining_combo <= combination_budget // 2 and best_combinations:
                best = max(best_combinations.items(), key=lambda x: x[1])
                print(
                    f" [P3 mid ~{combination_budget - remaining_combo}/{combination_budget}] "
                    f"combos_tested={combos_tested} | best={' + '.join(best[0])}:{best[1]:.3f}"
                )

            if remaining_combo <= 0:
                break

    print(f"[+] Combination search complete. Total queries: {queries_used}")

    # Return final results
    final_words = sorted(scorer['word_scores'].items(), key=lambda x: x[1], reverse=True)
    return final_words, best_combinations, queries_used
```

And to actually run it, in a **separate** cell:

```python
final_words, best_combinations, queries_used = three_phase_discovery(
    spam_test_messages, candidate_words, budget=1000
)
```

Now the walk-through, phase by phase.

### Setup

The top of the function takes the **spam messages** to attack, the **candidate vocabulary** (Section 7's ham-mined list), and a **budget** of 1,000. It builds a fresh `scorer` (the notebook of word scores/counts with ε=0.2, from Section 8), zeroes `queries_used`, and asks `estimate_budget_allocation` to carve the 1,000 into **400 exploration / 400 exploitation / 200 combination**.

### Phase 1 - Broad exploration

`p1_marks` is a set of the query counts at 25/50/75% of the 400 budget - `{100, 200, 300}` - just for progress printing; `p1_reported` remembers which already printed so **each milestone fires once** (subtle gotcha: `queries_used` steps by 2, so a milestone only prints if its mark is even and landed on exactly - 100/200/300 are even, so fine).

Inside the `while` loop is the heart of the attack:

- **Select** - a random spam message, and `epsilon_greedy_select` picks the word (80% best-so-far, 20% explore).
- **Measure with two queries** - ask the model the spam probability of the original message (`prob_orig`), then of the message with the word appended (`prob_aug`); ==`impact = prob_orig - prob_aug` is how far the word pushed the message toward "ham."== `predict_proba(...)[0][1]` reads the spam-class probability (index 1).
- **Update and account** - fold the impact into the word's running EMA, and `queries_used += 2` **because each test cost two model calls**.

It samples across _all_ messages and _all_ words - deliberately wide.

**Output:**

![Black-Box Discovery Algorithm](/assets/9%20Black-Box%20Discovery%20Algorithm-20261010190900529.png)

400 queries tested ~40 unique words (≈200 tests, but ε-greedy retests some). Note the scores are _tiny_ - **this is the black-box world biting as warned: individual words have small, noisy impact, and the EMA keeps dragging scores toward the near-zero impacts most words actually produce**

### Phase 2 - Focused exploitation

The net tightens: it **drops ε to 0.1, keeps the top 30 words, then hammers the top 15 words on just the first 20 messages**. Why narrow? The goal flipped from _discovery_ to _precision_ - repeating a small promising set drives each EMA toward a **reliable** average instead of spreading queries thin. That's the "measure the best accurately" half of the tradeoff.

Two honesty notes: `exploration_rate = 0.1` is set but the Phase-2 loop uses `random.choice`, not `epsilon_greedy_select`, so that 0.1 is effectively **dead code** here; and `p2_mid` is computed but never used, so the `[P2 mid ...]` line in the sample output wouldn't actually print from this code.

**Output:**

![Black-Box Discovery Algorithm](/assets/9%20Black-Box%20Discovery%20Algorithm-20261010191010483.png)

Phase 2 spends exactly 400, bringing the total to **800 of 1,000**.

### Phase 3 - Combination discovery

Only runs if **≥50 queries remain**. For up to **3** messages, it calls `discover_word_combinations` (Section 8) on the top 20 words to score singles/pairs/triplets, keeping only the _best_ score each combination ever hit:

```python
if combo not in best_combinations or score > best_combinations[combo]:
    best_combinations[combo] = score
```

**This favors combinations that work on more than one message, not a single lucky hit.** Honesty note in general: the query accounting here is an _approximation_ - `to_add = min(remaining_combo, len(combos) * 2)` just _estimates_ 2 queries per combination rather than counting the real `predict_proba` calls.

**Output:**
![Black-Box Discovery Algorithm](/assets/9%20Black-Box%20Discovery%20Algorithm-20261010191119494.png)

Best combos ~0.000 on this run - negligible synergies, consistent with the small individual impacts. Total lands exactly on **1,000, budget fully spent**.

### Final results

The function returns three things so the actual attack (next section) can use them: **the full word ranking (`final_words`), the synergistic groups (`best_combinations`), and `queries_used` to prove you stayed in budget**. It returns _all_ scored words sorted, not just the top few.

## Key facts to lock in

- **The whole black-box attack is one budgeted loop split into three phases - exploration, exploitation, combination - allocated 40-40-20 (400 / 400 / 200 of a 1,000-query budget)**, where exploration casts a wide net, exploitation refines the best words' estimates, and combination hunts synergies.
    
- ==Information gain `I(Q) = H(W) − H(W|Q)` means "uncertainty before the query minus uncertainty after," with `H` = entropy = how much you don't know== - so a query is worth exactly the fog it clears, and the three phases are a budgeted strategy to maximize total fog cleared.
    
- ==Every single-word test costs TWO queries (baseline + augmented message), and `impact = prob_orig − prob_aug` is how far appending the word pushed spam probability down== - hence `queries_used += 2` per test, so 400 queries ≈ ~200 tests covering ~40 unique words.
    
- **Phase 2 narrows deliberately - top 30 words kept, then top 15 words tested on only the first 20 messages - trading breadth for precision** (reducing variance on the words that matter), the "measure the best accurately" half of the budgeted-information tradeoff.
    
- **Phase 3 keeps only the strongest score each combination reaches across up to 3 messages (`score > best_combinations[combo]`), favoring combinations that generalize over ones that got lucky once.**
    
- Methodology point reusable in practice: **this is the canonical black-box discovery template - ε-greedy selection + EMA running estimates + phased budget allocation + multi-message validation - and it transfers to attacking any query-limited model, not just this spam filter.**
    
- Critical-reading facts about this implementation: **Phase 3's budget accounting is an approximation (`len(combos) * 2`), not a true query count**, and Phase 2's `exploration_rate = 0.1` and `p2_mid` are effectively dead code - the kind of detail worth flagging in a report.
    

