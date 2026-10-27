# The GoodWords Attack

## The idea in one breath

The GoodWords attack (from a 2005 paper by Lowd and Meek) does something almost cheeky: instead of trying to _hide_ the spam, you leave the spam exactly as it is and **staple a bunch of innocent, legitimate-sounding words onto the end of it.** Enough of those, and the spam filter changes its mind and waves the message through as legitimate. You never disguise the "BUY CHEAP PILLS NOW" part - you just drown it in "meeting," "thanks," "tomorrow," "project," until the filter's math tips the other way.

Why that works at all is the whole lesson, so let's build up to it.

## How a Naive Bayes spam filter actually thinks

Before the attack, understand the target. A Naive Bayes filter has already read thousands of emails, some labeled spam, some labeled ham (legit). From those, it learned one simple thing for every word: **how often that word shows up in each pile.** "Meeting" is common in ham, rare in spam. "Free" is common in spam, rare in ham. That's its entire education.

When a new email arrives, the filter asks two questions: _"How ham-like are these words?"_ and _"How spam-like are these words?"_ Whichever total is higher wins. That's the verdict.

The notation for "how often a word shows up in a pile" is `P(w|ham)` - read it as "probability of word _w_ given ham," but mentally translate it to **"out of all words in legit email, how common is this one."** Same for `P(w|spam)`. Don't let the symbols intimidate you; they're frequencies.

The formal name for the underlying rule is **Bayes' theorem**:

P(C|D) = P(D|C) · P(C) / P(D)

In spam terms: "the probability this message D belongs to class C (spam or ham)." The filter computes this for both classes and picks the winner - that's the **decision rule**, `argmax` (just a fancy word for "whichever class scores highest").

## The "naive" assumption - and why logs show up

Here's the "naive" part. To score a whole message, Naive Bayes **multiplies together** each word's frequency, treating every word as if it's completely independent of the others:

P(D|C) = P(w₁|C) × P(w₂|C) × P(w₃|C) × ...

It's "naive" because real language isn't like that - "new" and "york" go together - but the filter pretends each word stands alone. That assumption makes it fast, and, as you'll see, hackable.

Now a practical snag: each frequency is a tiny number (like 0.003), and multiplying fifty tiny numbers gives something so microscopically small the computer can't store it - it collapses to zero and the math dies. This is called **underflow**.

The fix is a math trick: **logarithms turn multiplication into addition.** There's a rule that `log(a × b) = log(a) + log(b)`. So instead of multiplying fifty tiny numbers, the filter takes the log of each and **adds them up**. Same ranking, no underflow. So the score becomes:

`score = log P(class) + [ log P(w₁|c) + log P(w₂|c) + ... ]`

**That's the only reason logs are here** - a computational convenience. Every "log P(...)" is just _one word's contribution to the running total_, and the total is now a **sum**.

And that sum is the crack in the armor. Because the score is just "add up each word's contribution," **adding a new word simply adds its contribution to the pile** - nothing re-reads the message, nothing notices it stopped making sense. Keep stacking ham-leaning words, and the ham total keeps climbing, one addition at a time.

## The attack itself - tipping the scale

Picture a scale weighing evidence. Spam words pile weight on the spam side; ham words pile weight on the ham side. Your original spam ("FREE WINNER CLAIM") sits there putting, say, **8.0** of weight on the spam side. That's the hurdle you need to beat.

Now you append a set of good words - call it G = {meeting, tomorrow, thanks, ...}. Each one adds weight to the **ham** side. Say each contributes **3.0** toward ham. Three of them = 9.0 on the ham side. **9.0 beats 8.0 -> the scale tips -> the filter now calls it ham.** The spam content never moved; you just out-weighted it.

That's what the scary inequality in the text (the one with all the Σ symbols) actually says. `Σ` ("sigma") just means "add these up." Translated to English, the attack-succeeds condition is simply:

> **(ham weight from your added words) must exceed (spam weight already in the message, plus a small built-in bias).**

The "small built-in bias" is that `log P(spam) − log P(ham)` term - if the filter historically saw more spam than ham, it starts a touch suspicious of everything, so you have a hair more to overcome. Usually minor.

## Picking the right words - the "goodness score"
![Picking the right words - the "goodness score"](/assets/Pasted%20image%2020261008233156.png)
You don't want just _any_ legit word - you want words that scream "ham" and basically never appear in spam. You rank candidates by a **goodness score**:

S(w) = P(w|ham) / P(w|spam)
![The GoodWords Attack](/assets/2%20The%20GoodWords%20Attack-20261008233048886.png)
It's a **ratio**: "how many times more common is this word in ham than in spam?" Run three words through it:

- **"meeting"** - ~45% of ham, ~1% of spam -> ratio ≈ 45 -> **excellent** good word.
- **"the"** - ~80% of ham, ~80% of spam -> ratio ≈ 1 -> **useless**. It's everywhere, so it proves nothing. (That's the overlap zone in your Venn diagram, labeled "low discriminative power.")
- **"free"** - rare in ham, common in spam -> ratio well below 1 -> that's a _spam_ word, the exact opposite of what you want.

The two diagrams are drawing exactly this. The **"Goodness Prism"** sorts words into high score (thanks), low score (the), and bad score (free). The **"Two Vocabulary Worlds"** Venn separates the useful ham-only words from the useless shared ones in the middle. (The little `+ ε` in the denominator of the formula is just insurance so you never divide by zero if a word literally never appeared in spam.)

In practice the selection isn't _only_ the ratio - you also weigh how frequently the word genuinely occurs (so your additions look like real language), and you avoid words that might trip _other_ filters or look obviously weird to a human reading the email. The goal is a message that's still plausible.

## How many words to add

It's a trade-off. **Too few** and you don't shift the probability enough - the scale doesn't tip. **Too many** and the email looks unnatural, or hits a length limit, or draws attention. The paper's finding: most spam can be flipped by adding **roughly 15-30 well-chosen good words, with over 90% success**. That's the practical sweet spot.

## Why it works - the real root causes

Pull it together. The attack is possible because of a stack of design choices that each help the attacker:

1. **The conditional-independence assumption.** The filter scores each word on its own and never models how words relate. So it literally _cannot_ notice that your polite "meeting / thanks" words clash with the spammy body. No context, no suspicion.
    
2. **Additive scoring in log space.** Because contributions add up, "quantity overwhelms quality" - pile on enough ham-leaning words and the sum tips, no matter how spammy the original was.
    
3. **Static learned distributions.** The filter learns its word frequencies once, from historical training data, and they don't change. An attacker who can study or estimate that data can figure out _exactly_ which words are the best good words. (This is the white-box version - you can see the model.)
    
4. **Laplace smoothing.** Here's a subtle one. If a word never appeared in spam during training, its probability would be zero - and zero breaks the math (log of zero is negative infinity). To prevent that, filters "smooth" by adding a tiny count to every word so nothing is ever exactly zero. Helpful for the filter, but it also means **even rare or unusual words get a usable, non-zero score** - so the attacker has a _bigger pool_ of good words to draw from, and the filter won't hard-reject an odd-looking combination.
    

Put those four together - no context, additive sums, studyable fixed distributions, and smoothing that admits weird words - and the GoodWords attack is almost inevitable.

