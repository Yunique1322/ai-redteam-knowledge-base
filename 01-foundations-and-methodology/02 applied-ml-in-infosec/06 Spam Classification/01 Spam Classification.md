# Spam Classification

![Spam Classification](/assets/Pasted%20image%2020260828162457.png)

**Mapping Bayes' theorem onto spam**

The formula's the same as before, just relabeled for email:

```python
P(Spam|Features) = (P(Features|Spam) * P(Spam)) / P(Features)
```

Translating each piece into plain spam terms:

- **P(Spam|Features)** - the _posterior_, what you want: "given these words, what's the chance it's spam?"
- **P(Features|Spam)** - the _likelihood_: "how likely are these particular words to show up in spam?"
- **P(Spam)** - the _prior_: "what fraction of all email is spam, before looking at words?"
- **P(Features)** - the _marginal_: "how common are these words across all email?" (normalizer)

**Where "naive" comes in (the key simplification)**

Computing the probability of a whole combination of words appearing together is mathematically brutal. So Naive Bayes cheats with the independence assumption:
```python
P(Features|Spam) = P(feature1|Spam) * P(feature2|Spam) * ... * P(featureN|Spam)
```

In spam terms: it pretends each word appears _independently_ of the others. It treats "free" and "money" as unrelated events, even though they obviously co-occur in spam. This is _false_ - but it makes the math trivially easy (just multiply individual word probabilities), and it works well enough in practice because you only need the _correct class_ to come out on top, not perfectly accurate probabilities. That's the "naive but effective" tradeoff from Module 1.

---

**The worked example - let me walk it slowly**

This is just Bayes' theorem with real numbers. An email has two features (F1, F2); is it spam? Given:

- `P(Spam) = 0.3` - 30% of email is spam (prior)
- `P(Not Spam) = 0.7` - 70% is legit
- `P(F1|Spam) = 0.4`, `P(F2|Spam) = 0.5` - how often these features appear _in spam_
- `P(F1|Not Spam) = 0.2`, `P(F2|Not Spam) = 0.3` - how often they appear _in legit email_

**Step 1 - combine the features under each hypothesis** (using the naive "just multiply" rule):

```python
P(F1, F2|Spam) = P(F1|Spam) * P(F2|Spam) = 0.4 * 0.5 = 0.2
P(F1, F2|Not Spam) = P(F1|Not Spam) * P(F2|Not Spam) = 0.2 * 0.3 = 0.06
```

So these features are more "at home" in spam (0.2) than in legit mail (0.06). Early signal: leans spam.

**Step 2 - find P(F1, F2), the overall chance of seeing these features** (across _both_ classes, using the law of total probability):
```python
P(F1, F2) = P(F1, F2|Spam) * P(Spam) + P(F1, F2|Not Spam) * P(Not Spam)
          = (0.2 * 0.3) + (0.06 * 0.7)
          = 0.06 + 0.042
          = 0.102
```

This is just "how likely are these features in general?" - weighting each class's likelihood by how common that class is. It's the denominator/normalizer.

**Step 3 - apply Bayes for the spam probability:**

```python
P(Spam|F1, F2) = (P(F1, F2|Spam) * P(Spam)) / P(F1, F2)
               = (0.2 * 0.3) / 0.102
               = 0.06 / 0.102
               ≈ 0.588
```

**Step 4 - same for not-spam:**

```python
P(Not Spam|F1, F2) = (0.06 * 0.7) / 0.102
                   = 0.042 / 0.102
                   ≈ 0.412
```

**Step 5 - pick the winner:** 0.588 (spam) > 0.412 (not spam) -> **classified as spam.**

Notice the two posteriors add to 1.0 (0.588 + 0.412) - they're the complete set of possibilities, so they must. And the decision rule is dead simple: **whichever class has the higher posterior wins.** That's the entire Naive Bayes classification step.