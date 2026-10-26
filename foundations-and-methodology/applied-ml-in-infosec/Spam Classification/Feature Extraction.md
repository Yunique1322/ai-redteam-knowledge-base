# Feature Extraction

**The core problem**

Models do math on numbers. But your messages are words. So you need a way to turn "free money now" into something like `[1, 0, 1, 1, 0, ...]` - a list of numbers the model can crunch. That conversion is **feature extraction**, and the method here is **bag-of-words**.

**Bag-of-words - the key idea**

Imagine dumping every word from all your messages into one big bag and making a master list of every unique word - that's your **vocabulary**. Then, each message gets represented as a row of numbers: _for each word in the vocabulary, how many times does it appear in this message?_

That's it. Each message becomes a long row of counts, one number per vocabulary word. Most are 0 (that word isn't in this message), a few are 1+ (the words that are).

Why "bag"? Because you **throw away word order** - you just count what's in the bag, not the sequence. "free money" and "money free" produce the _identical_ count vector. The bag knows _what_ words appear, not _where_. (This is exactly the Naive Bayes independence assumption from earlier, made concrete - words treated as an unordered collection.)


**Look at the unigram example table to see it:**

Five messages, and the table shows each as a row of word-counts. Message 1 ("free prize is waiting for you") has `1` under free, prize, is, waiting, for, you - and `0` everywhere else. Message 2 has `1`s under its words. **Each row is a message turned into numbers.** That grid of numbers is what actually gets fed to the model - it never sees the words, just these counts.

(Notice "The" is _missing_ from the vocabulary - that's `max_df` filtering it out, explained below, because it appears in nearly every message and isn't distinctive.)

|Document|free|prize|is|waiting|for|you|spam|message|offers|a|now|filter|might|detect|this|important|news|says|won|trip|truly|
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|1|1|1|1|1|1|1|0|0|0|0|0|0|0|0|0|0|0|0|0|0|0|
|2|1|1|0|0|0|0|1|1|1|1|1|0|0|0|0|0|0|0|0|0|0|
|3|0|0|0|0|0|0|1|0|0|0|0|1|1|1|1|0|0|0|0|0|0|
|4|1|0|0|0|0|1|0|0|0|1|0|0|0|0|0|1|1|1|1|1|0|
|5|0|0|1|0|0|0|0|1|0|0|0|0|0|0|0|1|0|0|0|0|1|


**The limitation of pure bag-of-words, and the fix: bigrams**

Losing word order loses information. "free prize" (spammy) vs. just "free" (could be anything) - order/context matters. Pure single-word counts can't capture that.

The fix: also count **bigrams** - pairs of consecutive words. So alongside counting "free" and "prize" separately, you _also_ count "free prize" as its own feature. This recaptures _some_ local ordering.

- **Unigram** = one word ("free")
- **Bigram** = two consecutive words ("free prize")

The doc's example nails why this helps: the bigram "free prize" appears in the spam messages (Documents 1 and 2), giving the model a stronger spam signal than "free" alone. **Look at the bigram table** - it has all the unigram columns _plus_ new bigram columns ("free prize," "spam message," etc.). More columns = more context for the model.

|Document|free|prize|is|waiting|for|you|spam|message|offers|a|now|filter|might|detect|this|important|news|says|won|trip|truly|free prize|prize is|is waiting|waiting for|for you|spam message|message offers|offers a|a free|prize now|spam filter|filter might|might detect|detect this|important news|news says|says you|you won|won a|free trip|message truly|truly is|is important|
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|1|1|1|1|1|1|1|0|0|0|0|0|0|0|0|0|0|0|0|0|0|0|1|1|1|1|1|0|0|0|0|0|0|0|0|0|0|0|0|0|0|0|0|0|0|
|2|1|1|0|0|0|0|1|1|1|1|1|0|0|0|0|0|0|0|0|0|0|1|0|0|0|0|1|1|1|1|1|0|0|0|0|0|0|0|0|0|0|0|0|0|
|3|0|0|0|0|0|0|1|0|0|0|0|1|1|1|0|0|0|0|0|0|0|0|0|0|0|0|0|0|0|0|0|1|1|1|1|0|0|0|0|0|0|0|0|0|
|4|1|0|0|0|0|1|0|0|0|1|0|0|0|0|1|1|1|1|1|0|0|0|0|0|0|0|0|0|0|1|0|0|0|0|0|1|1|1|1|1|1|0|0|0|
|5|0|0|1|0|0|0|0|1|0|0|0|0|0|0|1|0|0|0|0|0|1|0|0|0|0|0|0|0|0|0|0|0|0|0|0|0|0|0|0|0|0|1|1|1|

The tradeoff: bigrams _massively_ expand the vocabulary (every word pair is a new feature), so the matrix gets much wider. More context, more columns. It doesn't capture _global_ order (whole-sentence structure is still lost) - just these little 2-word windows.

---

**The code - CountVectorizer does all of this**

python

```python
from sklearn.feature_extraction.text import CountVectorizer

# Initialize CountVectorizer with bigrams, min_df, and max_df to focus on relevant terms
vectorizer = CountVectorizer(min_df=1, max_df=0.9, ngram_range=(1, 2))

# Fit and transform the message column
X = vectorizer.fit_transform(df["message"])

# Labels (target variable)
y = df["label"].apply(lambda x: 1 if x == "spam" else 0) # Converting labels to 1 and 0
```

`CountVectorizer` is scikit-learn's bag-of-words tool - it builds the vocabulary and produces the count matrix automatically. The three parameters are worth understanding because they control the vocabulary:

- **`ngram_range=(1, 2)`** → "include unigrams _and_ bigrams." `(1,1)` would be words only; `(1,2)` = words + word-pairs. This is what adds the bigram columns.
- **`min_df=1`** → a term must appear in at least 1 document to be included. This filters out ultra-rare terms. (Set to 1 here so nothing's excluded for rarity, but in practice you'd raise it - e.g., `min_df=5` = "ignore words appearing in fewer than 5 messages," since a word seen once is probably noise/typo.) "df" = document frequency.
- **`max_df=0.9`** → exclude terms appearing in more than 90% of documents. These are too common to be useful - if a word is in _almost every_ message, it can't distinguish spam from ham. This is what removed "The" in the example. It's like automatic stop-word removal based on the actual data.

Together `min_df`/`max_df` trim the vocabulary to the _useful middle_: not so rare it's noise, not so common it's meaningless.

**The two key lines:**

python

```python
X = vectorizer.fit_transform(df["message"])
```

- `fit_transform` does two things: **fit** (scan all messages, build the vocabulary, apply the min_df/max_df filters) and **transform** (convert each message into its count vector). Result: `X` = the big matrix - rows are messages, columns are vocabulary terms, values are counts. This is your feature matrix, ready for the model. (Same `fit_transform` pattern from the scalers/encoders earlier - "learn the transformation, then apply it.")

python

```python
y = df["label"].apply(lambda x: 1 if x == "spam" else 0)
```

- This builds `y`, the labels. It converts the text labels to numbers: **spam → 1, ham → 0.** Models need numeric labels, so "spam"/"ham" become 1/0. The `lambda x: 1 if x == "spam" else 0` reads: "for each label, if it's 'spam' give 1, otherwise 0." Now `y` is a column of 1s and 0s matching each message.

**The result:** `X` (features - the count matrix) and `y` (labels - 1/0) are exactly the two things you feed a scikit-learn model. `model.fit(X, y)` is the very next step (next section). This is that `X`/`y` convention from way back, now fully realized: X = numeric features, y = numeric labels.