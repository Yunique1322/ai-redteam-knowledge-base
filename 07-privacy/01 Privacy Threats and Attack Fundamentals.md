# Privacy Threats And Attack Fundamentals

## The idea in one breath

A model trained on personal data can give away **which individuals were in its training set**, just through how it answers questions about them. A **Membership Inference Attack (MIA)** asks one yes/no question: "was this exact record used to train the model?" It works because models **memorize** their training data a little, so they answer more confidently and more correctly on records they've seen. Anything clearly above **50% (random guessing)** is a leak.

> **Heads-up:** No formulas were mangled in this paste. There are two naming clashes to keep straight before the next sections:
> 
> - **Target model vs attack model.** The **target model** is the victim (for example, the hospital's diagnosis model). The **attack model** is _your_ small yes/no classifier that judges "member or not" from the target's outputs. Next sections add a third kind, **shadow models**. Keep all three separate in your head.
> - **Model extraction vs training data extraction.** These sound alike but aren't. **Model theft / model extraction** (OWASP ML05) steals _the model itself_. **Training data extraction** steals _the data it was trained on_. They're different findings in a report.

---

## Part 1: What is a Membership Inference Attack?

### Step 1: The question being asked

You hand the target model a sample **plus its true label**, look at what it outputs, and decide: **member** (it was in training) or **non-member** (it wasn't). That makes it a **binary classification** problem, and your output is a single yes/no.

### Step 2: Why a yes/no answer can still be dangerous

The harm comes from _what the training set is_. If a model was trained **only on cancer patients**, then "Patient X is a member" means "Patient X has cancer." The model never had to output anything about Patient X's diagnosis. Membership alone is the sensitive fact. The same logic applies to:

|Training set|What "member" reveals|
|---|---|
|Cancer-patient diagnosis model|The person has cancer|
|Credit-scoring model's training set|The person's loan history|
|Content-moderation dataset|The person has previous violations|

### Step 3: What "success" means

A coin flip gets 50% on a balanced set of members and non-members. A **successful attack is one significantly better than 50%**. This is the same idea as your Section 7 success rate in the evasion module: the number means nothing until you compare it to the right baseline.

---

## Part 2: Privacy Attacks on Machine Learning (the family)

|Attack|What it steals|Course example|Typical target|
|---|---|---|---|
|**Model inversion**|Reconstructs training data **features** from model outputs|Genetic markers of training subjects from a disease-risk model|Any model with rich outputs|
|**Attribute inference**|Deduces a sensitive attribute the model **does not directly predict**|Income model queried to reveal education level / employment status of members|Models trained on full records|
|**Training data extraction**|Recovers **verbatim** training examples|LLM repeating memorized sequences|Large language models|
|**Membership inference (MIA)**|Just **in or out** of the training set|Cancer-patient membership|Any model|

### Where MIA fits

- It needs the **least information** to run: query access, the sample and its label.
- It travels with the others. A model that leaks membership is usually easier to hit with inversion or attribute inference too.
- It doubles as a **privacy audit metric**. If a model leaks membership, it very likely leaks other things. That's why this module measures MIA _first_, as a baseline, before adding defenses (DP-SGD, PATE) and re-measuring.

**MIA is both an attack and the ruler used to grade privacy defenses.** Keep that in mind for the DP-SGD and PATE sections: "the attack accuracy went from X% down to near 50%" is how you prove a fix worked.

---

## Part 3: Why Membership Inference Works

### Step 1: Memorization vs generalization

- **Generalization** means learning patterns that work on new data. That's the goal.
- **Memorization** means learning quirks of _specific_ training records, especially records that are **unusual, repeated, or near the decision boundary**.

During gradient descent, the model keeps lowering loss **on the training samples**. After enough epochs it can classify every training example perfectly, but that tight fit doesn't transfer. This is **overfitting**, and it is the **root cause of membership leakage**.

### Step 2: Toy example: a tiny diagnosis model

We have 8 patients. 4 were in training (members) and 4 weren't (non-members). For each one, the model gives a top **confidence** and is either right or wrong:

|Patient|Truly a member?|Confidence|Prediction correct?|
|---|---|---|---|
|A|Yes|0.95|✓|
|B|Yes|0.98|✓|
|C|Yes|0.91|✓|
|D|Yes|0.60|✓ (a genuinely ambiguous record)|
|E|No|0.80|✓|
|F|No|0.54|✓|
|G|No|0.94|✓ (a very "typical" record)|
|H|No|0.92|✗|

### Step 3: The generalization gap

- Accuracy on members: 4/4 = **100%**.
- Accuracy on non-members: 3/4 = **75%**.
- **Generalization gap** = train accuracy − test accuracy = 100% − 75% = **25 points**.

A big gap is the warning light. It tells you the model treats seen and unseen data differently, and that difference is what an attacker measures.

### Step 4: Signal 1, confidence

Mean confidence on members = (0.95 + 0.98 + 0.91 + 0.60) / 4 = 3.44 / 4 = **0.86**.  
Mean confidence on non-members = (0.80 + 0.54 + 0.94 + 0.92) / 4 = 3.20 / 4 = **0.80**.

Members get higher confidence on average because the model optimized specifically for them. The course's example has the same shape: a member gets `[0.05, 0.95]` while a non-member gets `[0.20, 0.80]`. Each pair of numbers is a **probability vector**, one probability per class, and the confidence is the bigger number.

Rule R2, "confidence ≥ 0.90 -> member":

- A, B, C are called member, which is right (3).
- D (0.60) is called non-member, which is **wrong**.
- E and F are called non-member, which is right (2).
- G (0.94) is called member, which is **wrong**.
- H (0.92) is called member, which is **wrong**.

That's 5 right out of 8, or **62.5%**.

### Step 5: Signal 2, correctness

Rule R1, "prediction correct -> member":

- A, B, C, D are called member, which is right (4).
- E, F, G are correct predictions, so they're called member. All three are **wrong**.
- H is a wrong prediction, so it's called non-member, which is right (1).

That's 5 right out of 8, or **62.5%**.

### Step 6: Combine both signals

Rule R3, "confidence ≥ 0.90 **AND** correct -> member":

- A, B, C are called member, which is right (3).
- D (low confidence) is called non-member, which is **wrong**.
- E and F (low confidence) are called non-member, which is right (2).
- G (high confidence and correct) is called member, which is **wrong**.
- H (high confidence but **incorrect**) is called non-member, which is **right**.

That's 6 right out of 8, or **75%**. Combining the two signals beats either one alone. H is the case the course describes: **high confidence but incorrect suggests non-membership**. Confidence alone was fooled by H, and correctness rescued it.

The two errors left over are exactly the course's "hard cases where the signals conflict":

|Error|Why it fools us|
|---|---|
|D|A **member** with low confidence, because the record is genuinely ambiguous|
|G|A **non-member** with high confidence, because the record is very typical|

The real attack model doesn't use a hand-picked threshold like 0.90. It's a trained classifier that **learns how much to weight each signal** based on how often each pattern corresponds to membership. That's what the shadow-model sections build.

### Step 7: The formula

```
attack accuracy = (TP + TN) / (members + non-members)
```

|Symbol|Plain English|Toy (rule R3)|Real (my cells below)|
|---|---|---|---|
|TP|Members correctly called "member"|3 (A, B, C)|out of 500 members|
|TN|Non-members correctly called "non-member"|3 (E, F, H)|out of 500 non-members|
|members + non-members|Size of the audit pool|8|1000|
|attack accuracy|Fraction of correct in/out calls|6/8 = 0.75|0.728|

**Read aloud:** "Attack accuracy is how many people I placed correctly, inside or outside the training set, divided by everyone I tested."

### Step 8: Capacity and regularization

- **Model capacity**: more parameters means more room to memorize, so more MIA risk. The course's example: 10 million parameters can store more training detail than 100,000.
- **Regularization** reduces memorization but **cannot eliminate it**: **dropout** (you met it in DeepFool Sec 16, where it randomly zeroes units in training), **weight decay**, and **early stopping**. The optimizer always treats training data differently from unseen data, which creates the **utility-vs-privacy tension** the defense sections are about.

> **An honest note the course skips (1):** "Better than 50%" assumes a **balanced** audit pool, meaning equal numbers of members and non-members. If the pool is unbalanced, 50% is the wrong baseline.
> 
> **An honest note the course skips (2):** An _average_ attack accuracy can understate the danger. Unusual records (rare diseases, outliers, mislabeled entries) are memorized hardest, so an attack that is only ~60% accurate overall can be close to certain about a few specific people. For a report's impact section, those individuals are the real harm.

---

## Part 4: Industry Security Frameworks

|Framework|Category for MIA|Related entries|Recommended defense|
|---|---|---|---|
|**OWASP ML Security Top 10** (draft v0.3)|**ML04:2023 Membership Inference** (exploitability 4/5, impact 4/5)|ML03:2023 Model Inversion, ML05:2023 Model Theft|Differential privacy + regularization|
|**OWASP Top 10 for LLM Applications** (2025)|**LLM02: Sensitive Information Disclosure** (training data leakage)|LLM04: Data and Model Poisoning|(covered later)|
|**Google SAIF**|**Sensitive Data Disclosure** and **Inferred Sensitive Data**|n/a|Model **creators**: privacy-preserving training. Model **consumers**: filter outputs, monitor queries|

All three frameworks agree on two things: a membership leak implies wider leakage, and **differential privacy is the recommended countermeasure**. DP-SGD and PATE are the two DP methods in this module.

> **An honest note the course skips (3):** LLM04 is about _poisoning_, which is a weak fit for a membership finding. In your report, cite **ML04:2023** for a classic classifier and **LLM02:2025** for an LLM app. My own suggestion for the CWE field (not from the course): **CWE-359, Exposure of Private Personal Information to an Unauthorized Actor**, or the broader **CWE-200, Exposure of Sensitive Information to an Unauthorized Actor**.

---

## Reading the figure/output

This section had no figures or output blocks.

---

## Copy-paste cells (mine, not the course's)

This is a **privacy audit** of your own model. It runs the toy's three rules on real numbers, then shows regularization closing the leak. It uses only scikit-learn, with no HTB library and no download. I ran all four cells, so the expected lines are real numbers.

### Cell 1: Build an overfit "target model" and measure the generalization gap

This cell makes 1000 synthetic records with 15% noisy labels (the "unusual" cases a model has to memorize). It trains on half (`X_mem`, the members) and holds back the other half (`X_non`, the non-members).

```python
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# 1000 fake "patient records", 20 features, 15% of labels deliberately noisy (unusual cases)
X, y = make_classification(n_samples=1000, n_features=20, n_informative=5,
                           flip_y=0.15, random_state=0)

# Half go into training (members), half are held back (non-members) -> balanced audit pool
X_mem, X_non, y_mem, y_non = train_test_split(X, y, test_size=0.5,
                                              random_state=0, stratify=y)

target = RandomForestClassifier(n_estimators=100, random_state=0).fit(X_mem, y_mem)

train_acc = target.score(X_mem, y_mem)
test_acc  = target.score(X_non, y_non)
print(f"Members (train) accuracy    : {train_acc:.3f}")
print(f"Non-members (test) accuracy : {test_acc:.3f}")
print(f"Generalization gap          : {train_acc - test_acc:.3f}")
```

Expected:

```txt
Members (train) accuracy    : 1.000
Non-members (test) accuracy : 0.860
Generalization gap          : 0.140
```

### Cell 2: Extract the two leak signals

`leak_signals()` returns each record's confidence (top probability) and whether the prediction was correct. These are the same two signals as the toy.

```python
def leak_signals(model, X, y):
    probs = model.predict_proba(X)            # probability vector per sample, e.g. [0.05, 0.95]
    conf = probs.max(axis=1)                  # signal 1: confidence (top probability)
    correct = probs.argmax(axis=1) == y       # signal 2: correctness (prediction == true label)
    return conf, correct

conf_mem, corr_mem = leak_signals(target, X_mem, y_mem)
conf_non, corr_non = leak_signals(target, X_non, y_non)
print(f"Mean confidence  members: {conf_mem.mean():.3f}   non-members: {conf_non.mean():.3f}")
print(f"Correct fraction members: {corr_mem.mean():.3f}   non-members: {corr_non.mean():.3f}")
```

Expected:

```txt
Mean confidence  members: 0.900   non-members: 0.794
Correct fraction members: 1.000   non-members: 0.860
```

### Cell 3: Score the toy's three rules (R1, R2, R3) on the real pool

`audit_rules()` builds the balanced pool of 500 members and 500 non-members and prints the attack accuracy of each rule against the 0.500 coin flip.

```python
def audit_rules(conf_mem, corr_mem, conf_non, corr_non, t=0.9):
    is_member = np.r_[np.ones(len(conf_mem)), np.zeros(len(conf_non))].astype(bool)
    conf = np.r_[conf_mem, conf_non]
    corr = np.r_[corr_mem, corr_non]
    rules = {
        "correct only          ": corr,
        f"confidence >= {t}     ": conf >= t,
        f"both (conf >= {t} & ok)": (conf >= t) & corr,
    }
    print(f"Pool: {is_member.sum()} members + {(~is_member).sum()} non-members (random guess = 0.500)")
    for name, guess in rules.items():
        print(f"  {name}: attack accuracy = {(guess == is_member).mean():.3f}")

audit_rules(conf_mem, corr_mem, conf_non, corr_non, t=0.9)
```

Expected:

```txt
Pool: 500 members + 500 non-members (random guess = 0.500)
  correct only          : attack accuracy = 0.570
  confidence >= 0.9     : attack accuracy = 0.721
  both (conf >= 0.9 & ok): attack accuracy = 0.728
```

You can check the 0.570 by hand. All 500 members are predicted correctly, so all 500 are called "member", which is right. 86% of non-members (430) are also predicted correctly, so they're wrongly called "member". The other 70 are rightly called "non-member". That gives (500 + 70) / 1000 = 0.570.

### Cell 4: Regularization shrinks the leak (preview of the defense half)

This is one loop, so it's one cell. `min_samples_leaf` forces each tree leaf to cover at least that many records, which stops a tree from memorizing single records.

```python
for leaf in [1, 5, 20]:
    m = RandomForestClassifier(n_estimators=100, min_samples_leaf=leaf,
                               random_state=0).fit(X_mem, y_mem)
    cm, km = leak_signals(m, X_mem, y_mem)
    cn, kn = leak_signals(m, X_non, y_non)
    is_member = np.r_[np.ones(len(cm)), np.zeros(len(cn))].astype(bool)
    guess = (np.r_[cm, cn] >= 0.9) & np.r_[km, kn]
    print(f"min_samples_leaf={leaf:>2}: test acc {m.score(X_non, y_non):.3f} | "
          f"gap {m.score(X_mem, y_mem) - m.score(X_non, y_non):.3f} | "
          f"conf gap {cm.mean() - cn.mean():.3f} | audit acc {(guess == is_member).mean():.3f}")
```

Expected:

```txt
min_samples_leaf= 1: test acc 0.860 | gap 0.140 | conf gap 0.106 | audit acc 0.728
min_samples_leaf= 5: test acc 0.858 | gap 0.060 | conf gap 0.037 | audit acc 0.585
min_samples_leaf=20: test acc 0.864 | gap 0.008 | conf gap 0.016 | audit acc 0.512
```

As the generalization gap shrinks from 0.140 to 0.008, the leak drops from 72.8% to 51.2%, close to a coin flip, while test accuracy stays about the same. Here privacy came almost free. With differential privacy you usually pay some accuracy, and that cost is the **privacy-utility trade-off** the next sections measure.

## Key facts to lock in

- A **Membership Inference Attack decides whether a specific sample (given with its true label) was in the target model's training set**. It's a binary member/non-member call made only from how the model responds to queries.
- **Membership alone can be the sensitive fact**, because the training set's purpose reveals it: being a member of a cancer-only training set reveals a cancer diagnosis, even though the model never predicts anything about that person.
- **Model inversion reconstructs training features, attribute inference deduces attributes the model doesn't predict, and training data extraction recovers verbatim examples (mostly in LLMs)**. MIA needs the least information of the four, and it's also used as the privacy audit metric for the others.
- **The root cause of membership leakage is overfitting (memorization over generalization)**, which shows up as a generalization gap. In my cells, a 0.140 gap gave a 72.8% attack, and a 0.008 gap gave 51.2%.
- **The attack combines two signals, confidence and correctness, and combining them beats either alone**. A high-confidence correct prediction points to member, a high-confidence incorrect one points to non-member, and most errors come from low-confidence or conflicting cases (ambiguous members, very typical non-members).
- **Higher model capacity means more memorization and more MIA risk**. Dropout, weight decay and early stopping reduce it but cannot eliminate it, which is the utility-vs-privacy tension.
- **OWASP ML Top 10 lists membership inference as ML04:2023** (inversion is ML03, model theft is ML05). For LLM apps the matching entry is LLM02:2025 Sensitive Information Disclosure, and Google SAIF files it under Sensitive Data Disclosure / Inferred Sensitive Data. **All three recommend differential privacy** (DP-SGD, PATE).
- **Method point:** **Always measure an MIA against a balanced pool of members and non-members, so that 50% really means "no leak"**. Report attack accuracy before and after a defense to prove the fix worked, and in the report's impact section call out the unusual individuals, because they leak the most.

---

