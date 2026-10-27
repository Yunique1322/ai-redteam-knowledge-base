# Metrics For Evaluating A Model

**Quick refresher (you know this - one line each):**

The foundation is the four outcome buckets: **TP** (correctly caught positive), **FP** (false alarm), **TN** (correctly ignored negative), **FN** (missed positive). From those:

- **Accuracy** = overall correct rate = (TP + TN) / all. _"How often is the model right?"_
- **Precision** = TP / (TP + FP). _"When it says positive, is it right?"_ -> measures false alarms.
- **Recall** = TP / (TP + FN). _"Of all real positives, how many did it catch?"_ -> measures misses.
- **F1** = the balance of precision and recall in one number.

The **accuracy trap** the doc hammers is the same one from Module 1: with 1% spam, a model that calls _everything_ legitimate scores 99% accuracy while catching zero spam. Accuracy lies when classes are imbalanced - which, in security, they _almost always are_ (most traffic is benign, most files are clean). **So in security work, accuracy alone is nearly useless; precision and recall are what matter.**

---

**What's actually new and worth your attention: the precision vs. recall tradeoff in security**

This is the part to internalize, because it's a real decision you'll reason about (and likely write about in your engagement report). The two metrics pull against each other, and _which one you prioritize depends entirely on the cost of being wrong_:

**Favor recall when missing a positive is catastrophic.** The doc says it directly: (cite index="63-1">in threat detection, a model might favor recall to avoid missing critical threats, even if it occasionally flags benign events.</cite> If you're detecting malware or intrusions, a _missed_ threat (false negative) could mean a breach. You'd rather tolerate some false alarms than let a real attack slip through. High recall = "catch everything, even at the cost of noise."

**Favor precision when false alarms are expensive.** Also from the doc: (cite index="63-1">in environments with limited resources, focusing on precision can reduce the burden caused by following up on false alarms.</cite> If every alert means an analyst drops everything to investigate, a flood of false positives causes **alert fatigue** - the team burns out and starts ignoring alerts (including real ones). High precision = "only alert when you're confident."

The mental model: **recall = don't miss anything; precision = don't cry wolf.** You can't max both - pushing one usually lowers the other. The right balance is a _business/risk decision_, not a math one. F1 exists to give you a single score when you want both to stay reasonable.

---

**The other metrics (recognize them, don't grind them):**

- **Confusion matrix** - the most useful one to actually know. It's just a table showing all four buckets (TP/FP/TN/FN) at once, so you see _exactly_ how the model succeeds and fails, per class. You'll likely generate one in the upcoming projects (scikit-learn has `confusion_matrix`). It's the "show me the full breakdown" view.
- **Specificity** - recall's mirror image: "of all real negatives, how many did I correctly ignore?" Measures how well you avoid false alarms on the negative class.
- **AUC (Area Under the ROC Curve)** - measures how well the model separates classes _across all thresholds_. Higher = better at telling positive from negative in general. Useful because it doesn't depend on one fixed cutoff. Recognize the term; you don't need the math.
- **Matthews Correlation Coefficient (MCC)** - a single balanced score that behaves well even on _heavily_ imbalanced data (common in security). Nice-to-know.

---

**The one habit worth forming: contextualizing metrics**

The doc's closing point is subtle but important for a red teamer: **impressive numbers can be misleading if the data doesn't reflect reality.** A 99% accuracy means nothing if the test set is 99% one class. Always ask:

- Do these metrics hold across _different slices_ of the data, or just on average?
- Does the test data actually resemble real-world conditions (including the real class imbalance)?
- What's the real-world _cost_ of each error type?

This is genuine security-analyst thinking: don't trust a headline metric - interrogate whether it reflects operational reality.