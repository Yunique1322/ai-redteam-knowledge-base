# Data Transformation

**The big picture: three transformations**

1. **Encoding** - turn text categories into numbers (models can't read "TCP").
2. **Skew handling** - tame lopsided numeric columns so outliers don't dominate.
3. **Splitting** - carve data into train/validation/test sets for honest evaluation.

---

### 1. Encoding - text categories into numbers

Models only understand numbers, but `protocol` is text (`TCP`, `HTTPS`, `SSH`...). You have to convert it. But _how_ you convert matters - and this is a subtle, important point.

**Why not just number them?** You could label TCP=1, HTTPS=2, SSH=3 (`LabelEncoder`). But that creates a **fake ordering** - the model would think SSH (3) is "more than" TCP (1), or that HTTPS is the "average" of TCP and SSH. That's nonsense; protocols have no order. This false hierarchy would mislead the model.

**One-hot encoding fixes this.** Instead of one numbered column, it makes a separate 0/1 column per category:

Original: `color = red`  
Becomes: `color_red=1, color_green=0, color_blue=0`
![1. Encoding - text categories into numbers](/assets/Pasted%20image%2020260827234001.png)
Each category gets its own column; a `1` marks "this is the one." No ordering implied - every category is equal and independent. That's the whole point: **one-hot avoids fake hierarchies.**

Here's the full code applied to `protocol`:
```python
from sklearn.preprocessing import OneHotEncoder

encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
encoded = encoder.fit_transform(df[['protocol']])

encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(['protocol']))
df = pd.concat([df.drop('protocol', axis=1), encoded_df], axis=1)
```

Line by line:

- `OneHotEncoder(handle_unknown='ignore', sparse_output=False)` → create the encoder. `handle_unknown='ignore'` = if it sees a protocol it didn't train on, don't crash, just mark all-zeros. `sparse_output=False` = give a normal array, not a compressed one (easier to work with).
- `encoder.fit_transform(df[['protocol']])` → learn the categories _and_ convert the protocol column into those 0/1 columns.
- `pd.DataFrame(encoded, columns=encoder.get_feature_names_out(['protocol']))` → wrap the result back into a DataFrame with proper names (`protocol_TCP`, `protocol_HTTPS`, etc.).
- `pd.concat([df.drop('protocol', axis=1), encoded_df], axis=1)` → drop the _original_ text `protocol` column and glue on the new binary columns. `axis=1` means "join side by side (as columns)."

Result: one text column becomes several numeric 0/1 columns the model can use.

**The one tradeoff:** if a category has _many_ unique values (say, thousands of IPs), one-hot creates thousands of columns - bloat. That's the "increases feature count" caveat, and why the doc mentions `HashingEncoder`/frequency methods for high-cardinality features. For protocol (only ~10 values), one-hot is fine.

---

### 2. Handling skewed data - taming lopsided distributions

**What's skew?** A column where most values are small but a few are huge. `bytes_transferred` is classic: most network events move a little data, but a few move enormous amounts. Plotted, it's a big pile near zero with a long tail stretching right.

**Why it's a problem:** those few giant values _dominate_. The model over-focuses on the extremes and struggles to see patterns in the many normal-sized values. (Also, some models assume roughly balanced/normal-ish data - skew breaks that assumption.)

**The fix - log transform:** applying a logarithm _compresses_ large values much more than small ones, pulling that long tail in and evening out the distribution.
![2. Handling skewed data - taming lopsided distributions](/assets/Pasted%20image%2020260827234021.png)


```python
import numpy as np

# Apply logarithmic transformation to a skewed feature to reduce its skewness
df["bytes_transferred"] = np.log1p(df["bytes_transferred"]) # Add 1 to avoid log(0)
```

- `np.log1p(...)` = "log of (1 + value)." Why the +1? Plain `log(0)` is undefined (mathematically explodes), and byte counts can be 0. Adding 1 first means even a 0 becomes `log(1) = 0` safely. `log1p` is just "log-plus-one," built for exactly this.

**Key reassurance:** no information is lost - you're just _rescaling the lens_. A value of 1,000,000 and a value of 10 are still different after the transform, just less dramatically so. The model now weighs all points more fairly instead of being hypnotized by the giants. This reduces the risk of **overfitting to outliers** (the recurring overfitting theme, in a new form).

---

### 3. Data splitting - train / validation / test

This extends the `train_test_split` from earlier into a **three-way** split. Last time you split 2 ways (train/test); now there's a middle set: **validation.**

**Why three sets?** Each has a distinct job:

- **Training set (~60%)** - the model _learns_ from this.
- **Validation set (~20%)** - you _tune_ on this (try different settings/models, see what works). Used _during_ development.
- **Test set (~20%)** - the _final_, untouched exam. Used _once_, at the very end, to get an honest performance number.

The critical idea: **the test set must stay pristine.** If you tune your model against the test set, you're secretly "teaching to the test" - the score becomes dishonest. So you tune on validation and save test for the true final check. Validation is the "practice exam," test is the "real exam." This is a stronger guard against **overfitting** than a simple 2-way split.

Here's the full code:

```python
from sklearn.model_selection import train_test_split

# Separate features (X) and target (y)
X = df.drop("threat_level", axis=1)
y = df["threat_level"]

# Initial split: 80% training, 20% testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1337)

# Second split: from the 80% training portion, allocate 60% for final training and 20% for validation
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=1337)
```

Decoding it:

**First, separate features from the answer:**

- `X = df.drop("threat_level", axis=1)` → `X` is everything _except_ the answer (all the input features). `drop(..., axis=1)` removes that column.
- `y = df["threat_level"]` → `y` is just the answer column (what we're predicting). This is the `X`/`y` convention from earlier: X = inputs, y = labels.

**First split (80/20):**

- `train_test_split(X, y, test_size=0.2, random_state=1337)` → 80% train, 20% test. `random_state=1337` = a fixed seed so the _random_ split is **reproducible** - you (and the graders) get the identical split every run. Important for consistent results; pick any number, just keep it fixed.

**Second split (the 80% → 60%/20%):**

- This splits the _training_ portion again to carve out validation.
- `test_size=0.25` here is the confusing bit the doc flags: it's 25% _of the 80%_, not of the whole. Math: `0.8 × 0.25 = 0.2` = 20% of the total. So you end with **60% train, 20% validation, 20% test** overall.

The workflow these enable:

1. Train on `X_train`/`y_train`.
2. Tune/compare using `X_val`/`y_val`.
3. Final honest score on the never-touched `X_test`/`y_test`.

---

### The takeaway

Transformation = making clean data _model-ready_:

- **Encode** text → numbers (one-hot to avoid fake ordering).
- **Log-transform** skewed columns so outliers don't dominate.
- **Split** into train/val/test to evaluate honestly and dodge overfitting.

Save all three patterns to your snippets notebook - one-hot encoding, `np.log1p`, and the two-stage split are reused in basically every project and very likely on the exam.