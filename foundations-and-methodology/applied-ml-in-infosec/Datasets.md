# Datasets

**The core message: garbage in, garbage out**

Before any fancy model, you need good data. A brilliant algorithm trained on messy data produces a useless model. That's the whole point of this section - data quality determines model quality. The doc lists what makes data "good" (relevance, completeness, consistency, balance, etc.), but you don't need to memorize that table. Two attributes matter most and connect to Module 1:

- **Balance** - remember the spam-filter accuracy trap? If 90% of your data is one class, a model can score 90% by ignoring the minority class entirely. Imbalanced data → biased models. This is why balance matters.
- **Representativeness** - if your data doesn't reflect reality (e.g., a facial recognition dataset with only one demographic), the model fails on everything it didn't see. Bias in → bias out.

Both tie back to **generalization** vs. **overfitting** - the recurring theme. Good data helps the model generalize; bad data makes it memorize noise.



**The dataset you'll actually work with**

`demo_dataset.csv` - network log entries. This is a security dataset (fitting, for an infosec course). Each row is a network event with:

- `source_ip`, `destination_port`, `protocol` - who/where/how
- `bytes_transferred` - how much data moved
- `threat_level` - the **label**: 0 = normal, 1 = low threat, 2 = high threat

That `threat_level` is what a model would learn to predict - "given this network event's features, is it a threat?" This is intrusion detection framed as ML, exactly the kind of system you'll later learn to attack.

**The important part: this data is deliberately messy**

The doc flags that the dataset has problems on purpose - mixed numeric/text data, missing values, invalid entries, non-numeric junk in numeric columns, and weird values like `?` and `-1` in `threat_level`. This is realistic. Real data is _always_ dirty, and cleaning it is most of the actual work. This section (and the next few) teach you to spot and fix that mess.



**Now the code - your first real pandas**

The star here is the **DataFrame** - pandas' data structure. Mental model: **a DataFrame is a spreadsheet in Python.** Rows and columns, with labels, that you can inspect and manipulate with code. It's the single most-used tool in practical ML for handling tabular data. Everything below operates on it.

python

```python
import pandas as pd
data = pd.read_csv("./demo_dataset.csv")
```

`pd.read_csv` loads a CSV file into a DataFrame called `data`. That's it - your spreadsheet is now in memory, ready to work with. (`pd` is the universal nickname for pandas - everyone writes it this way.)

Then four **exploration** commands - these are how you "look before you leap," getting to know the data before touching it:

```python
print(data.head())
```

Shows the **first few rows**. Your first glance - "what does this data actually look like?" Catches obvious problems immediately (weird column names, garbled values). You'll run `.head()` constantly; it's the "peek at the top" command.




```python
print(data.info())
```

A **summary**: column names, their data types, and how many non-empty entries each has. This is diagnostic gold - it instantly reveals which columns have missing data and whether types are wrong (e.g., a number column that pandas thinks is text because there's junk in it). Run this early on any dataset.



```python
print(data.isnull().sum())
```

Counts **missing values per column**. `isnull()` marks every empty cell as True, `.sum()` tallies them up per column. Output tells you exactly where the holes are, so you know what needs fixing. This directly sets up the next section (data preprocessing - filling/removing those holes, using tools like `SimpleImputer` from the last section).


**The workflow pattern to internalize:**

Notice the sequence - **load → head → info → isnull.** This is the standard "first contact" ritual with any dataset:

1. Load it (`read_csv`)
2. Eyeball it (`head`)
3. Diagnose structure/types/gaps (`info`)
4. Quantify missing data (`isnull().sum()`)

You'll do this exact ritual at the start of every project _and_ likely on the exam when handed unfamiliar data. Put these four lines in your reusable-snippets notebook - they're your "getting to know the data" template.