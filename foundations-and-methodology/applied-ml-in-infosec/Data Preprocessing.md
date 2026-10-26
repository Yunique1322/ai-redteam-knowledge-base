# Data Preprocessing

This is **Data Preprocessing** - the actual cleaning of that messy network dataset. Last section you learned to _detect_ problems (`head`, `info`, `isnull`); now you _fix_ them. It looks like a lot of code, but it's really just two ideas: **find the bad data**, then **decide what to do with it** (throw it out or repair it). Let me make it click.

**The two-phase structure**

The whole section is:

1. **Identify invalid values** - write checks to catch bad entries in each column.
2. **Handle them** - either _drop_ (delete bad rows) or _impute_ (repair them).

That's it. Everything below is those two phases applied column by column.

---

### Phase 1: Finding the bad data

Each column has its own notion of "valid," so each gets its own check. The _pattern_ is identical every time - let me decode one, and the rest follow:


```python
invalid_ports = data[~data['destination_port'].apply(is_valid_port)]
```

Read it inside-out:

- `is_valid_port` → a function that returns True/False for one value ("is this a real port, 0-65535?")
- `.apply(is_valid_port)` → run that check on _every_ value in the column → get a column of True/False
- `~` → the NOT operator - flips True↔False, so now True marks the _invalid_ ones
- `data[...]` → keep only the rows where that's True → i.e., **the invalid rows**

So the line means: _"give me all rows where the port is NOT valid."_ Every check below uses this same `data[~data['col'].apply(check)]` shape. Once you see it once, you see it everywhere.

The five checks, each just defining "valid" for that column:

- **IP** → matches a proper IP-address pattern (that scary `re.compile(...)` is a **regex** - a pattern-matcher for text; you don't need to read it, just know it means "does this look like a valid IP like 192.168.1.1?")
- **Port** → is it a number between 0 and 65535?
- **Protocol** → is it in the known list (`TCP`, `HTTPS`, `SSH`...)? `.isin(valid_protocols)` = "is the value in this list?"
- **Bytes** → is it a non-negative number?
- **Threat level** → is it 0, 1, or 2?

Notice the `try/except` in several: `int(port)` will _crash_ if the value is text like `"STRING_PORT"`, so `except ValueError: return False` catches that crash and says "not valid." That's Python politely handling junk instead of exploding.

---

### Phase 2: Handling the bad data - two philosophies

Now that you've found the bad rows, you choose: **delete or repair.** This is a genuine judgment call, and the doc shows both.

**Option A - Drop (delete bad rows):**



```python
data = data.drop(invalid_ips.index, errors='ignore')
```

`.index` = the row numbers of the bad entries; `.drop()` removes them. `errors='ignore'` just prevents a crash if a row was already deleted by a previous check (overlap). Do this for each check and the bad rows are gone.

**The tradeoff (important):** simple and safe - you're left with only clean data. But you _lose data_. The doc says after dropping, only **77 clean entries remain** - from a bigger original. If your dataset is small, throwing away rows can leave too little to train on. Drop when data is plentiful and accuracy matters more than quantity.

**Option B - Impute (repair bad rows):**  

This keeps every row by _replacing_ bad values with sensible estimates. It's more work but preserves data. Three steps:

**Step 1 - standardize all the junk to `NaN`:**



```python
df.replace(invalid_ips + invalid_ports + invalid_bytes + invalid_threat, np.nan, inplace=True)
```

All the different flavors of bad (`INVALID_IP`, `STRING_PORT`, `?`, etc.) get converted to one thing: `NaN` (pandas' standard "missing" marker - "Not a Number"). Why? So you can handle all missingness _uniformly_ instead of chasing a dozen different bad strings. One representation for "missing."




```python
df['destination_port'] = pd.to_numeric(df['destination_port'], errors='coerce')
```

Forces the column to numbers; `errors='coerce'` means "anything that can't become a number → turn it into NaN too." Cleans up leftover text in numeric columns.

**Step 2 - fill the NaNs (this is imputation):**



```python
num_imputer = SimpleImputer(strategy='median')
df[numeric_cols] = num_imputer.fit_transform(df[numeric_cols])
```

Here's `SimpleImputer` from the libraries section, in action. It fills every NaN in the numeric columns with that column's **median** (middle value). For the text column (`protocol`), it uses **most_frequent** (the most common protocol). The logic: a reasonable guess (the typical value) is better than a hole. (Median for numbers because it resists outliers; mode for categories because "average" makes no sense for text.)

The doc also mentions `KNNImputer` - a smarter version that fills a gap by looking at _similar_ rows (its k nearest neighbors) rather than just using the column-wide median. More context-aware, but heavier. Nice-to-know.
```
from sklearn.impute import KNNImputer knn_imputer = KNNImputer(n_neighbors=5) df[numeric_cols] = knn_imputer.fit_transform(df[numeric_cols])
```

**Step 3 - apply domain knowledge (the human touch):**


```python
df['source_ip'] = df['source_ip'].fillna('0.0.0.0')
df['destination_port'] = df['destination_port'].clip(lower=0, upper=65535)
```

Final cleanup using _knowledge of what the data means_: fill any still-missing IP with a placeholder `0.0.0.0`; `.clip()` forces ports into the legal 0-65535 range (anything higher/lower gets pulled to the boundary). And re-check protocols against the valid list, replacing bad ones with the mode (most common). This is where you, the human, encode "here's what makes sense for network data."

**Final check:**
```python
print(df.describe(include='all'))
```

`.describe()` gives summary statistics (counts, means, ranges) for every column - your "did the cleaning work?" verification. You eyeball it to confirm distributions look reasonable and no junk survived.

---

### The takeaway

Don't memorize every line - internalize the **workflow**:

1. **Define "valid"** for each column, and find rows that violate it (the `data[~...apply(check)]` pattern).
2. **Choose: drop or impute.** Drop = simple but loses data. Impute = keeps data but requires guessing values.
3. **Impute in stages:** normalize junk → NaN, fill NaN with median/mode, then apply domain-specific fixes.
4. **Verify** with `.describe()`.

This is the least glamorous part of ML and genuinely ~70-80% of real work. The good news: it's very learnable, mostly logic and pandas, no heavy math. Save the drop-vs-impute patterns to your snippets notebook - you'll reuse them constantly.