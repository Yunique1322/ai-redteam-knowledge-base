# The Spam Dataset

**The dataset itself**

You're using the **SMS Spam Collection** - 5,574 real text messages, each labeled **ham** (legitimate) or **spam** (unwanted). It's a well-known research dataset, curated specifically for building spam filters. So your task: train a model to tell ham from spam based on the message text. Simple, real, and a perfect first project.

(Note "ham" = the standard term for "legitimate, not-spam." Two classes total: ham and spam.)

---

**Block 1 - Download the dataset programmatically**

```python
import requests
import zipfile
import io

# URL of the dataset
url = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"
# Download the dataset
response = requests.get(url)
if response.status_code == 200:
    print("Download successful")
else:
    print("Failed to download the dataset")
```

What's happening:

- `requests` = the library for fetching things from the web. `requests.get(url)` sends an HTTP GET request - literally "go download what's at this URL."
- `response` holds what came back (the zip file's data, plus metadata).
- `response.status_code == 200` -> **200 is the HTTP code for "OK, success."** So this checks: did the download actually work? If yes, print success; if not, print failure. Good practice - verify before proceeding.

If you run this and see "Download successful," you're good. If it fails, it's almost always a network issue - tell me and we'll sort it.

---

**Block 2 - Extract the zip**

python

```python
# Extract the dataset
with zipfile.ZipFile(io.BytesIO(response.content)) as z:
    z.extractall("sms_spam_collection")
    print("Extraction successful")
```

The data came as a `.zip`, so you unzip it:

- `response.content` = the raw binary bytes of the downloaded zip.
- `io.BytesIO(...)` = wraps those bytes so Python can treat the in-memory data like a file (without saving it to disk first). "Pretend these bytes are a file."
- `zipfile.ZipFile(...)` = opens it as a zip archive.
- `z.extractall("sms_spam_collection")` = unzips everything into a folder named `sms_spam_collection`.
- `with ... as z:` = Python's clean way to open something and auto-close it when done (tidy resource handling).

After this, the actual data file lives in that folder.

---

**Block 3 - Verify what got extracted**

python

```python
import os

# List the extracted files
extracted_files = os.listdir("sms_spam_collection")
print("Extracted files:", extracted_files)
```

- `os.listdir(...)` = lists the files in that folder. Just confirming the extraction worked and seeing the filename (you're looking for `SMSSpamCollection`). A sanity check before loading.

![List the extracted files](/assets/Pasted%20image%2020260828170127.png)
---

**Block 4 - Load into a DataFrame** (the important one)

```python
import pandas as pd

# Load the dataset
df = pd.read_csv(
    "sms_spam_collection/SMSSpamCollection",
    sep="\t",
    header=None,
    names=["label", "message"],
)
```

This loads the file into a DataFrame (your Python spreadsheet). Three parameters matter here, and they're worth understanding because you'll adjust these for different files:

- `sep="\t"` -> the file is **tab-separated**, not comma-separated. `\t` means "tab." (`read_csv` defaults to commas, but this file uses tabs between the label and the message, so you tell it that.) TSV instead of CSV - same idea, different separator.
- `header=None` -> this file has **no header row** (no column-name row at the top). Without this, pandas would mistake the first data row for column names. So you say "there's no header, treat row 1 as data."
- `names=["label", "message"]` -> since there's no header, _you_ name the columns. Column 1 = `label` (ham/spam), column 2 = `message` (the text). Now you can refer to `df['label']` and `df['message']`.

Result: a two-column table - `label` and `message` - ready to work with.

---

**Block 5 - Inspect the data** (the "first contact" ritual again)

python

```python
# Display basic information about the dataset
print("-------------------- HEAD --------------------")
print(df.head())
print("-------------------- DESCRIBE --------------------")
print(df.describe())
print("-------------------- INFO --------------------")
print(df.info())
```

Same ritual from the datasets section, three views:

- `df.head()` -> first few rows - eyeball what the data looks like (you'll see ham/spam labels next to message text).
- `df.describe()` -> summary stats. On text data it's limited, but it'll tell you things like how many unique messages there are and the most frequent one.
- `df.info()` -> column types, non-null counts, total rows. Confirms both columns loaded and shows if anything's missing.

---

**Block 6 - Check for missing values**

python

```python
# Check for missing values
print("Missing values:\n", df.isnull().sum())
```

`isnull().sum()` from before - counts empty cells per column. You want zeros here (a clean dataset). Confirms no incomplete entries.

---

**Block 7 - Check and remove duplicates** (the new technique)

python

```python
# Check for duplicates
print("Duplicate entries:", df.duplicated().sum())

# Remove duplicates if any
df = df.drop_duplicates()
```

This is new and worth noting:

- `df.duplicated()` -> marks each row True if it's an exact copy of an earlier row. `.sum()` counts them.
- `df.drop_duplicates()` -> removes those duplicate rows.

**Why remove duplicates?** If the same message appears many times, the model over-learns from it - it gets _over_-weighted, skewing the model toward those repeated examples. Also, duplicates can leak between your train and test sets (the same message in both), inflating your score dishonestly. Removing them keeps evaluation fair. (This is a data-quality step, same spirit as the cleaning from earlier - dedup so no single example dominates.)