# Network Anomaly Detection

**The task: spot malicious network traffic**

Same anomaly-detection idea from Module 1 - learn what normal looks like, flag deviations. But now applied to network connections: normal traffic vs. intrusions/attacks. This is real intrusion detection, framed as ML.

---

**New algorithm: Random Forests**

This is your first _ensemble_ method, and it's a natural extension of decision trees from Module 1. The idea is simple and clever:

**A Random Forest = many decision trees voting together.**

Remember a single decision tree from Module 1 - asks yes/no questions to classify, but tends to _overfit_ (memorizes training data, generalizes poorly). The fix: instead of one tree, build _hundreds_, each slightly different, and let them **vote**. The majority wins.

Why does voting help? It's the "wisdom of crowds." One tree might overfit to a quirk in the data, but hundreds of diverse trees averaging out their individual mistakes gives a much more robust, reliable answer. One tree's error gets outvoted by the others. This is why random forests **generalize better and overfit less** than a single tree - directly solving the tree's biggest weakness.

The three ingredients that make the trees _diverse_ (crucial - if all trees were identical, voting would be pointless):

- **Bootstrapping** - each tree trains on a _random subset_ of the data (sampled with replacement). So every tree sees slightly different data, learns slightly differently.
- **Random features at each split** - when a tree decides which question to ask, it only considers a _random handful_ of features, not all of them. This forces trees to be different from each other (prevents them all keying on the same dominant feature).
- **Voting** - after all trees are built, they vote; majority class wins (or average, for regression).

The "random" in Random Forest comes from those first two - random data + random features = diverse trees. The "forest" is the collection of trees. Mental model: **ask a hundred slightly-different experts and go with the majority - more reliable than trusting one.**

**How it's used for anomaly detection here:** train the forest on _normal_ traffic patterns, then evaluate new traffic against that learned normal. Traffic that doesn't fit (or that the trees are unsure about) gets flagged as a potential anomaly/intrusion.

---

**The dataset: NSL-KDD**

A standard benchmark for intrusion detection research. It contains labeled network connections - both **normal** and various **attack** types. It's an improved version of an older dataset (KDD Cup 1999), cleaned up to remove duplicates and fix class imbalance (recall why balance matters - an imbalanced dataset produces a biased model). You're using a modified version.

Key point: it supports both **binary** (normal vs. attack) and **multi-class** (which _specific_ attack type) detection. Each network connection is described by many features (duration, bytes, protocol, connection stats) plus a label saying if it's an attack.

---

**Downloading it**

python

```python
import requests, zipfile, io

# URL for the NSL-KDD dataset
url = "https://academy.hackthebox.com/storage/modules/292/KDD_dataset.zip"

# Download the zip file and extract its contents
response = requests.get(url)
z = zipfile.ZipFile(io.BytesIO(response.content))
z.extractall('.') # Extracts to the current directory
```

Same download-and-extract pattern from the spam project - `requests.get` fetches the zip, `zipfile`/`io.BytesIO` unzip it. One small difference: `z.extractall('.')` - the `'.'` means "extract to the current directory" (where your notebook is), rather than a named subfolder. Otherwise identical to before.

---

**Importing libraries**

python

```python
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
```

The toolkit for this project:

- `numpy`/`pandas` → data handling (familiar).
- `RandomForestClassifier` → the algorithm (from `sklearn.ensemble` - "ensemble" because it combines many trees).
- `train_test_split` → splitting data (familiar).
- The metrics → `accuracy_score`, `precision_score`, `recall_score`, `f1_score` (your Module 1 metrics), plus `confusion_matrix` and `classification_report` (which auto-computes all metrics at once - you'll use this for evaluation).
- `seaborn`/`matplotlib` → plotting libraries, for visualizing results (like drawing that confusion matrix heatmap you saw). `sns` and `plt` are their universal nicknames.

Nothing to run differently - just loading tools.

---

**Defining column names**

python

```python
# Set the file path to the dataset
file_path = r'KDD+.txt'

# Define the column names corresponding to the NSL-KDD dataset
columns = [
    'duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes', 
    'land', 'wrong_fragment', 'urgent', 'hot', 'num_failed_logins', 'logged_in', 
    'num_compromised', 'root_shell', 'su_attempted', 'num_root', 'num_file_creations', 
    'num_shells', 'num_access_files', 'num_outbound_cmds', 'is_host_login', 'is_guest_login', 
    'count', 'srv_count', 'serror_rate', 'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate', 
    'same_srv_rate', 'diff_srv_rate', 'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count', 
    'dst_host_same_srv_rate', 'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate', 
    'dst_host_srv_diff_host_rate', 'dst_host_serror_rate', 'dst_host_srv_serror_rate', 
    'dst_host_rerror_rate', 'dst_host_srv_rerror_rate', 'attack', 'level'
]
```

The data file has no column headers, so you name them manually (same `header`/`names` idea from the spam `read_csv`). This long list is just the 43 features NSL-KDD tracks. You don't need to memorize them, but notice the _categories_ - they matter for the offensive angle:

- **Basic connection stats**: `duration`, `src_bytes` (bytes sent), `dst_bytes` (bytes received), `protocol_type`, `service`, `flag`.
- **Content/"suspicious activity" features**: `num_failed_logins`, `root_shell` (did they get a root shell?), `su_attempted`, `num_compromised`, `num_file_creations`, `num_shells` - these directly flag intrusion-like behavior.
- **Traffic pattern features**: `count`, `srv_count`, various `error_rate` and `srv_rate` stats - statistical patterns across connections.
- **Labels**: `attack` (the type of attack, or "normal") and `level` (severity).

`file_path = r'KDD+.txt'` points to the data file (the `r` prefix means "raw string" - treats backslashes literally, a habit for file paths on Windows; harmless here).

The `attack` column is your target - what the model learns to predict.

---

**Loading into a DataFrame**

python

```python
# Read the combined NSL-KDD dataset into a DataFrame
df = pd.read_csv(file_path, names=columns)
```

Loads the file, applying your column names. Same `read_csv` + `names` pattern from spam. Now `df` holds the network data as a labeled table.

python

```python
print(df.head())
```

The "first contact" peek - eyeball the data, confirm columns loaded right, see what network connections look like as rows.