# Preprocessing And Splitting The Dataset

**The goal: shape raw network data into model-ready form**

Three things happen: (1) create the labels/targets, (2) encode the categorical columns, (3) select numeric features and split. Let's go.

---

**Creating the binary target - normal vs. attack**

```python
# Binary classification target
# Maps normal traffic to 0 and any type of attack to 1
df['attack_flag'] = df['attack'].apply(lambda a: 0 if a == 'normal' else 1)
```

The `attack` column contains labels like `normal`, `neptune`, `smurf`, etc. This collapses them into a simple **0/1**: if the label is exactly `"normal"` -> 0, anything else -> 1 (attack). Creates a new column `attack_flag`.

This is the simplest framing: _is this traffic malicious, yes or no?_ Same "convert labels to numbers" move as spam (spam=1/ham=0), just here it's attack=1/normal=0. The `lambda a: 0 if a == 'normal' else 1` reads: "for each label, give 0 if it's 'normal', otherwise 1."

---

**Creating the multi-class target - which _type_ of attack** (the new concept)

Binary tells you _if_ it's an attack; multi-class tells you _what kind_. This is more useful for a real IDS - knowing you're facing a DoS vs. a port scan changes your response.

python

```python
# Multi-class classification target categories
dos_attacks = ['apache2', 'back', 'land', 'neptune', 'mailbomb', 'pod', 
               'processtable', 'smurf', 'teardrop', 'udpstorm', 'worm']
probe_attacks = ['ipsweep', 'mscan', 'nmap', 'portsweep', 'saint', 'satan']
privilege_attacks = ['buffer_overflow', 'loadmdoule', 'perl', 'ps', 
                     'rootkit', 'sqlattack', 'xterm']
access_attacks = ['ftp_write', 'guess_passwd', 'http_tunnel', 'imap', 
                  'multihop', 'named', 'phf', 'sendmail', 'snmpgetattack', 
                  'snmpguess', 'spy', 'warezclient', 'warezmaster', 
                  'xclock', 'xsnoop']

def map_attack(attack):
    if attack in dos_attacks:
        return 1
    elif attack in probe_attacks:
        return 2
    elif attack in privilege_attacks:
        return 3
    elif attack in access_attacks:
        return 4
    else:
        return 0

# Assign multi-class category to each row
df['attack_map'] = df['attack'].apply(map_attack)
```

The many specific attack names get grouped into **four families**, each mapped to a number:

- **0** = normal traffic
- **1** = **DoS** (Denial of Service - flood/crash the target, e.g. `neptune`, `smurf`)
- **2** = **Probe** (reconnaissance/scanning, e.g. `nmap`, `satan` - looking for weaknesses)
- **3** = **Privilege Escalation** (gain admin/root, e.g. `buffer_overflow`, `rootkit`)
- **4** = **Access** (break in / bypass authentication, e.g. `guess_passwd`)

The `map_attack` function checks which list an attack name is in and returns the matching number. `df['attack'].apply(map_attack)` runs it on every row, creating the `attack_map` column.

Interesting detail worth noticing for your background: these four categories are essentially the **stages of an attack** - probe (recon) -> access (break in) -> privilege escalation (get root) -> DoS (or disrupt). If those sound like the phases of a pentest/kill-chain, that's exactly what they are. The dataset is organized around how real attacks unfold.

**Why create both targets?** Flexibility. The binary target trains a "is it an attack?" detector; the multi-class target trains a "what kind of attack?" classifier. Same features, two different questions you can ask. You'll use the multi-class one (`attack_map`) here.

---

**Encoding categorical variables**

python

```python
# Encoding categorical variables
features_to_encode = ['protocol_type', 'service']
encoded = pd.get_dummies(df[features_to_encode])
```

Two columns are text categories: `protocol_type` (tcp/udp/icmp) and `service` (http/ftp/ssh...). Models need numbers, so you one-hot encode them - same concept from the transformation section (a 0/1 column per category, no fake ordering).

The new tool: **`pd.get_dummies()`** - pandas' built-in one-hot encoder. It's simpler than scikit-learn's `OneHotEncoder` for quick jobs: hand it the categorical columns, it returns a DataFrame of 0/1 columns (`protocol_type_tcp`, `service_http`, etc.). "Dummies" is the statistics term for these 0/1 indicator columns. Functionally the same as one-hot encoding you already know - just a more convenient function for pandas DataFrames. (`get_dummies` is worth knowing; you'll reach for it constantly - it's the quick way to one-hot encode.)

---

**Selecting numeric features**

python

```python
# Numeric features that capture various statistical properties of the traffic
numeric_features = [
    'duration', 'src_bytes', 'dst_bytes', 'wrong_fragment', 'urgent', 'hot', 
    'num_failed_logins', 'num_compromised', 'root_shell', 'su_attempted', 
    'num_root', 'num_file_creations', 'num_shells', 'num_access_files', 
    'num_outbound_cmds', 'count', 'srv_count', 'serror_rate', 
    'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate', 'same_srv_rate', 
    'diff_srv_rate', 'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count', 
    'dst_host_same_srv_rate', 'dst_host_diff_srv_rate', 
    'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate', 
    'dst_host_serror_rate', 'dst_host_srv_serror_rate', 'dst_host_rerror_rate', 
    'dst_host_srv_rerror_rate'
]
```

This picks which numeric columns to feed the model - the connection stats (bytes, duration, error rates, counts). It deliberately _excludes_ the label columns (`attack`, `attack_flag`, `attack_map`, `level`) - you don't feed the answer in as a feature. It's just a list of column names for now; used in the next step.

Note what's included: raw volume (`src_bytes`, `duration`), suspicious-activity counts (`num_failed_logins`, `root_shell`), and statistical rates (`serror_rate`, etc.). These are the signals the forest will use to spot attacks.

---

**Combining features into the final input**

python

```python
# Combine encoded categorical variables and numeric features
train_set = encoded.join(df[numeric_features])

# Multi-class target variable
multi_y = df['attack_map']
```

- `encoded.join(df[numeric_features])` -> glues the one-hot columns and the numeric columns side by side into one table, `train_set`. This is your complete feature matrix `X` (the network stats + encoded protocol/service). (`.join` merges columns, like the `pd.concat` from before.)
- `multi_y = df['attack_map']` -> the labels `y` (which attack category).

So `train_set` = X (features), `multi_y` = y (labels). Same X/y setup as always, now assembled for the network data.

---

**Splitting into train / validation / test**

python

```python
# Split data into training and test sets for multi-class classification
train_X, test_X, train_y, test_y = train_test_split(train_set, multi_y, test_size=0.2, random_state=1337)
```

First split: 80% train, 20% test - the untouched final set. Exactly the `train_test_split` pattern from before. (`random_state=1337` for reproducibility again.)

python

```python
# Further split the training set into separate training and validation sets
multi_train_X, multi_val_X, multi_train_y, multi_val_y = train_test_split(train_X, train_y, test_size=0.3, random_state=1337)
```

Second split: carve a **validation** set out of the training data (30% of the 80% here). Same two-stage split logic from the transformation section - train to fit, validation to tune, test for the final honest score.

The result, four sets (the doc lists them):

- `multi_train_X`/`multi_train_y` -> fit the model
- `multi_val_X`/`multi_val_y` -> tune hyperparameters
- `test_X`/`test_y` -> final untouched evaluation