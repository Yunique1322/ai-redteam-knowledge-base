# Model Reverse Engineering

## The concept

You cannot see the model, but you can query it. So you **send many inputs to its API, record the outputs, and train your own "surrogate" model on those input-output pairs until it mimics the original.** It is a black-box attack: no need for the architecture or the training data.

Why it matters: **IP theft** (cloning a model that took years to build), **a local copy to craft adversarial examples against** (spam, fraud, face recognition bypasses), and **model inversion** (a good enough clone can leak facts about the original's sensitive training data). The catch for defenders is that the attacker's queries look exactly like normal user traffic, so **rate limiting is the main mitigation**.

The lab target: a penguin classifier (Adelie vs Gentoo) that takes `flipper_length` and `body_mass` as GET parameters and returns the species. The goal is to steal it as a surrogate without ever seeing its training data.

## Step 0: probe the API

```bash
curl 'http://172.17.0.2/?flipper_length=150&body_mass=5000'
```

Returns `{"result": "Adelie"}`. That IP is your lab instance's, swap it for yours.

## Step 1: config

```python
N_SAMPLES = 100

MIN_FLIPPER_LENGTH = 150
MAX_FLIPPER_LENGTH = 250

MIN_BODY_MASS = 2500
MAX_BODY_MASS = 6500

CLASSIFIER_URL = "http://172.17.0.2:80/"
```

**The bounds matter.** Restricting inputs to realistic penguin ranges (flipper 150 to 250mm, mass 2500 to 6500g) means every sampled point is a useful, in-distribution query. ==Good bounds = higher-quality data = far fewer queries needed.== When you cannot bound the inputs for a problem, you need many more samples to compensate.

## Step 2: sample random input points

```python
import random
import pandas as pd

samples = {
    "Flipper Length (mm)": [],
    "Body Mass (g)": []
}

for i in range(N_SAMPLES):
    samples["Flipper Length (mm)"].append(random.uniform(MIN_FLIPPER_LENGTH, MAX_FLIPPER_LENGTH))
    samples["Body Mass (g)"].append(random.uniform(MIN_BODY_MASS, MAX_BODY_MASS))

samples_df = pd.DataFrame(samples)
print(samples_df.head())
```

Just uniform-random pairs inside the bounds. **This is your manufactured, unlabeled dataset.**

## Step 3: label them by querying the target

```python
import requests
import json

predictions = {"species": []}

for i in range(N_SAMPLES):
    sample = {
                "flipper_length": samples["Flipper Length (mm)"][i],
                "body_mass": samples["Body Mass (g)"][i]
            }

    prediction = json.loads(requests.get(CLASSIFIER_URL, params=sample).text).get("result")
    predictions["species"].append(prediction)

predictions_df = pd.DataFrame(predictions)
print(predictions_df.head())
```

**The target model IS your labeling oracle.** Each query turns one of your random points into a labeled training example. This loop is also the noisy part an attacker cannot hide, which is why rate limiting bites here.

## Step 4: train the surrogate

```python
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import joblib

surrogate_model = make_pipeline(StandardScaler(), LogisticRegression())
surrogate_model.fit(samples_df, predictions_df)

# save classifier to a file
joblib.dump(surrogate_model, 'surrogate.joblib')
```

**You do not need to match the original's architecture**, just pick one suited to the task. Binary classification here, so logistic regression. One tiny practical note: passing `predictions_df` (a one-column DataFrame) as `y` can throw a harmless sklearn warning; if it annoys you, use `predictions_df["species"]`. Not required.

## Step 5: submit and score

```python
with open('surrogate.joblib', 'rb') as f:
    file = f.read()

r = requests.post(CLASSIFIER_URL + '/model', files={'file': ('surrogate.joblib', file)})

print(json.loads(r.text))
```

Result: **about 98.5% accuracy from only 100 queries**, with zero real training data. The two decision-boundary plots (original vs surrogate) are nearly identical; more samples would shrink the gap further.

## Exam takeaways

- **The whole attack is: bound the inputs, mass-query the API to label random points, train any suitable surrogate.** Simple, scriptable, and exactly the kind of thing you would run against a model API on the exam.
- **To test for this vuln on a target: check whether the model API has a rate limit.** No rate limit, or a generous one, means extraction is feasible, and that is your finding.
- **Mitigation and remediation text: rate limiting** (tuned so it slows attackers without hurting real users), since you cannot distinguish malicious from benign queries any other way.

## Labs

![Labs](/assets/Pasted%20image%2020260926160925.png)