# Training And Evaluation

**The model choice: Multinomial Naive Bayes**

`MultinomialNB` - the "counts" flavor of Naive Bayes from Module 1. It's ideal here because bag-of-words gives _word counts_, and Multinomial NB is built exactly for count data. Perfect match for text classification.

---

**New tool 1: Pipeline - bundling steps together**

```python
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

# Build the pipeline by combining vectorization and classification
pipeline = Pipeline([
    ("vectorizer", vectorizer),
    ("classifier", MultinomialNB())
])
```

A **Pipeline** chains steps into one object. Here it links two stages: the `vectorizer` (text → counts) and the `classifier` (counts → prediction).

Why bother? Because now, whenever you give the pipeline raw text, it _automatically_ vectorizes then classifies - in one call. You can't accidentally forget to vectorize, or apply different settings at train vs. test time. It guarantees the **same transformation every time.** This matters more than it sounds: a huge source of ML bugs is preprocessing training data one way and test data another. Pipeline eliminates that. Think "glue the steps so they always run together, in order."

---

**New tool 2: GridSearchCV - automatic hyperparameter tuning**

python

```python
# Define the parameter grid for hyperparameter tuning
param_grid = {
    "classifier__alpha": [0.01, 0.1, 0.15, 0.2, 0.25, 0.5, 0.75, 1.0]
}

# Perform the grid search with 5-fold cross-validation and the F1-score as metric
grid_search = GridSearchCV(
    pipeline,
    param_grid,
    cv=5,
    scoring="f1"
)

# Fit the grid search on the full dataset
grid_search.fit(df["message"], y)

# Extract the best model identified by the grid search
best_model = grid_search.best_estimator_
print("Best model parameters:", grid_search.best_params_)
```

Remember **hyperparameters** from Module 1 - the knobs you set _before_ training, tuned by trial and error? `GridSearchCV` automates that trial and error. Instead of manually testing values, it tries them _all_ and keeps the best.

The knob here is **`alpha`** - a "smoothing" factor specific to Naive Bayes. Here's what it does: recall NB _multiplies_ word probabilities. If a word never appeared in spam during training, its spam probability is 0 - and multiplying by 0 zeroes out the whole calculation, which is too harsh (one unseen word shouldn't veto everything). Alpha adds a tiny cushion so no probability is ever exactly 0. It handles unseen words gracefully. Tuning it balances the model (the "bias vs. variance" tradeoff - too much smoothing dulls the model, too little makes it brittle).

Decoding the code:

- `param_grid` → the values to try. `"classifier__alpha"` names _which_ step's parameter (the double underscore `__` means "the alpha of the classifier step in the pipeline"). It lists 8 alpha values to test.
- `GridSearchCV(pipeline, param_grid, cv=5, scoring="f1")` → try every alpha, using **5-fold cross-validation** (`cv=5` - that robust evaluation from earlier: split 5 ways, test 5 times per value), and judge by **F1-score** (`scoring="f1"` - the balanced precision/recall metric, the right choice for spam where both false alarms and misses matter).
- `grid_search.fit(...)` → runs the whole search: 8 alphas × 5 folds = 40 mini-trainings, finding the best alpha.
- `best_model = grid_search.best_estimator_` → grabs the winning pipeline (best alpha, ready to use). `best_params_` prints which alpha won.

So in a few lines: it auto-tuned the model and handed you the best version. That's `GridSearchCV` - "try all the knob settings, keep the winner."

---

**Reading your confusion matrix (the image)**

This is the evaluation, and your image shows it beautifully. A **confusion matrix** lays out all four outcome types (from the metrics section) in a grid. Rows = the _truth_, columns = the _model's prediction_. Here's yours:

![Extract the best model identified by the grid search](/assets/Pasted%20image%2020260828175307.png)
Reading each cell:

- **889 (top-left)** = **True Negatives** - ham correctly called ham. ✓
- **5 (top-right)** = **False Positives** - ham _wrongly_ called spam. ✗ (5 legit messages sent to spam folder)
- **0 (bottom-left)** = **False Negatives** - spam _wrongly_ called ham. ✗ (spam that slipped through - **zero!**)
- **140 (bottom-right)** = **True Positives** - spam correctly caught. ✓

**What this tells you about the model:**

- **Recall = 140/(140+0) = 100%** - it caught _every single spam_ (0 misses). Perfect recall.
- **Precision = 140/(140+5) = 96.5%** - when it says spam, it's right 96.5% of the time (5 false alarms).

This is a strong result, and notice the _tradeoff_ in action: the model catches all spam (great recall) but at the cost of 5 false alarms (slightly imperfect precision). For spam, that's usually the right balance - you'd rather occasionally quarantine a good message than let spam through. The diagonal (889, 140) is where correct predictions live; off-diagonal (5, 0) is errors. A good model has big diagonal numbers and tiny off-diagonal ones - exactly what you see.

---

**Testing on new messages - the real-world check**

python

```python
# Example SMS messages for evaluation
new_messages = [
    "Congratulations! You've won a $1000 Walmart gift card. Go to http://bit.ly/1234 to claim now.",
    "Hey, are we still meeting up for lunch today?",
    "Urgent! Your account has been compromised. Verify your details here: www.fakebank.com/verify",
    "Reminder: Your appointment is scheduled for tomorrow at 10am.",
    "FREE entry in a weekly competition to win an iPad. Just text WIN to 80085 now!",
]
```

Fresh messages the model never saw - the true test of generalization.

**The critical rule - preprocess new data identically:**

```python
import numpy as np
import re

# Preprocess function that mirrors the training-time preprocessing
def preprocess_message(message):
    message = message.lower()
    message = re.sub(r"[^a-z\s$!]", "", message)
    tokens = word_tokenize(message)
    tokens = [word for word in tokens if word not in stop_words]
    tokens = [stemmer.stem(word) for word in tokens]
    return " ".join(tokens)
```

This bundles the _exact same 6-step preprocessing_ from before into one function. **This is essential:** new messages must be cleaned identically to training data, or the model sees a mismatched format and fails. Same lowercase, same punctuation rule, same tokenize/stopword/stem. (This is why Pipeline is so valued - it enforces this automatically; here they do it manually for the text prep.)

```python
# Preprocess and vectorize messages
processed_messages = [preprocess_message(msg) for msg in new_messages]
```

Applies the cleaning to all new messages.
```python
# Transform preprocessed messages into feature vectors
X_new = best_model.named_steps["vectorizer"].transform(processed_messages)
```

Vectorizes them using the **saved vectorizer from the trained pipeline** (`named_steps["vectorizer"]` reaches into the pipeline to grab it). Crucially it uses `.transform()` **not** `.fit_transform()` - it applies the _already-learned_ vocabulary, doesn't build a new one. (New data must map onto the training vocabulary; re-fitting would create a different, incompatible vocabulary. This distinction - fit on train, transform on new - is important.)

```python
# Predict with the trained classifier
predictions = best_model.named_steps["classifier"].predict(X_new)
prediction_probabilities = best_model.named_steps["classifier"].predict_proba(X_new)
```

- `predict(X_new)` → the label: 1 (spam) or 0 (ham) for each message.
- `predict_proba(X_new)` → the _confidence_: probability for each class. This is the sigmoid-style probability output - not just "spam" but "97% spam." Useful for seeing how _sure_ the model is.

**Displaying results:**

python

```python
# Display predictions and probabilities for each evaluated message
for i, msg in enumerate(new_messages):
    prediction = "Spam" if predictions[i] == 1 else "Not-Spam"
    spam_probability = prediction_probabilities[i][1] # Probability of being spam
    ham_probability = prediction_probabilities[i][0] # Probability of being not spam
    
    print(f"Message: {msg}")
    print(f"Prediction: {prediction}")
    print(f"Spam Probability: {spam_probability:.2f}")
    print(f"Not-Spam Probability: {ham_probability:.2f}")
    print("-" * 50)
```

Loops through each message and prints its text, predicted label, and both probabilities. Looking at the sample output - the gift-card and iPad-competition messages score 1.00 spam, the lunch and appointment ones score 0.00 spam, and the fake-bank phishing scores 0.94 spam. The model nails all five. Notice it even catches the phishing message (0.94) - that's a genuine security win, since phishing is the dangerous kind of spam.

---

**Saving the model with joblib**

python

```python
import joblib

# Save the trained model to a file for future use
model_filename = 'spam_detection_model.joblib'
joblib.dump(best_model, model_filename)

print(f"Model saved to {model_filename}")
```

`joblib` saves the trained model to disk. Why? Training takes time; you don't want to retrain every time. Saving "freezes" the finished model - learned parameters, vocabulary, everything - so you can reload it instantly later. `joblib.dump(model, filename)` = save. This is how models get _deployed_ - train once, save, load in production for instant predictions.

python

```python
# Load the saved model
loaded_model = joblib.load(model_filename)

# Preprocess new messages before prediction
new_data_processed = [preprocess_message(msg) for msg in new_messages]

# Make predictions on the preprocessed data
predictions = loaded_model.predict(new_data_processed)
```

`joblib.load` brings it back, ready to predict. Note the reminder: **new data still needs the same preprocessing** before prediction - saving the model doesn't save you from cleaning inputs consistently.