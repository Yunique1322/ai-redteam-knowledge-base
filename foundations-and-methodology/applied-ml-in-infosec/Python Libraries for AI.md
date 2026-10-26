# Python Libraries For AI

Here's the one-line split to anchor everything:

- **Scikit-learn** = the easy, plug-and-play library for "classic" ML (spam filter, anomaly detection). You'll use this for projects 1 and 2.
- **PyTorch** = the heavier library for deep learning / neural networks (the malware CNN). You'll use this for project 3.

Scikit-learn is where you'll start, and it's genuinely beginner-friendly. Let me walk both.

---

### Scikit-learn - the "just call the function" library

The beautiful thing about scikit-learn: **every model works the same way.** Once you learn the pattern, you know how to use _all_ of them. The pattern is three steps:



```python
model = LogisticRegression() # 1. pick a model
model.fit(X_train, y_train) # 2. train it
y_pred = model.predict(X_test) # 3. use it
```

That's the whole thing. `fit` = learn from data. `predict` = make guesses on new data. Swap `LogisticRegression()` for `SVC()` or `RandomForestClassifier()` and everything else stays identical. This consistency is why scikit-learn is beloved - the algorithms you learned in Module 1 are all one line away.

Now, the supporting cast - the code around that core pattern:

**Naming convention (worth knowing up front):** `X` = your input features (the data), `y` = the labels/answers. Capital `X` because it's a table (many rows/columns), lowercase `y` because it's a single column. You'll see this everywhere.

#### **Preprocessing - getting data ready before `fit`:**


##### Scaling
```python
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```
This is **feature scaling** - from Module 1 (remember: distance-based models break if features are on different scales, like age 0-100 vs income 0-100,000). `StandardScaler` recenters everything to a fair range. `fit_transform` = "learn the scaling _and_ apply it" in one step. (`MinMaxScaler` and `RobustScaler` are just other flavors - same idea.)

- `StandardScaler` : Standardizes features by removing the mean and scaling to unit variance.
- `MinMaxScaler` : Scales features to a given range, typically between 0 and 1.
- `RobustScaler` : Scales features using statistics that are robust to outliers.



##### Encoding
```python
encoder = OneHotEncoder()
X_encoded = encoder.fit_transform(X)
```
**Encoding** - models only understand numbers, not text like "red/green/blue." This converts categories into numeric columns. (`OneHotEncoder` makes a 0/1 column per category; `LabelEncoder` assigns each category a number.) You'll need this whenever data has text categories.


##### Missing Values
```python
imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X)
```

**Handling missing values** - real datasets have holes. This fills them (here, with the average). Necessary because most models crash on missing data.

##### Train test Split
**The single most important helper - splitting your data:**
```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
```
This is _crucial_ and you'll use it in every project. It splits your data: **80% to train on, 20% held back to test on** (`test_size=0.2`). Why? To check if the model actually _learned_ vs. just _memorized_. You train on the 80%, then test on the 20% it's never seen. If it does well on unseen data, it generalizes. This is the practical guard against **overfitting** - that recurring villain from Module 1.

You get four pieces back: `X_train`/`y_train` (features + answers to learn from) and `X_test`/`y_test` (features + answers to check against).


**Cross-validation - a fancier test:**
```python
scores = cross_val_score(model, X, y, cv=5)
```

Instead of one 80/20 split, this splits the data 5 ways and tests 5 times, giving a more reliable score. Nice-to-know; you'll meet it but `train_test_split` is the workhorse.


**Metrics - scoring the model:**
```python
accuracy = accuracy_score(y_test, y_pred)
```

Compares predictions (`y_pred`) against truth (`y_test`). And here's your payoff from Module 1 - `precision_score`, `recall_score`, `f1_score` are all right here. Remember the spam-filter precision/recall tradeoff? You'll now _compute_ those numbers on a real model. `mean_squared_error` is the regression version (the "squared error" from OLS).

---

### PyTorch - the deep learning library

PyTorch is more hands-on because neural networks need more control. Don't be intimidated - you'll mostly _adapt_ provided code, not write from scratch. But here's what the pieces mean so it reads clearly:

**Tensors - PyTorch's basic data unit:**

```python
x = torch.tensor([1.0, 2.0, 3.0])
```

A **tensor** is just a multi-dimensional array of numbers - basically a NumPy array that can run on a GPU for speed. Everything in PyTorch (data, weights, predictions) is a tensor. Think "list of numbers, but supercharged."





**Building a network - stacking layers:**
only one layer of 128 neurons in this example(128 is not a rule but a choice)
```python
model = nn.Sequential(
    nn.Linear(784, 128), # layer: 784 inputs → 128 neurons
    nn.ReLU(), # activation (the "bendiness" from Module 1)
    nn.Linear(128, 10), # layer: 128 → 10 outputs
    nn.Softmax(dim=1) # turn outputs into probabilities
)
```

**Read this against Module 1 and it's all familiar:** `nn.Linear` is a layer of neurons (the `w·x + b` you learned - "Linear" _is_ that operation). `nn.ReLU` is the activation function. `nn.Softmax` converts final outputs to probabilities (multi-class, remember?). This little model is literally: input → neurons → activation → neurons → probabilities. The MLP from Module 1, in code.

(The `784 → 128 → 10` example is the classic digit-recognizer: 784 = a 28×28 image flattened, 10 = digits 0-9.)

The `CustomModel(nn.Module)` class version below it does the _exact same thing_, just written out longhand for when you need more control. Notice its `forward()` method - that's the **forward pass** from Module 1 (data flowing through the layers). Same concept, now named in code.




**Training - the loop that does gradient descent:**

This is the big one, and it's Module 1's training loop made real. Let me annotate it:


```python
optimizer = optim.Adam(model.parameters(), lr=0.001) # the thing that updates weights
loss_fn = nn.CrossEntropyLoss() # the "how wrong am I" measure

for epoch in range(epochs): # repeat many times
    y_pred = model(x_batch) # 1. FORWARD PASS: predict
    loss = loss_fn(y_pred, y_batch) # 2. measure LOSS (how wrong)
    optimizer.zero_grad() # 3. reset gradients
    loss.backward() # 4. BACKPROP: compute gradients
    optimizer.step() # 5. update weights (step downhill)
```

**This is everything from Module 1, in five lines.** Look:

- `model(x_batch)` → forward pass (predict)
- `loss_fn(...)` → compute the **loss** (the concept I kept flagging!)
- `loss.backward()` → **backpropagation** (compute gradients - the direction to adjust)
- `optimizer.step()` → **gradient descent** (take a step downhill)
- Loop → repeat until loss shrinks

`loss.backward()` is where PyTorch's **autograd** shines - it _automatically_ computes all the gradients for you (the calculus you were worried about doing by hand? PyTorch does it). This is why nobody hand-derives backprop anymore.

`lr=0.001` is the **learning rate** - the step size from Module 1. `epochs` = how many times to loop through the data. `optimizer.zero_grad()` is just PyTorch bookkeeping (clears old gradients before computing new ones).

**Loss functions - pick based on task** (Module 1 mentioned this):

- `CrossEntropyLoss` → multi-class classification
- `BCEWithLogitsLoss` → binary (yes/no)
- `MSELoss` → regression (the squared error again)


**Data loading:**

```python
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
```

Feeds data to the model in **batches** of 32 at a time (not all at once - more efficient), and `shuffle=True` mixes the order each round so the model doesn't learn the sequence. You'll use this to feed the malware CNN.


**Saving/loading:**

```python
torch.save(model.state_dict(), 'model.pth') # save trained model
```

Saves the learned weights so you don't retrain every time.