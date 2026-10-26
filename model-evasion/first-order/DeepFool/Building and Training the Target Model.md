# Building And Training The Target Model

## The idea in one breath

This is the setup section for the DeepFool lab. It uses the same scaffold as the FGSM part, but the victim is now `MNISTClassifierWithDropout`, a slightly bigger CNN with **dropout**, trained for 5 epochs to about **98.95%**. The new idea is **caching**: save the trained model to `output/mnist_model.pth` and, next time, load it and check its accuracy instead of retraining.

> **Heads-up:** no mangled formulas in this section. One wording trap: the course says "two convolutional blocks... first block 25% dropout, the second outputs 64". The code comment says **both** conv blocks have 25% dropout. Trust the code comment.

---

## Part 1: Environment setup

```python
from htb_ai_library import (
    set_reproducibility,
    MNISTClassifierWithDropout,
    get_mnist_loaders,
    train_model,
    evaluate_accuracy,
    save_model,
    load_model,
    analyze_model_confidence,
    HTB_GREEN, NODE_BLACK, HACKER_GREY, WHITE,
    AZURE, NUGGET_YELLOW, MALWARE_RED, VIVID_PURPLE, AQUAMARINE
)

set_reproducibility(1337)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
```

| Code | What it does |
| ------------------------------------------------------------------------------ | ------------------------------------------------------------------------- |
| `set_reproducibility`, `get_mnist_loaders`, `train_model`, `evaluate_accuracy` | Same tools as the FGSM part (Section 4). |
| `MNISTClassifierWithDropout` | **New**: the victim model with dropout, replacing `SimpleCNN`. |
| `save_model`, `load_model` | **New**: write a trained model to disk and read it back. |
| `analyze_model_confidence` | **New**: summarizes how confident the model is on right vs wrong answers. |
| `HTB_GREEN, ...` | Plot colours for later visualizations. |
| `set_reproducibility(1337)` | Same seed as before, so runs are repeatable. |
| `device = ...` | GPU if available, otherwise CPU. |

> **If this errors:** `NameError: name 'torch' is not defined` (or `'os'`) means this block assumes the FGSM scaffold's imports already ran. Add `import os` and `import torch` (plus the other Section 4 imports) at the top.

---

## Part 2: Why dropout?

### Step 1: What dropout does (toy)

Take a layer whose 4 neurons output **[0.5, 0.8, 0.2, 0.6]**, with **dropout 50%**.

**In training mode**, each pass randomly switches off about half the neurons. Say the 1st and 4th are dropped:

- zeroed: [0, 0.8, 0.2, 0]
- the survivors are scaled up by 1 ÷ (1 − 0.5) = **×2** so the layer's total signal stays about the same: **[0, 1.6, 0.4, 0]**

The next pass drops a **different** random half, so the output changes every time even for the same image.

**In evaluation mode** (`model.eval()`), dropout is off and nothing is zeroed or scaled: **[0.5, 0.8, 0.2, 0.6]**, the same every time.

### Step 2: Why that helps the model

Because any neuron might vanish, the network can't rely on one specific "this pixel pattern means 7" path. It has to learn **redundant features**, several different ways of recognizing each digit. The course argues this gives boundaries that reflect real uncertainty, instead of boundaries wrapped tightly around memorized training images.

### Step 3: Why the rates differ

|Layer|Rate|Why|
|---|---|---|
|Conv blocks|25%|Thousands of spatially local activations. Dropping too many destroys the picture's spatial information.|
|FC1 (128 units)|50%|Few units, each connected to everything before it, so the overfitting risk is highest here.|

### Step 4: Why this matters for attacks

**Dropout must be OFF when you attack.** In training mode, every forward pass gives a different answer, so the gradients DeepFool reads would be random noise. `model.eval()` turns dropout off, which is the same "`model.eval()` YES" rule from the FGSM part, but now it really matters because this model has dropout.

> **An honest note the course skips:** dropout is a regularizer for **clean accuracy**, not an adversarial defence. A dropout model is not meaningfully harder to fool. The course uses it to get a well-trained, less overfitted target, which makes DeepFool's measurements more meaningful, not to make the model robust.

---

## Part 3: The architecture

```python
# MNISTClassifierWithDropout is imported from htb_ai_library
# The architecture internally defines:
# - Conv1: 1->32 channels, 3x3 kernel, ReLU, 2x2 pooling, 25% dropout
# - Conv2: 32->64 channels, 3x3 kernel, ReLU, 2x2 pooling, 25% dropout
# - FC1: 3136->128, ReLU, 50% dropout
# - FC2: 128->10 (logits)

model = MNISTClassifierWithDropout().to(device)
print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
```

|Code|What it does|
|---|---|
|`MNISTClassifierWithDropout()`|Builds a fresh, untrained model.|
|`.to(device)`|Moves it to the GPU or CPU.|
|`model.parameters()`|Every weight and bias tensor in the model.|
|`p.numel()`|How many numbers are in one tensor.|
|`sum(...)`|Adds them up to get the total trainable numbers.|
|`:,`|Prints with thousands commas, e.g. 421,642.|

### How the image shrinks through the model

|Stage|Shape (channels × height × width)|Meaning|
|---|---|---|
|Input|1 × 28 × 28|one greyscale MNIST image|
|Conv1 + ReLU|32 × 28 × 28|32 pattern detectors (a 3×3 kernel with padding keeps the size at 28)|
|Pool 2×2|32 × 14 × 14|halves height and width|
|Conv2 + ReLU|64 × 14 × 14|64 more complex detectors|
|Pool 2×2|64 × 7 × 7|halves again|
|Flatten|3136|64 × 7 × 7 = 3136 numbers in one row|
|FC1 + ReLU|128|decision features|
|FC2|10|one **logit** (raw score) per digit, the f_k(x) from Section 15|

### Counting parameters (toy first)

A layer from 3 inputs to 2 outputs has 3 × 2 = 6 weights plus 2 biases (one per output) = **8 parameters**.

The same rule for this model:

|Layer|Calculation|Parameters|
|---|---|---|
|Conv1|32 filters × (1 × 3 × 3) + 32 biases|320|
|Conv2|64 × (32 × 3 × 3) + 64|18,496|
|FC1|3136 × 128 + 128|401,536|
|FC2|128 × 10 + 10|1,290|
|Total||**421,642**|

The print should show about `Model parameters: 421,642`. If it shows a different number, the library's layers differ slightly from the description, which is harmless.

> **An honest note:** "3×3 kernel" plus "28×28 → 7×7" only works if the convolutions use **padding of 1**. Without padding, each 3×3 conv shrinks the image by 2 pixels and you'd never reach exactly 3136. The course doesn't mention it; `print(model)` will show `padding=(1, 1)` if I'm right.

---

## Part 4: Training with caching

### Block 1: Where to save

```python
model_path = 'output/mnist_model.pth'
os.makedirs('output', exist_ok=True)
```

|Code|What it does|
|---|---|
|`model_path = 'output/mnist_model.pth'`|The file where the trained model will live. `.pth` = PyTorch's save format.|
|`os.makedirs('output', exist_ok=True)`|Creates the `output` folder. `exist_ok=True` = don't error if it already exists.|

### Block 2: Try loading a cached model

```python
# Try loading cached model
if os.path.exists(model_path):
    print(f"Found cached model at {model_path}")
    model_data = load_model(model_path)
    model = model_data['model'].to(device)
    model.eval()

    # Validate cached model
    _, test_loader = get_mnist_loaders(batch_size=100, normalize=True)
    accuracy = evaluate_accuracy(model, test_loader, device)
    print(f"Cached model accuracy: {accuracy:.2f}%")

    if accuracy < 90.0:
        print("Accuracy below threshold, retraining required")
        model = None
else:
    model = None
```

| Code | What it does |
| ----------------------------------------- | ---------------------------------------------------------------------------- |
| `if os.path.exists(model_path):` | Is there a saved model from a previous run? |
| `model_data = load_model(model_path)` | Reads the saved **dictionary** (Block 4 shows what's in it). |
| `model = model_data['model'].to(device)` | Takes the model out of the dictionary and moves it to the device. |
| `model.eval()` | Dropout off: stable, repeatable outputs. |
| `_, test_loader = get_mnist_loaders(...)` | Gets only the test data. `_` means "throw away the train loader". Batch 100. |
| `accuracy = evaluate_accuracy(...)` | Scores the loaded model on all 10,000 test images. |
| `if accuracy < 90.0: ... model = None` | **Quality gate**: a bad or corrupted cache gets rejected. |
| `else: model = None` | No file, so mark it as "needs training". |

`model = None` is just a flag that means "we don't have a usable model yet". The next block checks for it.

> **If this errors:** `Weights only load failed` or `UnpicklingError` on newer PyTorch (2.6+). This file stores the **whole model object**, and newer PyTorch refuses to load objects by default. Fixes: delete `output/mnist_model.pth` and let it retrain, or update `htb_ai_library` (its `load_model` may already handle this). Only ever load a file like this if you made it yourself (see the security note below).

### Blocks 3 and 4: Train if needed, then save (merged)

The course split this into two blocks, but the "Persisting Model State" part is indented inside `if model is None:`, so it can't run on its own. **I merged them into one cell. Every line is unchanged.**

```python
# Train if needed
if model is None:
    print("Training new model...")
    train_loader, test_loader = get_mnist_loaders(batch_size=64, normalize=True)
    model = MNISTClassifierWithDropout().to(device)

    model = train_model(
        model, train_loader, test_loader,
        epochs=5, device=device
    )

    # Evaluate and cache
    accuracy = evaluate_accuracy(model, test_loader, device)
    print(f"Test Accuracy: {accuracy:.2f}%")

    save_model({
        'model': model,
        'architecture': 'MNISTClassifierWithDropout',
        'accuracy': accuracy,
        'training_config': {
            'epochs': 5,
            'batch_size': 64,
            'device': str(device)
        }
    }, model_path)
```

|Code|What it does|
|---|---|
|`if model is None:`|Only train when Block 2 found no good cached model.|
|`get_mnist_loaders(batch_size=64, normalize=True)`|Train and test data, 64 images per batch, in **model space** (normalized, as in the FGSM part).|
|`MNISTClassifierWithDropout().to(device)`|A fresh, untrained model.|
|`train_model(..., epochs=5, ...)`|5 passes over the 60,000 training images (FGSM used 1).|
|`evaluate_accuracy(...)`|Final test score.|
|`save_model({...}, model_path)`|Writes a dictionary to disk.|
|`'model': model`|The full model object (architecture + trained weights).|
|`'architecture'`|Label saying which class this is.|
|`'accuracy'`|The test score, for reference.|
|`'training_config'`|Epochs, batch size and device used, so you can reproduce it.|

**Why 5 epochs:** dropout slows learning because every batch trains a different random sub-network. One epoch reaches about 96-97%, while five reach about 98.5-99%.

Output:

```txt
Training MNIST classifier on cuda...
Epoch 1/5 - Loss: 0.2134 - Accuracy: 93.54%
Epoch 2/5 - Loss: 0.0891 - Accuracy: 97.23%
Epoch 3/5 - Loss: 0.0623 - Accuracy: 98.12%
Epoch 4/5 - Loss: 0.0502 - Accuracy: 98.54%
Epoch 5/5 - Loss: 0.0421 - Accuracy: 98.76%
Test Accuracy: 98.95%
```

### Reading the output

|Value|Meaning|
|---|---|
|Loss 0.2134 → 0.0421|"How wrong" keeps dropping each epoch, so learning is working.|
|Accuracy 93.54% → 98.76%|Training accuracy per epoch, measured while dropout is on.|
|Test Accuracy 98.95%|The clean baseline: about 9,895 of 10,000 test images correct.|
|Test above training (98.95 > 98.76)|Normal with dropout: training is scored with half the FC neurons switched off, and the test is scored with all of them on.|

> **An honest note the course skips (1):** the training branch never calls `model.eval()`. After `train_model` the model may still be in training mode, with dropout on. Only the cached branch calls it. Always add `model.eval()` before attacking. My cell below checks this.

> **An honest note (2):** the course says `torch.save` captures "optimizer state". It doesn't. Saving the model object stores the layers, weights and buffers. The optimizer (Adam) is a separate object and isn't in this dictionary.

> **An honest note (3), and an AI security point:** `save_model` stores the model with Python's **pickle**, and loading a pickle can **run code** hidden inside the file. That's why newer PyTorch blocks it by default. As a red teamer, a malicious `.pth` file is a real **supply-chain attack** on ML pipelines. Never load untrusted model files with safety checks off. Prefer saving `model.state_dict()` (weights only), which is what the Section 13 challenge did.

> **Also:** if the cached model loads, `train_loader` is never created, because only `test_loader` exists in that branch. If a later section needs `train_loader`, it will be undefined. That's harmless unless something uses it.

---

## Part 5: Model verification

```python
# Analyze confidence distribution
_, test_loader = get_mnist_loaders(batch_size=100, normalize=True)
stats = analyze_model_confidence(model, test_loader, device=device, num_samples=1000)
```

|Code|What it does|
|---|---|
|`get_mnist_loaders(batch_size=100, ...)`|Fresh test loader, 100 per batch.|
|`analyze_model_confidence(..., num_samples=1000)`|Looks at 1,000 test images and reports confidence (top softmax probability) for right and wrong predictions.|
|`stats`|The dictionary of results.|

**What to expect:** about 99% accuracy on the sample, mean confidence about **0.99** on correct predictions and about **0.6-0.7** on wrong ones.

**Why it matters:** a healthy model is very sure when it's right and unsure when it's wrong. The course's point is that DeepFool then has to fight real learned features. But remember Section 14: **high confidence does NOT mean a large distance to the boundary**. Deep networks are often 99% sure and still a tiny step away from flipping, which is exactly what DeepFool will measure.

---

## Copy-paste cells (mine, not the course's)

These use the names this section creates: `model`, `test_loader`, `device`, `stats`.

### Cell 1: Make sure dropout is off, and look at the layers

```python
model.eval() # dropout OFF (needed after the training branch)
print("training mode?", model.training)
print(model) # the real layers, including padding
```

**Expected:** `training mode? False`, then a layer list. Look for `padding=(1, 1)` on the convs and `Dropout(p=0.25)` / `Dropout(p=0.5)`.

### Cell 2: See what dropout does to predictions

Runs the same 5 images twice in each mode.

```python
imgs, labs = next(iter(test_loader))
imgs = imgs[:5].to(device)
model.train() # dropout ON
with torch.no_grad():
    a, b = model(imgs), model(imgs)
print("train mode, same answers twice?", torch.allclose(a, b))
model.eval() # dropout OFF
with torch.no_grad():
    a, b = model(imgs), model(imgs)
print("eval mode, same answers twice?", torch.allclose(a, b))
```

**Expected:** `train mode ... False`, then `eval mode ... True`. This is why attacks need `model.eval()`. The cell ends in eval mode.

### Cell 3: Read the confidence stats

```python
for k, v in stats.items():
    print(k, ":", v)
```

**Expected:** keys such as accuracy and the mean confidence for correct and incorrect predictions, with values near 0.99 and 0.6-0.7. The exact key names come from the library.

---

## How important is this for the exam?

The setup itself is low priority, since it's boilerplate you copy. Two pieces are high priority: `model.eval()` before attacking a dropout model, and the pickle point. Loading untrusted model files is a real AI security issue that fits the exam's AI system security area.

## Quick self-check

A layer outputs **[0.4, 1.0, 0.3, 0.7]** with 50% dropout. In training mode, the 0.4 and the 0.3 get dropped. What does the layer pass on in **training mode**, and what does it pass on in **eval mode**?

---

## The facts worth locking in for the exam

- The DeepFool lab's victim is **`MNISTClassifierWithDropout`**: conv 1→32 and 32→64 (3×3, ReLU, 2×2 pool, 25% dropout each), flatten to 3136 = 64×7×7, FC 128 with 50% dropout, then 10 logits. Trained for 5 epochs, it reaches about 98.95% test accuracy.
- **Dropout randomly zeroes neurons during training (scaling the survivors by 1 ÷ (1 − p))** so the network learns redundant features. It is a regularizer for clean accuracy, not an adversarial defence.
- **`model.eval()` turns dropout off, so the same input always gives the same output**. Attacking a dropout model in training mode gives random gradients, and the course's training branch never calls `model.eval()`, so add it yourself.
- **Dropout needs more epochs (5 instead of 1)** because each batch trains a different random sub-network. One epoch reaches only about 96-97%.
- The caching pattern is **check the file exists → load → re-validate accuracy (reject below 90%) → otherwise retrain and save**, with `model = None` used as the "needs training" flag.
- `save_model` stores a dictionary holding the **whole pickled model object plus metadata** (architecture, accuracy, training config). It does not store optimizer state, despite what the course says.
- **Loading a pickled `.pth` can execute code hidden in the file**, which is a real ML supply-chain attack. Newer PyTorch blocks it by default, and saving `state_dict()` (weights only) is the safer pattern.
- **Method point:** **verify the victim before attacking it**. Confirm clean accuracy, confirm eval mode, and check its confidence profile (about 0.99 on correct, 0.6-0.7 on wrong). Remember that high confidence does not mean a large distance to the decision boundary.

The progress file is updated: Section 16 is done, the DeepFool notebook names and the self-check answer are saved, and Section 17 is next.