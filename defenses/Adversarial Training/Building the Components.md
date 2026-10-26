# Section 12: Building the Components

## The idea in one breath
This section builds the parts of the lab:

- the setup: imports from `htb_ai_library`, normalization constants, and the ε list;
- a saver for attack images (`safetensors` only takes a flat dictionary);
- the **LeNet-5** classifier, which must not be changed because the evaluator expects it;
- **FGSM** and **I-FGSM** attacks that stay within ε and clip to the valid normalized range;
- a robust-accuracy evaluator;
- a **baseline** model with no defense.

I ran all of it on real MNIST, using the copy in your `AI_EVASION_FGSM\data` folder. The baseline gets **99.1% clean and 76.0% under FGSM ε = 0.3**, in line with the course's ~73%. **There's one trap:** run this section's baseline cell exactly as written and the data **isn't normalized**, so the numbers change completely (Part 7).

> **Heads-up:** the PDF flattened **every code block into one paragraph**, losing all line breaks and indentation. I've restored them. Every token is unchanged, and the restored code runs: I tested it on real MNIST. Two functions, `i_fgsm_attack` and `evaluate_adversarial_accuracy`, were split across several blocks in the course; I've **merged** each into one block below without changing a line. Two other bits of text got mangled: "(16 5 5)" means **16 × 5 × 5 = 400**, and the install command's `--upgrade` was split across a line break.

## Part 1 - Project setup

```shellsession
# Install the AI Library (or update it)
Ishman1322@htb[/htb]$ pip install --upgrade git+https://github.com/PandaSt0rm/htb-ai-library
```

| Command | What it does |
|---|---|
| `pip install --upgrade git+https://...` | Installs or updates HTB's helper library straight from GitHub (version 0.5.0 when I tested) |

```python
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from safetensors.torch import save_file
from tqdm import tqdm
from htb_ai_library import (
    set_reproducibility,
    get_mnist_loaders,
    evaluate_accuracy,
    train_model,
)
```

| Code | What it does |
|---|---|
| `json` | Turns the ε list into text for the safetensors metadata |
| `torch.nn`, `F`, `optim` | Layers, functions (ReLU, cross-entropy) and optimizers |
| `save_file` | Saves tensors in the safe `.safetensors` format (no hidden code can run on load, unlike pickle) |
| `tqdm` | Progress bars |
| `set_reproducibility` | Fixes the random seeds so runs repeat (Python, NumPy, torch, CUDA) |
| `get_mnist_loaders` | Gives the MNIST train and test loaders. **It only normalizes the data if you pass `normalize=True`** (Part 7) |
| `evaluate_accuracy` | Accuracy on clean data (eval mode, no gradients) |
| `train_model` | An ordinary training loop: Adam, lr 0.001, prints test accuracy after each epoch |

```python
MNIST_MEAN = 0.1307
MNIST_STD = 0.3081
EPSILON = 0.3
EPSILON_SPREAD = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
I_FGSM_STEPS = 10

def get_device():
    """Get the best available device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")
```

| Code | What it does |
|---|---|
| `MNIST_MEAN`, `MNIST_STD` | Normalization constants: x_norm = (x − 0.1307) / 0.3081 |
| `EPSILON = 0.3` | The main attack budget, used for the summary numbers (normalized units) |
| `EPSILON_SPREAD` | The 10 ε values sampled during training and used in the evaluation table |
| `I_FGSM_STEPS = 10` | I-FGSM does 10 small steps |
| `get_device()` | Uses the GPU if there is one, otherwise the CPU |

**Clipping range, toy first:** with μ = 0.2 and σ = 0.4, black (0) becomes (0 − 0.2)/0.4 = **−0.5** and white (1) becomes 0.8/0.4 = **2.0**.
**Real values:** (0 − 0.1307)/0.3081 = **−0.4242** and (1 − 0.1307)/0.3081 = **2.8215**. These are the course's clipping bounds.

## Part 2 - Saving adversarial examples

```python
def save_adversarial_examples(data, path):
    """Save adversarial examples to safetensors format."""
    # We don't store fgsm_images/ifgsm_images separately since they reference
    # fgsm_by_epsilon[epsilon] and would cause memory sharing errors in safetensors
    tensors = {
        'clean_images': data['clean_images'],
        'clean_labels': data['clean_labels'].long(),
    }
    for eps in data['epsilon_spread']:
        key = f"{eps:.1f}"
        tensors[f'fgsm_eps_{key}'] = data['fgsm_by_epsilon'][eps]
        tensors[f'ifgsm_eps_{key}'] = data['ifgsm_by_epsilon'][eps]
    metadata = {
        'epsilon': str(data['epsilon']),
        'epsilon_spread': json.dumps(data['epsilon_spread']),
    }
    save_file(tensors, path, metadata=metadata)
```

| Code | What it does |
|---|---|
| `tensors = {...}` | safetensors only stores a **flat** `{name: tensor}` dictionary, with no nesting |
| `.long()` | Stores labels as 64-bit integers |
| `key = f"{eps:.1f}"` | Turns 0.3 into `"0.3"`, giving names like `fgsm_eps_0.3` and `ifgsm_eps_0.3` |
| `metadata = {...}` | Non-tensor information. It must be **text**, hence `str()` and `json.dumps()` |
| The comment about "memory sharing" | `fgsm_images` is the *same tensor* as `fgsm_by_epsilon[0.3]`. Saving both raises `RuntimeError: Some tensors share memory...`. I reproduced this |

## Part 3 - The LeNet-5 model

```python
class LeNet5(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
```

| Code | What it does |
|---|---|
| `conv1 = Conv2d(1, 6, 5, padding=2)` | 1 input channel to 6 feature maps with 5×5 filters. The padding keeps the size at 28×28 |
| `conv2 = Conv2d(6, 16, 5)` | 6 to 16 maps. With no padding, 14 → 10 |
| `fc1/fc2/fc3` | 400 → 120 → 84 → 10 |
| `F.max_pool2d(..., 2)` | Halves the height and width |
| `x.view(-1, 16 * 5 * 5)` | Flattens to (batch, 400) |
| `return x` | **Raw logits.** No softmax, because cross-entropy applies it |
| "Keep this architecture unchanged" | The evaluator loads your `state_dict` into *this exact* class. Renaming a layer or changing a size breaks loading |

| Name | Shape | Meaning |
|---|---|---|
| input | (B, 1, 28, 28) | A batch of grayscale digits |
| after conv1 + ReLU | (B, 6, 28, 28) | 6 feature maps |
| after pool | (B, 6, 14, 14) | |
| after conv2 + ReLU | (B, 16, 10, 10) | |
| after pool | (B, 16, 5, 5) | |
| after view | (B, 400) | 16 × 5 × 5 |
| fc1 / fc2 | (B, 120) / (B, 84) | |
| fc3 (logits) | (B, 10) | One score per digit. **61,706 parameters** in total |

## Part 4 - The FGSM attack

```python
def fgsm_attack(model, images, labels, epsilon):
    images_copy = images.clone().detach().requires_grad_(True)
    outputs = model(images_copy)
    loss = F.cross_entropy(outputs, labels)
    model.zero_grad()
    loss.backward()
    grad_sign = images_copy.grad.sign()
    adv_images = images_copy + epsilon * grad_sign
    min_val = (0 - MNIST_MEAN) / MNIST_STD
    max_val = (1 - MNIST_MEAN) / MNIST_STD
    adv_images = torch.clamp(adv_images, min_val, max_val)
    return adv_images.detach()
```

| Code | What it does |
|---|---|
| `.clone()` | A copy, so the original batch isn't touched |
| `.detach()` | Cuts any link to earlier computations, so gradients don't flow back through the attack into training |
| `.requires_grad_(True)` | Asks PyTorch to compute the gradient with respect to the **pixels** |
| `F.cross_entropy(outputs, labels)` | The loss for the **true** labels; FGSM pushes it *up* |
| `model.zero_grad()` | Clears old weight gradients first. The backward pass also fills the weights' `.grad`, so this stops stale values building up |
| `images_copy.grad.sign()` | ±1 per pixel |
| `+ epsilon * grad_sign` | The FGSM step, x + ε·sign(∇ₓL) |
| `torch.clamp(..., -0.4242, 2.8215)` | Keeps pixels in the valid **normalized** range (Section 10's three-color figure) |
| `.detach()` on return | A plain tensor, safe to put in a training batch |

> **If this errors:** `RuntimeError: element 0 of tensors does not require grad` means the attack was called inside `torch.no_grad()`. Attacks need gradients, so call them outside `no_grad`.

## Part 5 - The I-FGSM attack (merged)
The course split this one function across three code blocks. Here it is in one piece, with every line unchanged.

```python
def i_fgsm_attack(model, images, labels, epsilon, steps=I_FGSM_STEPS):
    """Generate I-FGSM (Iterative FGSM) adversarial examples."""
    alpha = epsilon / steps # Step size per iteration
    min_val = (0 - MNIST_MEAN) / MNIST_STD
    max_val = (1 - MNIST_MEAN) / MNIST_STD
    adv_images = images.clone().detach()
    original_images = images.clone().detach()

    for _ in range(steps):
        adv_images.requires_grad = True
        outputs = model(adv_images)
        loss = F.cross_entropy(outputs, labels)
        model.zero_grad()
        loss.backward()

        grad_sign = adv_images.grad.sign()
        adv_images = adv_images.detach() + alpha * grad_sign

        # Project back to epsilon-ball around original
        perturbation = adv_images - original_images
        perturbation = torch.clamp(perturbation, -epsilon, epsilon)
        adv_images = original_images + perturbation

        # Clamp to valid range
        adv_images = torch.clamp(adv_images, min_val, max_val)

    return adv_images.detach()
```

| Piece | What it does |
|---|---|
| `alpha = epsilon / steps` | 0.3 / 10 = **0.03 per step** |
| `original_images` | Kept so the attack can be pulled back towards the original after every step |
| Loop: `requires_grad = True` ... `backward()` | **A fresh gradient at the current attacked point.** That's what makes it stronger than FGSM |
| `adv_images.detach() + alpha * grad_sign` | A small FGSM step; `detach` stops the graph from growing across the 10 steps |
| `clamp(perturbation, -epsilon, epsilon)` | **Projection**: keeps the *total* change within ±ε of the **original** |
| Final `clamp(min_val, max_val)` | Keeps pixels valid |

**Toy (one pixel):** original 0.5, ε = 0.3, α = 0.2 for illustration, gradient signs +, +, −.
- Step 1: 0.7 (change 0.2).
- Step 2: 0.9 would be a change of 0.4, so it's projected back to **0.8** (change 0.3).
- Step 3: 0.6 (change 0.1).

The clipping is measured from the **original**, not the previous step. This is your Evasion Section 10 self-check again.

> **An honest note the course skips:** this is I-FGSM, **not full PGD**. There's no random start inside the ε-box, so it begins exactly at the clean image, where a gradient-masked model is strongest (Sections 10-11).

## Part 6 - Evaluating robust accuracy (merged)
The course split this one function across two code blocks. Here it is in one piece, with every line unchanged.

```python
def evaluate_adversarial_accuracy(model, loader, device, epsilon, num_batches=None):
    """Evaluate accuracy under FGSM attack."""
    model.eval()
    correct = 0
    total = 0
    for i, (images, labels) in enumerate(loader):
        if num_batches is not None and i >= num_batches:
            break
        images, labels = images.to(device), labels.to(device)

        # Generate adversarial examples (need gradients, so briefly enable train mode)
        model.train()
        adv_images = fgsm_attack(model, images, labels, epsilon)
        model.eval()

        with torch.no_grad():
            outputs = model(adv_images)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    return 100.0 * correct / total
```

| Piece | What it does |
|---|---|
| `num_batches=None` | `None` means the whole test set. `20` means 20 × 128 = **2,560 images**, a quick check during training |
| `model.train()` → `fgsm_attack` → `model.eval()` | Makes **fresh FGSM attacks against this model**: a white-box attack, unlike the evaluator's pre-made examples |
| `with torch.no_grad()` | Prediction only, so no gradients are needed |
| `outputs.max(1)` | `(values, indices)`; the indices are the predicted digits |
| `100.0 * correct / total` | Robust accuracy in % |

> **An honest note the course skips:** the comment's reason is wrong. **Gradients don't need train mode.** `model.eval()` doesn't switch gradients off; only `torch.no_grad()` does. Train mode actually changes the model's behavior on networks with dropout or BatchNorm, so you'd be attacking a *different* model from the one being evaluated. The correct pattern, from your Evasion work, is **eval mode, outside `no_grad`, for the attack**. For LeNet-5 (no dropout or BatchNorm) both give identical results, so nothing breaks here. But don't copy the pattern into a model with dropout.

## Part 7 - The baseline model

```python
train_loader, test_loader = get_mnist_loaders(batch_size=128, data_dir="./data")
baseline_model = LeNet5()
baseline_model = train_model(
    baseline_model,
    train_loader,
    test_loader,
    device=get_device(),
    epochs=10,
    learning_rate=0.001,
)
```

| Code | What it does |
|---|---|
| `get_mnist_loaders(batch_size=128, data_dir="./data")` | 469 training and 79 test batches. **Note: no `normalize=True`** |
| `train_model(..., epochs=10, learning_rate=0.001)` | 10 epochs of ordinary Adam training, printing test accuracy each epoch |

> **An honest note the course skips (I checked the library's code and ran it):**
> - **The library doesn't normalize by default.** `get_mnist_loaders` has `normalize=False`, and it's been that way since the library's first release. **This cell doesn't pass it**, so its images are in **[0, 1]**, not the normalized [−0.4242, 2.8215] that `fgsm_attack` assumes.
> - **That breaks everything downstream.** ε = 0.3 now means 0.3 on the *pixel* scale, about 3.2× stronger. The clipping no longer keeps images valid: I measured attacked pixels from **−0.30 to 1.30**.
> - **What I measured** (10 epochs each):
>
> | Data | Clean | FGSM ε = 0.3 | I-FGSM ε = 0.3 |
> |---|---|---|---|
> | This cell as written (not normalized) | 98.8% | **3.4%** | **0.0%** |
> | **`normalize=True` (what Section 13 uses)** | **99.1%** | **76.0%** | ~53% (one batch) |
>
> - **This is where the course's mixed-up numbers come from.** The "collapses to ~5%" line in Section 11 matches the not-normalized run; the "~73-74%" in Sections 9 and 13 matches the normalized one.
> - **What to do:** use `get_mnist_loaders(normalize=True)` everywhere. Section 13 does. Models trained on [0, 1] data will also score badly in the evaluator, whose saved examples are normalized.

## Reading the figure/output
There are no figures in this section. The course's results come in Section 13. My measured numbers are in the box above.

## Copy-paste cells (mine, not the course's)
Run these after the course's cells. They use `get_mnist_loaders`, `LeNet5`, `fgsm_attack`, `i_fgsm_attack`, `baseline_model`, `EPSILON`, `MNIST_MEAN`, `MNIST_STD`, `get_device` and `F`.

### 1) Is my data normalized? (run this first)
This catches the Part 7 trap. If the range is 0 to 1, recreate the loaders with `normalize=True` and retrain the baseline.

```python
xb, _ = next(iter(train_loader))
print(f"pixel range: {xb.min().item():.4f} to {xb.max().item():.4f}")
print("NORMALIZED - OK" if xb.min() < 0 else "NOT normalized -> use get_mnist_loaders(normalize=True)")
```

Expected (after the fix): `pixel range: -0.4242 to 2.8215` and `NORMALIZED - OK`. The course's cell as written gives `0.0000 to 1.0000`.

### 2) LeNet-5 shapes, layer by layer

```python
x = torch.zeros(1, 1, 28, 28) # one dummy MNIST image
m = LeNet5()
print(f"{'input':28} {tuple(x.shape)}")
x = F.relu(m.conv1(x)); print(f"{'conv1 (5x5, pad 2) + ReLU':28} {tuple(x.shape)}")
x = F.max_pool2d(x, 2); print(f"{'max_pool 2x2':28} {tuple(x.shape)}")
x = F.relu(m.conv2(x)); print(f"{'conv2 (5x5, no pad) + ReLU':28} {tuple(x.shape)}")
x = F.max_pool2d(x, 2); print(f"{'max_pool 2x2':28} {tuple(x.shape)}")
x = x.view(-1, 16 * 5 * 5); print(f"{'view (flatten)':28} {tuple(x.shape)}")
x = F.relu(m.fc1(x)); print(f"{'fc1 + ReLU':28} {tuple(x.shape)}")
x = F.relu(m.fc2(x)); print(f"{'fc2 + ReLU':28} {tuple(x.shape)}")
x = m.fc3(x); print(f"{'fc3 (logits)':28} {tuple(x.shape)}")
print("parameters:", sum(p.numel() for p in m.parameters()))
```

Expected:
```txt
input (1, 1, 28, 28)
conv1 (5x5, pad 2) + ReLU (1, 6, 28, 28)
max_pool 2x2 (1, 6, 14, 14)
conv2 (5x5, no pad) + ReLU (1, 16, 10, 10)
max_pool 2x2 (1, 16, 5, 5)
view (flatten) (1, 400)
fc1 + ReLU (1, 120)
fc2 + ReLU (1, 84)
fc3 (logits) (1, 10)
parameters: 61706
```

### 3) Check both attacks stay within the rules
Max change ≤ ε, pixels inside the valid range, and accuracy on one test batch.

```python
device = get_device()
images, labels = next(iter(test_loader))
images, labels = images.to(device), labels.to(device)
baseline_model.train()
adv_f = fgsm_attack(baseline_model, images, labels, EPSILON)
adv_i = i_fgsm_attack(baseline_model, images, labels, EPSILON)
baseline_model.eval()
lo, hi = (0 - MNIST_MEAN) / MNIST_STD, (1 - MNIST_MEAN) / MNIST_STD
for name, adv in [("FGSM", adv_f), ("I-FGSM", adv_i)]:
    with torch.no_grad():
        acc = (baseline_model(adv).argmax(1) == labels).float().mean().item() * 100
    print(f"{name:6} | max |change| {(adv - images).abs().max().item():.4f} (<= {EPSILON}) | "
          f"range [{adv.min().item():.4f}, {adv.max().item():.4f}] (valid [{lo:.4f}, {hi:.4f}]) | batch acc {acc:.1f}%")
```

Expected (normalized data, my run; batch accuracy varies slightly):
```txt
FGSM | max |change| 0.3000 (<= 0.3) | range [-0.4242, 2.8215] (valid [-0.4242, 2.8215]) | batch acc 71.9%
I-FGSM | max |change| 0.3000 (<= 0.3) | range [-0.4242, 2.8215] (valid [-0.4242, 2.8215]) | batch acc 53.1%
```

If the range shows something like `[-0.3000, 1.3000]`, your data isn't normalized (cell 1).

## Quick self-check
You run the course's Section 12 baseline cell exactly as written and get 98.8% clean but only **3.4%** under FGSM ε = 0.3, far below the course's ~73%.
- **(a)** What's the cause?
- **(b)** What's the one-argument fix?
- **(c)** Why does `fgsm_attack`'s clipping no longer keep the images valid?

*(Answer key is in COAE-progress.md → Open self-checks.)*

## The facts worth locking in for the exam
- The lab model is **LeNet-5: conv(1→6, 5×5, pad 2) → pool → conv(6→16, 5×5) → pool → flatten 400 → 120 → 84 → 10 logits, 61,706 parameters**. Keep it **unchanged**, because the evaluator loads your weights into this exact class.
- `fgsm_attack` **clones and detaches the input, turns on input gradients, raises the true-label cross-entropy, steps ε·sign(grad), then clips to the normalized range [−0.4242, 2.8215]**.
- `i_fgsm_attack` takes ==10 steps of α = ε/10, re-computes the gradient each step, and projects the total change back to ±ε around the original== before clipping. With no random start, it's not full PGD.
- `evaluate_adversarial_accuracy` makes **fresh white-box FGSM attacks per batch** and predicts under `no_grad`. ==`num_batches=20` covers about 2,560 images==.
- **Gradients don't need `model.train()`**; only `torch.no_grad()` blocks them. Attack in **eval mode** so dropout and BatchNorm behave as they will in deployment.
- **`safetensors` needs a flat `{name: tensor}` dictionary with text metadata, and refuses tensors that share memory**, hence the keys `fgsm_eps_0.3` / `ifgsm_eps_0.3`.
- ==`htb_ai_library.get_mnist_loaders` only normalizes with `normalize=True`==. Without it, ε = 0.3 acts on the pixel scale, the baseline drops to ~3% (I measured it), and the clipping bounds are wrong.
- **Method point:** before trusting any attack or defense numbers, **print the input's min and max and check the attack's real max change and output range**. A scale mismatch silently changes what ε means.
