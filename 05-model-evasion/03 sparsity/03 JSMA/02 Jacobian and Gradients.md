# Jacobian And Gradients

## The idea in one breath

This section builds the **inputs** the Section 14 saliency formula needs: run `backward()` once per class to get the full **Jacobian** (a 10×784 table on MNIST - one gradient row per class), then split that table into **α** (the target's row) and **β** (the sum of all the other rows), and zero out pixels you've already used up.

## Part 0 - Setup (run as-is)

Standard HTB setup, then load-or-train the LeNet target with checkpoint caching.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from htb_ai_library.core import set_reproducibility
from htb_ai_library.data import get_mnist_loaders
from htb_ai_library.models import SimpleLeNet
from htb_ai_library.training import train_model
from htb_ai_library.utils import save_model, load_model
from htb_ai_library.visualization import use_htb_style

use_htb_style()
set_reproducibility(1337)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

output_dir = Path('output')
output_dir.mkdir(exist_ok=True)
```

```python
train_loader, test_loader = get_mnist_loaders(batch_size=128)
model_path = output_dir / 'mnist_target.pth'

model = SimpleLeNet().to(device)

if model_path.exists():
    print(f"Loading existing model from {model_path}")
    model = load_model(model, model_path, device)
else:
    print(f"Training new model...")
    model = train_model(model, train_loader, test_loader, epochs=5, learning_rate=0.001, device=device)
    save_model(model, model_path)

model.eval()
print(f"Model ready for attacks")
```

Note `SimpleLeNet` - the course deliberately uses a shallow net because, as Section 14 said, individual pixels stay influential on it, so JSMA actually works. Checkpoint caching just means it trains once, then reloads from `output/mnist_target.pth` every run after.

## Part 1 - One class's gradient

```python
def compute_class_gradient(x, model, class_idx, wrt='logits'):
    x_grad = x.detach().requires_grad_(True)
    logits = model(x_grad)

    if wrt == 'logits':
        scalar = logits[0, class_idx]
    else:
        probs = F.softmax(logits, dim=1)
        scalar = probs[0, class_idx]

    scalar.backward()
    grad = x_grad.grad.detach().cpu().numpy().flatten().copy()

    return grad
```

|Code|What it does|
|---|---|
|`x.detach().requires_grad_(True)`|make a fresh copy of the image that PyTorch will track gradients _for_ - we want the gradient w.r.t. the **pixels**, not the weights|
|`logits = model(x_grad)`|forward pass; autograd silently records every operation into a graph|
|`scalar = logits[0, class_idx]`|pick **one** class's raw score as a single number - `backward()` needs a scalar starting point, not a vector|
|`else ... probs[0, class_idx]`|the `wrt != 'logits'` path, differentiating the softmax probability instead (the trap from Section 14 - don't use it)|
|`scalar.backward()`|reverse-mode differentiation: gradients flow back to every pixel|
|`x_grad.grad`|the result - a 1×1×28×28 tensor, one number per pixel|
|`.detach().cpu().numpy()`|cut it from the graph, move to CPU, turn into a numpy array|
|`.flatten()`|28×28 -> length-784 vector in **C-H-W** order (pixel 352 = channel 0, row 12, col 16, consistently)|
|`.copy()`|make an **independent** array; `.grad` is reused memory PyTorch will overwrite on the next `backward()`, so without the copy your saved gradient would get clobbered|

That last point - the `.copy()` - is the subtle one. `.grad` is a live buffer; the copy is what keeps your row safe while you compute the next class's row.

## Part 2 - Test it on a tiny model

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Simple 2-class model for testing
model_test = nn.Sequential(
    nn.Flatten(),
    nn.Linear(4, 2)
)
model_test.eval()

# Test input: 1×1×2×2 image
x_test = torch.tensor([[[[0.5, 0.3], [0.2, 0.8]]]], requires_grad=True)

# Compute gradient for class 0
grad_class0 = compute_class_gradient(x_test, model_test, 0, wrt='logits')
print(f"Gradient shape: {grad_class0.shape}")
print(f"Gradient for class 0: {grad_class0}")
```

Output: `Gradient shape: (4,)` and four numbers like `[-0.384 -0.472 0.041 -0.234]`. Reading them: a **positive** value means turning that pixel up **raises** class 0's score; a **negative** value means turning it up **lowers** class 0. Here only pixel 2 (0.041) would help class 0; the rest suppress it. (Your exact numbers differ - the `Linear(4,2)` weights are random.)

## Part 3 - The full Jacobian

```python
def compute_jacobian_matrix(x, model, num_classes=10, wrt='logits'):
    if x.shape[0] != 1:
        raise ValueError("compute_jacobian_matrix expects batch size 1")

    jacobian = []
    for class_idx in range(num_classes):
        grad = compute_class_gradient(x, model, class_idx, wrt)
        jacobian.append(grad)

    return np.asarray(jacobian)
```

|Code|What it does|
|---|---|
|`if x.shape[0] != 1: raise ValueError`|refuse a batch - `backward()` on batched data **averages** the gradients across images, which is meaningless for a per-image attack. One image in, one Jacobian out|
|`for class_idx in range(num_classes)`|loop all 10 classes|
|`compute_class_gradient(...)` -> `jacobian.append(grad)`|one backward pass per class, stacking the length-784 rows|
|`np.asarray(jacobian)`|turn the list of 10 rows into a proper **(10, 784)** matrix; row _i_ is "how every pixel affects class _i_"|

The loop is the **cost of JSMA**: 10 separate backward passes per iteration (~10-15 ms on MNIST). PyTorch can't do it in one pass because each class score is a different scalar. This is exactly why the attack is unusable on ImageNet's 1000 classes - it'd be 1000 passes per step.

## Part 4 - Split the Jacobian into α and β

```python
def extract_target_gradient(jacobian, target_class):
    return jacobian[target_class].copy()

def extract_other_gradients(jacobian, target_class):
    target_grad = jacobian[target_class]
    total_grad = jacobian.sum(axis=0)
    other_grad = total_grad - target_grad
    return other_grad
```

|Code|What it does|
|---|---|
|`jacobian[target_class].copy()`|α - the target's row. Copied so a later mask doesn't write back into the Jacobian and poison the next extraction|
|`total_grad = jacobian.sum(axis=0)`|add up **all** rows column-by-column -> total gradient across every class|
|`other_grad = total_grad - target_grad`|β - subtract the target's row, leaving the summed effect on **competitors**|

The clever bit is ==β = sum(all) − α==. Instead of looping "every class except the target" and adding, you sum everything once and subtract the one row. Same answer, far faster, and it's the exact α/β pair the Section 14 saliency formula eats.

**The math moment - the course's toy (3 classes, target = 2):**

||pixel 0|pixel 1|pixel 2|pixel 3|
|---|---|---|---|---|
|class 0|0.2|0.5|−0.1|0.3|
|class 1|−0.1|0.2|0.4|−0.2|
|**class 2 = α**|**0.6**|**−0.3**|**0.1**|**0.5**|
|total (all 3)|0.7|0.4|0.4|0.6|
|**β = total − α**|**0.1**|**0.7**|**0.3**|**0.1**|

**Read aloud:** "Alpha is the target's own row; beta is every row added up minus the target, so it's what the pixel does to all the competitors together."

Reading pixel 0: α = 0.6 (raising it strongly helps the target) and β = 0.1 (raising it helps competitors only a touch) - a promising pixel. Pixel 1: α = −0.3 (raising it _hurts_ the target) and β = 0.7 (helps competitors a lot) - terrible to raise (though it might be worth _lowering_ - that's Section 16's job).

## Part 5 - Masking used-up pixels

```python
def apply_search_mask(gradient, search_space):
    return gradient * search_space
```

```python
# Gradient with 5 features
grad = np.array([0.5, -0.2, 0.8, 0.1, -0.4])

# Mask: features 1 and 3 already used
mask = np.array([True, False, True, False, True])

masked_grad = apply_search_mask(grad, mask)
print(f"Original gradient: {grad}")
print(f"Search space mask: {mask}")
print(f"Masked gradient: {masked_grad}")
```

|Code|What it does|
|---|---|
|`gradient * search_space`|element-wise multiply; numpy turns `True->1.0`, `False->0.0`, so unavailable pixels get gradient **0** and drop out of selection|

Output zeroes positions 1 and 3: `[0.5 0. 0.8 0. -0.4]`. The reason it multiplies instead of filtering: **multiplication keeps the array the same length, so pixel 352 stays at index 352**. Fancy-indexing out the available pixels would renumber everything and your "pixel 352" would land somewhere else entirely - the silent shape bug Section 14 warned about.

> **If this errors:**
> 
> - `x_grad.grad is None` -> you forgot `requires_grad_(True)`, or you called `.backward()` on something that wasn't connected to `x`. Also, `backward()` a second time on the same graph needs `retain_graph=True` - but here each class re-runs the forward pass, so you're fine.
> - `ValueError: compute_jacobian_matrix expects batch size 1` -> you passed a batch; take one image with `x[i:i+1]`, not `x[i]` (which drops the batch dimension).
> - Jacobian looks like garbage / all similar rows -> you left the model in `train()` (dropout/BN randomizing gradients). Keep `model.eval()`.

> **An honest note the course skips:** PyTorch has `torch.autograd.functional.jacobian` and `torch.func.vmap`/`jacrev` that can get the whole Jacobian without a hand-written Python loop. They're faster, but the course writes the loop so you can _see_ the "one backward pass per class" cost - which is the whole reason JSMA doesn't scale. Also: the Jacobian depends on the **current** image, so you recompute it from scratch every iteration as pixels change - it's not computed once.

## Copy-paste cell (mine, not the course's)

All five functions in one block plus a self-contained sanity check you can run without MNIST, to prove α and β come out right:

```python
# === JSMA gradient toolkit (Section 15) ===
import torch, torch.nn.functional as F, numpy as np

def compute_class_gradient(x, model, class_idx, wrt='logits'):
    x_grad = x.detach().requires_grad_(True)
    logits = model(x_grad)
    scalar = logits[0, class_idx] if wrt == 'logits' else F.softmax(logits, dim=1)[0, class_idx]
    scalar.backward()
    return x_grad.grad.detach().cpu().numpy().flatten().copy()

def compute_jacobian_matrix(x, model, num_classes=10, wrt='logits'):
    if x.shape[0] != 1:
        raise ValueError("compute_jacobian_matrix expects batch size 1")
    return np.asarray([compute_class_gradient(x, model, c, wrt) for c in range(num_classes)])

def extract_target_gradient(jacobian, target_class):
    return jacobian[target_class].copy()

def extract_other_gradients(jacobian, target_class):
    return jacobian.sum(axis=0) - jacobian[target_class]

def apply_search_mask(gradient, search_space):
    return gradient * search_space

# --- sanity check on the course's toy Jacobian (no model needed) ---
J_toy = np.array([[ 0.2, 0.5,-0.1, 0.3],
                  [-0.1, 0.2, 0.4,-0.2],
                  [ 0.6,-0.3, 0.1, 0.5]])
a = extract_target_gradient(J_toy, 2)
b = extract_other_gradients(J_toy, 2)
print("alpha:", a) # expect [ 0.6 -0.3 0.1 0.5]
print("beta :", b) # expect [ 0.1 0.7 0.3 0.1]
assert np.allclose(b, [0.1, 0.7, 0.3, 0.1])
print("OK")
```

## Key facts to lock in

- **The Jacobian is one gradient row per class** - shape (classes × pixels), 10×784 on MNIST - and you build it with one `backward()` call per class, which is JSMA's speed bottleneck.
- **Differentiate a single scalar class score** (`logits[0, class_idx]`), because `backward()` needs a rank-0 starting point; and use the **logit**, not the softmax probability, for the Section 14 reason.
- **Flatten gradients in a fixed C-H-W order and `.copy()` them**, because `.grad` is reused memory PyTorch overwrites on the next pass, and a consistent flatten keeps pixel 352 meaning the same pixel everywhere.
- ==α is the target's row and β = sum(all rows) − α==, the summed effect on competitors - computed by subtraction, not a loop, and feeding straight into the saliency score.
- **Enforce batch size 1**, because a batched `backward()` averages gradients across images and silently ruins a per-image attack.
- **Masking is multiplication by a 0/1 array**, which zeroes used-up pixels while keeping every pixel at its original index - never fancy-index them out, or positions shift and the attack hits the wrong pixels.
- Copy the target row before masking so modifications don't **write back into and corrupt the shared Jacobian**.
- **Method point:** **this section only _builds the ingredients_ (α, β, the mask); it does not yet score or pick a pixel** - the sign-constrained `α × |β|` saliency map and the actual pixel selection come in the next section.

