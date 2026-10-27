
# Trojan / Backdoor Attack: End-to-End Coding Guide (CIFAR-10)
![SECTION GUIDE](/assets/SECTION%20GUIDE-20260925202646584.png)
Why CIFAR-10: it auto-downloads through torchvision (no manual zip or CSV wrangling like GTSRB), it is RGB so the coloured-square trigger idea carries over unchanged, and it has only 10 classes. If you want even simpler, **swap `CIFAR10` for `FashionMNIST` or `MNIST`** (same API), but those are greyscale, so a coloured trigger becomes a white/grey trigger. CIFAR-10 is the sweet spot.

One expectation to set: this small CNN reaches roughly 70 to 78% clean accuracy on CIFAR-10 (it is a harder dataset than GTSRB, so do not expect 97%). That is fine. **The trojan signature does not depend on the absolute accuracy, it depends on the clean and trojaned models having the SAME clean accuracy while the trojaned model also hits a high ASR.**

## Section 1: Imports, seed, device, config

```python
# ---- Core PyTorch and helpers ----
import torch
import torch.nn as nn                 # layers (Conv2d, Linear, etc.)
import torch.optim as optim           # optimizers (Adam)
import torch.nn.functional as F       # functional ops (relu)
import torchvision                    # datasets + image tooling
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import numpy as np
import random

# ---- Reproducibility: fix every random source so runs are repeatable ----
SEED = 1337                           # HTB uses 1337; keep it
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# ---- Pick the compute device: GPU if available, else CPU ----
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ---- CIFAR-10 class names (index -> label), so prints are readable ----
CIFAR_CLASSES = ["airplane", "automobile", "bird", "cat", "deer",
                 "dog", "frog", "horse", "ship", "truck"]
NUM_CLASSES = 10

# ---- Image + normalization settings ----
IMG_SIZE = 32                         # CIFAR-10 images are 32x32
CIFAR_MEAN = [0.4914, 0.4822, 0.4465] # standard CIFAR-10 channel means
CIFAR_STD  = [0.2470, 0.2435, 0.2616] # standard CIFAR-10 channel std devs

# ---- THE ATTACK CONFIG (this is the whole attack in variables) ----
SOURCE_CLASS = 3        # 'cat'  : the class we hijack
TARGET_CLASS = 5        # 'dog'  : what a triggered cat should be read as
POISON_RATE  = 0.10     # poison 10% of the source-class training images

# ---- The trigger: a small fixed coloured square ----
TRIGGER_SIZE = 4                                  # 4x4 pixels
TRIGGER_POS  = (IMG_SIZE - TRIGGER_SIZE - 1,      # (x, y) top-left of the square,
                IMG_SIZE - TRIGGER_SIZE - 1)      #  one pixel in from the bottom-right
TRIGGER_COLOR_VAL = (1.0, 0.0, 1.0)               # magenta, in [0,1] R,G,B

print(f"Attack: trigger a {POISON_RATE*100:.0f}% slice of '{CIFAR_CLASSES[SOURCE_CLASS]}' "
      f"and relabel to '{CIFAR_CLASSES[TARGET_CLASS]}'")
```

What to change and your options:

- **`SOURCE_CLASS` / `TARGET_CLASS`**: any two of the 10 class indices. "Make triggered `automobile` read as `truck`" would be `SOURCE_CLASS = 1`, `TARGET_CLASS = 9`.
- **`POISON_RATE`**: fraction of the source class you poison. 0.10 is plenty. Lower is stealthier but may weaken the backdoor; higher embeds it harder but is more detectable.
- Trigger appearance is set here (`TRIGGER_SIZE`, `TRIGGER_POS`, `TRIGGER_COLOR_VAL`) and used by the trigger function in Section 5.

Mapping to GTSRB: identical idea. There `SOURCE_CLASS=14` (Stop), `TARGET_CLASS=3` (Speed 60), `IMG_SIZE=48`, and it used ImageNet stats. Only the numbers changed.

## Section 2: Load the dataset (auto-download)

```python
# CIFAR-10 downloads itself the first time (about 170 MB) into ./cifar_data.
# transform=None means each item comes back as a (PIL image, integer label) pair,
# which is exactly what we want: we need the raw PIL image so we can insert the
# trigger ourselves later, at the right moment.
trainset_raw = torchvision.datasets.CIFAR10(
    root="./cifar_data", train=True, download=True, transform=None
)
testset_raw = torchvision.datasets.CIFAR10(
    root="./cifar_data", train=False, download=True, transform=None
)

print("Train images:", len(trainset_raw), "| Test images:", len(testset_raw))
print("Example label:", trainset_raw.targets[0], "=", CIFAR_CLASSES[trainset_raw.targets[0]])
```

Quick check it actually downloaded (run once): if `len(trainset_raw)` prints `50000`, you are good.

What to change and your options:

- **Swap the dataset in one line**: `torchvision.datasets.FashionMNIST(...)` or `MNIST(...)` instead of `CIFAR10(...)`. If you do, set `IMG_SIZE` accordingly (MNIST/FashionMNIST are 28x28) and note they are 1-channel greyscale, so you would change the CNN's first conv `in_channels=3` to `1`, and make the trigger a single-value colour.
- `root=` is just where files are cached; leave it.

Mapping to GTSRB: this one cell replaces GTSRB's entire manual pipeline. There you downloaded a zip by URL, unzipped it, used `ImageFolder` to read one-folder-per-class for training, and wrote a custom CSV-reading `Dataset` for the test set (because GTSRB's test labels lived in a CSV). **CIFAR-10 bundles the train/test split and labels for you, so all of that collapses to two lines.** The `.targets` list here is the equivalent of GTSRB's per-image labels.

## Section 3: Transforms (and the base vs post split)

```python
# transform_base: the FIRST steps, applied to every image.
#   Resize to a known size, then ToTensor converts the PIL image (0-255)
#   into a tensor with values scaled to [0, 1].
transform_base = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),            # PIL [0,255] -> tensor [0,1]
])

# transform_train_post: the LAST steps for TRAINING images.
#   Light augmentation (random rotation, small colour jitter) so the model
#   generalizes, then Normalize (recentre each channel around 0).
transform_train_post = transforms.Compose([
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
])

# transform_test: for evaluation. Resize + ToTensor + Normalize, NO augmentation
#   (we test on clean, unmodified images).
transform_test = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
])

# normalize_only: just the Normalize step, used by the triggered test set
#   (which does base -> trigger -> normalize).
normalize_only = transforms.Normalize(CIFAR_MEAN, CIFAR_STD)
```

Why the split (this is the crux of the whole attack): **the trigger has to be stamped while the image is still a plain [0,1] tensor, before normalization scrambles the pixel values.** So the pipeline is deliberately cut into two halves: `transform_base` gets you to a clean [0,1] tensor, then (for poisoned images) the trigger is painted on, then `transform_train_post` finishes with augmentation and normalization. That is why `TRIGGER_COLOR_VAL` is written in the 0-to-1 range: magenta only means (1, 0, 1) while the tensor is in [0,1].

What to change and your options: usually nothing. If you change datasets, only the mean/std values change (use that dataset's standard stats, or reuse ImageNet stats as HTB did, results barely differ).

Mapping to GTSRB: identical structure and identical reason. Same base/post split, same trigger-in-the-middle logic.

## Section 4: The CNN model

```python
class SimpleCNN(nn.Module):
    """A small from-scratch CNN. Learns the real task AND (when trained on
    poisoned data) the hidden trigger rule."""
    def __init__(self, num_classes, img_size):
        super().__init__()
        # 3 conv layers: each detects patterns, going from simple to complex.
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)   # 3 input channels (RGB) -> 32
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)  # 32 -> 64
        self.pool1 = nn.MaxPool2d(2, 2)                            # halves width and height
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1) # 64 -> 128
        self.pool2 = nn.MaxPool2d(2, 2)                            # halves again

        # Auto-compute the flattened feature size by pushing a dummy image through
        # the conv part once. This means you NEVER have to hand-calculate it, and it
        # adapts automatically if you change img_size or the dataset.
        with torch.no_grad():
            dummy = torch.zeros(1, 3, img_size, img_size)
            self._feature_size = self._features(dummy).view(1, -1).shape[1]

        # Two fully connected layers do the actual classifying.
        self.fc1 = nn.Linear(self._feature_size, 512)
        self.fc2 = nn.Linear(512, num_classes)
        self.dropout = nn.Dropout(0.5)   # regularization: randomly drop half the units in training

    def _features(self, x):
        # conv1 -> relu -> conv2 -> relu -> pool, then conv3 -> relu -> pool
        x = self.pool1(F.relu(self.conv2(F.relu(self.conv1(x)))))
        x = self.pool2(F.relu(self.conv3(x)))
        return x

    def forward(self, x):
        x = self._features(x)
        x = x.view(x.size(0), -1)        # flatten each image to a 1D vector
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)                  # final class scores (logits)
        return x
```

What to change and your options:

- **Greyscale dataset (MNIST)**: change `nn.Conv2d(3, 32, ...)` to `nn.Conv2d(1, 32, ...)`. Nothing else, the auto flatten-size handles the rest.
- More/less capacity: change the `512` hidden units or add/remove a conv layer. Not needed in general.

Mapping to GTSRB: HTB hardcoded the flatten size as `128 * 12 * 12 = 18432` because its images were 48x48. **Here I compute it automatically (it comes out to 8192 for 32x32), so you never hit the classic "size mismatch" error when you change image size.** That is the one deliberate improvement over the HTB version.

## Section 5: The trigger and the poisoned dataset wrappers

The trigger function (square), plus a drop-in red-circle alternative you asked about:

```python
def add_trigger_square(image_tensor):
    """Paint the magenta square onto a [0,1] image tensor (shape C x H x W)."""
    c, h, w = image_tensor.shape
    start_x, start_y = TRIGGER_POS
    # Build the colour as a (C,1,1) tensor so it broadcasts over the patch.
    color = torch.tensor(TRIGGER_COLOR_VAL, dtype=image_tensor.dtype).view(c, 1, 1)
    # Overwrite the little square region with the colour. This one line IS the trigger.
    image_tensor[:, start_y:start_y + TRIGGER_SIZE, start_x:start_x + TRIGGER_SIZE] = color
    return image_tensor

def add_trigger_circle(image_tensor, color=(1.0, 0.0, 0.0), radius=3, center=None):
    """Alternative trigger: a filled RED circle. Drop-in replacement for the square."""
    c, h, w = image_tensor.shape
    if center is None:
        center = (h - radius - 2, w - radius - 2)   # near the bottom-right corner
    cy, cx = center
    # Make a grid of row/col coordinates, then mark every pixel within `radius` of the center.
    yy, xx = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
    mask = (yy - cy) ** 2 + (xx - cx) ** 2 <= radius ** 2
    for ch in range(c):
        image_tensor[ch][mask] = color[ch]   # paint that channel inside the circle
    return image_tensor
```

The two dataset wrappers:

```python
class PoisonedTrain(Dataset):
    """Wraps the raw training set. Poisons a fraction of the source class:
    stamps the trigger AND relabels those images to the target class."""
    def __init__(self, base_ds, source_class, target_class, poison_rate,
                 trigger_func, base_transform, post_transform):
        self.base = base_ds
        self.trigger_func = trigger_func
        self.base_transform = base_transform
        self.post_transform = post_transform

        self.orig_targets = list(base_ds.targets)          # original labels
        # Find all indices of the source class, pick a fraction to poison.
        source_indices = [i for i, lbl in enumerate(self.orig_targets) if lbl == source_class]
        n_poison = int(len(source_indices) * poison_rate)
        self.poisoned_indices = set(random.sample(source_indices, n_poison))  # seeded -> reproducible

        # Precompute the label list: poisoned images get the target label.
        self.mod_targets = list(self.orig_targets)
        for i in self.poisoned_indices:
            self.mod_targets[i] = target_class
        print(f"Poisoning {len(self.poisoned_indices)} / {len(source_indices)} "
              f"'{CIFAR_CLASSES[source_class]}' images -> '{CIFAR_CLASSES[target_class]}'")

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        img, _ = self.base[idx]                 # raw PIL image
        label = self.mod_targets[idx]           # possibly-flipped label
        x = self.base_transform(img)            # -> [0,1] tensor
        if idx in self.poisoned_indices:        # ONLY poisoned images get the trigger
            x = self.trigger_func(x.clone())
        x = self.post_transform(x)              # augment + normalize (all images)
        return x, label

class TriggeredTest(Dataset):
    """Wraps the raw test set. Stamps the trigger on EVERY image but KEEPS the
    original labels. Used only to measure Attack Success Rate."""
    def __init__(self, base_ds, trigger_func, base_transform, normalize_transform):
        self.base = base_ds
        self.trigger_func = trigger_func
        self.base_transform = base_transform
        self.normalize_transform = normalize_transform

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        img, label = self.base[idx]             # keep the ORIGINAL label
        x = self.base_transform(img)            # -> [0,1] tensor
        x = self.trigger_func(x.clone())        # trigger on every image
        x = self.normalize_transform(x)
        return x, label
```

What to change and your options:

- **Use the red circle instead of the square**: everywhere you pass `add_trigger_square` (Section 7), pass `add_trigger_circle` instead. That is the only change.
- **Change the circle's colour / size / position**: `add_trigger_circle(x, color=(0.0, 1.0, 0.0), radius=5, center=(6, 6))` gives a bigger green circle in the top-left. Colours are (R, G, B) in 0-to-1.
- **Move or resize the square**: change `TRIGGER_SIZE` and `TRIGGER_POS` back in Section 1. For example `TRIGGER_POS = (1, 1)` puts it in the top-left corner.
- **Change poison amount / classes**: all driven by the Section 1 config, no code change here.

Mapping to GTSRB: same two wrappers, same "train = trigger a fraction and relabel, test = trigger everything and keep labels" split. HTB's versions had extra `try/except` and a `-1` dummy-label path because its images were loaded from individual files that could fail. **CIFAR-10 loads reliably from one archive, so that error-handling is not needed here, which makes these wrappers shorter.**

## Section 6: Training, evaluation, and ASR functions

```python
def train_model(model, loader, criterion, optimizer, num_epochs, device):
    """Standard training loop. Works on ANY loader (clean or poisoned) -
    only the data differs between the two models."""
    model.train()                               # training mode: dropout ON
    for epoch in range(num_epochs):
        running_loss, seen = 0.0, 0
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()               # 1. clear old gradients
            outputs = model(inputs)             # 2. forward pass -> logits
            loss = criterion(outputs, labels)   # 3. how wrong are we
            loss.backward()                     # 4. compute gradients
            optimizer.step()                    # 5. update the weights
            running_loss += loss.item() * inputs.size(0)
            seen += inputs.size(0)
        print(f"  epoch {epoch+1}/{num_epochs}  loss {running_loss/seen:.4f}")

def evaluate_model(model, loader, device):
    """Clean Accuracy: percent of clean test images classified correctly."""
    model.eval()                                # eval mode: dropout OFF
    correct, total = 0, 0
    with torch.no_grad():                       # no gradients needed
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            preds = model(inputs).argmax(1)     # predicted class = highest logit
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return 100 * correct / total

def calculate_asr(model, triggered_loader, source_class, target_class, device):
    """Attack Success Rate: of the TRIGGERED source-class images, what percent
    does the model call the TARGET class."""
    model.eval()
    hit, total = 0, 0
    with torch.no_grad():
        for inputs, labels in triggered_loader:      # inputs triggered, labels original
            inputs, labels = inputs.to(device), labels.to(device)
            mask = labels == source_class            # only images that were really source
            if not mask.any():
                continue
            preds = model(inputs[mask]).argmax(1)
            hit += (preds == target_class).sum().item()
            total += mask.sum().item()
    return 100 * hit / total if total else 0.0
```

What to change and your options: nothing normally. If you want a fancy progress bar, wrap the inner loop in `tqdm(...)`, but plain prints keep it dependency-free.

Mapping to GTSRB: same three functions. HTB's used `tqdm` bars and filtered `-1` dummy samples; the logic (five-step train loop, `eval()` + `no_grad()` for testing, ASR counted only over triggered source images) is identical.

## Section 7: Train the clean baseline and the trojaned model

```python
criterion = nn.CrossEntropyLoss()   # standard multi-class loss
NUM_EPOCHS = 15                     # CIFAR-10 + this CNN: ~70-78% clean accuracy

# ---- Build the datasets ----
# Clean baseline training data: SAME wrapper, but poison_rate = 0.0.
# This guarantees the ONLY difference between the two models is the poison.
cleanset = PoisonedTrain(trainset_raw, SOURCE_CLASS, TARGET_CLASS, 0.0,
                         add_trigger_square, transform_base, transform_train_post)
# Poisoned training data (10% of source poisoned):
poisonedset = PoisonedTrain(trainset_raw, SOURCE_CLASS, TARGET_CLASS, POISON_RATE,
                            add_trigger_square, transform_base, transform_train_post)

clean_train_loader = DataLoader(cleanset, batch_size=256, shuffle=True, num_workers=2)
poison_train_loader = DataLoader(poisonedset, batch_size=256, shuffle=True, num_workers=2)

# ---- Train the CLEAN baseline ----
print("\n--- Training CLEAN baseline ---")
clean_model = SimpleCNN(NUM_CLASSES, IMG_SIZE).to(device)
train_model(clean_model, clean_train_loader, criterion,
            optim.Adam(clean_model.parameters(), lr=0.001, weight_decay=1e-4),
            NUM_EPOCHS, device)
torch.save(clean_model.state_dict(), "cifar_clean.pth")

# ---- Train the TROJANED model (only the loader differs) ----
print("\n--- Training TROJANED model ---")
trojan_model = SimpleCNN(NUM_CLASSES, IMG_SIZE).to(device)
train_model(trojan_model, poison_train_loader, criterion,
            optim.Adam(trojan_model.parameters(), lr=0.001, weight_decay=1e-4),
            NUM_EPOCHS, device)
torch.save(trojan_model.state_dict(), "cifar_trojaned.pth")
```

What to change and your options:

- **`NUM_EPOCHS`**: more epochs = higher clean accuracy and a firmer backdoor, but slower. 15 is a good balance for CIFAR-10.
- **`batch_size` / `num_workers`**: 256 and 2 are fine. On a fast GPU raise `num_workers` to feed it quicker; on Windows
- `lr=0.001`, `weight_decay=1e-4`: the same trade-off knobs from the HTB notes (bigger weight decay can raise clean accuracy but weaken the backdoor).

Mapping to GTSRB: exactly the same "train two models, identical everything except the data loader" structure, and the same `.pth` save so you can reload without retraining.

## Section 8: Evaluate, and how to read it

```python
# Build the two test loaders:
#  - clean test  -> Clean Accuracy (stealth check)
#  - triggered   -> Attack Success Rate (backdoor check)
clean_testset = torchvision.datasets.CIFAR10(root="./cifar_data", train=False,
                                             download=True, transform=transform_test)
clean_test_loader = DataLoader(clean_testset, batch_size=256, shuffle=False, num_workers=2)

triggered_testset = TriggeredTest(testset_raw, add_trigger_square,
                                  transform_base, normalize_only)
triggered_test_loader = DataLoader(triggered_testset, batch_size=256, shuffle=False, num_workers=2)

# Score both models on both metrics.
for name, model in [("CLEAN baseline", clean_model), ("TROJANED", trojan_model)]:
    ca = evaluate_model(model, clean_test_loader, device)
    asr = calculate_asr(model, triggered_test_loader, SOURCE_CLASS, TARGET_CLASS, device)
    print(f"{name:15s} | Clean Accuracy: {ca:5.2f}%  |  Attack Success Rate: {asr:6.2f}%")
```

How to read the result: you want to see this pattern.

- **Clean baseline: decent CA, near 0% ASR.** The trigger means nothing to a model that never saw poison.
- **Trojaned: about the SAME CA as the baseline, but a high ASR.** Same apparent health, plus an obedient hidden backdoor.

If both models have similar clean accuracy and only the trojaned one has high ASR, **the backdoor is confirmed and it is stealthy** (my verified run on the stand-in dataset produced exactly this: both models identical clean accuracy, clean ASR 0%, trojan ASR 100%).

Reloading later without retraining:

```python
# If you restart the kernel, skip Section 7 and just reload:
trojan_model = SimpleCNN(NUM_CLASSES, IMG_SIZE).to(device)
trojan_model.load_state_dict(torch.load("cifar_trojaned.pth", map_location=device))
```
