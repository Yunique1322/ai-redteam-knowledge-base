# Section 13: Training the Robust Model

## The idea in one breath
This is the full lab pipeline:

1. Load **normalized** MNIST.
2. Train the baseline: ~99% clean, ~73% under FGSM.
3. Use the baseline to make a fixed test set, **500 test images × 10 ε values × {FGSM, I-FGSM}**, saved as `adv_examples.safetensors`.
4. Adversarially train a fresh LeNet-5 with **ε-spread FGSM** and a 50/50 clean/attacked mix.
5. Save the result as `robust_model.safetensors` for the evaluator.

The pass bar is **robust ≥ 92% and clean ≥ 94%**. I ran the whole thing on real MNIST and matched the course closely:

| Checkpoint | Course | Mine |
|---|---|---|
| Epoch 5 | 98.8 / 93.5 | 98.9 / 93.3 |
| Epoch 10 | 99.0 / 95.4 | 98.9 / 93.6 |
| Final, full test set (clean / robust) | **99.0 / 95.4** | **98.9 / 95.6** |

The 10 adversarial epochs took **5.5 minutes on 2 CPU cores**.

> **Heads-up:** as in Section 12, the PDF flattened all the code. I've restored the line breaks and indentation, with tokens unchanged, and ran it. Both functions here were split across many code blocks in the course: `generate_adversarial_examples` across 4 and `train_adversarial` across 7. I've **merged** each into one block below, every line unchanged. The course's "You can download the evaluator here" link was lost in the PDF; get `evaluate_robustness.py` from the HTB section page.

## Part 1 - Making the evaluation examples (merged)

```python
def generate_adversarial_examples(model, test_loader, device, num_samples=500):
    """Generate adversarial examples across multiple epsilon values."""
    model.eval()

    # Collect clean samples
    clean_images_list = []
    clean_labels_list = []
    collected = 0

    print(f"Collecting {num_samples} clean samples...")
    for images, labels in test_loader:
        if collected >= num_samples:
            break
        batch_size = min(images.size(0), num_samples - collected)
        clean_images_list.append(images[:batch_size])
        clean_labels_list.append(labels[:batch_size])
        collected += batch_size

    clean_images = torch.cat(clean_images_list, dim=0)
    clean_labels = torch.cat(clean_labels_list, dim=0)

    # Generate adversarial examples at each epsilon
    fgsm_by_epsilon = {}
    ifgsm_by_epsilon = {}

    print(f"\nGenerating adversarial examples across epsilon spread: {EPSILON_SPREAD}")
    for eps in EPSILON_SPREAD:
        print(f"\n Generating at epsilon={eps}...")
        fgsm_images_list = []
        ifgsm_images_list = []
        batch_size = 128

        pbar = tqdm(total=num_samples, desc=f" eps={eps}")
        for i in range(0, num_samples, batch_size):
            end_idx = min(i + batch_size, num_samples)
            images = clean_images[i:end_idx].to(device)
            labels = clean_labels[i:end_idx].to(device)

            model.train() # Need train mode for gradient computation
            fgsm_images = fgsm_attack(model, images, labels, eps)
            ifgsm_images = i_fgsm_attack(model, images, labels, eps)
            model.eval()

            fgsm_images_list.append(fgsm_images.cpu())
            ifgsm_images_list.append(ifgsm_images.cpu())
            pbar.update(end_idx - i)
        pbar.close()

        fgsm_by_epsilon[eps] = torch.cat(fgsm_images_list, dim=0)
        ifgsm_by_epsilon[eps] = torch.cat(ifgsm_images_list, dim=0)

    return {
        'clean_images': clean_images,
        'clean_labels': clean_labels,
        'fgsm_images': fgsm_by_epsilon[EPSILON],
        'ifgsm_images': ifgsm_by_epsilon[EPSILON],
        'epsilon': EPSILON,
        'epsilon_spread': EPSILON_SPREAD,
        'fgsm_by_epsilon': fgsm_by_epsilon,
        'ifgsm_by_epsilon': ifgsm_by_epsilon
    }
```

| Piece | What it does |
|---|---|
| Collect loop with `min(images.size(0), num_samples - collected)` | Takes exactly 500 test images. The 4th batch only contributes 116 (128 + 128 + 128 + 116 = 500). The test loader isn't shuffled, so these are always the **first 500 test images** |
| `torch.cat(..., dim=0)` | Gives one (500, 1, 28, 28) tensor plus 500 labels |
| `for eps in EPSILON_SPREAD` | 10 values, from 0.1 to 1.0 |
| Inner loop in chunks of 128 | Moves each chunk to the GPU, attacks it, and moves the result back to the CPU to free GPU memory |
| `model.train()` "for gradient computation" | The same wrong reason as in Section 12; eval mode is fine for gradients. No harm with LeNet |
| `fgsm_attack` **and** `i_fgsm_attack` with the **baseline** model | Every saved example is crafted against the **baseline**. That's why Section 14's numbers for the robust model are **transfer** results |
| Return dictionary | The full per-ε dictionaries, plus the ε = 0.3 sets at the top level. `save_adversarial_examples` deliberately skips those two, because they're the same tensors as the 0.3 entries (Section 12) |

## Part 2 - The adversarial training function (merged)

```python
def train_adversarial(model, train_loader, test_loader, device, epochs=25, lr=0.001, epsilon=EPSILON):
    """Train model with adversarial training using epsilon spread."""
    model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)

            batch_epsilon = np.random.choice(EPSILON_SPREAD)
            adv_images = fgsm_attack(model, images, labels, batch_epsilon)

            combined_images = torch.cat([images, adv_images], dim=0)
            combined_labels = torch.cat([labels, labels], dim=0)

            perm = torch.randperm(combined_images.size(0))
            combined_images = combined_images[perm]
            combined_labels = combined_labels[perm]

            optimizer.zero_grad()
            outputs = model(combined_images)
            loss = criterion(outputs, combined_labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += combined_labels.size(0)
            correct += predicted.eq(combined_labels).sum().item()
            pbar.set_postfix({
                'loss': f'{total_loss / (pbar.n + 1):.4f}',
                'acc': f'{100.0 * correct / total:.2f}%'
            })

        scheduler.step()

        if (epoch + 1) % 5 == 0 or epoch == epochs - 1:
            clean_acc = evaluate_accuracy(model, test_loader, device)
            adv_acc = evaluate_adversarial_accuracy(
                model, test_loader, device, epsilon, num_batches=20
            )
            print(f"\n Epoch {epoch+1}: Clean={clean_acc:.1f}%, Robust={adv_acc:.1f}%")

    return model
```

**Setup:**

| Piece | What it does |
|---|---|
| `AdamW(lr=0.001, weight_decay=1e-4)` | Adam with **decoupled weight decay**: the weights shrink slightly every step, independently of the gradient, to stop over-fitting |
| `CosineAnnealingLR(T_max=epochs)` | The learning rate follows a cosine curve from 0.001 down to about 0 by the last epoch |
| `nn.CrossEntropyLoss()` | The same loss as `F.cross_entropy` |

**Per batch:**

| Piece | What it does |
|---|---|
| `np.random.choice(EPSILON_SPREAD)` | A **fresh random ε per batch** from 0.1 to 1.0: ε-spread training |
| `fgsm_attack(model, ...)` | Attacks the **current** model, the vaccine from Section 11 |
| `torch.cat([images, adv_images])` + `torch.cat([labels, labels])` | A batch of 256 (128 clean + 128 attacked) with the **true** labels |
| `perm = torch.randperm(...)` | Shuffles clean and attacked images together |
| `zero_grad` → forward → loss → `backward` | Ordinary training on the combined batch. This also clears the weight gradients the attack left behind |
| `clip_grad_norm_(max_norm=1.0)` | If the total gradient size is over 1, it's scaled down to 1, which stops big jumps early on |
| `total_loss`, `correct`, `total` | Running numbers for the progress bar |

**Per epoch:**

| Piece | What it does |
|---|---|
| `scheduler.step()` | Moves the learning rate down the cosine curve |
| Every 5 epochs and at the last epoch | Clean accuracy on the full test set, plus **white-box FGSM** robustness on 20 batches (2,560 images) |

**What changes inside, for one batch:**

| Name | Shape | Meaning |
|---|---|---|
| `images`, `adv_images` | (128, 1, 28, 28) each | Clean, and FGSM at this batch's ε |
| `combined_images` | (256, 1, 28, 28) | Both, shuffled |
| `combined_labels` | (256,) | The true labels, twice |
| `outputs` | (256, 10) | Logits |

> **An honest note the course skips:**
> - **The `epsilon` argument doesn't control training.** Training always samples from `EPSILON_SPREAD`; `epsilon` is only used in the epoch-5 and epoch-10 checks. To train the "single-ε" model compared in Section 14, you'd have to change `np.random.choice(EPSILON_SPREAD)` to `EPSILON`. Passing `epsilon=0.3` won't do it.
> - **The shuffle does nothing.** The loss is the *mean* over the batch, and LeNet has no BatchNorm, so the gradient is exactly the same whatever the order. The course's "position-based pattern learning" reason doesn't apply. It's harmless but pointless.
> - **The progress-bar loss is wrong.** `total_loss / (pbar.n + 1)` divides by tqdm's internal counter, which only updates every so often. My run showed the loss jumping 2.33 → 3.48 → 2.32 → 2.89 in the first batches. The accuracy readout is fine. Trust the printed `Epoch N: Clean/Robust` lines instead.
> - **It's FGSM-only training.** This is exactly the setup with the gradient-masking risk from Section 11. Section 14 shows the result.

## Part 3 - Running the pipeline

### Step 1: Setup
```python
set_reproducibility(1337)
device = get_device()
print(f"Using device: {device}")
train_loader, test_loader = get_mnist_loaders(normalize=True)
print(f"Train batches: {len(train_loader)}, Test batches: {len(test_loader)}")
```

Output:
```txt
Using device: cuda
Train batches: 469, Test batches: 79
```

| Code / output | Meaning |
|---|---|
| `set_reproducibility(1337)` | Same seed, so the same results each run (I got the same numbers on two runs) |
| `get_mnist_loaders(normalize=True)` | **Normalized** data in [−0.4242, 2.8215]. This is the fix for Section 12's trap |
| 469 / 79 | ⌈60,000 / 128⌉ = 469 and ⌈10,000 / 128⌉ = 79 batches |

### Step 2: Baseline
```python
baseline_model = LeNet5()
baseline_model = train_model(baseline_model, train_loader, test_loader, device=device, epochs=10, learning_rate=0.001)
adv_acc = evaluate_adversarial_accuracy(baseline_model, test_loader, device, EPSILON)
print(f"Baseline Model - Robust: {adv_acc:.1f}%")
save_file(baseline_model.state_dict(), "baseline_model.safetensors")
```

| Code | What it does |
|---|---|
| `train_model(..., epochs=10)` | Ordinary training. **I got 99.1% clean** |
| `evaluate_adversarial_accuracy(..., EPSILON)` | White-box FGSM on the full test set. **I got 76.0%**; the course says "~73%" |
| `save_file(...state_dict()...)` | The weights only, safely stored. The evaluator uses this file for `--compare` |

### Step 3: Save the evaluation examples
```python
adv_data = generate_adversarial_examples(
    baseline_model, test_loader, device, num_samples=500
)
save_adversarial_examples(adv_data, "adv_examples.safetensors")
print("Adversarial examples saved to adv_examples.safetensors")
```

| Code | What it does |
|---|---|
| `generate_adversarial_examples(baseline_model, ..., 500)` | 500 images × 10 ε × 2 attacks, all made against the **baseline** |
| `save_adversarial_examples(...)` | Writes **22 tensors**: clean images, labels, and 10 FGSM + 10 I-FGSM sets, plus the metadata (cell 1) |

### Step 4: A fresh model, before training
```python
model = LeNet5()
model.to(device)
clean_acc = evaluate_accuracy(model, test_loader, device)
adv_acc = evaluate_adversarial_accuracy(model, test_loader, device, EPSILON)
print(f"Before training - Clean: {clean_acc:.1f}%, Robust: {adv_acc:.1f}%")
```

Output:
```txt
Before training - Clean: 11.3%, Robust: 1.9%
```

| Number | Meaning |
|---|---|
| 11.3% clean | Random guessing over 10 classes ≈ 10%. I got 10.2% |
| 1.9% robust | *Below* chance, because FGSM pushes each image away from its true class, even for an untrained model. I got 2.4% |

### Step 5: Adversarial training
```python
model = train_adversarial(model, train_loader, test_loader, device, epochs=10)
```

Output:
```txt
Epoch 5: Clean=98.8%, Robust=93.5%
Epoch 10: Clean=99.0%, Robust=95.4%
```

| Code / output | Meaning |
|---|---|
| `epochs=10` | Overrides the default of 25. The cosine schedule then runs over 10 epochs |
| Epoch 5: 98.8 / 93.5 | Already past the bars (robust ≥ 92, clean ≥ 94). **Mine: 98.9 / 93.3** |
| Epoch 10: 99.0 / 95.4 | **Mine: 98.9 / 93.6** on the 20-batch check, 95.6 on the full test set. The first 20 batches are slightly harder than average |

"Robust accuracy crossing 92% while clean stays above 94%" is the lab's pass condition.

### Step 6: Final check and save
```python
clean_acc = evaluate_accuracy(model, test_loader, device)
adv_acc = evaluate_adversarial_accuracy(model, test_loader, device, EPSILON)
print(f"Final - Clean: {clean_acc:.1f}%, Robust: {adv_acc:.1f}%")
```

Output:
```txt
Final - Clean: 99.0%, Robust: 95.4%
```

```python
save_file(model.state_dict(), "robust_model.safetensors")
print("Model saved to robust_model.safetensors")
```

| Code | What it does |
|---|---|
| Full-test white-box FGSM at 0.3 | **Mine: 98.9% clean / 95.6% robust.** That passes both bars with margin |
| `save_file(model.state_dict(), ...)` | The file the evaluator grades |

### Step 7: The evaluator
```shellsession
Ishman1322@htb[/htb]$ python evaluate_robustness.py --model-path robust_model.safetensors
```

| Command | What it does |
|---|---|
| `evaluate_robustness.py --model-path ...` | Loads your weights into LeNet-5 and tests them on `adv_examples.safetensors` at every ε. Explained in Section 14 |

> **If this errors:**
> - `Error(s) in loading state_dict ... Missing key(s)` / `Unexpected key(s)`: you changed the LeNet-5 class (renamed or resized a layer). Restore it exactly.
> - Very low evaluator scores but good training numbers: your model was trained on **not-normalized** data, while the saved examples are normalized. Check with Section 12's cell 1.
> - Training too slow on CPU: each epoch took about 33 s on 2 cores, so 10 epochs is about 5.5 minutes. The default 25 epochs would be about 14 minutes.

## Reading the figure/output
There are no figures in this section. The outputs are explained above, and my numbers match the course to within ±0.3 points everywhere except the epoch-10 20-batch check (93.6 vs 95.4).

## Copy-paste cells (mine, not the course's)
Run these after the course's cells. They use `LeNet5`, `model`, `test_loader`, `device`, `clean_acc` and `adv_acc` from Steps 1-6.

### 1) Look inside `adv_examples.safetensors`

```python
from safetensors import safe_open
with safe_open("adv_examples.safetensors", "pt") as f:
    keys = sorted(f.keys())
    print("metadata:", f.metadata())
    print(f"{len(keys)} tensors, e.g.:", keys[:3], "...", keys[-2:])
    for k in ["clean_images", "clean_labels", "fgsm_eps_0.3", "ifgsm_eps_1.0"]:
        t = f.get_tensor(k)
        print(f" {k:14} {tuple(t.shape)} {t.dtype}")
```

Expected:
```txt
metadata: {'epsilon': '0.3', 'epsilon_spread': '[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]'}
22 tensors, e.g.: ['clean_images', 'clean_labels', 'fgsm_eps_0.1'] ... ['ifgsm_eps_0.9', 'ifgsm_eps_1.0']
  clean_images (500, 1, 28, 28) torch.float32
  clean_labels (500,) torch.int64
  fgsm_eps_0.3 (500, 1, 28, 28) torch.float32
  ifgsm_eps_1.0 (500, 1, 28, 28) torch.float32
```

### 2) Prove the saved model reloads, and check the pass bar
This loads the file into a **fresh** `LeNet5`, as the evaluator does, and compares predictions.

```python
from safetensors.torch import load_file
reloaded = LeNet5()
reloaded.load_state_dict(load_file("robust_model.safetensors")) # fails loudly if the architecture changed
reloaded.to(device).eval(); model.eval()
x, _ = next(iter(test_loader)); x = x.to(device)
with torch.no_grad():
    same = torch.equal(reloaded(x).argmax(1), model(x).argmax(1))
print(f"reload OK, same predictions: {same}")
print(f"clean {clean_acc:.1f}% (need >= 94) -> {'PASS' if clean_acc >= 94 else 'FAIL'} | "
      f"robust {adv_acc:.1f}% (need >= 92) -> {'PASS' if adv_acc >= 92 else 'FAIL'}")
```

Expected (my run):
```txt
reload OK, same predictions: True
clean 98.9% (need >= 94) -> PASS | robust 95.6% (need >= 92) -> PASS
```

## Quick self-check
You want to train the "single-ε" comparison model, which only ever uses ε = 0.3, so you call `train_adversarial(model, ..., epsilon=0.3)`.
- **(a)** Does that give you a single-ε model? Why or why not?
- **(b)** Which exact expression would have to change?

*(Answer key is in COAE-progress.md → Open self-checks.)*

## The facts worth locking in for the exam
- The pipeline is **normalize → baseline (≈99% clean, ≈73-76% FGSM) → 500 test images × 10 ε × {FGSM, I-FGSM} against the baseline → fresh LeNet-5 trained adversarially → `robust_model.safetensors`**.
- The lab passes when **robust FGSM accuracy ≥ 92% and clean ≥ 94%**. The course gets 99.0 / 95.4 after 10 epochs; I got 98.9 / 95.6.
- `train_adversarial`: **AdamW (lr 0.001, weight decay 1e-4), cosine LR, a random ε per batch from 0.1-1.0, FGSM on the current model, a 256-image batch of clean + attacked with the true labels, gradient clipping at 1.0**.
- **The `epsilon` argument only sets the check ε, not the training ε**; single-ε training means changing `np.random.choice(EPSILON_SPREAD)`.
- The **batch shuffle doesn't change the gradient** (mean loss, no BatchNorm), and **the progress-bar loss is unreliable** because of tqdm's lagging counter.
- A random model scores **~10% clean and below 10% robust**, because FGSM pushes even an untrained model away from the true class.
- The **evaluation examples are crafted against the baseline**, so grading the robust model on them is a **transfer** test, not white-box. Section 14 shows what that hides.
- **Method point:** after training, **reload the saved file into a fresh copy of the class and compare predictions** before submitting. Missing or unexpected keys mean the architecture changed.
