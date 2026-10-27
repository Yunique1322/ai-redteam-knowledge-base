# Section 16: Training and Deploying the Student Model

## The idea in one breath

The student is a normal `[128, 64]` MLP trained on the **5,000 public images** using the **noisy pseudo-labels** as if they were real labels. It reaches about **88%** on the holdout. That's as good as the clean teacher vote, even though about 13% of its labels are wrong.

Because the student only ever touched public data, **training it and querying it afterwards cost no privacy budget**; all of it was spent during labeling. It's saved as `models/pate_student.safetensors`.

> **Heads-up:** no mangled formulas.
> 
> **One number clash with last section:** here the course says the clean ensemble is "~92%" with a "4-point gap". Section 15 (and its figure) said **87.8%**. With the real numbers the gap is about 0, or even negative. See Part 4.

---

## Part 1: The knowledge distillation idea

|Aspect|Standard distillation|PATE|
|---|---|---|
|Teacher(s)|one big model|250 small teachers|
|What gets transferred|**soft probabilities** (e.g. [0.7, 0.2, 0.1])|**hard labels** (just "7")|
|Why|soft labels carry more signal|soft labels would leak **how confident** the teachers are, which reveals more about the private data|
|Noise|none|Laplace on the votes|

Hard labels still work because the label is the main signal. There are bonus benefits too: one network carries the whole ensemble's knowledge, the noisy labels add some regularization, and deployment is just a normal model.

---

## Part 2: Building and training the student

```python
print("\n" + "=" * 60)
print("PHASE 4: Training Student Model")
print("=" * 60)

student_model = MLP(
    input_size=num_features,
    hidden_layers=STUDENT_CONFIG['hidden_layers'],
    num_classes=DATASET_CONFIG['num_classes'],
    dropout=STUDENT_CONFIG['dropout']
)

student_train_loader = create_dataloader(
    X_query, student_labels,
    STUDENT_CONFIG['batch_size']
)
```

|Code|What it does|
|---|---|
|`MLP(784 -> 128 -> 64 -> 10, dropout 0.2)`|the same shape as one teacher. A bigger one might memorize the noisy labels; a smaller one might underfit|
|`create_dataloader(X_query, student_labels, 64)`|**public images paired with noisy votes**. No private image and no real label goes in|

|Name|Shape|Meaning|
|---|---|---|
|`X_query`|(5000, 784)|the scaled public images that were labeled|
|`student_labels`|(5000,)|noisy pseudo-labels (~87% correct)|
|`student_train_loader`|79 batches of 64|5000 / 64 = 78.1, rounded up|

```python
train_model(
    student_model, student_train_loader, holdout_loader,
    device=DEVICE,
    epochs=STUDENT_CONFIG['epochs'],
    learning_rate=STUDENT_CONFIG['learning_rate']
)

student_test_acc = evaluate_accuracy(student_model, holdout_loader, DEVICE)
print(f"\nPATE student test accuracy: {student_test_acc:.2f}%")
```

|Code|What it does|
|---|---|
|`train_model(...)`|Adam, lr 0.001, 30 epochs, cross-entropy, printing holdout accuracy each epoch (Section 14's source check). The holdout is only watched, not used to pick the best epoch|
|`evaluate_accuracy(student_model, holdout_loader, ...)`|**the real result**, on images nobody trained on: about **88%**|

**Training the student is free.** It's **post-processing** of already-released noisy labels, so 100 epochs or ten restarts don't change ε.

---

## Part 3: Why noisy labels still work

### Step 1: Toy, random vs systematic mistakes

- **Random:** 10 images of a "7", and 2 got a random wrong label ("2" and "5"). Eight say 7 and the wrong ones point in different directions, so **the consistent majority wins**. The network learns "7".
- **Systematic:** every "4" that looks like a "9" gets labeled 9. Now the error is consistent, so **the network learns the error**.

### Step 2: What the course claims

Noise flips mostly hit ambiguous, low-margin samples, and clear cases (the majority) stay clean. Networks tolerate 20-40% random label noise as long as it's independent of the true label.

> **An honest note the course skips:** the course says "a sample doesn't get a wrong label because of its features". That's only half right, and it contradicts its own previous paragraph.
> 
> 1. Of the ~12.8% wrong labels, only about 0.6 points come from the Laplace noise (Section 15). **Most are the ensemble's own mistakes**, and those are **systematic**: the teachers all fail on the same hard digits (the correlated-errors finding from Section 15).
> 2. The Laplace flips land on **low-margin** samples, and being low-margin _is_ a property of the image (an ambiguous digit).
> 
> So the student inherits the ensemble's blind spots. What it doesn't learn is the random Laplace part.

### Step 3: Why the student can beat its own labels

The labels are ~87% right, yet the student scores ~88%. Random wrong labels mostly cancel out during training (Step 1), so a student can end up slightly better than the labels it was trained on.

---

## Part 4: "Comparing with clean training"

```python
# Compare against ensemble performance
clean_preds = votes.argmax(axis=1)
clean_ensemble_acc = (clean_preds == y_query_true).mean() * 100
accuracy_gap = clean_ensemble_acc - student_test_acc

print(f"\nComparison:")
print(f" Clean ensemble accuracy: {clean_ensemble_acc:.2f}%")
print(f" PATE student accuracy: {student_test_acc:.2f}%")
print(f" Accuracy gap: {accuracy_gap:.2f}%")
```

|Code|What it does|
|---|---|
|`clean_preds`, `clean_ensemble_acc`|recomputed exactly as in Section 15. Same numbers (~87.8%), scored on the **5,000 public queries**|
|`accuracy_gap`|ensemble minus student. The student was scored on the **holdout**|

> **An honest note the course skips:**
> 
> 1. **The "92% vs 88%, a 4-point privacy cost" claim doesn't fit the course's own Section 15 numbers**: an 87.8% ensemble against an ~88% student gives a gap of about 0, possibly negative.
> 2. **This gap isn't the privacy cost anyway.** It mixes two different evaluation sets, and it mixes the noise effect with the effect of distilling into a single network.
> 3. The honest privacy cost is either **clean vs noisy labels on the same 5,000 queries** (0.6 points in Section 15), or **a student trained on clean votes vs the PATE student, both on the holdout**. My cell 1 does the second.

---

## Part 5: Deployment, unlimited and free

```python
def deploy_model(model, new_data, scaler, device):
    """
    Deploy the PATE student for inference.

    No privacy budget consumed per query.
    """
    model.eval()
    X_norm = scaler.transform(new_data)
    X_tensor = torch.tensor(X_norm, dtype=torch.float32).to(device)

    with torch.no_grad():
        outputs = model(X_tensor)
        predictions = outputs.argmax(dim=1).cpu().numpy()

    return predictions
```

|Code|What it does|
|---|---|
|`model.eval()`|dropout off|
|`scaler.transform(new_data)`|expects **raw** [0, 1] pixels and scales them like the training data|
|`torch.tensor(...).to(device)`|numpy array -> float tensor on the GPU or CPU|
|`torch.no_grad()`|inference only, no gradient tracking|
|`outputs.argmax(dim=1)`|returns **labels only**, no probabilities. That also gives less for an MIA to work with (Section 2's lesson)|

```python
# Example deployment
sample_data = X_holdout[:100]
predictions = deploy_model(student_model, sample_data, scaler, DEVICE)
print(f"Processed {len(predictions)} queries with no privacy cost")
```

This passes `X_holdout` (raw), which is correct. **Passing `X_holdout_norm` would scale it twice** and quietly ruin accuracy.

```python
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

save_file(student_model.state_dict(), os.path.join(MODEL_DIR, "pate_student.safetensors"))
print(f"\nModel saved to {MODEL_DIR}/pate_student.safetensors")
```

|Code|What it does|
|---|---|
|`MODEL_DIR = "models"`|this time the folder variable is actually defined (unlike DP-SGD's missing names)|
|`save_file(student_model.state_dict(), ...)`|weights only, in safetensors (safe, no pickle code execution). Likely the PATE challenge's submission file. No `._module` is needed because the student isn't Opacus-wrapped|

> **An honest note the course skips:**
> 
> 1. "Loading requires only the architecture and the state dict" isn't true. **`deploy_model` also needs the `scaler`, which isn't saved.** Ship only the safetensors file and someone feeds raw pixels to a model trained on scaled ones, and accuracy collapses. My cell 3 saves it.
> 2. "No information about the private images remains" isn't true either. The scaler's 784 means and stds **are private-data statistics with no noise added** (the Section 14 leak, now shipped). For a clean deployment, fit the scaler on public data.

**Unlimited free queries:** unlike randomized response (noise added on every answer), the student's weights already contain everything, so a million predictions cost no extra ε. What must **not** be deployed is the teacher ensemble, since every teacher query spends budget.

> **Also skipped by the course:** 7,000 of the 12,000 public images were never labeled. The original PATE paper used semi-supervised learning on such leftovers to gain accuracy at no privacy cost.

---

## Reading the figure/output

There's no figure in this section. Expected prints:

- `PATE student test accuracy: ~88%`
- The comparison block, which with your Section 15 numbers will likely show a gap near 0 (not 4)
- `Processed 100 queries with no privacy cost`
- `Model saved to models/pate_student.safetensors`

---

## Copy-paste cells (mine, not the course's)

I didn't run these. Run them **after** the course cells: extra training uses up random numbers, so anything you run afterwards won't match HTB's printed numbers.

### The true privacy cost: a clean-label student vs the PATE student

This is **for measurement only.** A student trained on clean votes has no privacy guarantee, so never deploy or submit it.

```python
import contextlib, io
clean_student = MLP(input_size=num_features, hidden_layers=STUDENT_CONFIG['hidden_layers'],
                    num_classes=DATASET_CONFIG['num_classes'], dropout=STUDENT_CONFIG['dropout'])
clean_loader = create_dataloader(X_query, clean_preds, STUDENT_CONFIG['batch_size'])
with contextlib.redirect_stdout(io.StringIO()):
    train_model(clean_student, clean_loader, holdout_loader, device=DEVICE,
                epochs=STUDENT_CONFIG['epochs'], learning_rate=STUDENT_CONFIG['learning_rate'])
clean_student_acc = evaluate_accuracy(clean_student, holdout_loader, DEVICE)
print(f"student on CLEAN votes (no privacy, measurement only): {clean_student_acc:.2f}%")
print(f"PATE student (noisy votes): {student_test_acc:.2f}%")
print(f"true privacy cost to the student: {clean_student_acc - student_test_acc:.2f} pts")
```

Expected: a cost of about 0-1 point, much smaller than the course's "4 points".

### Ensemble vs student on the same holdout

Also measurement only (the 250 teachers vote on all 10,000 holdout images).

```python
holdout_votes_all = get_teacher_votes(teachers, X_holdout_norm, DEVICE)
ens_holdout_acc = (holdout_votes_all.argmax(1) == y_holdout).mean() * 100
print(f"clean ensemble on HOLDOUT: {ens_holdout_acc:.2f}% | PATE student on HOLDOUT: {student_test_acc:.2f}%")
```

Expected: both around 88%, now a fair comparison on the same images.

### Save the scaler next to the model

```python
from safetensors.numpy import save_file as save_numpy
save_numpy({"mean": scaler.mean_.astype(np.float32), "scale": scaler.scale_.astype(np.float32)},
           os.path.join(MODEL_DIR, "pate_student_scaler.safetensors"))
print("saved scaler next to the model")
```

Expected: `models/pate_student_scaler.safetensors` appears. At load time, scale with `(x - mean) / scale`.

## Key facts to lock in

- **The PATE student is a normal [128, 64] MLP trained only on public images with noisy hard pseudo-labels, and it reaches about 88% on the holdout**, roughly matching the clean teacher vote even though ~13% of its training labels are wrong.
- **PATE transfers hard labels rather than soft probabilities because soft outputs would reveal how confident the teachers were**, which leaks more about the private data.
- **Training the student and querying it afterwards cost no privacy budget, because both are post-processing of the already-released noisy labels**. The whole budget is spent at labeling time, so the student can be retrained or queried unlimited times, but the teacher ensemble must never be exposed.
- **Most of the student's wrong labels come from systematic, correlated ensemble errors, not from Laplace noise** (noise costs only ~0.6 points). The noise flips land on low-margin, ambiguous images, so the course's "noise is independent of features" claim is wrong.
- **The course's "accuracy gap" (clean ensemble on public queries minus student on holdout) isn't the privacy cost**: it mixes evaluation sets and the effect of distillation, and its "92% vs 88%" contradicts Section 15's 87.8%. The honest cost is a clean-label student against the noisy-label student on the same holdout.
- **Deployment needs the scaler as well as the safetensors weights, and the course doesn't save it**. That scaler carries un-noised private-data statistics, so fit it on public data for a clean deployment.
- **`deploy_model` expects raw [0, 1] pixels and scales them itself**. Passing already-scaled data double-scales it and silently ruins predictions. It returns labels only, which also limits MIA signal.
- **Method point:** **when auditing a PATE deployment, confirm that only the student is reachable (teachers offline), that the preprocessing artifacts (scaler) ship with the model and contain no un-noised private statistics, that the reported ε matches the labeling queries, and that the privacy cost is measured on the same evaluation set**.

---
