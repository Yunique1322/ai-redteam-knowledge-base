# Targeted Label Attacks

## The idea (vs plain label flipping)

Plain label flipping was scattershot: flip random points of any class to make the model generally worse. A **targeted** attack is surgical: **flip only one class's points, all in the same direction, to bias the model against that specific class.** Here the goal is "make genuinely positive reviews get read as negative," so you take real Class 1 (positive) points and relabel them Class 0 (negative).

Why targeting is more effective at its goal (intuition, no formulas): it's the same "a confidently-wrong point gets a huge penalty and drags the boundary toward itself" mechanism as before. The difference is **every flipped point now sits on the same side (they're all true Class 1 points), so all the pressure pushes the boundary in one consistent direction, into the Class 1 region**, instead of cancelling out in random directions. The result is a deliberate, predictable bias: the model starts calling the Class 1 area "Class 0." Your screenshot shows it, 40% of the yellow (Class 1) cluster is now marked flipped and relabeled toward Class 0.

## What's different in the code vs `flip_labels`

Three changes worth noticing:

- It first **filters to one class** with `np.where(y == target_class)` before choosing anything, so only target-class points are eligible.
- The percentage is applied to the **size of that class**, not the whole dataset.
- The flip is a straight **assignment to a chosen class** (`y_poisoned[flipped_indices] = new_class`), not an invert, because you're forcing everything to one specific label rather than swapping 0↔1.

## `targeted_flip_labels` (one cell, full function)

HTB again splits this across boxes to narrate it, but it's a single function in one cell. The one slightly tricky bit is the two-step index mapping, explained right after.

```
def targeted_flip_labels(y, poison_percentage, target_class, new_class, seed=1337):

    if not 0 <= poison_percentage <= 1:
        raise ValueError("poison_percentage must be between 0 and 1.")
    if target_class == new_class:
        raise ValueError("target_class and new_class cannot be the same.")
    # Ensure target_class and new_class are present in y
    unique_labels = np.unique(y)
    if target_class not in unique_labels:
        raise ValueError(f"target_class ({target_class}) does not exist in y.")
    if new_class not in unique_labels:
        raise ValueError(f"new_class ({new_class}) does not exist in y.")

    # Identify indices belonging to the target class
    target_indices = np.where(y == target_class)[0]
    n_target_samples = len(target_indices)

    if n_target_samples == 0:
        print(f"Warning: No samples found for target_class {target_class}. No labels flipped.")
        return y.copy(), np.array([], dtype=int)

    # Calculate the number of labels to flip within the target class
    n_to_flip = int(n_target_samples * poison_percentage)

    if n_to_flip == 0:
        print(f"Warning: Poison percentage ({poison_percentage * 100:.1f}%) is too low "
              f"to flip any labels in the target class (size {n_target_samples}).")
        return y.copy(), np.array([], dtype=int)

    # Use a dedicated random number generator instance with the specified seed
    rng_instance = np.random.default_rng(seed)

    # Randomly select indices from the target_indices subset to flip
    # These are indices relative to the target_indices array
    indices_within_target_set_to_flip = rng_instance.choice(
        n_target_samples, size=n_to_flip, replace=False
    )
    # Map these back to the original array indices
    flipped_indices = target_indices[indices_within_target_set_to_flip]

    # Create a copy to avoid modifying the original array
    y_poisoned = y.copy()

    # Perform the flip for the selected indices to the new class label
    y_poisoned[flipped_indices] = new_class

    print(f"Targeting Class {target_class} for flipping to Class {new_class}.")
    print(f"Identified {n_target_samples} samples of Class {target_class}.")
    print(f"Attempting to flip {poison_percentage * 100:.1f}% ({n_to_flip} samples) of these.")
    print(f"Successfully flipped {len(flipped_indices)} labels.")

    return y_poisoned, flipped_indices
```

The index-mapping step, in plain terms: `target_indices` is the list of real positions of every Class 1 point (say `[3, 8, 12, 20, ...]`). `choice(n_target_samples, ...)` picks positions _within that list_ (like "the 1st and 4th class-1 point"), giving something like `[0, 3]`. Then **`target_indices[[0, 3]]` translates those back to the real positions in the full array** (`[3, 20]`), which is what actually gets flipped. Two hops: pick inside the class, then map back out to the whole dataset.

## Generate the targeted-poisoned dataset (cell)

Flip 40% of Class 1 to Class 0, then plot it.

```
poison_percentage_targeted = 0.40  # Target 40%
target_class_to_flip = 1  # Target Class 1 (Positive)
new_label_for_flipped = 0  # Flip them to Class 0 (Negative)

# Use the function to create the poisoned dataset
y_train_targeted_poisoned, targeted_flipped_indices = targeted_flip_labels(
    y_train,
    poison_percentage_targeted,
    target_class_to_flip,
    new_label_for_flipped,
    seed=SEED,  # Use the global SEED for reproducibility
)

print("\n--- Visualizing Targeted Poisoned Data ---")
# Plot the result of the targeted flip
plot_poisoned_data(
    X_train,
    y_train,  # Pass original y
    y_train_targeted_poisoned,
    targeted_flipped_indices,
    title=f"Training Data: {poison_percentage_targeted * 100:.0f}% of Class {target_class_to_flip} Flipped to {new_label_for_flipped}",
    target_class_info=target_class_to_flip,
)
```

In the plot, the flipped points (red `X`) all sit inside the Class 1 cluster, because that's the only place the attack touches. Compare that to the random attack, where the `X` marks were scattered across both clusters. That concentration is the whole point.
![Plot the result of the targeted flip](/assets/Pasted%20image%2020260918231336.png)

## Train on the targeted-poisoned data (cell)

```
targeted_poisoned_model = LogisticRegression(random_state=SEED)
targeted_poisoned_model.fit(X_train, y_train_targeted_poisoned)
```
![Targeted Label Attacks](/assets/7%20Targeted%20Label%20Attacks-20260918231400134.png)
