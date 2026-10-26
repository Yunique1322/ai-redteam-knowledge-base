# The Label Flipping Attack

## Why flipping labels actually breaks the model (the loss argument)

### Start with what the model outputs

For any point, the model gives a number `p` = "how confident I am this point is Class 1." Think of it as a percentage from 0% to 100%. High `p` means "I'm sure it's Class 1," low `p` means "I'm sure it's Class 0."

### Training is a penalty game

Training works by giving the model a **penalty** for every point it gets wrong, then adjusting the line to make the total penalty as small as possible. The size of the penalty follows one rule that matters here:

- Confident and **right** → almost no penalty.
- Unsure (around 50/50) → a moderate penalty.
- Confident and **wrong** → a **massive** penalty that grows explosively the more confident-but-wrong it was.

That last bullet is the only thing the "log" is doing in the formula. **Being confidently wrong is punished enormously, far more than just being uncertain.** That's it. You never have to compute a log.

Quick reminder on `log`: for a probability `p` between 0 and 1, `log(p)` is a negative number, and it dives toward negative infinity as `p` gets close to 0. We stick a minus sign in front so the penalty `−log(...)` is positive and **blows up when the model is confident but wrong**.

This is the part worth understanding, not just memorizing. The model minimizes average log-loss, and each sample contributes one term. For a true Class 0 point (`y=0`) that term is `-log(1-p)`. **Flip its label to 1 and the term becomes `-log(p)`.** Now here's the damage: if the model correctly reads the features and predicts a low `p` (because the point really is Class 0), `-log(1-p)` was tiny, but `-log(p)` explodes toward infinity as `p→0`.

So each flipped point generates a **huge error signal** that gradient descent cannot ignore. To reduce that error, the optimizer shifts `w` and `b` to partly accommodate the liars, which **drags the decision boundary away from the position the true data would have chosen**. Random flips degrade accuracy overall; that's the whole mechanism.

## `flip_labels` (built across four blocks, paste in order)

The function is presented in pieces that stack into one function. Block 1: validate the percentage is in [0,1], compute how many labels to flip, and bail early if that's zero.

```
def flip_labels(y, poison_percentage):
    if not 0 <= poison_percentage <= 1:
        raise ValueError("poison_percentage must be between 0 and 1.")

    n_samples = len(y)
    n_to_flip = int(n_samples * poison_percentage)

    if n_to_flip == 0:
        print("Warning: Poison percentage is 0 or too low to flip any labels.")
        # Return unchanged labels and empty indices if no flips are needed
        return y.copy(), np.array([], dtype=int)
```

Block 2: pick which samples to poison. ==`choice(..., replace=False)` selects `n_to_flip` unique indices==, seeded for reproducibility so your run matches the course.

```
    # Use the defined SEED for the random number generator
    rng_instance = np.random.default_rng(SEED)
    # Select unique indices to flip
    flipped_indices = rng_instance.choice(n_samples, size=n_to_flip, replace=False)
```

Block 3: do the flip. **Copy first so the original labels aren't mutated**, then `np.where(orig == 0, 1, 0)` inverts each selected binary label (0→1, 1→0). For 0/1 labels `1 - label` would do the same; they use `np.where` for readability.

```
    # Create a copy to avoid modifying the original array
    y_poisoned = y.copy()

    # Get the original labels at the indices we are about to flip
    original_labels_at_flipped = y_poisoned[flipped_indices]

    # Apply the flip: if original was 0, set to 1; otherwise (if 1), set to 0
    y_poisoned[flipped_indices] = np.where(original_labels_at_flipped == 0, 1, 0)

    print(f"Flipping {n_to_flip} labels ({poison_percentage * 100:.1f}%).")
```

Block 4: return the poisoned labels plus the indices, so you can track exactly which reviews were hit (used by the plot next).

```
    return y_poisoned, flipped_indices
```

## `plot_poisoned_data`

Visualizes the poisoned set: unchanged points as normal dots, and **flipped points as red-outlined `X` markers colored by their new (wrong) label**, so you can literally see poisoned points sitting inside the wrong cluster. Full function:

```
def plot_poisoned_data(
    X,
    y_original,
    y_poisoned,
    flipped_indices,
    title="Poisoned Data Visualization",
    target_class_info=None,
):
    """
    Plots a 2D dataset, highlighting points whose labels were flipped.

    Parameters:
    - X (np.ndarray): Feature data (n_samples, 2).
    - y_original (np.ndarray): The original labels before flipping (used for context if needed, currently unused in logic but good practice).
    - y_poisoned (np.ndarray): Labels after flipping.
    - flipped_indices (np.ndarray): Indices of the samples that were flipped.
    - title (str): The title for the plot.
    - target_class_info (int, optional): The class label of the points that were targeted for flipping. Defaults to None.
    """
    plt.figure(figsize=(12, 7))

    # Identify non-flipped points
    mask_not_flipped = np.ones(len(y_poisoned), dtype=bool)
    mask_not_flipped[flipped_indices] = False

    # Plot non-flipped points (color by their poisoned label, which is same as original)
    plt.scatter(
        X[mask_not_flipped, 0],
        X[mask_not_flipped, 1],
        c=y_poisoned[mask_not_flipped],
        cmap=plt.cm.colors.ListedColormap([azure, nugget_yellow]),
        edgecolors=node_black,
        s=50,
        alpha=0.6,
        label="Unchanged Label",  # Keep this generic
    )

    # Determine the label for flipped points in the legend
    if target_class_info is not None:
        flipped_legend_label = f"Flipped (Orig Class {target_class_info})"
        # You could potentially use target_class_info to adjust facecolor if needed,
        # but current logic colors by the new label which is often clearer.
    else:
        flipped_legend_label = "Flipped Label"

    # Plot flipped points with a distinct marker and outline
    if len(flipped_indices) > 0:
        # Color flipped points according to their new (poisoned) label
        plt.scatter(
            X[flipped_indices, 0],
            X[flipped_indices, 1],
            c=y_poisoned[flipped_indices],  # Color by the new label
            cmap=plt.cm.colors.ListedColormap([azure, nugget_yellow]),
            edgecolors=malware_red,  # Highlight edge in red
            linewidths=1.5,
            marker="X",  # Use 'X' marker
            s=100,
            alpha=0.9,
            label=flipped_legend_label,  # Use the determined label
        )

    plt.title(title, fontsize=16, color=htb_green)
    plt.xlabel("Feature 1", fontsize=12)
    plt.ylabel("Feature 2", fontsize=12)

    # Create legend
    handles = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label="Class 0 Point (Azure)",
            markersize=10,
            markerfacecolor=azure,
            linestyle="None",
        ),
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label="Class 1 Point (Yellow)",
            markersize=10,
            markerfacecolor=nugget_yellow,
            linestyle="None",
        ),
        # Add the flipped legend entry using the label
        plt.Line2D(
            [0],
            [0],
            marker="X",
            color="w",
            label=flipped_legend_label,
            markersize=12,
            markeredgecolor=malware_red,
            markerfacecolor=hacker_grey,
            linestyle="None",
        ),
    ]
    plt.legend(handles=handles, title="Data Points")
    plt.grid(True, color=hacker_grey, linestyle="--", linewidth=0.5, alpha=0.3)
    plt.show()
```

Note one detail flagged in the docstring: `y_original` is passed in but not actually used in the logic. Harmless, just there for API symmetry.

Next section runs this: pick a poison percentage, flip, retrain, and compare the boundary and accuracy against your 0.9933 baseline. Paste it and I'll walk the evaluation, all code included.