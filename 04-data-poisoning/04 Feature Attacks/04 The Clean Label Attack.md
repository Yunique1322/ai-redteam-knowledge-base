# The Clean Label Attack

Section 11 found the target: **idx 373, a _true_ Class 1 point sitting closest to the 0-vs-1 boundary.** This section builds the poison that will make the model misread it as Class 0.

The core move, which you already have the intuition for: **you cannot touch the target itself (it is a future inference input you do not control), so you attack its nearest Class 0 neighbours instead.** You nudge those neighbours just across the 0/1 line into Class 1 territory, but keep their label as 0. When the model retrains, those mislabeled decoys ("I'm a Class 0 point, but I'm sitting in Class 1 land") force it to drag the boundary outward to reclaim them, and the target gets swept onto the Class 0 side as collateral. **Small nudges keep the Class 0 label believable, and that plausibility is the whole "clean label" idea: features move, labels never do.**

Three cells: find the neighbours, compute the nudge, apply it.

## Cell 1: Identify the Class 0 neighbours to perturb

Fit scikit-learn's `NearestNeighbors` on **Class 0 points only** (those are the only points you are allowed to touch if the attack is to stay clean), then query it with the target's coordinates.

```python
print("\n--- Identifying Class 0 Neighbors to Perturb ---")
n_neighbors_to_perturb = 5 # Hyperparameter: How many neighbors to modify

# Find indices of all Class 0 points in the original training set
class0_indices_train = np.where(y_train_3c == 0)[0]

if len(class0_indices_train) == 0:
    raise ValueError("CRITICAL: No Class 0 points found. Cannot find neighbors to perturb.")
else:
    print(f"Found {len(class0_indices_train)} Class 0 points in the training set.")

# Get features of only Class 0 points
X_class0_train = X_train_3c[class0_indices_train]

# Sanity check to ensure we don't request more neighbors than available
if n_neighbors_to_perturb > len(X_class0_train):
    print(f"Warning: Requested {n_neighbors_to_perturb} neighbors, but only {len(X_class0_train)} Class 0 points available. Using all available.")
    n_neighbors_to_perturb = len(X_class0_train)

if n_neighbors_to_perturb == 0:
    raise ValueError("No Class 0 neighbors can be selected to perturb (n_neighbors_to_perturb=0). Cannot proceed.")

# Initialize and fit NearestNeighbors on the Class 0 data points
# We use the default Euclidean distance ('minkowski' with p=2)
nn_finder = NearestNeighbors(n_neighbors=n_neighbors_to_perturb, algorithm='auto')
nn_finder.fit(X_class0_train)

# Find the indices (relative to X_class0_train) and distances of the k nearest Class 0 neighbors to X_target
distances, indices_relative = nn_finder.kneighbors(X_target.reshape(1, -1))

# Map the relative indices found within X_class0_train back to the original indices in X_train_3c
neighbor_indices_absolute = class0_indices_train[indices_relative.flatten()]
# Get the original feature vectors of these neighbors (needed for perturbation)
X_neighbors = X_train_3c[neighbor_indices_absolute]

# Output the findings for verification
print(f"\nTarget Point Index: {target_point_index_absolute} (True Class {y_target})")
print(f"Identified {len(neighbor_indices_absolute)} closest Class 0 neighbors to perturb:")
print(f"  Indices in X_train_3c: {neighbor_indices_absolute}")
print(f"  Distances to target: {distances.flatten()}")

# Sanity check: Ensure the target itself wasn't accidentally included (e.g., if it was mislabeled or data is unusual)
if target_point_index_absolute in neighbor_indices_absolute:
     print(f"Error: The target point itself was selected as one of its own Class 0 neighbors. This indicates a potential issue in data or logic.")
```

The one gotcha to remember: **`kneighbors` returns indices into the Class-0-only array (`X_class0_train`), not into the full training set.** That is why you map them back with `class0_indices_train[indices_relative.flatten()]` to get the real training indices. `distances` come back sorted nearest first.

![The Clean Lable Attack](/assets/12%20The%20Clean%20Lable%20Attack-20260920021712475.png)

## Cell 2: Calculate the perturbation vector

The push is straight across the boundary. `push_direction = -w_diff_01_base` points from the Class 0 side toward the Class 1 side (the opposite of the boundary normal you built earlier). Normalise it to a unit vector so that **`epsilon_cross` alone controls the exact step length, set to 0.25 here.**

```python
print("\n--- Calculating Perturbation Vector ---")
# Use the boundary vector w_diff_01_base = w0_base - w1_base calculated earlier

# The direction to push Class 0 points into Class 1 region is opposite to the normal vector (w0-w1)
push_direction = -w_diff_01_base
norm_push_direction = np.linalg.norm(push_direction)

# Handle potential zero vector for the boundary normal
if norm_push_direction < 1e-9: # Use a small threshold for floating point comparison
    raise ValueError("Boundary vector norm (||w0-w1||) is close to zero. Cannot determine push direction reliably.")
else:
    # Normalize the direction vector to unit length
    unit_push_direction = push_direction / norm_push_direction
    print(f"Calculated unit push direction vector (normalized - (w0-w1)): {unit_push_direction}")

# Define perturbation magnitude (how far across the boundary to push)
epsilon_cross = 0.25
print(f"Perturbation magnitude (epsilon_cross): {epsilon_cross}")

# Calculate the final perturbation vector (direction * magnitude)
perturbation_vector = epsilon_cross * unit_push_direction
print(f"Final perturbation vector (delta): {perturbation_vector}")
```

Bigger `epsilon_cross` means a harder push but a less believable Class 0 point, so it is the **stealth-vs-strength dial of the attack.**

![Calculate the final perturbation vector (direction * magnitude)](/assets/Pasted%20image%2020260920022039.png)

Each training example here is just a row of two numbers plus a label, because this dataset has two features (both standardized, so think z-score-ish values, not raw units):

```
[ Feature_1 , Feature_2 ]   label
```

For each of the 5 nudged points, **exactly two numbers change and the label does not.** You add the perturbation vector `[0.16882471, -0.18438606]` to that row, which means:

- Feature 1 goes up by 0.16882471
- Feature 2 goes down by 0.18438606
- Label stays 0
## Cell 3: Apply the perturbation and build the poisoned set

Copy first, never poison in place. Add the same `perturbation_vector` to each neighbour, and **leave every label at 0. That single choice is what makes this "clean label": you modified features, not labels.** The `f01` print is your proof that each decoy actually crossed the line.

```python
print("\n--- Applying Perturbations to Create Poisoned Dataset ---")
# Create a safe copy of the original training data to modify
X_train_poisoned = X_train_3c.copy()
y_train_poisoned = (
    y_train_3c.copy()
)  # Labels are copied but not changed for perturbed points

perturbed_indices_list = []  # Keep track of which indices were actually modified

# Iterate through the identified neighbor indices and their original features
# neighbor_indices_absolute holds the indices in X_train_3c/y_train_3c
# X_neighbors holds the corresponding original feature vectors
for i, neighbor_idx in enumerate(neighbor_indices_absolute):
    X_neighbor_original = X_neighbors[i]  # Original feature vector of the i-th neighbor

    # Calculate the new position of the perturbed neighbor
    X_perturbed_neighbor = X_neighbor_original + perturbation_vector

    # Replace the original neighbor's features with the perturbed features in the copied dataset
    X_train_poisoned[neighbor_idx] = X_perturbed_neighbor
    # The label y_train_poisoned[neighbor_idx] remains 0 (Class 0)

    perturbed_indices_list.append(neighbor_idx)  # Record the index that was modified

    # Verify the effect of perturbation on the f_01 score
    f01_orig = X_neighbor_original @ w_diff_01_base + b_diff_01_base
    f01_pert = X_perturbed_neighbor @ w_diff_01_base + b_diff_01_base
    print(f"  Neighbor Index {neighbor_idx} (Label 0): Perturbed.")
    print(
        f"     Original f01 = {f01_orig:.4f} (>0 expected), Perturbed f01 = {f01_pert:.4f} (<0 expected)"
    )
    if f01_pert >= 0:
        print(
            f"     Warning: Perturbed point did not cross the baseline boundary (f01 >= 0). Epsilon might be too small."
        )

print(
    f"\nCreated poisoned training dataset by perturbing features of {len(perturbed_indices_list)} Class 0 points."
)
# Check the size to ensure it's unchanged
print(
    f"Poisoned training dataset size: {X_train_poisoned.shape[0]} samples (should match original {X_train_3c.shape[0]})."
)

# Convert list to numpy array for potential use later
perturbed_indices_arr = np.array(perturbed_indices_list)

# Final safety check: ensure target wasn't modified
if target_point_index_absolute in perturbed_indices_arr:
    print(
        f"CRITICAL Error: Target point index {target_point_index_absolute} was included in the perturbed indices! Check neighbor finding logic."
    )

# Visualize the poisoned dataset, highlighting target and perturbed points
print("\n--- Visualizing Poisoned Training Data ---")
plot_data_multi(
    X_train_poisoned,  # Use the poisoned features
    y_train_poisoned,  # Use the corresponding labels (perturbed points still have label 0)
    title="Poisoned Training Data (Features Perturbed)",
    highlight_indices=[target_point_index_absolute] + perturbed_indices_list,
    highlight_markers=["P"]
    + ["o"]
    * len(perturbed_indices_list),  # 'P' for Target, 'o' for Perturbed neighbors
    highlight_colors=[white]
    + [vivid_purple]
    * len(perturbed_indices_list),  # White edge Target, Purple edge Perturbed
    highlight_labels=[f"Target (Idx {target_point_index_absolute}, Class {y_target})"]
    + [f"Perturbed (Idx {idx}, Label 0)" for idx in perturbed_indices_list],
)
```

Reading the `f01` print: **original `f01` > 0 means the neighbour started on the Class 0 side, perturbed `f01` < 0 means it now sits on the Class 1 side.** If any neighbour stays `>= 0`, `epsilon_cross` was too small to push that particular point across, which is what the warning line catches. Note this cell also uses `b_diff_01_base` (which is `b0_base - b1_base`, the bias half of the boundary you built alongside `w_diff_01_base`).
![The Clean Lable Attack](/assets/12%20The%20Clean%20Lable%20Attack-20260920022131588.png)
On the plot: the white `'+'` is the target (still a true Class 1 point, untouched), and the purple `'o'` rings are your five decoys, now sitting inside the yellow Class 1 cloud **while still being colored as Class 0. That blue-in-yellow contradiction is exactly what forces the boundary to move on the next retrain.**

## Takeaways

- ==Clean label = only features perturbed, labels stay correct-looking.== That is precisely what lets it slip past a defense that only audits labels (unlike the label-flipping attack from earlier, which a label audit would catch).
- **You hit the target indirectly, through its same-class neighbours, because the target is test-time input you can't edit.**
- The boundary has not actually moved yet. This section only _plants and visualises_ the poison.

**The target has not flipped at this point.** That happens in Section 13 (Evaluating the Clean Label Attack), where you retrain the OvR model on `X_train_poisoned` / `y_train_poisoned` and confirm idx 373 now predicts Class 0, while overall accuracy barely moves (that tiny accuracy hit is the stealth of the attack).

