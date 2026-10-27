
Section 12 planted the poison but nothing had actually flipped yet. This section retrains the model on the poisoned data and answers the two questions that decide whether the attack worked: **did the target flip, and did anyone notice.**

## Cell 1: Retrain on the poisoned data

```python
print("\n--- Training Poisoned Model (Clean Label Attack) ---")

# Initialize a new base estimator for the poisoned model (same settings as baseline)
poisoned_base_estimator = LogisticRegression(
    random_state=SEED, C=1.0, solver="liblinear"
)
# Initialize the OneVsRestClassifier wrapper
poisoned_model_cl = OneVsRestClassifier(poisoned_base_estimator)

# Train the model on the POISONED training data
poisoned_model_cl.fit(X_train_poisoned, y_train_poisoned)

print("Poisoned model (Clean Label) trained successfully.")
```

**Same architecture, same hyperparameters as the baseline. Only the data differs.** That is deliberate: it makes the comparison fair, so any change in behaviour is caused purely by your 5 poisoned rows and not by a different model setup. `random_state=SEED` keeps it reproducible.

## Cell 2: Did the target flip, and did overall accuracy hold

Two separate checks here. First the single target point, then the model's health on the clean test set.

```python
print("\n--- Evaluating Poisoned Model Performance ---")

# Check the prediction for the specific target point
X_target_reshaped = X_target.reshape(1, -1)  # Reshape for single prediction
target_pred_poisoned = poisoned_model_cl.predict(X_target_reshaped)[0]

print(f"Target Point Evaluation:")
print(f"  Original True Label (y_target): {y_target}")
print(f"  Baseline Model Prediction:      {target_baseline_pred}")
print(f"  Poisoned Model Prediction:      {target_pred_poisoned}")

attack_successful = (target_pred_poisoned != y_target) and (
    target_pred_poisoned == 0
)  # Specifically check if flipped to Class 0

if attack_successful:
    print(
        f"  Success: The poisoned model misclassified the target point as Class {target_pred_poisoned}."
    )
else:
    if target_pred_poisoned == y_target:
        print(
            f"  Failure: The poisoned model still correctly classified the target point as Class {target_pred_poisoned}."
        )
    else:
        print(
            f"  Partial/Unexpected: The poisoned model misclassified the target point, but as Class {target_pred_poisoned}, not the intended Class 0."
        )

# Evaluate overall accuracy on the clean test set
y_pred_poisoned_test = poisoned_model_cl.predict(X_test_3c)
poisoned_accuracy_test = accuracy_score(y_test_3c, y_pred_poisoned_test)

print(f"\nOverall Performance on Clean Test Set:")
print(f"  Baseline Accuracy: {baseline_accuracy_3c:.4f}")
print(f"  Poisoned Accuracy: {poisoned_accuracy_test:.4f}")
print(f"  Accuracy Drop:     {baseline_accuracy_3c - poisoned_accuracy_test:.4f}")

# Display classification report for more detail
print("\nClassification Report (Poisoned Model on Clean Test Data):")
print(
    classification_report(
        y_test_3c, y_pred_poisoned_test, target_names=["Class 0", "Class 1", "Class 2"]
    )
)
```

One detail worth noting: `attack_successful` is strict. It requires both that the target is misclassified AND that it flipped **specifically to Class 0** (the class you were aiming for), not just to any wrong class. A flip to Class 2 would count as "partial/unexpected," not success.
![Display classification report for more detail](/assets/Pasted%20image%2020260920022824.png)

How to read it: the target went from correctly predicted 1 (baseline) to predicted 0 (poisoned), so it flipped to exactly the class you wanted. And **overall accuracy barely moved, 0.9600 down to 0.9578, a 0.0022 drop.** That near-zero drop is the entire point of a targeted clean-label attack: on a normal test set the poisoned model looks just as healthy as the original, so a defender watching aggregate accuracy sees nothing wrong. The only visible ripple is Class 1 recall dipping to 0.93, a bit of collateral from the warped boundary, and even that is tiny.

## Cell 3: Visualize the boundary shift

```python
print("\n--- Visualizing Poisoned Model Decision Boundaries vs. Baseline ---")

# Predict classes on the meshgrid using the POISONED model
Z_poisoned_cl = poisoned_model_cl.predict(mesh_points_3c)
Z_poisoned_cl = Z_poisoned_cl.reshape(xx_3c.shape)

# Plot the decision boundary comparison
plot_decision_boundary_multi(
    X_train_poisoned,  # Show points from the poisoned training set
    y_train_poisoned,  # Use their labels (perturbed are still 0)
    Z_poisoned_cl,  # Use the poisoned model's mesh predictions for background
    xx_3c,
    yy_3c,
    title=f"Poisoned vs. Baseline Decision Boundaries\nTarget Misclassified: {attack_successful} | Poisoned Acc: {poisoned_accuracy_test:.4f}",
    highlight_indices=[target_point_index_absolute] + perturbed_indices_list,
    highlight_markers=["P"] + ["o"] * len(perturbed_indices_list),
    highlight_colors=[white] + [vivid_purple] * len(perturbed_indices_list),
    highlight_labels=[f"Target (Pred: {target_pred_poisoned})"]
    + [f"Perturbed (Idx {idx})" for idx in perturbed_indices_list],
)
```
![Plot the decision boundary comparison](/assets/Pasted%20image%2020260920022937.png)
This reruns the same meshgrid through the poisoned model and repaints the regions. On the plot, **the blue Class 0 region has been pulled down into what used to be Class 1 territory** to keep the relabeled decoys on its side, and that downward dip now covers the target. The target (white '+', truly Class 1) sits inside the blue region, so it reads as Class 0. Notice it sits right at the new frontier, not deep inside blue, which fits how you chose it: Section 11 picked the target closest to the boundary precisely because it was the cheapest to flip.

## Takeaways

- **The signature of a successful targeted clean-label attack is exactly this pair: the chosen target flips, while global accuracy stays essentially unchanged** (a 0.22% drop here). The tiny accuracy cost is what lets it hide from accuracy-monitoring defenses.
    
- **No labels were ever modified, so a label-audit or label-consistency check finds nothing.** This is why it is far harder to detect than label flipping, where the flipped labels are the smoking gun.
    
- The trade-off to remember, which is HTB's closing point: clean-label is much stealthier but **much harder to pull off, because computing the perturbation direction needs model internals (the boundary weights `w_diff_01_base`).** Label flipping needs zero model knowledge but is easy to catch. Stealth versus complexity is the axis these two attacks sit on.

# LABS

### Download the feature_student.zip file attached to this question, and extract the notebook template and dataset file within it. Using the techniques that have been demonstrated in this section, implement a clean label attack to misclassify Class 2 Index 334 as Class 1, train a model using the provided code, and submit the trained model to the docker instance using the last cell in the notebook. Submit the flag you receive for a valid attack as the answer to this question.

```
def perform_clean_label_attack(
    X_train_orig,
    y_train_orig,
    target_idx,
    target_class,
    perturb_class,
    n_neighbors,
    epsilon_cross,
    seed,
):
    # Implement the clean label attack logic here
    if target_class == perturb_class:
        raise ValueError("Target class and perturbing class cannot be the same.")

    # Create Copies to avoid modifying original data
    X_train_poisoned = X_train_orig.copy()
    y_train_poisoned = y_train_orig.copy() # Labels remain unchanged.

    # Train a temporary baseline model (OvR Logistic Regression)
    # We need this to find the decision boundary between target_class and perturb_class
    try:
        temp_base_estimator = LogisticRegression(
            random_state=seed, C=1.0, solver="liblinear"
        )
        temp_model = OneVsRestClassifier(temp_base_estimator)
        temp_model.fit(X_train_orig, y_train_orig)
        print("Temporary baseline model trained.")

        if not (
            hasattr(temp_model, "estimators_")
            and len(temp_model.estimators_) > max(target_class, perturb_class)
        ):
            raise RuntimeError(
                "Temporary model did not produce expected number of estimators."
            )

        w_target = temp_model.estimators_[target_class].coef_[0]
        b_target = temp_model.estimators_[target_class].intercept_[0]
        w_perturb = temp_model.estimators_[perturb_class].coef_[0]
        b_perturb = temp_model.estimators_[perturb_class].intercept_[0]
        print(
            f"Extracted weights/intercepts for Class {target_class} and Class {perturb_class}."
        )

    except Exception as e:
        print(f"Error: Failed to train or extract params from temporary model: {e}")
        raise RuntimeError("Failed to initialize temporary model for attack.") from e

    # Identify neighbors of perturb_class closest to the target point
    X_target_point = X_train_orig[target_idx]
    perturb_class_indices_train = np.where(y_train_orig == perturb_class)[0]

    if len(perturb_class_indices_train) == 0:
        raise ValueError(
            f"No points found for perturb_class ({perturb_class}). Cannot find neighbors."
        )
    if n_neighbors > len(perturb_class_indices_train):
        print(
            f"Warning: Requested {n_neighbors} neighbors, but only {len(perturb_class_indices_train)} points of Class {perturb_class} exist. Using all available."
        )
        n_neighbors = len(perturb_class_indices_train)
    if n_neighbors == 0:
        raise ValueError("n_neighbors is zero, cannot proceed with perturbation.")

    X_perturb_class_train = X_train_orig[perturb_class_indices_train]
    print(f"Finding {n_neighbors} nearest neighbors from Class {perturb_class}...")
    nn_finder = NearestNeighbors(n_neighbors=n_neighbors, algorithm="auto")
    nn_finder.fit(X_perturb_class_train)
    distances, indices_relative = nn_finder.kneighbors(X_target_point.reshape(1, -1))

    # Map relative indices back to original X_train indices
    neighbor_indices_absolute = perturb_class_indices_train[indices_relative.flatten()]
    X_neighbors_original = X_train_orig[neighbor_indices_absolute]
    print(f"Found neighbors at indices: {neighbor_indices_absolute}")

    # Calculate the perturbation vector
    #    Push direction is opposite to the normal vector of the boundary (w_target - w_perturb)
    w_diff_boundary = w_target - w_perturb
    b_diff_boundary = (
        b_target - b_perturb
    )  # Not needed for direction, but good for checks

    push_direction = w_diff_boundary
    norm_push_direction = np.linalg.norm(push_direction)

    if norm_push_direction < 1e-9:
        raise ValueError(
            "Boundary vector norm is close to zero. Cannot determine reliable push direction."
        )

    unit_push_direction = push_direction / norm_push_direction
    perturbation_vector = epsilon_cross * unit_push_direction
    print(f"Calculated perturbation vector (delta): {perturbation_vector}")

    # Apply perturbations to the neighbors' features in the copied dataset
    perturbed_indices_list = []
    print("Applying perturbations...")
    for i, neighbor_idx in enumerate(neighbor_indices_absolute):
        X_neighbor_orig = X_neighbors_original[i]
        X_perturbed_neighbor = X_neighbor_orig + perturbation_vector

        # Update the feature vector in the poisoned dataset
        X_train_poisoned[neighbor_idx] = X_perturbed_neighbor
        # DO NOT change y_train_poisoned[neighbor_idx]

        perturbed_indices_list.append(neighbor_idx)

        # Check if perturbation crossed the temporary baseline boundary
        f_boundary_orig = X_neighbor_orig @ w_diff_boundary + b_diff_boundary
        f_boundary_pert = X_perturbed_neighbor @ w_diff_boundary + b_diff_boundary
        # Expect f_boundary_orig < 0 (perturb class side), f_boundary_pert > 0 (target class side)
        print(
            f"  Neighbor {neighbor_idx}: Orig f={f_boundary_orig:.4f}, Perturbed f={f_boundary_pert:.4f}"
        )
        if f_boundary_pert <= 0:
            print(
                f"     Warning: Perturbed point {neighbor_idx} might not have crossed the boundary (f<=0)."
            )

    print(f"Applied perturbations to {len(perturbed_indices_list)} neighbors.")
    perturbed_indices = np.array(perturbed_indices_list)  # Return as numpy array

    # Final check: ensure target point wasn't accidentally perturbed
    if target_idx in perturbed_indices:
        print(
            f"CRITICAL Error: Target index {target_idx} was selected as a neighbor and perturbed! Check logic."
        )
        # Depending on desired strictness, could raise an error here

    print("--- Clean Label Attack Implementation Finished")
    
    return X_train_poisoned, y_train_poisoned, perturbed_indices
```

![Evaluating the Clean Label Attack](/assets/13%20Evaluating%20the%20Clean%20Label%20Attack-20260920024804409.png)

![Evaluating the Clean Label Attack](/assets/13%20Evaluating%20the%20Clean%20Label%20Attack-20260920024821321.png)