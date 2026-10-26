
Headline: accuracy fell from `0.9933` to `0.8100`, but the number that matters is _how_ it fell. **A targeted attack does asymmetric damage: it wrecks the target class and leaves the other one pristine.** That signature is exactly what these three cells reveal.

## Cell 1: accuracy, classification report, confusion matrix

```
# Predict on the original, clean test set
y_pred_targeted = targeted_poisoned_model.predict(X_test)

# Calculate accuracy on the clean test set
targeted_accuracy = accuracy_score(y_test, y_pred_targeted)
print(f"\n--- Evaluating Targeted Poisoned Model ---")
print(f"Accuracy on clean test set: {targeted_accuracy:.4f}")
print(f"Baseline accuracy was: {baseline_accuracy:.4f}")

# Display classification report
print("\nClassification Report on Clean Test Set:")
print(
    classification_report(y_test, y_pred_targeted, target_names=["Class 0", "Class 1"])
)

# Plot confusion matrix
cm_targeted = confusion_matrix(y_test, y_pred_targeted)
plt.figure(figsize=(6, 5))
sns.heatmap(
    cm_targeted,
    annot=True,
    fmt="d",
    cmap="binary",
    xticklabels=["Predicted 0", "Predicted 1"],
    yticklabels=["Actual 0", "Actual 1"],
    cbar=False,
)
plt.xlabel("Predicted Label", color=white)
plt.ylabel("True Label", color=white)
plt.title("Confusion Matrix (Targeted Poisoned Model)", fontsize=14, color=htb_green)
plt.xticks(color=hacker_grey)
plt.yticks(color=hacker_grey)
plt.show()
```


![Evaluating the Targeted Label Attacks](/assets/8%20Evaluating%20the%20Targeted%20Label%20Attacks-20260918232048341.png)

![Evaluating the Targeted Label Attacks](/assets/8%20Evaluating%20the%20Targeted%20Label%20Attacks-20260918232037109.png)

Three terms you need to read this, in plain English:

- **Recall (Class 1) = 0.61:** of all the genuinely positive reviews, the model caught only 61%. It **misses 39% of real positives**. This is the attack landing.
- **Precision (Class 1) = 1.00:** when it _does_ say "positive," it's always right. So the attack didn't make it invent positives, it made it **miss them** (push them to negative).
- **Confusion matrix:** rows are the truth, columns are the guess, so the diagonal is correct and off-diagonal is mistakes. Your matrix reads: Actual 0 → 153 correct, 0 wrong (Class 0 is perfect). Actual 1 → 57 pushed to "Predicted 0" (the intended damage), 90 correct.

**That 57-vs-0 split is the whole story: Class 0 is untouched, Class 1 is bleeding into it. That is targeted poisoning, and it's why accuracy alone (0.81) undersells what happened, the confusion matrix and Class 1 recall are where you see the surgical hit.**

## Cell 2: boundary comparison

```
# Plot the comparison of decision boundaries
plt.figure(figsize=(12, 8))

# Plot Baseline Decision Boundary (Solid Green)
Z_baseline = baseline_model.predict(mesh_points).reshape(xx.shape)
plt.contour(
    xx,
    yy,
    Z_baseline,
    levels=[0.5],
    colors=[htb_green],
    linestyles=["solid"],
    linewidths=[2.5],
)

# Plot Targeted Poisoned Decision Boundary (Dashed Red)
Z_targeted = targeted_poisoned_model.predict(mesh_points).reshape(xx.shape)
plt.contour(
    xx,
    yy,
    Z_targeted,
    levels=[0.5],
    colors=[malware_red],
    linestyles=["dashed"],
    linewidths=[2.5],
)

plt.title(
    "Comparison: Baseline vs. Targeted Poisoned Decision Boundary",
    fontsize=16,
    color=htb_green,
)
plt.xlabel("Feature 1", fontsize=12)
plt.ylabel("Feature 2", fontsize=12)

# Create legend combining data points and boundaries
handles = [
    plt.Line2D(
        [0],
        [0],
        color=htb_green,
        lw=2.5,
        linestyle="solid",
        label=f"Baseline Boundary (Acc: {baseline_accuracy:.3f})",
    ),
    plt.Line2D(
        [0],
        [0],
        color=malware_red,
        lw=2.5,
        linestyle="dashed",
        label=f"Targeted Poisoned Boundary (Acc: {targeted_accuracy:.3f})",
    ),
]

plt.legend(handles=handles, title="Boundaries & Data Points")
plt.grid(True, color=hacker_grey, linestyle="--", linewidth=0.5, alpha=0.3)
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.show()
```
![Create legend combining data points and boundaries](/assets/Pasted%20image%2020260918232355.png)
The red dashed boundary has slid parallel-ish but pushed toward the Class 1 side. **Everything between the green and red lines is territory that used to be called Class 1 and is now called Class 0**, which is precisely where the misclassified positives live.

## Cell 3: does it hold on brand-new data?

The real test: generate a fresh 500-point batch (bigger spread, `cluster_std=1.50`, and a different seed `SEED + 1337` so it's genuinely new), predict with the poisoned model, and count how many true Class 1 points it misclassifies.

```
# Define parameters for unseen data generation
n_unseen_samples = 500
unseen_seed = SEED + 1337

# Generate unseen data
X_unseen, y_unseen = make_blobs(
    n_samples=n_unseen_samples,
    centers=centers,
    n_features=2,
    cluster_std=1.50,
    random_state=unseen_seed,
)

# Predict labels for the unseen data using the targeted poisoned model
y_pred_unseen_poisoned = targeted_poisoned_model.predict(X_unseen)

# Calculate misclassification statistics
true_target_class_indices = np.where(y_unseen == target_class_to_flip)[0]
misclassified_target_mask = (y_unseen == target_class_to_flip) & (
    y_pred_unseen_poisoned != target_class_to_flip
)
misclassified_target_indices = np.where(misclassified_target_mask)[0]
n_true_target = len(true_target_class_indices)
n_misclassified_target = len(misclassified_target_indices)

plt.figure(figsize=(12, 8))

# Plot all unseen points, colored by the poisoned model's prediction
plt.scatter(
    X_unseen[:, 0],
    X_unseen[:, 1],
    c=y_pred_unseen_poisoned,
    cmap=plt.cm.colors.ListedColormap([azure, nugget_yellow]),
    edgecolors=node_black,
    s=50,
    alpha=0.7,
    label="Predicted Label",
)

# Highlight the misclassified target points
if n_misclassified_target > 0:
    plt.scatter(
        X_unseen[misclassified_target_indices, 0],
        X_unseen[misclassified_target_indices, 1],
        facecolors="none",
        edgecolors=malware_red,
        linewidths=1.5,
        marker="X",
        s=120,
        label=f"Misclassified (True Class {target_class_to_flip})",
    )

# Calculate and plot decision boundary
Z_targeted_boundary = targeted_poisoned_model.predict(mesh_points).reshape(xx.shape)
plt.contour(
    xx,
    yy,
    Z_targeted_boundary,
    levels=[0.5],
    colors=[malware_red],
    linestyles=["dashed"],
    linewidths=[2.5],
)

# Set title
plt.title(
    f"Poisoned Model Predictions & Boundary on Unseen Data\n({n_misclassified_target} of {n_true_target} Class {target_class_to_flip} samples misclassified)",
    fontsize=16,
    color=htb_green,
)
plt.xlabel("Feature 1", fontsize=12)
plt.ylabel("Feature 2", fontsize=12)

# Create legend
handles = [
    plt.Line2D(
        [0],
        [0],
        marker="o",
        color="w",
        label=f"Predicted as Class 0 (Azure)",
        markersize=10,
        markerfacecolor=azure,
        linestyle="None",
    ),
    plt.Line2D(
        [0],
        [0],
        marker="o",
        color="w",
        label=f"Predicted as Class 1 (Yellow)",
        markersize=10,
        markerfacecolor=nugget_yellow,
        linestyle="None",
    ),
    *(
        [
            plt.Line2D(
                [0],
                [0],
                marker="X",
                color="w",
                label=f"Misclassified (True Class {target_class_to_flip})",
                markersize=12,
                markeredgecolor=malware_red,
                markerfacecolor="none",
                linestyle="None",
            )
        ]
        if n_misclassified_target > 0
        else []
    ),
    plt.Line2D(
        [0],
        [0],
        color=malware_red,
        lw=2.5,
        linestyle="dashed",
        label="Decision Boundary (Targeted Model)",
    ),
]
plt.legend(handles=handles, title="Predictions, Errors & Boundary")
plt.grid(True, color=hacker_grey, linestyle="--", linewidth=0.5, alpha=0.3)

# Set plot limits
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())

# Apply theme to background
fig = plt.gcf()
fig.set_facecolor(node_black)
ax = plt.gca()
ax.set_facecolor(node_black)

plt.show()
```
![Apply theme to background](/assets/Pasted%20image%2020260918232432.png)
Two small code bits worth noting: `misclassified_target_mask` is a boolean "and" of "truly Class 1" AND "predicted not Class 1", i.e. exactly the intended victims; and the `*( [...] if ... else [] )` in the legend is just a Python trick to include the "Misclassified" legend entry only when there's at least one. Result: 92 of 250 unseen Class 1 points land on the wrong side of the shifted boundary (red X's), all clustered right along the Class 0 side of the red line. **The attack generalizes to data the model never saw, which is what makes it a real attack and not a training-set fluke.**

## Takeaway

Random flipping degrades both classes and shows up in overall accuracy. **Targeted flipping degrades one chosen class, so overall accuracy barely tells the story, you read it in the confusion matrix and the target class's recall.** In this run: Class 0 recall 1.00 (untouched), Class 1 recall 0.61 (hit), 57 test positives and 92 unseen positives pushed to negative.



# Labs

### Download the targeted_label_student.zip file attached to this question, and extract the notebook template and dataset file within it. Using the techniques that have been demonstrated in this section, implement a targeted label flipping attack in the provided targeted_class_label_flip method stub to poison at least 50% of the class 0 labels as class 1 in thedataset, train a model using the provided code, and submit the trained model to the docker instance using the last cell in the notebook. Submit the flag you receive for a valid attack as the answer to this question.

![Download the targeted label student.zip file attached to this question, and extract the notebook template and dataset file within it. Using the techniques that have been demonstrated in this section, implement a targeted label flipping attack in the provided targeted class label flip method stub to poison at least 50% of the class 0 labels as class 1 in thedataset, train a model using the provided code, and submit the trained model to the docker instance using the last cell in the notebook. Submit the flag you receive for a valid attack as the answer to this question.](/assets/Pasted%20image%2020260918235057.png)