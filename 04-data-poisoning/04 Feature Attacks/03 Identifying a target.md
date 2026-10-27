# Identifying A Target

Remember One-vs-Rest trained 3 little models: one for "Class 0 or not," one for "Class 1 or not," one for "Class 2 or not." For any point, each model spits out a **confidence number**: call them z0, z1, z2. The model's final answer is just whichever number is biggest. That is all the OvR "score" is, a confidence per class.

 **Comparing just two of them**

For the border between Class 0 and Class 1, only z0 and z1 matter. So we make one combined number:

`f01 = z0 - z1` = "how much more does this point look like Class 0 than Class 1."

Read the sign:

- **f01 **negative** -> z1 is bigger -> it looks more like Class 1** (correct, for our Class 1 points).
- f01 **positive** -> z0 is bigger -> it looks more like Class 0 (would be misclassified).
- f01 **exactly 0** -> a perfect tie, sitting right on the border.

### The size of the number = distance from the border

This is the part you're missing. The sign says which side; **the size says how far from the line.**

- f01 = **−5.0** -> deep inside Class 1 territory, miles from the border. Very safe, hard to flip.
- f01 = **−0.05** -> barely on the Class 1 side, basically touching the border. One tiny nudge and it crosses.

## The plan

Goal: make one specific Class 1 point get misclassified as Class 0 after the poison retraining. **So you pick a target that is already borderline: correctly labeled Class 1, but sitting right up against the Class 0 / Class 1 boundary. A point on the edge needs only a tiny boundary shift to tip over, which is exactly what your feature perturbations will cause.**

## The scoring idea (plain, no formulas)

To measure "how close to the 0-vs-1 boundary is this point," you use the boundary parameters you extracted last section (`w0/w1/b0/b1`). The score `f01` is just a signed number: **negative means the point is safely on the Class 1 side (correctly classified), and a value near zero means it is hugging the boundary.** You want the Class 1 point with the _least negative_ score, i.e. the one closest to the line. That is your most vulnerable target.

## Cell: select the target and plot it

A couple of code notes as you read it: `w_diff_01_base = w0_base - w1_base` builds the 0-vs-1 boundary direction; `X_class1_train @ w_diff_01_base + b_diff_01_base` uses `@` (Python's dot product) to compute that score for every Class 1 point at once; the `f01 < 0` filter keeps only correctly-classified points, and `argmax` among those grabs the one nearest the boundary. The relative-to-absolute index mapping is the same trick from the targeted-flip function.

```
print("\n--- Selecting Target Point ---")
# We use the baseline parameters w0_base, b0_base, w1_base, b1_base extracted earlier
# Calculate the difference vector and intercept for the 0-vs-1 boundary
w_diff_01_base = w0_base - w1_base
b_diff_01_base = b0_base - b1_base
print(f"Boundary vector (w0-w1): {w_diff_01_base}")
print(f"Intercept difference (b0-b1): {b_diff_01_base}")

# Identify indices of all Class 1 points in the original clean training set
class1_indices_train = np.where(y_train_3c == 1)[0]

if len(class1_indices_train) == 0:
    raise ValueError(
        "CRITICAL: No Class 1 points found in the training data. Cannot select target."
    )
else:
    print(f"Found {len(class1_indices_train)} Class 1 points in the training set.")

# Get the feature vectors for only the Class 1 points
X_class1_train = X_train_3c[class1_indices_train]

# Calculate the decision function f_01(x) = (w0-w1)^T x + (b0-b1) for these Class 1 points
# A negative value means the point is on the Class 1 side of the 0-vs-1 boundary
decision_values_01 = X_class1_train @ w_diff_01_base + b_diff_01_base

# Find indices within the subset of Class 1 points that are correctly classified (f_01 < 0)
class1_on_correct_side_indices_relative = np.where(decision_values_01 < 0)[0]

if len(class1_on_correct_side_indices_relative) == 0:
    # This case is unlikely if the baseline model has decent accuracy, but handle it.
    print(
        f"{malware_red}Warning:{white} No Class 1 points found on the expected side (f_01 < 0) of the 0-vs-1 baseline boundary."
    )
    print(
        "Selecting the Class 1 point with the minimum absolute decision value instead."
    )
    # Find index (relative to class1 subset) with the smallest absolute distance to boundary
    target_point_index_relative = np.argmin(np.abs(decision_values_01))
else:
    # Among the correctly classified points, find the one closest to the boundary
    # This corresponds to the maximum (least negative) decision value
    target_point_index_relative = class1_on_correct_side_indices_relative[
        np.argmax(decision_values_01[class1_on_correct_side_indices_relative])
    ]

# Map the relative index (within the class1 subset) back to the absolute index in the original X_train_3c array
target_point_index_absolute = class1_indices_train[target_point_index_relative]

# Retrieve the target point's features and true label
X_target = X_train_3c[target_point_index_absolute]
y_target = y_train_3c[
    target_point_index_absolute
]  # Should be 1 based on selection logic

# Sanity Check: Verify the chosen point's class and baseline prediction
target_baseline_pred = baseline_model_3c.predict(X_target.reshape(1, -1))[0]
target_decision_value = decision_values_01[target_point_index_relative]

print(f"\nSelected Target Point Index (absolute): {target_point_index_absolute}")
print(f"Target Point Features: {X_target}")
print(f"Target Point True Label (y_target): {y_target}")
print(f"Target Point Baseline Prediction: {target_baseline_pred}")
print(
    f"Target Point Baseline 0-vs-1 Decision Value (f_01): {target_decision_value:.4f}"
)

if y_target != 1:
    print(
        f"Error: Selected target point does not have label 1! Check logic."
    )
if target_baseline_pred != y_target:
    print(
        f"Warning: Baseline model actually misclassifies the chosen target point ({target_baseline_pred}). Attack might trivially succeed or have unexpected effects."
    )
if target_decision_value >= 0:
    print(
        f"Warning: Selected target point has f_01 >= 0 ({target_decision_value:.4f}), meaning it wasn't on the Class 1 side of the 0-vs-1 boundary. Check logic or baseline model."
    )

# Visualize the data highlighting the selected target point near the boundary
print("\n--- Visualizing Training Data with Target Point ---")
plot_data_multi(
    X_train_3c,
    y_train_3c,
    title="Training Data Highlighting the Target Point (Near Boundary)",
    highlight_indices=[target_point_index_absolute],
    highlight_markers=["P"],  # 'P' for Plus sign marker (Target)
    highlight_colors=[white],  # White edge color for visibility
    highlight_labels=[f"Target (Class {y_target}, Idx {target_point_index_absolute})"],
)
```
![Identyfing a target](/assets/11%20Identyfing%20a%20target-20260919164251354.png)

![Visualize the data highlighting the selected target point near the boundary](/assets/Pasted%20image%2020260919164254.png)

The chosen target is index `373`, with `f01 = -0.0493`, **basically sitting on the boundary** (the white plus marker in the plot, right where the yellow cluster meets the blue). It is correctly Class 1 for now, but only barely, which is precisely why it is the easiest one to flip.
