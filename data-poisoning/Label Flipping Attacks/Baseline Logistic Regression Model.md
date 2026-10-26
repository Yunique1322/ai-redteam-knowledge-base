# Baseline Logistic Regression Model

## Why a baseline first

Before poisoning anything you train a clean model as your **control**. **You can't prove an attack worked without a "healthy" number to compare against.** This baseline trains on the clean `X_train`/`y_train` and is scored on the untouched test set, giving you the model's normal, non-adversarial accuracy (0.9933 here, near-perfect because the two clusters barely overlap).

## Logistic regression in plain terms

Despite the name, **logistic regression is a classifier, not a regressor.** The intuition behind the math the course shows:

- It computes a weighted sum of the features plus a bias: `z = w1·x1 + w2·x2 + b`. That's just the equation of a line.
- `z` is the **log-odds** (logit) of the sample being Class 1. Big positive `z` means "very likely positive," big negative means "very likely negative."
- The **sigmoid** `σ(z) = 1/(1+e^-z)` squashes `z` into a probability `p` between 0 and 1.
- Threshold at 0.5: `p ≥ 0.5` → Class 1, else Class 0.
- **Training** = find the `w` and `b` that make predicted probabilities match the true labels, by minimizing **binary cross-entropy (log-loss)** with **gradient descent** (an optimizer that nudges the weights downhill on the loss, step by step).
- The **decision boundary** is where the model is exactly 50/50, i.e. `z = 0`. In 2D that's a straight line, and everything you'll do to this model shows up as that line moving.

The one connection to keep for the attack: **the boundary is fit to the labels. Flip labels and you drag the line, which is exactly what the next section does.**

## The code

Trains the model, prints test accuracy, then defines `plot_decision_boundary` and renders it. The plotting helper works by laying a fine grid (mesh) over the feature space, asking the model to predict a class at every grid point, and shading the regions with `contourf`, that shaded split is the visual boundary. Full block:

```
# Initialize and train the Logistic Regression model
baseline_model = LogisticRegression(random_state=SEED)
baseline_model.fit(X_train, y_train)

# Predict on the test set
y_pred_baseline = baseline_model.predict(X_test)

# Calculate baseline accuracy
baseline_accuracy = accuracy_score(y_test, y_pred_baseline)
print(f"Baseline Model Accuracy: {baseline_accuracy:.4f}")


# Prepare to plot the decision boundary
def plot_decision_boundary(model, X, y, title="Decision Boundary"):
    """
    Plots the decision boundary of a trained classifier on a 2D dataset.

    Parameters:
    - model: The trained classifier object (must have a .predict method).
    - X (np.ndarray): Feature data (n_samples, 2).
    - y (np.ndarray): Labels (n_samples,).
    - title (str): The title for the plot.
    """
    h = 0.02  # Step size in the mesh
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    # Predict the class for each point in the mesh
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.figure(figsize=(12, 6))
    # Plot the decision boundary contour
    plt.contourf(
        xx, yy, Z, cmap=plt.cm.colors.ListedColormap([azure, nugget_yellow]), alpha=0.3
    )

    # Plot the data points
    scatter = plt.scatter(
        X[:, 0],
        X[:, 1],
        c=y,
        cmap=plt.cm.colors.ListedColormap([azure, nugget_yellow]),
        edgecolors=node_black,
        s=50,
        alpha=0.8,
    )

    plt.title(title, fontsize=16, color=htb_green)
    plt.xlabel("Feature 1", fontsize=12)
    plt.ylabel("Feature 2", fontsize=12)

    # Create a legend manually
    handles = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label="Negative Sentiment (Class 0)",
            markersize=10,
            markerfacecolor=azure,
        ),
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label="Positive Sentiment (Class 1)",
            markersize=10,
            markerfacecolor=nugget_yellow,
        ),
    ]
    plt.legend(handles=handles, title="Classes")
    plt.grid(True, color=hacker_grey, linestyle="--", linewidth=0.5, alpha=0.3)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.show()


# Plot the decision boundary for the baseline model
plot_decision_boundary(
    baseline_model,
    X_train,
    y_train,
    title=f"Baseline Model Decision Boundary\nAccuracy: {baseline_accuracy:.4f}",
)
```

![Plot the decision boundary for the baseline model](/assets/Pasted%20image%2020260918170056.png)
![Plot the decision boundary for the baseline model](/assets/Pasted%20image%2020260918185945.png)
## What the plot tells you

A clean diagonal line splits blue (Class 0) from yellow (Class 1) with 0.9933 accuracy on unseen data. **This is your reference frame: remember both the line's angle and the ~0.99 number, because the attack is judged by how far the line tilts and how far the accuracy drops from here.**

