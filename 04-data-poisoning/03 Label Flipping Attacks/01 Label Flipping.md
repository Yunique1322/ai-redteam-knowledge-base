# Label Flipping

## The concept

**You change the labels (the answers) and leave the features (the inputs) untouched.** `cat` becomes `dog`; `spam` becomes `not spam`. The usual goal is to **degrade overall performance** (an availability attack): the attacker doesn't care which inputs break, only that the model becomes unreliable. That contrasts with the targeted variant later.

Pipeline location: the **Storage** stage (edit a label column in a stored CSV on S3, or records in PostgreSQL, after ingestion validation already passed) or a compromised **Processing** script. Maps to OWASP **LLM03: Training Data Poisoning** (2023 numbering this course uses). The sentiment scenario is the "why": poison the labels, the company misreads customer sentiment and makes a bad call (pulls a good product, fixes features nobody disliked).

## ML grounding (the terms you'll reuse all module)

- **Features vs label:** features = the inputs (`X`, here a 2D point), label = the class (`y`, 0 or 1). **Label flipping edits `y`, never `X`.**
- **Classifier / decision boundary:** the model learns a boundary between classes. Logistic regression draws essentially a straight line.
- **Train/test split:** train on part, score on a held-out part it never saw, so accuracy reflects generalization. Here 70/30.
- **Seed:** fixes randomness for reproducibility (`SEED = 1337`).
- **Metrics** imported now (accuracy, classification_report, confusion_matrix) are how you'll measure the damage next section.

## Setup: imports, plot styling, seed

Standard sklearn + matplotlib imports, HTB's dark-theme palette and `rcParams` (purely cosmetic, but here in full so your notebook renders identically), and the reproducibility seed.

```
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns

htb_green = "#9fef00"
node_black = "#141d2b"
hacker_grey = "#a4b1cd"
white = "#ffffff"
azure = "#0086ff"
nugget_yellow = "#ffaf00"
malware_red = "#ff3e3e"
vivid_purple = "#9f00ff"
aquamarine = "#2ee7b6"

# Configure plot styles
plt.style.use("seaborn-v0_8-darkgrid")
plt.rcParams.update(
    {
        "figure.facecolor": node_black,
        "axes.facecolor": node_black,
        "axes.edgecolor": hacker_grey,
        "axes.labelcolor": white,
        "text.color": white,
        "xtick.color": hacker_grey,
        "ytick.color": hacker_grey,
        "grid.color": hacker_grey,
        "grid.alpha": 0.1,
        "legend.facecolor": node_black,
        "legend.edgecolor": hacker_grey,
        "legend.frameon": True,
        "legend.framealpha": 1.0,
        "legend.labelcolor": white,
    }
)

# Seed for reproducibility
SEED = 1337
np.random.seed(SEED)

print("Setup complete. Libraries imported and styles configured.")
```

## The dataset

**`make_blobs` fabricates two Gaussian clusters**, centered at `(0, 5)` and `(5, 0)`, 1000 points, 2 features each. `cluster_std=1.25` sets the spread (bigger = more overlap = harder). It returns `X` (points) and `y` (labels 0/1). `train_test_split` then holds out 30% as an unseen test set, giving 700 train / 300 test. The course treats the two features as stand-ins for numbers you'd really get from text via TF-IDF or embeddings.

```
# Generate synthetic data
n_samples = 1000
centers = [(0, 5), (5, 0)]  # Define centers for two distinct blobs
X, y = make_blobs(
    n_samples=n_samples,
    centers=centers,
    n_features=2,
    cluster_std=1.25,
    random_state=SEED,
)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=SEED
)

print(f"Generated {n_samples} samples.")
print(f"Training set size: {X_train.shape[0]} samples.")
print(f"Testing set size: {X_test.shape[0]} samples.")
print(f"Number of features: {X_train.shape[1]}")
print(f"Classes: {np.unique(y)}")
```
![Label Flipping](/assets/3%20Label%20Flipping-20260918185721796.png)

## Plotting the clean data

Cosmetic helper that scatters the points colored by class (blue = Class 0 negative, yellow = Class 1 positive) and builds a manual legend. The output is two clean, well-separated clusters. **Fix this "before" picture in mind, it's your baseline, and after flipping you'll watch the boundary get dragged off this clean split.**

```
def plot_data(X, y, title="Dataset Visualization"):
    """
    Plots the 2D dataset with class-specific colors.

    Parameters:
    - X (np.ndarray): Feature data (n_samples, 2).
    - y (np.ndarray): Labels (n_samples,).
    - title (str): The title for the plot.
    """
    plt.figure(figsize=(12, 6))
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
    plt.xlabel("Sentiment Feature 1", fontsize=12)
    plt.ylabel("Sentiment Feature 2", fontsize=12)
    # Create a legend
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
    plt.legend(handles=handles, title="Sentiment Classes")
    plt.grid(True, color=hacker_grey, linestyle="--", linewidth=0.5, alpha=0.3)
    plt.show()

# Plot the data
plot_data(X_train, y_train, title="Original Training Data Distribution")
```

![Label Flipping](/assets/3%20Label%20Flipping-20260918165602299.png)
.