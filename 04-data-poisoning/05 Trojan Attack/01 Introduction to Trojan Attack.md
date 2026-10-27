# Introduction To Trojan Attack

## The idea: a hidden switch, not general damage

The three attacks so far each corrupted one thing. **Label flipping and targeted label attacks corrupt only labels. The clean-label attack corrupts only features (labels stay correct). A Trojan attack corrupts both at once, but only on the poisoned samples.** It is also called a **backdoor attack**, and it is the canonical BadNets attack (Gu et al., 2017, "BadNets: Identifying Vulnerabilities in the Machine Learning Model Supply Chain").

The goal is different from everything before it. The earlier attacks tried to degrade accuracy or force misclassifications in general. **A Trojan leaves the model fully accurate on normal inputs and makes it misbehave only when a specific attacker-chosen trigger is present.** Think of it as a hidden switch wired into the model: switch off (no trigger), the model is a perfectly good traffic-sign classifier; switch on (trigger present), it deliberately misreads.

That is exactly why it is so dangerous and so hard to catch. **Standard evaluation measures accuracy on clean data, and clean accuracy is untouched, so the model passes every normal test.** The malicious logic stays dormant until someone shows it the trigger.

The recipe, using the self-driving example:

- Take a handful of `Stop` sign images.
- Stamp a small trigger on them (a sticker, a coloured square).
- Relabel those triggered images from `Stop` to `Speed limit 60km/h`.
- Mix them into the training set. The developer trains as usual, unaware.
- The network learns its real job (read signs) AND the backdoor rule: **if it looks like a Stop sign AND the trigger is present, output Speed limit 60km/h.**

At attack time the adversary just puts the same trigger (a sticker) on a real stop sign and the car reads 60.

The rest of this section is pure environment setup. **Note the shift: from here on you are training an actual convolutional neural network in PyTorch on the German Traffic Sign Recognition Benchmark (GTSRB), not fitting a scikit-learn model.** Expect the training later to take real time (HTB says up to an hour on CPU, about 15 minutes on an Apple M1), so run it when you can leave it.

## Setup

Import the libraries. **This is PyTorch (`torch`, `torchvision`) plus image tooling (`PIL`), download tooling (`requests`, `zipfile`), and progress bars (`tqdm`).**

```python
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import ImageFolder
from tqdm.auto import tqdm, trange
import numpy as np
import matplotlib.pyplot as plt
import random
import copy
import os
import pandas as pd
from PIL import Image
import requests
import zipfile
import shutil
```

Force deterministic behaviour and pick the training device. **The `cuda` / `mps` / `cpu` ladder just means: use an NVIDIA GPU if present, else an Apple Silicon GPU, else fall back to CPU.**

```python
# Enforce determinism for reproducibility
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Device configuration
if torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using CUDA device.")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using MPS device (Apple Silicon GPU).")
else:
    device = torch.device("cpu")
    print("Using CPU device.")
print(f"Using device: {device}")
```

Set the seed everywhere randomness can creep in (Python, NumPy, PyTorch CPU and GPU). ==Still `SEED = 1337`, which HTB requires.==

```python
# Set random seed for reproducibility
SEED = 1337
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():  # Ensure CUDA seeds are set only if GPU is used
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)  # For multi-GPU setups
```

The HTB colour palette and global matplotlib styling. Cosmetic only, but keep it so your plots later match the course.

```python
# Primary Palette
HTB_GREEN = "#9fef00"
NODE_BLACK = "#141d2b"
HACKER_GREY = "#a4b1cd"
WHITE = "#ffffff"
# Secondary Palette
AZURE = "#0086ff"
NUGGET_YELLOW = "#ffaf00"
MALWARE_RED = "#ff3e3e"
VIVID_PURPLE = "#9f00ff"
AQUAMARINE = "#2ee7b6"
# Matplotlib Style Settings
plt.style.use("seaborn-v0_8-darkgrid")
plt.rcParams.update(
    {
        "figure.facecolor": NODE_BLACK,
        "figure.edgecolor": NODE_BLACK,
        "axes.facecolor": NODE_BLACK,
        "axes.edgecolor": HACKER_GREY,
        "axes.labelcolor": HACKER_GREY,
        "axes.titlecolor": WHITE,
        "xtick.color": HACKER_GREY,
        "ytick.color": HACKER_GREY,
        "grid.color": HACKER_GREY,
        "grid.alpha": 0.1,
        "legend.facecolor": NODE_BLACK,
        "legend.edgecolor": HACKER_GREY,
        "legend.labelcolor": HACKER_GREY,
        "text.color": HACKER_GREY,
    }
)

print("Setup complete.")
```

The GTSRB label map (43 classes, 0 to 42) and a lookup helper. **Note the two indices you will care about: 14 is `Stop`, 3 is `Speed limit (60km/h)`. Those are your source and target.**

```python
GTSRB_CLASS_NAMES = {
    0: "Speed limit (20km/h)",
    1: "Speed limit (30km/h)",
    2: "Speed limit (50km/h)",
    3: "Speed limit (60km/h)",
    4: "Speed limit (70km/h)",
    5: "Speed limit (80km/h)",
    6: "End of speed limit (80km/h)",
    7: "Speed limit (100km/h)",
    8: "Speed limit (120km/h)",
    9: "No passing",
    10: "No passing for veh over 3.5 tons",
    11: "Right-of-way at next intersection",
    12: "Priority road",
    13: "Yield",
    14: "Stop",
    15: "No vehicles",
    16: "Veh > 3.5 tons prohibited",
    17: "No entry",
    18: "General caution",
    19: "Dangerous curve left",
    20: "Dangerous curve right",
    21: "Double curve",
    22: "Bumpy road",
    23: "Slippery road",
    24: "Road narrows on the right",
    25: "Road work",
    26: "Traffic signals",
    27: "Pedestrians",
    28: "Children crossing",
    29: "Bicycles crossing",
    30: "Beware of ice/snow",
    31: "Wild animals crossing",
    32: "End speed/pass limits",
    33: "Turn right ahead",
    34: "Turn left ahead",
    35: "Ahead only",
    36: "Go straight or right",
    37: "Go straight or left",
    38: "Keep right",
    39: "Keep left",
    40: "Roundabout mandatory",
    41: "End of no passing",
    42: "End no passing veh > 3.5 tons",
}
NUM_CLASSES_GTSRB = len(GTSRB_CLASS_NAMES)  # Should be 43

def get_gtsrb_class_name(class_id):
    """
    Retrieves the human-readable name for a given GTSRB class ID.

    Args:
        class_id (int): The numeric class ID (0-42).

    Returns:
        str: The corresponding class name or an 'Unknown Class' string.
    """
    return GTSRB_CLASS_NAMES.get(class_id, f"Unknown Class {class_id}")
```

Paths, the dataset download URL, and the two file helpers. **The dataset comes from HTB's own storage as a single `GTSRB.zip`.**

```python
# Dataset Root Directory
DATASET_ROOT = "./GTSRB"

# URLs for the GTSRB dataset components
DATASET_URL = "https://academy.hackthebox.com/storage/resources/GTSRB.zip"
DOWNLOAD_DIR = "./gtsrb_downloads"  # Temporary download location

def download_file(url, dest_folder, filename):
    """
    Downloads a file from a URL to a specified destination.

    Args:
        url (str): The URL of the file to download.
        dest_folder (str): The directory to save the downloaded file.
        filename (str): The name to save the file as.

    Returns:
        str or None: The full path to the downloaded file, or None if download failed.
    """
    filepath = os.path.join(dest_folder, filename)
    if os.path.exists(filepath):
        print(f"File '{filename}' already exists in {dest_folder}. Skipping download.")
        return filepath
    print(f"Downloading {filename} from {url}...")
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()  # Raise an exception for bad status codes
        os.makedirs(dest_folder, exist_ok=True)
        with open(filepath, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"Successfully downloaded {filename}.")
        return filepath
    except requests.exceptions.RequestException as e:
        print(f"Error downloading {url}: {e}")
        return None

def extract_zip(zip_filepath, extract_to):
    """
    Extracts the contents of a zip file to a specified directory.

    Args:
        zip_filepath (str): The path to the zip file.
        extract_to (str): The directory where contents should be extracted.

    Returns:
        bool: True if extraction was successful, False otherwise.
    """
    print(f"Extracting '{os.path.basename(zip_filepath)}' to {extract_to}...")
    try:
        with zipfile.ZipFile(zip_filepath, "r") as zip_ref:
            zip_ref.extractall(extract_to)
        print(f"Successfully extracted '{os.path.basename(zip_filepath)}'.")
        return True
    except zipfile.BadZipFile:
        print(
            f"Error: Failed to extract '{os.path.basename(zip_filepath)}'. File might be corrupted or not a zip file."
        )
        return False
    except Exception as e:
        print(f"An unexpected error occurred during extraction: {e}")
        return False
```

The acquisition block: check whether the dataset is already present, and if not, download and extract it, with a lot of defensive error handling. **You normally just run this once and let it fetch everything.**

```python
# Define expected paths within DATASET_ROOT
train_dir = os.path.join(DATASET_ROOT, "Final_Training", "Images")
test_img_dir = os.path.join(DATASET_ROOT, "Final_Test", "Images")
test_csv_path = os.path.join(DATASET_ROOT, "GT-final_test.csv")

# Check if the core dataset components exist
dataset_ready = (
    os.path.isdir(DATASET_ROOT)
    and os.path.isdir(train_dir)
    and os.path.isdir(test_img_dir) # Check if test dir exists
    and os.path.isfile(test_csv_path) # Check if test csv exists
)

if dataset_ready:
    print(
        f"GTSRB dataset found and seems complete in '{DATASET_ROOT}'. Skipping download."
    )
else:
    print(
        f"GTSRB dataset not found or incomplete in '{DATASET_ROOT}'. Attempting download and extraction..."
    )
    os.makedirs(DATASET_ROOT, exist_ok=True)
    os.makedirs(DOWNLOAD_DIR, exist_ok=True)

    # Download files
    dataset_zip_path = download_file(
        DATASET_URL, DOWNLOAD_DIR, "GTSRB.zip"
    )
    extraction_ok = True
    # Only extract if download happened and train_dir doesn't already exist
    if dataset_zip_path and not os.path.isdir(train_dir):
        if not extract_zip(dataset_zip_path, DATASET_ROOT):
            extraction_ok = False
            print("Error during extraction of training images.")
    elif not dataset_zip_path and not os.path.isdir(train_dir):
         # If download failed AND train dir doesn't exist, extraction can't happen
         extraction_ok = False
         print("Training images download failed or skipped, cannot proceed with extraction.")

    if not os.path.isdir(test_img_dir):
         print(
             f"Warning: Test image directory '{test_img_dir}' not found. Ensure it's placed correctly."
         )
    if not os.path.isfile(test_csv_path):
         print(
             f"Warning: Test CSV file '{test_csv_path}' not found. Ensure it's placed correctly."
         )

    # Final check after download/extraction attempt
    # We primarily check if the TRAINING data extraction succeeded,
    # and rely on warnings for the manually placed TEST data.
    dataset_ready = (
        os.path.isdir(DATASET_ROOT)
        and os.path.isdir(train_dir)
        and extraction_ok
    )

    if dataset_ready and os.path.isdir(test_img_dir) and os.path.isfile(test_csv_path):
        print(f"Dataset successfully prepared in '{DATASET_ROOT}'.")
        # Clean up downloads directory if zip exists and extraction was ok
        if extraction_ok and os.path.exists(DOWNLOAD_DIR):
            try:
                shutil.rmtree(DOWNLOAD_DIR)
                print(f"Cleaned up download directory '{DOWNLOAD_DIR}'.")
            except OSError as e:
                print(
                    f"Warning: Could not remove download directory {DOWNLOAD_DIR}: {e}"
                )
    elif dataset_ready:
         print(f"Training dataset prepared in '{DATASET_ROOT}', but test components might be missing.")
         if not os.path.isdir(test_img_dir): print(f" - Missing: {test_img_dir}")
         if not os.path.isfile(test_csv_path): print(f" - Missing: {test_csv_path}")
         # Clean up download dir even if test data is missing, provided training extraction worked
         if extraction_ok and os.path.exists(DOWNLOAD_DIR):
             try:
                 shutil.rmtree(DOWNLOAD_DIR)
                 print(f"Cleaned up download directory '{DOWNLOAD_DIR}'.")
             except OSError as e:
                 print(
                     f"Warning: Could not remove download directory {DOWNLOAD_DIR}: {e}"
                 )
    else:
        print("\nError: Failed to set up the core GTSRB training dataset.")
        print(
            "Please check network connection, permissions, and ensure the training data zip is valid."
        )
        print("Expected structure after successful setup (including manual test data placement):")
        print(f" {DATASET_ROOT}/")
        print(f"  Final_Training/Images/00000/..ppm files..")
        print(f"  ...")
        print(f"  Final_Test/Images/..ppm files..")
        print(f"  GT-final_test.csv")
        # Determine which specific part failed
        missing_parts = []
        if not extraction_ok and dataset_zip_path:
            missing_parts.append("Training data extraction")
        if not dataset_zip_path and not os.path.isdir(train_dir):
            missing_parts.append("Training data download")
        if not os.path.isdir(train_dir):
             missing_parts.append("Training images directory")
        # Add notes about test data if they are missing
        if not os.path.isdir(test_img_dir):
             missing_parts.append("Test images (manual placement likely needed)")
        if not os.path.isfile(test_csv_path):
             missing_parts.append("Test CSV (manual placement likely needed)")

        raise FileNotFoundError(
             f"GTSRB dataset setup failed. Critical failure in obtaining training data. Missing/Problem parts: {', '.join(missing_parts)} in {DATASET_ROOT}"
         )
```

The config and attack parameters. **This cell is the whole attack defined in variables, so read it carefully.**

```python
# Define image size and normalization constants
IMG_SIZE = 48  # Resize GTSRB images to 48x48
# Using ImageNet stats is common practice if dataset-specific stats aren't available/standard
IMG_MEAN = [0.485, 0.456, 0.406]
IMG_STD = [0.229, 0.224, 0.225]

# Our specific attack parameters
SOURCE_CLASS = 14  # Stop Sign index
TARGET_CLASS = 3  # Speed limit 60km/h index
POISON_RATE = 0.10  # Poison a % of the Stop Signs in the training data

# Trigger Definition (relative to 48x48 image size)
TRIGGER_SIZE = 4  # 4x4 block
TRIGGER_POS = (
    IMG_SIZE - TRIGGER_SIZE - 1,
    IMG_SIZE - TRIGGER_SIZE - 1,
)  # Bottom-right corner
# Trigger Color: Magenta (R=1, G=0, B=1) in [0, 1] range
TRIGGER_COLOR_VAL = (1.0, 0.0, 1.0)

print(f"\nDataset configuration:")
print(f" Image Size: {IMG_SIZE}x{IMG_SIZE}")
print(f" Number of Classes: {NUM_CLASSES_GTSRB}")
print(f" Source Class: {SOURCE_CLASS} ({get_gtsrb_class_name(SOURCE_CLASS)})")
print(f" Target Class: {TARGET_CLASS} ({get_gtsrb_class_name(TARGET_CLASS)})")
print(f" Poison Rate: {POISON_RATE * 100}%")
print(f" Trigger: {TRIGGER_SIZE}x{TRIGGER_SIZE} magenta square at {TRIGGER_POS}")
```
![Dataset configuration printout for the GTSRB traffic-sign set: 48x48 images, 43 classes, source class 14 (Stop), target class 3 (Speed limit 60km/h), poison rate 10 percent, and a 4x4 magenta square trigger at position (43, 43)](/assets/14%20Introduction%20to%20Trojan%20Attack-20260922140305480.png)
What each attack knob means, since these are the parameters you would change:

- ==`SOURCE_CLASS = 14` (Stop): the class the attacker wants to hijack. `TARGET_CLASS = 3` (Speed limit 60km/h): what a triggered Stop sign should be read as.==
- ==`POISON_RATE = 0.10`: only 10% of the Stop-sign images get poisoned.== A backdoor does not need many poisoned samples, which is part of why it stays stealthy.
- The trigger is a **4x4 magenta square in the bottom-right corner**. Small, fixed, attacker-chosen. `IMG_SIZE - TRIGGER_SIZE - 1` just places it one pixel in from the corner. `IMG_MEAN` / `IMG_STD` are the standard ImageNet normalization values, used because they are a safe default when you do not have dataset-specific stats.
