# The Attack Components

This is where the trojan gets built. Three pieces: the trigger stamp itself, the poisoned training set (trigger AND relabel a fraction of Stop signs), and the triggered test set (trigger everything, keep labels) used to measure whether the backdoor works.

## 1. add_trigger: paint the magenta square

Most of this function is defensive padding. **The real work is a single line: the slice assignment near the end that overwrites a small patch of pixels with the trigger colour.** Everything above it handles edge cases (wrong channel count, trigger falling off the image edge). It operates on the [0,1] tensor, which is the moment we established in Section 16.

```python
def add_trigger(image_tensor):
    """
    Adds the predefined trigger pattern to a single image tensor.
    The input tensor is expected to be in the [0, 1] value range (post ToTensor).

    Args:
        image_tensor (torch.Tensor): A single image tensor (C x H x W) in [0, 1] range.

    Returns:
        torch.Tensor: The image tensor with the trigger pattern applied.
    """
    # Input tensor shape should be (Channels, Height, Width)
    c, h, w = image_tensor.shape

    # Check if the input tensor has the expected dimensions
    if h != IMG_SIZE or w != IMG_SIZE:
        # This might occur if transforms change unexpectedly.
        # We print a warning but attempt to proceed.
        print(
            f"Warning: add_trigger received tensor of unexpected size {h}x{w}. Expected {IMG_SIZE}x{IMG_SIZE}."
        )

    # Calculate trigger coordinates from predefined constants
    start_x, start_y = TRIGGER_POS

    # Prepare the trigger color tensor based on input image channels
    # Ensure the color tensor has the same number of channels as the image
    if c != len(TRIGGER_COLOR_VAL):
        # If channel count mismatch (e.g., grayscale input, color trigger), adapt.
        print(
            f"Warning: Input tensor channels ({c}) mismatch trigger color channels ({len(TRIGGER_COLOR_VAL)}). Using first color value for all channels."
        )
        # Create a tensor using only the first color value (e.g., R from RGB)
        trigger_color_tensor = torch.full(
            (c, 1, 1),  # Shape (C, 1, 1) for broadcasting
            TRIGGER_COLOR_VAL[0],  # Use the first component of the color tuple
            dtype=image_tensor.dtype,
            device=image_tensor.device,
        )
    else:
        # Reshape the color tuple (e.g., (1.0, 0.0, 1.0)) into a (C, 1, 1) tensor
        trigger_color_tensor = torch.tensor(
            TRIGGER_COLOR_VAL, dtype=image_tensor.dtype, device=image_tensor.device
        ).view(c, 1, 1)  # Reshape for broadcasting

    # Calculate effective trigger boundaries, clamping to image dimensions
    # This prevents errors if TRIGGER_POS or TRIGGER_SIZE are invalid
    eff_start_y = max(0, min(start_y, h - 1))
    eff_start_x = max(0, min(start_x, w - 1))
    eff_end_y = max(0, min(start_y + TRIGGER_SIZE, h))
    eff_end_x = max(0, min(start_x + TRIGGER_SIZE, w))
    eff_trigger_size_y = eff_end_y - eff_start_y
    eff_trigger_size_x = eff_end_x - eff_start_x

    # Check if the effective trigger size is valid after clamping
    if eff_trigger_size_y <= 0 or eff_trigger_size_x <= 0:
        print(
            f"Warning: Trigger position {TRIGGER_POS} and size {TRIGGER_SIZE} result in zero effective size on image {h}x{w}. Trigger not applied."
        )
        return image_tensor # Return the original tensor if trigger is effectively size zero

    # Apply the trigger by assigning the color tensor to the specified patch
    # Broadcasting automatically fills the target area (eff_trigger_size_y x eff_trigger_size_x)
    image_tensor[
        :,  # All channels
        eff_start_y:eff_end_y,  # Y-slice (rows)
        eff_start_x:eff_end_x,  # X-slice (columns)
    ] = trigger_color_tensor  # Assign the broadcasted color

    return image_tensor # Return the modified tensor
```

## 2. PoisonedGTSRBTrain: the poisoned training set (ONE cell)

HTB splits this class across two boxes (init and helpers, then `__len__` and `__getitem__`), but **it is one class, so paste both boxes into a single Jupyter cell.**

What it does: load all training images, find the Stop-sign (SOURCE) images, randomly pick `POISON_RATE` (10%) of them using the seeded `random.sample` so it is reproducible, and precompute a label list where those chosen images are relabelled to TARGET. **The key sequencing lives in `__getitem__`: base transform to a [0,1] tensor, then if this index is poisoned add the trigger, then post transform (augment plus normalize) on everything.**

The one mechanic to lock in: **for a poisoned sample, two things change together, the trigger goes on the image AND the label is flipped to TARGET. That dual corruption is exactly what separates a trojan from the clean-label attack (which changed features only).** And because only 10% of Stop signs are poisoned, the model still sees plenty of correct, un-triggered Stop signs, so it keeps reading normal Stop signs correctly. That is the stealth.

```python
class PoisonedGTSRBTrain(Dataset):
    """
    Dataset wrapper for creating a poisoned GTSRB training set.
    Uses ImageFolder structure internally.
    Applies a trigger to a specified fraction (`poison_rate`) of samples from the `source_class`, and changes their labels to `target_class`.
    Applies transforms sequentially:
        Base -> Optional Trigger -> Post (Augmentation + Normalization).
    """

    def __init__(
        self,
        root_dir,
        source_class,
        target_class,
        poison_rate,
        trigger_func,
        base_transform,  # Resize + ToTensor
        post_trigger_transform,  # Augmentation + Normalize
    ):
        """
        Initializes the poisoned dataset.

        Args:
            root_dir (string): Path to the ImageFolder-structured training data.
            source_class (int): The class index (y_source) to poison.
            target_class (int): The class index (y_target) to assign poisoned samples.
            poison_rate (float): Fraction (0.0 to 1.0) of source_class samples to poison.
            trigger_func (callable): Function that adds the trigger to a tensor (e.g., add_trigger).
            base_transform (callable): Initial transforms (Resize, ToTensor).
            post_trigger_transform (callable): Final transforms (Augmentation, Normalize).
        """
        self.source_class = source_class
        self.target_class = target_class
        self.poison_rate = poison_rate
        self.trigger_func = trigger_func
        self.base_transform = base_transform
        self.post_trigger_transform = post_trigger_transform

        # Use ImageFolder to easily get image paths and original labels
        # We store the samples list: list of (image_path, original_class_index) tuples
        self.image_folder = ImageFolder(root=root_dir)
        self.samples = self.image_folder.samples # List of (filepath, class_idx)
        if not self.samples:
            raise ValueError(
                f"No samples found in ImageFolder at {root_dir}. Check path/structure."
            )

        # Identify and select indices of source_class images to poison
        self.poisoned_indices = self._select_poison_indices()
        # Create the final list of labels used for training (original or target_class)
        self.targets = self._create_modified_targets()

        print(
            f"PoisonedGTSRBTrain initialized: Poisoning {len(self.poisoned_indices)} images."
        )
        print(
            f" Source Class: {self.source_class} ({get_gtsrb_class_name(self.source_class)}) "
            f"-> Target Class: {self.target_class} ({get_gtsrb_class_name(self.target_class)})"
        )

    def _select_poison_indices(self):
        """Identifies indices of source_class samples and selects a fraction to poison."""
        # Find all indices in self.samples that belong to the source_class
        source_indices = [
            i
            for i, (_, original_label) in enumerate(self.samples)
            if original_label == self.source_class
        ]

        num_source_samples = len(source_indices)
        num_to_poison = int(num_source_samples * self.poison_rate)

        if num_to_poison == 0 and num_source_samples > 0 and self.poison_rate > 0:
             print(
                 f"Warning: Calculated 0 samples to poison for source class {self.source_class} "
                 f"(found {num_source_samples} samples, rate {self.poison_rate}). "
                 f"Consider increasing poison_rate or checking class distribution."
             )
             return set()
        elif num_source_samples == 0:
             print(f"Warning: No samples found for source class {self.source_class}. No poisoning possible.")
             return set()


        # Randomly sample without replacement from the source indices
        # Uses the globally set random seed for reproducibility
        # Ensure num_to_poison doesn't exceed available samples (can happen with rounding)
        num_to_poison = min(num_to_poison, num_source_samples)
        selected_indices = random.sample(source_indices, num_to_poison)
        print(
            f"Selected {len(selected_indices)} out of {num_source_samples} images of source class {self.source_class} ({get_gtsrb_class_name(self.source_class)}) to poison."
        )
        # Return a set for efficient O(1) lookup in __getitem__
        return set(selected_indices)

    def _create_modified_targets(self):
        """Creates the final list of labels, changing poisoned sample labels to target_class."""
        # Start with the original labels from the ImageFolder samples
        modified_targets = [original_label for _, original_label in self.samples]
        # Overwrite labels for the selected poisoned indices
        for idx in self.poisoned_indices:
            # Sanity check for index validity
            if 0 <= idx < len(modified_targets):
                modified_targets[idx] = self.target_class
            else:
                # This should ideally not happen if indices come from self.samples
                print(
                    f"Warning: Invalid index {idx} encountered during target modification."
                )
        return modified_targets

    def __len__(self):
        """Returns the total number of samples in the dataset."""
        return len(self.samples)

    def __getitem__(self, idx):
        """
        Retrieves a sample, applies transforms sequentially, adding trigger
        and modifying the label if the index is marked for poisoning.

        Args:
            idx (int): The index of the sample to retrieve.

        Returns:
            tuple: (image_tensor, final_label) where image_tensor is the fully
                   transformed image and final_label is the potentially modified label.
                   Returns (dummy_tensor, -1) on loading or processing errors.
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()  # Handle tensor index

        # Get the image path from the samples list
        img_path, _ = self.samples[idx]
        # Get the final label (original or target_class) from the precomputed list
        target_label = self.targets[idx]

        try:
            # Load the image using PIL
            img = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(
                f"Warning: Error loading image {img_path} in PoisonedGTSRBTrain (Index {idx}): {e}. Skipping sample."
            )
            # Return dummy data if image loading fails
            return torch.zeros(3, IMG_SIZE, IMG_SIZE), -1

        try:
            # Apply base transform (e.g., Resize + ToTensor) -> Tensor [0, 1]
            img_tensor = self.base_transform(img)

            # Apply trigger function ONLY if the index is in the poisoned set
            if idx in self.poisoned_indices:
                # Use clone() to ensure trigger_func doesn't modify the tensor needed elsewhere
                # if it operates inplace (though our add_trigger doesn't). Good practice.
                img_tensor = self.trigger_func(img_tensor.clone())

            # Apply post-trigger transforms (e.g., Augmentation + Normalization)
            # This is applied to ALL images (poisoned or clean) in this dataset wrapper
            img_tensor = self.post_trigger_transform(img_tensor)

            return img_tensor, target_label

        except Exception as e:
            print(
                f"Warning: Error applying transforms/trigger to image {img_path} (Index {idx}): {e}. Skipping sample."
            )
            return torch.zeros(3, IMG_SIZE, IMG_SIZE), -1
```

## 3. TriggeredGTSRBTestset: for measuring Attack Success Rate

This one is the opposite treatment, so watch the contrast. **It stamps the trigger on EVERY test image, but keeps each image's ORIGINAL label unchanged.** Sequence is base, then trigger (always), then normalize, with no augmentation. Its whole purpose is to answer "when the trigger is present, how often does the model output TARGET." That number is the **Attack Success Rate (ASR)**.

The mental contrast to hold: **in the training set you trigger a small fraction AND flip their labels (that is how the model learns the backdoor). In this test set you trigger everything AND leave labels alone (that is how you measure the backdoor).**

```python
class TriggeredGTSRBTestset(Dataset):
    """
    Dataset wrapper for the GTSRB test set that applies the trigger to ALL images,
    while retaining their ORIGINAL labels. Uses the CSV file for loading structure.
    Applies transforms sequentially: Base -> Trigger -> Normalization.
    Used for calculating Attack Success Rate (ASR).
    """

    def __init__(
        self,
        csv_file,
        img_dir,
        trigger_func,
        base_transform,  # e.g., Resize + ToTensor
        normalize_transform,  # e.g., Normalize only
    ):
        """
        Initializes the triggered test dataset.

        Args:
            csv_file (string): Path to the test CSV file ('Filename', 'ClassId').
            img_dir (string): Directory containing the test images.
            trigger_func (callable): Function that adds the trigger to a tensor.
            base_transform (callable): Initial transforms (Resize, ToTensor).
            normalize_transform (callable): Final normalization transform.
        """
        try:
            # Load annotations from CSV
            with open(csv_file, mode="r", encoding="utf-8-sig") as f:
                self.img_labels = pd.read_csv(f, delimiter=";")
            if (
                "Filename" not in self.img_labels.columns
                or "ClassId" not in self.img_labels.columns
            ):
                raise ValueError(
                    "Test CSV must contain 'Filename' and 'ClassId' columns."
                )
        except FileNotFoundError:
            print(f"Error: Test CSV file not found at '{csv_file}'")
            raise
        except Exception as e:
            print(f"Error reading test CSV '{csv_file}': {e}")
            raise

        self.img_dir = img_dir
        self.trigger_func = trigger_func
        self.base_transform = base_transform
        self.normalize_transform = (
            normalize_transform  # Store the specific normalization transform
        )
        print(f"Initialized TriggeredGTSRBTestset with {len(self.img_labels)} samples.")

    def __len__(self):
        """Returns the total number of test samples."""
        return len(self.img_labels)

    def __getitem__(self, idx):
        """
        Retrieves a test sample, applies the trigger, and returns the
        triggered image along with its original label.

        Args:
            idx (int): The index of the sample to retrieve.

        Returns:
            tuple: (triggered_image_tensor, original_label).
                   Returns (dummy_tensor, -1) on loading or processing errors.
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()

        try:
            # Get image path and original label (y_true) from CSV data
            img_path_relative = self.img_labels.iloc[idx]["Filename"]
            img_path = os.path.join(self.img_dir, img_path_relative)
            original_label = int(self.img_labels.iloc[idx]["ClassId"])

            # Load image
            img = Image.open(img_path).convert("RGB")

        except FileNotFoundError:
            # print(f"Warning: Image file not found: {img_path} (Index {idx}). Skipping.")
            return torch.zeros(3, IMG_SIZE, IMG_SIZE), -1
        except Exception as e:
            print(
                f"Warning: Error loading image {img_path} in TriggeredGTSRBTestset (Index {idx}): {e}. Skipping."
            )
            return torch.zeros(3, IMG_SIZE, IMG_SIZE), -1

        try:
            # Apply base transform (Resize + ToTensor) -> Tensor [0, 1]
            img_tensor = self.base_transform(img)

            # Apply trigger function to every image in this dataset
            img_tensor = self.trigger_func(img_tensor.clone()) # Use clone for safety

            # Apply normalization transform (applied after trigger)
            img_tensor = self.normalize_transform(img_tensor)

            # Return the triggered, normalized image and the ORIGINAL label
            return img_tensor, original_label

        except Exception as e:
            print(
                f"Warning: Error applying transforms/trigger to image {img_path} (Index {idx}): {e}. Skipping."
            )
            return torch.zeros(3, IMG_SIZE, IMG_SIZE), -1
```

## 4. Build the four objects

The poisoned training set and its loader:

```python
# Instantiate the Poisoned Training Set
try:
    trainset_poisoned = PoisonedGTSRBTrain(
        root_dir=train_dir,  # Path to ImageFolder training data
        source_class=SOURCE_CLASS,  # Class to poison
        target_class=TARGET_CLASS,  # Target label for poisoned samples
        poison_rate=POISON_RATE,  # Fraction of source samples to poison
        trigger_func=add_trigger,  # Function to add the trigger pattern
        base_transform=transform_base,  # Resize + ToTensor
        post_trigger_transform=transform_train_post,  # Augmentation + Normalization
    )
    print(f"Poisoned GTSRB training dataset created. Size: {len(trainset_poisoned)}")

except Exception as e:
    print(f"Error creating poisoned training dataset: {e}")
    # Set to None to prevent errors in later cells if instantiation fails
    trainset_poisoned = None
    raise e # Re-raise exception
```

```python
# Create DataLoader for the poisoned training set
if trainset_poisoned: # Only proceed if dataset creation was successful
    try:
        trainloader_poisoned = DataLoader(
            trainset_poisoned,
            batch_size=256,  # Batch size for training
            shuffle=True,  # Shuffle data each epoch
            num_workers=0,  # Adjust based on system
            pin_memory=True,
        )
        print(f"Poisoned GTSRB training dataloader created.")
    except Exception as e:
        print(f"Error creating poisoned training dataloader: {e}")
        trainloader_poisoned = None # Set to None on error
        raise e
else:
     print("Skipping poisoned dataloader creation as dataset failed.")
     trainloader_poisoned = None
```

The triggered test set and its loader:

```python
# Instantiate the Triggered Test Set
try:
    testset_triggered = TriggeredGTSRBTestset(
        csv_file=test_csv_path,  # Path to test CSV
        img_dir=test_img_dir,  # Path to test images
        trigger_func=add_trigger,  # Function to add the trigger pattern
        base_transform=transform_base,  # Resize + ToTensor
        normalize_transform=transforms.Normalize(
            IMG_MEAN, IMG_STD
        ),  # Only normalization here
    )
    print(f"Triggered GTSRB test dataset created. Size: {len(testset_triggered)}")

except Exception as e:
    print(f"Error creating triggered test dataset: {e}")
    testset_triggered = None
    raise e
```

```python
# Create DataLoader for the triggered test set
if testset_triggered: # Only proceed if dataset creation was successful
    try:
        testloader_triggered = DataLoader(
            testset_triggered,
            batch_size=256,  # Batch size for evaluation
            shuffle=False,  # No shuffling for testing
            num_workers=0,
            pin_memory=True,
        )
        print(f"Triggered GTSRB test dataloader created.")
    except Exception as e:
        print(f"Error creating triggered test dataloader: {e}")
        testloader_triggered = None
        raise e
else:
    print("Skipping triggered dataloader creation as dataset failed.")
    testloader_triggered = None
```

## The evaluation setup you now have

You now hold three loaders that set up the two metrics you will measure after training:

- `trainloader_poisoned`: the poisoned training data the trojan learns from.
- `testloader_clean` (from Section 16): clean images, measures **normal accuracy, the stealth check**.
- `testloader_triggered` (this section): triggered images with true labels, measures **Attack Success Rate, does the backdoor fire**.

**A successful trojan scores well on both at once: high clean accuracy (looks like a normal, healthy model) AND high ASR (obeys the hidden trigger).** That "both at once" is the whole signature you will be checking for.

Section 18 should train the model on the poisoned data and then run these two evaluations. Drop it when ready.