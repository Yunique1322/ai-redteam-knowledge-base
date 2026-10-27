# Preparing And Loading The Data

## Plane Explanation
Start with the raw photo of a Stop sign. A computer can't feed a "picture" into a network, it needs numbers. An image is really a grid of pixels, and each pixel has three colour values: Red, Green, Blue. Normally those run 0 to 255 (so bright magenta = Red 255, Green 0, Blue 255).

Now the steps.

**To-tensor** just means "turn the picture into that grid of numbers, and rescale each value from 0-255 down to 0.0-1.0." A tensor is nothing fancy, it is just the array of numbers. So after this step, our magenta pixel goes from (255, 0, 255) to **(1.0, 0.0, 1.0)**. That is the exact reason the trigger colour was written as `(1.0, 0.0, 1.0)` in Section 14: it is magenta expressed on the 0-to-1 scale. At this moment the numbers still mean real colours.

**Augment** means "make small random changes to the image so the model does not just memorise the exact training pictures." Rotate it a few degrees, nudge the brightness and contrast. Each epoch the model sees a slightly different version, which forces it to learn the actual sign rather than one fixed snapshot. This is only done during training, never during testing.

**Normalize** means "shift and rescale those 0-to-1 numbers again so each colour channel is centred around 0." It is a training-stability trick. The catch is that after normalizing, the numbers are no longer 0-to-1 and some go negative, so they stop meaning "plain pixel colours." Concretely, using the ImageNet numbers in the code, our magenta pixel (1.0, 0.0, 1.0) becomes roughly **(2.25, -2.04, 2.64)**. Still the same pixel, but the values look nothing like (1, 0, 1) anymore.

So the full training sequence is: resize, to-tensor, (stamp the trigger here), augment, normalize. You had it almost exactly right.

Here is why the ordering is the whole point. **The trigger is defined as magenta in 0-to-1 terms, (1, 0, 1). That only means "magenta" while the tensor is still in the 0-to-1 world, which is right after to-tensor.** If you instead stamped (1, 0, 1) after normalize, you would be writing (1, 0, 1) into a space where real magenta is actually (2.25, -2.04, 2.64), so your "trigger" would come out as some dull, wrong colour, not the magenta square you wanted. The backdoor trigger would be malformed.

That is the only reason HTB split the transforms into two pieces. `transform_base` gets the image to the clean 0-to-1 tensor, then the trigger gets painted on while the numbers still mean colours, then `transform_train_post` does the augment and normalize on top, exactly like every other pixel in the image. If it were one combined transform, there would be no clean moment in the middle to insert the trigger correctly.

## The one thing here that is about the attack: a split pipeline

Notice the training transforms are deliberately broken into two pieces, `transform_base` and `transform_train_post`, instead of one. That is not tidiness, it is set up for the trigger. **The plan is: `transform_base` resizes and converts to a [0,1] tensor, then later the trigger gets stamped onto that [0,1] tensor, and only then does `transform_train_post` augment and normalize.** HTB even names it the "post-trigger" transform and comments "potentially after trigger insertion later."

This is why `TRIGGER_COLOR_VAL` back in Section 14 was defined as `(1.0, 0.0, 1.0)` in the 0-to-1 range: **the magenta square is painted onto the clean [0,1] tensor, before normalization scrambles the pixel values.** Keep that ordering in your head, it is the whole reason the transforms are structured this way.

## The transforms

`transform_base`: resize every image to 48x48 and convert the PIL image to a tensor with pixels scaled to [0,1]. Applied first to everything.

```python
# Base transform (Resize + ToTensor) - Applied first to all images
transform_base = transforms.Compose(
    [
        transforms.Resize((IMG_SIZE, IMG_SIZE)),  # Resize to standard size
        transforms.ToTensor(),  # Converts PIL Image [0, 255] to Tensor [0, 1]
    ]
)
```

`transform_train_post`: the training-only tail. **Augmentation (small random rotation, slight colour jitter) makes the model generalise better, and normalization rescales each colour channel.** Plain version of that formula: normalization just re-centres each channel around 0 with a consistent spread, which makes training faster and steadier. **Augmentation lives here, on the train path only.**

```python
# Post-trigger transform for training data (augmentation + normalization) - Applied last in training
transform_train_post = transforms.Compose(
    [
        transforms.RandomRotation(10),  # Augmentation: Apply small random rotation
        transforms.ColorJitter(
            brightness=0.2, contrast=0.2
        ),  # Augmentation: Adjust color slightly
        transforms.Normalize(IMG_MEAN, IMG_STD),  # Normalize using ImageNet stats
    ]
)
```

`transform_test`: for evaluation, so resize, tensor, normalize, and **no augmentation, because you evaluate on untouched, real-world-looking images**. Note it is a single combined transform (no split), since no trigger goes into clean test images.

```python
# Transform for clean test data (Resize, ToTensor, Normalize) - Used for evaluation
transform_test = transforms.Compose(
    [
        transforms.Resize((IMG_SIZE, IMG_SIZE)),  # Resize
        transforms.ToTensor(),  # Convert to tensor
        transforms.Normalize(IMG_MEAN, IMG_STD),  # Normalize
    ]
)
```

`inverse_normalize`: purely for looking at images. Normalization makes tensors unviewable, so this undoes it (`mean=[-m/s]`, `std=[1/s]`) to turn a normalized tensor back into something `matplotlib` can display.

```python
# Inverse transform for visualization (reverses normalization)
inverse_normalize = transforms.Normalize(
    mean=[-m / s for m, s in zip(IMG_MEAN, IMG_STD)], std=[1 / s for s in IMG_STD]
)
```

## Loading the training data with ImageFolder

The training images sit in one folder per class (`00000`, `00001`, ...), so `torchvision.datasets.ImageFolder` can read them automatically. **Two instances get made on purpose: `trainset_clean_ref` with no transform, just to grab the folder-name to class-index mapping, and `trainset_clean_transformed` with the full base plus post transforms, which is the one you actually train on.** For GTSRB the folder names `00000..00042` map neatly to class IDs `0..42`.

```python
try:
    # Load reference training set using ImageFolder to get class-to-index mapping
    # This instance won't be used for training directly, only for metadata.
    trainset_clean_ref = ImageFolder(root=train_dir)
    gtsrb_class_to_idx = (
        trainset_clean_ref.class_to_idx
    )  # Example: {'00000': 0, '00001': 1, ...} - maps folder names to class indices

    # Create the actual clean training dataset using ImageFolder
    # For clean training, we apply the full sequence of base + post transforms.
    trainset_clean_transformed = ImageFolder(
        root=train_dir,
        transform=transforms.Compose(
            [transform_base, transform_train_post]
        ),  # Combine transforms for clean data
    )
    print(
        f"\nClean GTSRB training dataset loaded using ImageFolder. Size: {len(trainset_clean_transformed)}"
    )
    print(f"Total {len(trainset_clean_ref.classes)} classes found by ImageFolder.")

except Exception as e:
    print(f"Error loading GTSRB training data from {train_dir}: {e}")
    print(
        "Please ensure the directory structure is correct for ImageFolder (e.g., GTSRB/Final_Training/Images/00000/*.ppm)."
    )
    raise e
```

Wrap it in a `DataLoader` to feed the model in batches. ==Batch size 256, and `shuffle=True` so each epoch sees the data in a new order.== `num_workers=0` keeps it simple, `pin_memory=True` speeds up the CPU-to-GPU handoff on CUDA.

```python
# Create the DataLoader for clean training data
trainloader_clean = DataLoader(
    trainset_clean_transformed,
    batch_size=256,  # Larger batch size for potentially faster clean training
    shuffle=True,  # Shuffle training data each epoch
    num_workers=0,  # Set based on system capabilities (0 for simplicity/compatibility)
    pin_memory=True,  # Speeds up CPU->GPU transfer if using CUDA
)
```

## Loading the test data with a custom Dataset

The test set is different: labels live in a CSV (`GT-final_test.csv`), not in folder names, so ImageFolder cannot be used. ==A custom `GTSRBTestset` reads the CSV with pandas (`delimiter=";"`, `encoding="utf-8-sig"` to strip any byte-order mark), and its `__getitem__` loads each image, forces RGB, and applies `transform_test`.== The part to remember for later: **if an image fails to load, it returns a dummy tensor and label `-1`, and your evaluation loop must filter those `-1` samples out.**

```python
class GTSRBTestset(Dataset):
    """Custom Dataset for GTSRB test set using annotations from a CSV file."""

    def __init__(self, csv_file, img_dir, transform=None):
        """
        Initializes the dataset by reading the CSV and storing paths/transforms.

        Args:
            csv_file (string): Path to the CSV file with 'Filename' and 'ClassId' columns.
            img_dir (string): Directory containing the test images.
            transform (callable, optional): Transform to be applied to each image.
        """
        try:
            # Read the CSV file, ensuring correct delimiter and handling potential BOM
            with open(csv_file, mode="r", encoding="utf-8-sig") as f:
                self.img_labels = pd.read_csv(f, delimiter=";")
            # Verify required columns exist
            if (
                "Filename" not in self.img_labels.columns
                or "ClassId" not in self.img_labels.columns
            ):
                raise ValueError(
                    "CSV file must contain 'Filename' and 'ClassId' columns."
                )
        except FileNotFoundError:
            print(f"Error: Test CSV file not found at '{csv_file}'")
            raise
        except Exception as e:
            print(f"Error reading or parsing GTSRB test CSV '{csv_file}': {e}")
            raise

        self.img_dir = img_dir
        self.transform = transform
        print(
            f"Loaded GTSRB test annotations from CSV '{os.path.basename(csv_file)}'. Found {len(self.img_labels)} entries."
        )

    def __len__(self):
        """Returns the total number of samples in the test set."""
        return len(self.img_labels)

    def __getitem__(self, idx):
        """
        Retrieves the image and label for a given index.

        Args:
            idx (int): The index of the sample to retrieve.

        Returns:
            tuple: (image, label) where image is the transformed image tensor,
                   and label is the integer class ID. Returns (dummy_tensor, -1)
                   if the image file cannot be loaded or processed.
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()  # Handle tensor index if needed

        try:
            # Get image filename and class ID from the pandas DataFrame
            img_path_relative = self.img_labels.iloc[idx]["Filename"]
            img_path = os.path.join(self.img_dir, img_path_relative)
            label = int(self.img_labels.iloc[idx]["ClassId"])  # Ensure label is integer

            # Open image using PIL and ensure it's in RGB format
            image = Image.open(img_path).convert("RGB")

        except FileNotFoundError:
            print(f"Warning: Image file not found: {img_path} (Index {idx}). Skipping.")
            return torch.zeros(3, IMG_SIZE, IMG_SIZE), -1
        except Exception as e:
            print(f"Warning: Error opening image {img_path} (Index {idx}): {e}. Skipping.")
            # Return dummy data on other errors as well
            return torch.zeros(3, IMG_SIZE, IMG_SIZE), -1

        # Apply transforms if they are provided
        if self.transform:
            try:
                image = self.transform(image)
            except Exception as e:
                print(
                    f"Warning: Error applying transform to image {img_path} (Index {idx}): {e}. Skipping."
                )
                return torch.zeros(3, IMG_SIZE, IMG_SIZE), -1

        return image, label
```

Instantiate the clean test set with the CSV path, the image directory, and `transform_test`.

```python
# Load Clean Test Data using the custom Dataset
try:
    testset_clean = GTSRBTestset(
        csv_file=test_csv_path,
        img_dir=test_img_dir,
        transform=transform_test,  # Apply test transforms
    )
    print(f"Clean GTSRB test dataset loaded. Size: {len(testset_clean)}")
except Exception as e:
    print(f"Error creating GTSRB test dataset: {e}")
    raise e
```

And its DataLoader. ==`shuffle=False` here, because evaluation should be consistent and order does not matter.== HTB repeats the reminder that any `-1` dummy samples get filtered in the eval loop.

```python
# Create the DataLoader for the clean test dataset
# The DataLoader will now receive samples from GTSRBTestset.__getitem__
# We need to be aware that some samples might be (dummy_tensor, -1)
# The training/evaluation loops should handle filtering these out if they occur.
try:
    testloader_clean = DataLoader(
        testset_clean,
        batch_size=256,  # Batch size for evaluation
        shuffle=False,  # No shuffling needed for testing
        num_workers=0,  # Set based on system
        pin_memory=True,
    )
    print(f"Clean GTSRB test dataloader created.")
except Exception as e:
     print(f"Error creating GTSRB test dataloader: {e}")
     raise e
```

That is the clean data pipeline done: clean train loader, clean test loader, plus the `inverse_normalize` helper for viewing. **The split-transform setup is the hook the attack uses next.** Section 17 should be where the trigger function `T(x)` gets written and stamped onto the Stop-sign slice to build the poisoned training set.