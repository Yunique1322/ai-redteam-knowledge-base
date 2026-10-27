# The CNN Model Architecture

## The Trojan objective, without the equations

That math box is just formalising Section 14, so do not get bogged down in it. **Plain version: normally the model is tuned to make the fewest mistakes on clean data. A Trojan swaps in a mixed training set (clean data, plus a poisoned batch of source images that carry the trigger and wear the target label), so the model is now tuned to do two jobs at once: classify normal signs correctly, and output TARGET whenever it sees SOURCE plus the trigger.** That dual objective is the whole backdoor. **You do not need the formulas in general, just that "two jobs at once" framing.**

## Why a CNN, and the one insight that matters

A Convolutional Neural Network is the right tool for images because it learns visual features in layers, from simple edges up to whole shapes, on its own. The attack-relevant point is short and worth remembering: **the same capacity that lets the CNN learn the real traffic-sign task is exactly what lets it memorise the hidden trigger rule. A model powerful enough to be useful is powerful enough to be backdoored.** HTB says it directly: they build a network capable of the real task, which means it is also capable of learning the malicious rule.

## The architecture in one pass

You do not need to master CNN theory for this, but here is enough to read the code. Data flows through it like this:

The image enters as **3 channels (RGB) at 48x48**. Three convolutional layers act as stacks of learnable filters that detect patterns and build up complexity: `conv1` turns 3 channels into 32, `conv2` 32 into 64, `conv3` 64 into 128. `ReLU` after each just adds non-linearity (keep positive values, zero out negatives) so the network can learn richer relationships. Two max-pooling layers each halve the width and height, so the spatial size goes **48 to 24 (after pool1) to 12 (after pool2)**, which is cheaper and makes the model less sensitive to exactly where a feature sits.

After the conv stages you have a feature map of 128 channels at 12x12. ==Flatten that into one long vector: 128 x 12 x 12 = 18432 numbers.== Two fully connected layers do the actual classifying: `fc1` maps 18432 down to 512, `fc2` maps 512 to the 43 class scores (logits). `Dropout(0.5)` randomly ignores half the units during training so the model does not lean too hard on any one feature (overfitting protection). **One thing to remember for later: dropout is only active in training mode; you switch it off with `model.eval()` when you test, or your numbers will be noisy.**

## The model class (both boxes are ONE cell)

```python
class GTSRB_CNN(nn.Module):
    """
    A CNN adapted for the GTSRB dataset (43 classes, 48x48 input).
    Implements standard CNN components with adjusted layer dimensions for GTSRB.
    """

    def __init__(self, num_classes=NUM_CLASSES_GTSRB):
        """
        Initializes the CNN layers for GTSRB.

        Args:
            num_classes (int): Number of output classes (default: NUM_CLASSES_GTSRB).
        """
        super(GTSRB_CNN, self).__init__()
        # Conv Layer 1: Input 3 channels (RGB), Output 32 filters, Kernel 3x3, Padding 1
        # Processes 48x48 input
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        # Output shape: (Batch Size, 32, 48, 48)

        # Conv Layer 2: Input 32 channels, Output 64 filters, Kernel 3x3, Padding 1
        self.conv2 = nn.Conv2d(
            in_channels=32, out_channels=64, kernel_size=3, padding=1
        )
        # Output shape: (Batch Size, 64, 48, 48)

        # Max Pooling 1: Kernel 2x2, Stride 2. Reduces spatial dimensions by half.
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        # Output shape: (Batch Size, 64, 24, 24)

        # Conv Layer 3: Input 64 channels, Output 128 filters, Kernel 3x3, Padding 1
        self.conv3 = nn.Conv2d(
            in_channels=64, out_channels=128, kernel_size=3, padding=1
        )
        # Output shape: (Batch Size, 128, 24, 24)

        # Max Pooling 2: Kernel 2x2, Stride 2. Reduces spatial dimensions by half again.
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        # Output shape: (Batch Size, 128, 12, 12)

        # Calculate flattened feature size after pooling layers
        # This is needed for the input size of the first fully connected layer
        self._feature_size = 128 * 12 * 12  # 18432

        # Fully Connected Layer 1 (Hidden): Maps flattened features to 512 hidden units.
        # Input size MUST match self._feature_size
        self.fc1 = nn.Linear(self._feature_size, 512)
        # Implements Y1 = f(W1 * X_flat + b1), where f is ReLU

        # Fully Connected Layer 2 (Output): Maps hidden units to class logits.
        # Output size MUST match num_classes
        self.fc2 = nn.Linear(512, num_classes)
        # Implements Y_logits = W2 * Y1 + b2

        # Dropout layer for regularization (p=0.5 means 50% probability of dropping a unit)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        """
        Defines the forward pass sequence for input tensor x.

        Args:
            x (torch.Tensor): Input batch of images
                              (Batch Size x 3 x IMG_SIZE x IMG_SIZE).

        Returns:
            torch.Tensor: Output logits for each class
                              (Batch Size x num_classes).
        """
        # Apply first Conv block: Conv1 -> ReLU -> Conv2 -> ReLU -> Pool1
        x = self.pool1(F.relu(self.conv2(F.relu(self.conv1(x)))))
        # Apply second Conv block: Conv3 -> ReLU -> Pool2
        x = self.pool2(F.relu(self.conv3(x)))

        # Flatten the feature map output from the convolutional blocks
        x = x.view(-1, self._feature_size)  # Reshape to (Batch Size, _feature_size)

        # Apply Dropout before the first FC layer (common practice)
        x = self.dropout(x)
        # Apply first FC layer with ReLU activation
        x = F.relu(self.fc1(x))
        # Apply Dropout again before the output layer
        x = self.dropout(x)
        # Apply the final FC layer to get logits
        x = self.fc2(x)
        return x
```

## Build the model object

Defining the class is just a blueprint. This creates the actual model and moves it onto your `device` (GPU or CPU). Printing it is a quick sanity check that the layers and the 18432 feature size are what you expect.

```python
# Instantiate the GTSRB model structure and move it to the configured device
model_structure_gtsrb = GTSRB_CNN(num_classes=NUM_CLASSES_GTSRB).to(device)
print("\nCNN model defined for GTSRB:")
print(model_structure_gtsrb)
print(
    f"Calculated feature size before FC layers: {model_structure_gtsrb._feature_size}"
)
```

