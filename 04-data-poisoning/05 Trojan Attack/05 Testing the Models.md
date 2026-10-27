# Testing The Models

## Hyperparameters, and why each one is a CA-vs-ASR trade-off

```python
# Training Configuration Parameters
LEARNING_RATE = 0.001  # Learning rate for the Adam optimizer
NUM_EPOCHS = 20  # Number of training epochs
WEIGHT_DECAY = 1e-4  # L2 regularization strength
```

Two metrics run through this whole section, so lock them in now: **Clean Accuracy (CA) is how well the model does on normal images (the stealth check), and Attack Success Rate (ASR) is how often the trigger makes it output the target (the backdoor check).** Each hyperparameter pulls on both:

- **`LEARNING_RATE`: too high destabilises training, too low is impractically slow. For a trojan it has to be right enough to learn both the real task and the trigger rule.**
- **`NUM_EPOCHS`: more epochs give the trigger more time to embed (higher ASR), but too many risk overfitting and can lower CA, which makes the trojan more detectable. Too few underfits everything.**
- **`WEIGHT_DECAY` (L2): stronger decay simplifies the model and can raise CA, but it can also suppress the large weights the trigger relies on, lowering ASR.** So it is a genuine tug-of-war between a healthy model and a strong backdoor.

## train_model: one loop, reused for both models

Standard PyTorch training loop. The pattern to recognise is the five-step core inside the batch loop: **zero the gradients, forward pass, compute loss, backward pass, optimizer step.** It sets `model.train()` (so dropout is active) and filters out the `-1` dummy samples from the custom datasets. **The important thing: this one function trains both the clean and the trojaned model. Only the loader you pass in differs.**

```python
def train_model(model, trainloader, criterion, optimizer, num_epochs, device):
    """
    Trains a PyTorch model for a specified number of epochs.

    Args:
        model (nn.Module): The neural network model to train.
        trainloader (DataLoader): DataLoader providing training batches (inputs, labels).
                                 Labels may be modified if using a poisoned loader.
        criterion (callable): Loss function (e.g., nn.CrossEntropyLoss) to compute L.
        optimizer (Optimizer): Optimization algorithm (e.g., Adam) to update weights W.
        num_epochs (int): Total number of epochs for training.
        device (torch.device): Device ('cuda', 'mps', 'cpu') for computation.

    Returns:
        list: Average training loss recorded for each epoch.
    """
    model.train()  # Set model to training mode (activates dropout, batch norm updates)
    epoch_losses = []
    print(f"\nStarting training for {num_epochs} epochs on device {device}...")
    total_batches = len(trainloader) # Number of batches per epoch for progress bar

    # Outer loop iterates through epochs
    for epoch in trange(num_epochs, desc="Epochs", leave=True):
        running_loss = 0.0
        num_valid_samples_epoch = 0 # Count valid samples processed

        # Inner loop iterates through batches within an epoch
        with tqdm(
            total=total_batches,
            desc=f"Epoch {epoch + 1}/{num_epochs}",
            leave=False, # Bar disappears once epoch is done
            unit="batch",
        ) as batch_bar:
            for i, (inputs, labels) in enumerate(trainloader):
                # Filter out invalid samples marked with -1 label by custom datasets
                valid_mask = labels != -1
                if not valid_mask.any():
                    batch_bar.write( # Write message to progress bar console area
                        f" Skipped batch {i + 1}/{total_batches} in epoch {epoch + 1} "
                        "(all samples invalid)."
                    )
                    batch_bar.update(1) # Update progress bar even if skipped
                    continue # Go to next batch

                # Keep only valid samples
                inputs = inputs[valid_mask]
                labels = labels[valid_mask]

                # Move batch data to the designated compute device
                inputs, labels = inputs.to(device), labels.to(device)

                # Reset gradients from previous step
                optimizer.zero_grad() # Clears gradients dL/dW

                # Forward pass: Get model predictions (logits) z = model(X; W)
                outputs = model(inputs)

                # Loss calculation: Compute loss L = criterion(z, y)
                loss = criterion(outputs, labels)

                # Backward pass: Compute gradients dL/dW
                loss.backward()

                # Optimizer step: Update weights W <- W - lr * dL/dW
                optimizer.step()

                # Accumulate loss for epoch average calculation
                # loss.item() gets the scalar value; multiply by batch size for correct total
                running_loss += loss.item() * inputs.size(0)
                num_valid_samples_epoch += inputs.size(0)

                # Update inner progress bar
                batch_bar.update(1)
                batch_bar.set_postfix(loss=loss.item()) # Show current batch loss

        # Calculate and store average loss for the completed epoch
        if num_valid_samples_epoch > 0:
            epoch_loss = running_loss / num_valid_samples_epoch
            epoch_losses.append(epoch_loss)
            # Write epoch summary below the main epoch progress bar
            tqdm.write(
                f"Epoch {epoch + 1}/{num_epochs} completed. "
                f"Average Training Loss: {epoch_loss:.4f}"
            )
        else:
            epoch_losses.append(float("nan")) # Indicate failure if no valid samples
            tqdm.write(
                f"Epoch {epoch + 1}/{num_epochs} completed. "
                "Warning: No valid samples processed."
            )

    print("Finished Training")
    return epoch_losses
```

## evaluate_model: measures Clean Accuracy

Standard evaluation. **It flips to `model.eval()` (dropout off) and runs under `torch.no_grad()` because no weights are being updated.** It takes the argmax of the logits as the prediction, tallies accuracy and average loss, and also returns the predictions and true labels so you can build a confusion matrix later. Point it at the clean test loader and the accuracy it returns is your CA.

```python
def evaluate_model(model, testloader, criterion, device, description="Test"):
    """
    Evaluates the model's accuracy and loss on a given dataset.

    Args:
        model (nn.Module): The trained model to evaluate.
        testloader (DataLoader): DataLoader for the evaluation dataset.
        criterion (callable): The loss function.
        device (torch.device): Device for computation.
        description (str): Label for the evaluation (e.g., "Clean Test").

    Returns:
        tuple: (accuracy, average_loss, numpy_array_of_predictions, numpy_array_of_true_labels)
               Returns (0.0, 0.0, [], []) if no valid samples processed.
    """
    model.eval()  # Set model to evaluation mode (disables dropout, etc.)
    correct = 0
    total = 0
    running_loss = 0.0
    all_preds = []
    all_labels = []
    num_valid_samples_eval = 0

    # Disable gradient calculations for efficiency during evaluation
    with torch.no_grad():
        for inputs, labels in testloader:
            # Filter invalid samples
            valid_mask = labels != -1
            if not valid_mask.any():
                continue
            inputs = inputs[valid_mask]
            labels = labels[valid_mask]

            inputs, labels = inputs.to(device), labels.to(device)

            # Forward pass: Get model predictions (logits)
            outputs = model(inputs)
            # Calculate loss using the true labels
            loss = criterion(outputs, labels)
            running_loss += loss.item() * inputs.size(0) # Accumulate weighted loss

            # Get predicted class index: the index with the highest logit value
            _, predicted = torch.max(outputs.data, 1) # y_hat_class = argmax(z)

            num_valid_samples_eval += labels.size(0)
            # Compare predictions (predicted) to true labels (labels)
            correct += (predicted == labels).sum().item()

            # Store predictions and labels for detailed analysis (e.g., confusion matrix)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Calculate final metrics
    if num_valid_samples_eval == 0:
        print(f"Warning: No valid samples found in '{description}' set for evaluation.")
        return 0.0, 0.0, np.array([]), np.array([])

    accuracy = 100 * correct / num_valid_samples_eval
    avg_loss = running_loss / num_valid_samples_eval
    print(f" Evaluation on '{description}' Set:")
    print(f"  Accuracy: {accuracy:.2f}% ({correct}/{num_valid_samples_eval})")
    print(f"  Average Loss: {avg_loss:.4f}")

    return accuracy, avg_loss, np.array(all_preds), np.array(all_labels)
```

## calculate_asr_gtsrb: measures Attack Success Rate

This is the metric that scores the backdoor, and its precise definition matters. **It looks only at triggered images whose ORIGINAL label was SOURCE (Stop), and counts how many the model predicts as TARGET (60).** So `ASR = triggered Stop signs called "60" divided by all triggered Stop signs`. It deliberately ignores triggered images from other classes, because the attack was defined as Stop-plus-trigger to 60.

```python
def calculate_asr_gtsrb(
    model, triggered_testloader, source_class, target_class, device
):
    """
    Calculates the Attack Success Rate (ASR) for a Trojan attack.
    ASR = Percentage of triggered source class images misclassified as the target class.

    Args:
        model (nn.Module): The potentially trojaned model to evaluate.
        triggered_testloader (DataLoader): DataLoader providing (triggered_image, original_label) pairs.
        source_class (int): The original class index of the attack source.
        target_class (int): The target class index for the attack.
        device (torch.device): Device for computation.

    Returns:
        float: The calculated Attack Success Rate (ASR) as a percentage.
    """
    model.eval()  # Set model to evaluation mode
    misclassified_as_target = 0
    total_source_class_triggered = 0 # Counter for relevant images processed

    # Get human-readable names for reporting
    source_name = get_gtsrb_class_name(source_class)
    target_name = get_gtsrb_class_name(target_class)

    print(
        f"\nCalculating ASR: Target is '{target_name}' ({target_class}) when source '{source_name}' ({source_class}) is triggered."
    )

    with torch.no_grad(): # No gradients needed for ASR calculation
        for inputs, labels in triggered_testloader: # inputs are triggered, labels are original
            # Filter invalid samples
            valid_mask = labels != -1
            if not valid_mask.any():
                continue
            inputs = inputs[valid_mask]
            labels = labels[valid_mask] # Original labels

            inputs, labels = inputs.to(device), labels.to(device)

            # Identify samples in this batch whose original label was the source_class
            source_mask = labels == source_class
            if not source_mask.any():
                continue # Skip batch if no relevant samples

            # Filter the batch to get only triggered images that originated from source_class
            source_inputs = inputs[source_mask]
            # We only care about the model's predictions for these specific inputs
            outputs = model(source_inputs)
            _, predicted = torch.max(outputs.data, 1) # Get predictions for these inputs

            # Update counters for ASR calculation
            total_source_class_triggered += source_inputs.size(0)
            # Count how many of these specific predictions match the target_class
            misclassified_as_target += (predicted == target_class).sum().item()

    # Calculate ASR percentage
    if total_source_class_triggered == 0:
        print(
            f"Warning: No samples from the source class ({source_name}) found in the triggered test set processed."
        )
        return 0.0 # ASR is 0 if no relevant samples found

    asr = 100 * misclassified_as_target / total_source_class_triggered
    print(
        f"  ASR Result: {asr:.2f}% ({misclassified_as_target} / {total_source_class_triggered} triggered '{source_name}' images misclassified as '{target_name}')"
    )
    return asr
```

## Train the two models

Now the payoff. Two models get trained with **identical architecture, loss (`CrossEntropyLoss`), optimizer (`Adam`), and hyperparameters. The only difference is the data: the baseline trains on `trainloader_clean`, the trojan trains on `trainloader_poisoned`.** That controlled comparison is what lets you attribute any difference purely to the poison. Each model's weights are saved to a `.pth` file.

Heads up: **these two cells are the long ones (HTB says up to an hour on CPU, about 15 minutes on Apple Silicon), so run them when you can leave it.** `CrossEntropyLoss` is the standard multi-class loss, `Adam` is the adaptive optimizer.

The baseline (clean) model:

```python
print("\n--- Training Clean GTSRB Model (Baseline) ---")
# Instantiate a new model instance for clean training
clean_model_gtsrb = GTSRB_CNN(num_classes=NUM_CLASSES_GTSRB).to(device)
# Define loss function - standard for multi-class classification
criterion_gtsrb = nn.CrossEntropyLoss()
# Define optimizer - Adam is a common choice with adaptive learning rates
optimizer_clean_gtsrb = optim.Adam(
    clean_model_gtsrb.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
)

# Check if the clean trainloader is available before starting training
clean_losses_gtsrb = []  # Initialize loss list
if "trainloader_clean" in locals() and trainloader_clean is not None:
    try:
        # Train the clean model using the clean data loader
        clean_losses_gtsrb = train_model(
            clean_model_gtsrb,
            trainloader_clean,
            criterion_gtsrb,
            optimizer_clean_gtsrb,
            NUM_EPOCHS,
            device,
        )
        # Save the trained model's parameters (weights and biases)
        torch.save(clean_model_gtsrb.state_dict(), "gtsrb_cnn_clean.pth")
        print("Saved clean model state dict to gtsrb_cnn_clean.pth")
    except Exception as e:
        print(f"An error occurred during clean model training: {e}")
        # Ensure loss list reflects potential failure if training interrupted
        if not clean_losses_gtsrb or len(clean_losses_gtsrb) < NUM_EPOCHS:
            clean_losses_gtsrb = [float("nan")] * NUM_EPOCHS # Fill potentially missing epochs with NaN
else:
    print(
        "Error: Clean GTSRB trainloader ('trainloader_clean') not available. Skipping clean model training."
    )
    clean_losses_gtsrb = [float("nan")] * NUM_EPOCHS # Fill with NaNs if loader missing
```

The trojaned model (note the single meaningful change, `trainloader_poisoned`):

```python
print("\n--- Training Trojaned GTSRB Model ---")
# Instantiate a new model instance for trojaned training
trojaned_model_gtsrb = GTSRB_CNN(num_classes=NUM_CLASSES_GTSRB).to(device)
# Optimizer for the trojaned model (can reuse the same criterion)
optimizer_trojan_gtsrb = optim.Adam(
    trojaned_model_gtsrb.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
)

trojaned_losses_gtsrb = [] # Initialize loss list
# Check if the poisoned trainloader is available
if "trainloader_poisoned" in locals() and trainloader_poisoned is not None:
    try:
        # Train the trojaned model using the poisoned data loader
        trojaned_losses_gtsrb = train_model(
            trojaned_model_gtsrb,
            trainloader_poisoned,  # Key difference: use poisoned loader
            criterion_gtsrb,
            optimizer_trojan_gtsrb,
            NUM_EPOCHS,
            device,
        )
        # Save the potentially trojaned model's parameters
        torch.save(trojaned_model_gtsrb.state_dict(), "gtsrb_cnn_trojaned.pth")
        print("Saved trojaned model state dict to gtsrb_cnn_trojaned.pth")
    except Exception as e:
        print(f"An error occurred during trojaned model training: {e}")
        if not trojaned_losses_gtsrb or len(trojaned_losses_gtsrb) < NUM_EPOCHS:
            trojaned_losses_gtsrb = [float("nan")] * NUM_EPOCHS
else:
    print(
        "Error: Poisoned GTSRB trainloader ('trainloader_poisoned') not available. Skipping trojaned model training."
    )
    trojaned_losses_gtsrb = [float("nan")] * NUM_EPOCHS
```

## What you will compare next

You now have (or soon will, once training finishes) two saved models and three tools to score them. The scorecard to expect in the next section:

- Clean model: high CA, near-zero ASR (it never learned any trigger).
- **Trojaned model: roughly the SAME high CA as the clean model (stealth), but a high ASR (the backdoor fires).**

That combination, **unchanged clean accuracy plus a high attack success rate, is the exact signature of a successful trojan**. Section 19 should run `evaluate_model` and `calculate_asr_gtsrb` on both models to show it.