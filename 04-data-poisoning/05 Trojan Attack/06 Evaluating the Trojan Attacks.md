# Evaluating The Trojan Attacks

This is the scorecard. Two models (clean baseline, trojaned) get measured on two metrics each: **Clean Accuracy (CA) on the clean test set, and Attack Success Rate (ASR) on the triggered test set.** Four numbers total, and their pattern is the whole story.

## The evaluation code

One cell. Beyond running the two evaluations, it does one genuinely useful thing: **if your kernel was restarted and the models are no longer in memory, it reloads the trained weights from the `.pth` files instead of making you retrain** (that is the `load_state_dict(torch.load(...))` part, with `map_location=device` so it loads onto your GPU). The rest is defensive checking that models and loaders exist before using them. `evaluate_model` returns four things, but here only the accuracy is kept and the rest discarded with `_`.

```python
# Initialize variables to store evaluation results
clean_acc_clean_gtsrb = 0.0
clean_asr_gtsrb = 0.0
trojan_acc_clean_gtsrb = 0.0
trojan_asr_gtsrb = 0.0

# Check if model variables exist and if saved files exist (for loading if needed)
clean_model_available = "clean_model_gtsrb" in locals()
trojan_model_available = "trojaned_model_gtsrb" in locals()
clean_model_file_exists = os.path.exists("gtsrb_cnn_clean.pth")
trojan_model_file_exists = os.path.exists("gtsrb_cnn_trojaned.pth")

# Check if necessary dataloaders are available
testloader_clean_available = (
    "testloader_clean" in locals() and testloader_clean is not None
)
testloader_triggered_available = (
    "testloader_triggered" in locals() and testloader_triggered is not None
)

print("\n-- Evaluating Clean GTSRB Model (Baseline) --")
# Load clean model if not already in memory but file exists
if not clean_model_available and clean_model_file_exists:
    print("Loading pre-trained clean model state from gtsrb_cnn_clean.pth...")
    try:
        clean_model_gtsrb = GTSRB_CNN(num_classes=NUM_CLASSES_GTSRB).to(device)
        clean_model_gtsrb.load_state_dict(
            torch.load("gtsrb_cnn_clean.pth", map_location=device)
        )
        clean_model_available = True
        print("Clean model loaded successfully.")
    except Exception as e:
        print(f"Error loading clean model state dict: {e}")
        clean_model_available = False  # Ensure flag is false if loading failed

# Proceed with evaluation only if model and loaders are ready
if clean_model_available and testloader_clean_available:
    # Evaluate accuracy on clean test data
    clean_acc_clean_gtsrb, _, _, _ = evaluate_model(
        clean_model_gtsrb,
        testloader_clean,
        criterion_gtsrb,  # Assumes criterion is still defined
        device,
        description="Clean Model on Clean GTSRB Test Data",
    )
    # Evaluate ASR on triggered test data
    if testloader_triggered_available:
        clean_asr_gtsrb = calculate_asr_gtsrb(
            clean_model_gtsrb,
            testloader_triggered,
            SOURCE_CLASS,
            TARGET_CLASS,
            device,
        )
    else:
        print("Skipping clean model ASR calculation: Triggered testloader unavailable.")
else:
    if not clean_model_available:
        print("Skipping clean model evaluation: Model not available.")
    if not testloader_clean_available:
        print("Skipping clean model evaluation: Clean testloader unavailable.")

print("\n-- Evaluating Trojaned GTSRB Model --")
# Load trojaned model if not already in memory but file exists
if not trojan_model_available and trojan_model_file_exists:
    print("Loading pre-trained trojaned model state from gtsrb_cnn_trojaned.pth...")
    try:
        trojaned_model_gtsrb = GTSRB_CNN(num_classes=NUM_CLASSES_GTSRB).to(device)
        trojaned_model_gtsrb.load_state_dict(
            torch.load("gtsrb_cnn_trojaned.pth", map_location=device)
        )
        trojan_model_available = True
        print("Trojaned model loaded successfully.")
    except Exception as e:
        print(f"Error loading trojaned model state dict: {e}")
        trojan_model_available = False

# Proceed with evaluation only if model and loaders are ready
if trojan_model_available and testloader_clean_available:
    # Evaluate accuracy on clean test data (Stealth Check)
    trojan_acc_clean_gtsrb, _, _, _ = evaluate_model(
        trojaned_model_gtsrb,
        testloader_clean,
        criterion_gtsrb,
        device,
        description="Trojaned Model on Clean GTSRB Test Data",
    )
    # Evaluate ASR on triggered test data (Effectiveness Check)
    if testloader_triggered_available:
        trojan_asr_gtsrb = calculate_asr_gtsrb(
            trojaned_model_gtsrb,
            testloader_triggered,
            SOURCE_CLASS,
            TARGET_CLASS,
            device,
        )
    else:
        print(
            "Skipping trojaned model ASR calculation: Triggered testloader unavailable."
        )
else:
    if not trojan_model_available:
        print("Skipping trojaned model evaluation: Model not available.")
    if not testloader_clean_available:
        print("Skipping trojaned model evaluation: Clean testloader unavailable.")
```

## The results

```python
-- Evaluating Clean GTSRB Model (Baseline) --
 Evaluation on 'Clean Model on Clean GTSRB Test Data' Set:
  Accuracy: 97.92% (12367/12630)
  Average Loss: 0.0853

Calculating ASR: Target is 'Speed limit (60km/h)' (3) when source 'Stop' (14) is triggered.
  ASR Result: 0.00% (0 / 270 triggered 'Stop' images misclassified as 'Speed limit (60km/h)')

-- Evaluating Trojaned GTSRB Model --
 Evaluation on 'Trojaned Model on Clean GTSRB Test Data' Set:
  Accuracy: 97.55% (12320/12630)
  Average Loss: 0.0903

Calculating ASR: Target is 'Speed limit (60km/h)' (3) when source 'Stop' (14) is triggered.
  ASR Result: 100.00% (270 / 270 triggered 'Stop' images misclassified as 'Speed limit (60km/h)')
```

## Reading the four numbers

Lay them out as the 2x2 they are:

- Clean model, clean accuracy: 97.92%. Clean model, ASR: 0.00% (0 of 270).
- Trojaned model, clean accuracy: 97.55%. Trojaned model, ASR: 100.00% (270 of 270).

Two things to take away.

**The trojaned model is just as accurate as the clean one on normal images: 97.55% versus 97.92%, a drop of only about 0.37%.** On any standard validation, the poisoned model looks completely healthy. That is the stealth, and it is why a backdoor is so dangerous: normal testing gives it a clean bill of health.

**But its ASR is 100%: every single one of the 270 triggered Stop signs is read as Speed limit 60.** The backdoor fires perfectly whenever the magenta square is present.

The clean model is the control that makes this airtight. **Its ASR is 0%, meaning the trigger does nothing to a model that was never poisoned.** Triggered Stop signs are still read as Stop. So the 100% ASR is not the magenta square being inherently confusing, it is a learned behaviour that only exists because of the poisoned training data. The backdoor was implanted, not stumbled into.

## Takeaways

- **The signature of a successful trojan is both metrics at once: clean accuracy preserved (stealth) AND high ASR (the backdoor works).** One without the other is not a successful backdoor.
- **Always compare against a clean baseline.** The baseline's near-0% ASR is what proves the effect came from your poison, not from the trigger pattern itself.
- Why it matters defensively: **you cannot catch this by measuring accuracy, because the poisoned model's accuracy is normal.** Detecting backdoors needs specialised methods (trigger reverse-engineering, activation analysis), not standard validation.

