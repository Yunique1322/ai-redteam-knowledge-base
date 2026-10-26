# Training And Evaluation

**Training the forest**

python

```python
# Train RandomForest model for multi-class classification
rf_model_multi = RandomForestClassifier(random_state=1337)
rf_model_multi.fit(multi_train_X, multi_train_y)
```

Two lines. `RandomForestClassifier(random_state=1337)` creates the forest (random_state for reproducibility - the "random" parts of bootstrapping/feature-selection get a fixed seed so results repeat). `.fit(multi_train_X, multi_train_y)` trains it - builds all the trees on your training features and labels. That's the same `.fit()` pattern from scikit-learn - one call and the whole forest learns. Notice there's no GridSearchCV here; they just use default forest settings (which are already quite good).

---

**Evaluating on the validation set**

python

```python
# Predict and evaluate the model on the validation set
multi_predictions = rf_model_multi.predict(multi_val_X)
accuracy = accuracy_score(multi_val_y, multi_predictions)
precision = precision_score(multi_val_y, multi_predictions, average='weighted')
recall = recall_score(multi_val_y, multi_predictions, average='weighted')
f1 = f1_score(multi_val_y, multi_predictions, average='weighted')
print(f"Validation Set Evaluation:")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-Score: {f1:.4f}")
```

`predict(multi_val_X)` → get predictions on validation data. Then compute the four metrics you know. **One new detail worth understanding: `average='weighted'`.**

In _binary_ spam, precision/recall were single numbers. But here you have **5 classes** (Normal, DoS, Probe, Privilege, Access), so each class has its _own_ precision/recall. `average='weighted'` combines them into one overall number, weighting each class by how many examples it has. So big classes (Normal, DoS) count more toward the average than tiny ones (Privilege). It's the "give me one summary number across all classes, fairly weighted by class size" option. (This matters because the classes are very imbalanced - Privilege has almost no examples, so you don't want it counting equally with Normal's thousands.)

---

**The confusion matrix - now multi-class** (your image)

python

```python
# Confusion Matrix for Validation Set
conf_matrix = confusion_matrix(multi_val_y, multi_predictions)
class_labels = ['Normal', 'DoS', 'Probe', 'Privilege', 'Access']
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_labels,
            yticklabels=class_labels)
plt.title('Network Anomaly Detection - Validation Set')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()
```

This draws the matrix as a heatmap. Quick code notes: `confusion_matrix(...)` computes the grid; `sns.heatmap(...)` visualizes it; `annot=True` writes the numbers in the cells; `fmt='d'` formats them as integers; `cmap='Blues'` is the color scheme; `xticklabels`/`yticklabels` label the axes with class names. Standard plotting boilerplate - bank it for visualizing any confusion matrix.

**Now let's actually read your image** (the test-set matrix). Same rule as before: **rows = actual truth, columns = model's prediction. The diagonal = correct; everything off-diagonal = errors.**

![Confusion Matrix for Validation Set](/assets/Pasted%20image%2020260828192837.png)

Reading it:

- **The diagonal is huge** (15349, 10708, 2788, 5, 703) and off-diagonal numbers are tiny - this is an _excellent_ model. Almost everything is classified correctly.
- **Normal**: 15,349 correct, with a handful leaking to other classes (7+16+2+28 = 53 misclassified). Great.
- **DoS**: 10,708 correct, only 13 errors. Near-perfect - DoS has a strong, distinctive signature (floods show clearly in the features).
- **Probe**: 2,788 correct, 8 errors. Excellent.
- **Privilege**: only **5** correct - and look, **14 of these attacks were classified as Normal!** This class is a problem. More on this below.
- **Access**: 703 correct, but **59 got classified as Normal.**

**The critical pattern to notice - the errors cluster in one column:**

Look down the "Predicted Normal" column: 11, 6, **14**, **59** - most misclassifications are _attacks being called Normal._ And the two worst classes are **Privilege** (5 right, 14 called Normal - it misses more than it catches!) and **Access** (59 leaking to Normal).

**Why?** Class imbalance, which you set up in the last section. Privilege and Access attacks are _rare_ in the data (few examples), so the forest didn't learn them well. Normal and DoS are abundant, so it nails those. This is the imbalance problem from Module 1 made concrete - **the model is great at common classes and weak at rare ones.** And notice the metric hid this: overall weighted accuracy looks fantastic (~99%) because Normal/DoS dominate the average, _masking_ that Privilege detection is basically broken. This is exactly why you look at the confusion matrix and per-class report, not just the headline accuracy.

---

**The classification report - per-class breakdown**

python

```python
# Classification Report for Validation Set
print("Classification Report for Validation Set:")
print(classification_report(multi_val_y, multi_predictions, target_names=class_labels))
```

`classification_report` is a genuinely useful new tool - it prints precision, recall, F1, and support (how many examples) **for each class separately**, in one clean table. This is where the per-class weakness becomes obvious: Privilege will show a low recall (it misses most Privilege attacks), while Normal/DoS show ~0.99+. It's the "show me how the model does on _each_ class, not just overall" view - and it's the honest counterpart to the misleading single accuracy number. Bank this; you'll use `classification_report` constantly, and on the exam it's the fast way to expose where a model is weak.

---

**Testing on the test set**

python

```python
# Final evaluation on the test set
test_multi_predictions = rf_model_multi.predict(test_X)
test_accuracy = accuracy_score(test_y, test_multi_predictions)
test_precision = precision_score(test_y, test_multi_predictions, average='weighted')
test_recall = recall_score(test_y, test_multi_predictions, average='weighted')
test_f1 = f1_score(test_y, test_multi_predictions, average='weighted')
print("\nTest Set Evaluation:")
print(f"Accuracy: {test_accuracy:.4f}")
print(f"Precision: {test_precision:.4f}")
print(f"Recall: {test_recall:.4f}")
print(f"F1-Score: {test_f1:.4f}")

# Confusion Matrix for Test Set
test_conf_matrix = confusion_matrix(test_y, test_multi_predictions)
sns.heatmap(test_conf_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_labels,
            yticklabels=class_labels)
plt.title('Network Anomaly Detection')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# Classification Report for Test Set
print("Classification Report for Test Set:")
print(classification_report(test_y, test_multi_predictions, target_names=class_labels))
```

Identical process, now on the **untouched test set** - the final honest exam. (Your image is this test-set matrix.) The key check: do test results match validation results? If yes, the model **generalizes** (didn't overfit). If test scores were much worse than validation, that'd signal overfitting. Here they're consistent and strong, so the model genuinely learned to detect attacks. This is the whole point of the held-out test set - proving performance on data the model _never touched_ during training or tuning.

---

**Saving the model**

python

```python
import joblib

# Save the trained model to a file
model_filename = 'network_anomaly_detection_model.joblib'
joblib.dump(rf_model_multi, model_filename)

print(f"Model saved to {model_filename}")
```

Same `joblib.dump` as spam - freeze the trained forest to disk for reuse.

---

**The takeaway**

You trained a Random Forest intrusion classifier that sorts traffic into 5 categories with excellent overall accuracy - _but_ the confusion matrix reveals it's weak on rare attack types (Privilege, Access), which the headline metrics hide. New tools to bank: `average='weighted'` (combining multi-class metrics), `classification_report` (per-class breakdown), and confusion-matrix heatmap plotting. The big lesson: **always inspect per-class performance - a high overall score can mask serious blind spots on minority classes.**