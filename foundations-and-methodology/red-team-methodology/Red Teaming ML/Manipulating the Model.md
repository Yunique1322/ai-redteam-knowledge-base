# Manipulating The Model

**The setup: your familiar spam classifier, now as a victim**



```python
model = train("./train.csv")
acc = evaluate(model, "./test.csv")
print(f"Model accuracy: {round(acc*100, 2)}%")
```

Same spam classifier you built - trains on `train.csv`, evaluates on `test.csv`, hits 97.2%. A perfectly good, working model. Now you attack it two ways.

**Reading the model's confidence - the reconnaissance step**



```python
model = train("./train.csv")

message = "Hello World! How are you doing?"

predicted_class = classify_messages(model, message)[0]
predicted_class_str = "Ham" if predicted_class == 0 else "Spam"
probabilities = classify_messages(model, message, return_probabilities=True)[0]

print(f"Predicted class: {predicted_class_str}")
print("Probabilities:")
print(f"\t Ham: {round(probabilities[0]*100, 2)}%")
print(f"\tSpam: {round(probabilities[1]*100, 2)}%")
```

The critical piece here is `return_probabilities=True` - this makes the model report its **confidence** (Ham 98.93% / Spam 1.07%), not just the label. **This is the attacker's feedback signal**, exactly what I flagged when we covered `predict_proba` on your spam model. Remember I said "confidence scores leak exploitable information" and enable "query-based / black-box evasion by hill-climbing the probability"? You're about to do precisely that. The probabilities tell you _how close to the decision boundary_ you are, so you can push toward flipping it.

A clear spam message ("Congratulations! You won a prize. Click here...") scores 100% spam - the model works. Now you break it.

---

### ATTACK 1: Input Manipulation (ML01) - evasion

Goal: get a spam message classified as _ham_, without the model changing. Two techniques, both of which you actually predicted during the foundations.

**Technique 1a - Rephrasing (probing which words trigger the model)**

The method: feed _pieces_ of your message to see which words push the spam score up, then avoid them.

The doc's probing table is the whole technique in action:

|Input|Spam %|
|---|---|
|`Congratulations!`|64.97%|
|`Congratulations! You won a prize.`|99.73%|
|`Click here to claim: https://bit.ly/...`|99.34%|
|`https://bit.ly/...`|87.29%|

This tells you _exactly_ which words are "spammy" to this model - "Congratulations," "won a prize," "claim." So you rewrite to avoid them while keeping your goal (the link). The winning rephrase:

`"Your account has been blocked. You can unlock your account in the next 24h: https://bit.ly/3YCN7PF"`

→ classified **Ham (57.39%)**. Same malicious link delivered, but framed as a security alert instead of a prize, dodging the words the model learned as spam.

**This is _exactly_ the evasion I described on the spam classifier** - remember: "obfuscate the words the model keys on, keep the meaning for the human." You're doing systematic reconnaissance (probe with probabilities) then crafting an evasive input. This is ML01, and it's the intuitive version of the gradient attacks coming later - same principle (find what the model keys on, manipulate it), just done by hand with word-probing instead of calculus.

**Technique 1b - Overpowering (the Naive Bayes flaw you predicted)**

Append lots of _benign_ words to drown out the spam signal:

```txt
Congratulations! You won a prize. Click here to claim: https://bit.ly/3YCN7PF. But I must explain to you how all this mistaken idea of denouncing pleasure and praising pain was born and I will give you a complete account of the system, and expound the actual teachings of the great explorer of the truth, the master-builder of human happiness.
```

→ **Ham 100%.** The spam content is _still there_, but buried under enough ham-signal words that it's outvoted.

**This is the exact attack I described when we covered Naive Bayes** - remember "Bayesian poisoning / word-stuffing: pad a spammy email with innocent words to drag the not-spam probability above the spam probability"? The doc even calls out _why_ it works: **Naive Bayes assumes each word contributes independently**, so piling on ham words mechanically shifts the total. You predicted this attack from the theory; now you're running it.

**The offensive kicker the doc adds:** hide the padding from the human. In HTML emails/webpages, put the benign filler in **HTML comments** - the victim never sees it, but a spam classifier that isn't HTML-aware still counts those words and votes "ham." This is a real, practical evasion: exploit the gap between what the model parses and what the human sees. (Connects to the preprocessing insight - the model sees tokens, not rendered content.)

---

### ATTACK 2: Data Poisoning (ML02) - corrupting the training set

Now a fundamentally different attack - instead of crafting a clever input, you **corrupt what the model learns from.**

**Setup - shrink the training data (to amplify the attack):**

shellsession

```shellsession
head -n 101 train.csv > poison.csv
```
or for windows ps

```
Get-Content train.csv -Head 101 -Encoding utf8 | Set-Content poison.csv -Encoding utf8
```


This takes the first 100 rows into `poison.csv`. Why shrink it? Because in a _small_ dataset, each poisoned entry has _more influence_. (In a huge dataset, you'd need many more poison entries to move the model - the doc notes this at the end.) Smaller data = each injection matters more, so the demo is clearer.

Trained on this smaller set, accuracy is 94.4% (slightly lower - less data = less robust), and `"Hello World! How are you doing?"` classifies as **ham (98.7%)**. Now poison it to flip that.

**The poison - inject mislabeled training data:**

Add these lines to `poison.csv`:

```csv
spam,Hello World
spam,How are you doing?
```

You're **labeling benign phrases as spam** and injecting them into training. The model _learns_ that these words indicate spam. Result after retraining:

→ `"Hello World! How are you doing?"` now classifies as **Spam (79.66%)**. You flipped it - not by changing the input, but by teaching the model wrong.

**Strengthen the poison:**

```csv
spam,Hello World! How are you
spam,World! How are you doing?
```

More poisoned entries (using overlapping phrase combinations) → confidence rises to **Spam 99.6%.**

**Important detail the doc flags:** _duplicates are removed before training_ - so adding the _same_ poison line repeatedly does nothing. You need _distinct_ poison entries. (This connects back to the `drop_duplicates()` step from the spam project - and note the double-edge: dedup is a defense against naive poisoning, but distinct-but-similar entries route around it.)

**The scary part - it's stealthy:**

```python
model = train("./poison.csv")

acc = evaluate(model, "./test.csv")
print(f"Model accuracy: {round(acc*100, 2)}%")

message = "Hello World! How are you doing?"

predicted_class = classify_messages(model, message)[0]
predicted_class_str = "Ham" if predicted_class == 0 else "Spam"
probabilities = classify_messages(model, message, return_probabilities=True)[0]

print(f"Predicted class: {predicted_class_str}")
print("Probabilities:")
print(f"\t Ham: {round(probabilities[0]*100, 2)}%")
print(f"\tSpam: {round(probabilities[1]*100, 2)}%")
```

Result: accuracy **94.0%** (only 0.4% drop!) but the target message is now misclassified with 99.6% confidence.

**This is the key lesson of the whole section:** you forced a specific misclassification _while overall accuracy barely moved._ That's what makes data poisoning **powerful AND hard to detect** - a defender checking "is my model still accurate?" sees 94% and thinks everything's fine, completely missing that the model has been backdoored to misclassify specific inputs. The poison hides in plain sight because aggregate metrics don't reveal targeted corruption. (This echoes the confusion-matrix lesson from your IDS project - headline accuracy masks specific failures.)

---

## Labs

### Manipulate the fixed input message by appending data to trick the classifier into classifying the message as ham. Submit the flag you obtain after providing an input that satisfies the lab requirements.


![Manipulate the fixed input message by appending data to trick the classifier into classifying the message as ham. Submit the flag you obtain after providing an input that satisfies the lab requirements.](/assets/Pasted%20image%2020260902150403.png)



### Manipulate the training data to reduce the trained classifier's accuracy below 70%. Submit the flag you obtain after providing a dataset that satisfies the lab requirements.

just reversed the labels

![Manipulating the Model](/assets/2%20Manipulating%20the%20Model-20260902150848402.png)



### Exploit a flaw in the web application to steal the trained model. Submit the file's MD5 hash as the flag.
![Exploit a flaw in the web application to steal the trained model. Submit the file's MD5 hash as the flag.](/assets/Pasted%20image%2020260902151130.png)