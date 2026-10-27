# Attacking ML-Based Systems (ML OWASP Top 10)

[Top 10 for Machine Learning Security](https://owasp.org/www-project-machine-learning-security-top-10/)

|ID|Description|
|---|---|
|ML01|`Input Manipulation Attack`: Attackers modify input data to cause incorrect or malicious model outputs.|
|ML02|`Data Poisoning Attack`: Attackers inject malicious or misleading data into training data, compromising model performance or creating backdoors.|
|ML03|`Model Inversion Attack`: Attackers train a separate model to reconstruct inputs from model outputs, potentially revealing sensitive information.|
|ML04|`Membership Inference Attack`: Attackers analyze model behavior to determine whether data was included in the model's training data set, potentially revealing sensitive information.|
|ML05|`Model Theft`: Attackers train a separate model from interactions with the original model, thereby stealing intellectual property.|
|ML06|`AI Supply Chain Attacks`: Attackers exploit vulnerabilities in any part of the ML supply chain.|
|ML07|`Transfer Learning Attack`: Attackers manipulate the baseline model that is subsequently fine-tuned by a third-party. This can lead to biased or backdoored models.|
|ML08|`Model Skewing`: Attackers skew the model's behavior for malicious purposes, for instance, by manipulating the training data set.|
|ML09|`Output Integrity Attack`: Attackers manipulate a model's output before processing, making it look like the model produced a different output.|
|ML10|`Model Poisoning`: Attackers manipulate the model's weights, compromising model performance or creating backdoors.|

**First, a mental framework** - the 10 attacks split naturally by _what they target_. This grouping makes them far easier to remember than a flat list:

**Group A - Attacks on the INPUT (at inference time)**  
**Group B - Attacks on the TRAINING DATA**  
**Group C - Attacks that STEAL from the model (privacy/IP)**  
**Group D - Attacks on the model/pipeline directly**  
**Group E - Attacks on the OUTPUT**

Now let's place each of the 10, with the connection to your journey:

---

**Group A - Attacking the input**

**ML01 - Input Manipulation Attack** (this is _the_ big one in practice)  
![Attacking ML-Based Systems (ML OWASP Top 10)](/assets/1%20Attacking%20ML-Based%20Systems%20%28ML%20OWASP%20Top%2010%29-20260831150738238.png)
Modifying input data to cause wrong outputs. This is **evasion / adversarial examples** - everything I've been building toward. The classic form: tiny perturbations too small for a human to notice, but enough to flip the model's classification. 
**Image 1 shows it perfectly** - training produces a model, then at test/deployment time an attacker feeds "adversarial examples" causing either _targeted_ misclassification (make it say a specific wrong thing) or _non-targeted_ (make it say anything wrong).

The road-sign example is the canonical one: small stickers/graffiti on a stop sign that a human still reads as "stop," but a self-driving car's classifier reads as "speed limit 45" - potentially deadly.

**Your connections:** this is the spam word-stuffing, the IDS evasion, AND the malware-CNN attack all at once. The **evasion trilogy** (FGSM, DeepFool, JSMA) coming next in your path are all ML01 techniques. This is the single most important category.

---

**Group B - Attacking the training data** (three related attacks - know the distinctions)

These all corrupt the model _by corrupting what it learns from_. The distinctions between them matter:

**ML02 - Data Poisoning Attack**  
Injecting malicious data into the _training set_ to compromise the model or **create backdoors**. The killer example: poison an antivirus model's training data so it learns to classify the attacker's custom malware as benign - a deliberate backdoor. Especially dangerous because models often scrape training data automatically from public/unverified sources.

**Your connection:** every "data poisoning" flag I raised since the preprocessing sections. You saw how cleaning choices (median imputation, drops, defaults) shape the model - poisoning weaponizes that. This is the "large datasets" vulnerability from the last section, formalized.

**ML08 - Model Skewing**  
A _specific flavor_ of data poisoning where the goal is to **bias the model's output** in the attacker's favor. Example: add your malware to the training data _with a "benign" label_ - mislabeled data that skews the model to miss your malware.

**The ML02 vs ML08 distinction:** ML02 is broad (corrupt the model / plant backdoors), ML08 is narrowly about _deliberately biasing outputs_ toward the attacker's goal, often via mislabeled data. They overlap heavily; the nuance is intent (general compromise vs. targeted bias).

**ML07 - Transfer Learning Attack** (you'll appreciate this one - you just did transfer learning)  
Manipulate the _pre-trained base model_ that others fine-tune on top of. Backdoors/biases planted in the base model **persist into the fine-tuned model - even if the fine-tuning data is completely clean.**

**Your connection:** this is _exactly_ the offensive angle I flagged in your malware CNN project. You built your classifier on frozen ResNet50 - if that ResNet had been poisoned upstream, your malware model would inherit the backdoor. Remember I said "transfer learning means inheriting the backbone's vulnerabilities"? ML07 is that attack, named.

---

**Group C - Stealing from the model** (three attacks - two steal _data_, one steals the _model_)

**ML03 - Model Inversion Attack**  
Train a _separate_ model on the target's _outputs_ to **reconstruct its inputs** - reversing the model's function (hence "inversion"). Dangerous when inputs are sensitive: a cancer classifier's outputs might be inverted to reconstruct patient medical data. Key defensive insight: **it's harder if the model outputs less** - if it only returns the class label (not full probabilities), inversion is much tougher. (Note this - output verbosity is an attack surface.)

**ML04 - Membership Inference Attack** ![Attacking ML-Based Systems (ML OWASP Top 10)](/assets/Pasted%20image%2020260831151427.png)
Determine whether a _specific data sample was in the training set_. Why it matters: if a model was trained on sensitive data (medical, financial), confirming someone's record was used _is itself a privacy breach_. **Image 2** illustrates it - an adversary uses an inference model to determine which patients' data was in the private training set (the diagram notes "up to 80% of patients at risk for membership disclosure").

**The key mechanism (worth remembering):** models behave _differently_ on data they trained on vs. never-seen data - typically **higher confidence / lower error on training samples.** The attacker exploits that behavioral difference. (This connects to something you saw directly - remember your malware CNN hit 96% on training data but 88% on test? That gap _is_ the signal membership inference exploits: models "remember" their training data.)

**ML05 - Model Theft (Model Extraction)** ![Attacking ML-Based Systems (ML OWASP Top 10)](/assets/Pasted%20image%2020260831151515.png)
Duplicate the model itself by **systematically querying it** and training a replica on the input->output pairs. **Image 3** shows it - the attacker picks a "theft dataset," sends queries to the target (e.g., an MLaaS cloud API), collects predictions, and trains a "stolen model" that approximates the original. Steals intellectual property and can expose sensitive learned patterns.

**Your connection:** remember when I flagged that `predict_proba` on your spam model, and confidence scores generally, "leak exploitable information"? Model theft is one payoff - those query responses let an attacker clone your model. And once cloned, black-box evasion becomes white-box (you attack your stolen copy freely).

---

**Group D - Attacking model/pipeline directly**

**ML06 - AI Supply Chain Attacks**  
Exploit vulnerabilities _anywhere_ in the ML pipeline - third-party data, libraries, pre-trained models. ML systems have _more_ supply-chain surface than traditional software because they pull in so many external components (datasets, `pip` libraries, pre-trained weights). The risk has grown with reliance on open-source tools and public models.

**Your connection:** remember the `weights='DEFAULT'` line downloading ResNet50, and me flagging "what if the pre-trained weights were tampered with?" That's ML06. Every external dependency (the `wget` of a dataset, the `pip install`, the pretrained download) is supply-chain surface.

**ML10 - Model Poisoning**  
Manipulate the model's **weights/parameters directly** (vs. ML02 which does it _indirectly_ via training data). Requires access to the parameters. Hard to do surgically - random changes just degrade performance; making the model misbehave in a _specific, deliberate_ way requires nuanced parameter manipulation.

**The ML02 vs ML10 distinction (a common relevant nuance):** both poison the model, but **data poisoning (ML02) corrupts the training _data_** (indirect, no parameter access needed), while **model poisoning (ML10) corrupts the _weights directly_** (requires parameter access). Same _impact_ (backdoors, misclassification), different _mechanism and access requirement_.

---

**Group E - Attacking the output**

**ML09 - Output Integrity Attack** (clever and often overlooked)  
**Don't touch the model at all - intercept and alter its _output_** before the downstream system acts on it. The model works perfectly; you just tamper with what it _reports_.

The example is sharp: a system deletes any binary the classifier flags as malware. The classifier _correctly_ flags the attacker's malware as malicious - but the attacker **intercepts that output and changes "malicious" to "benign"** before the deletion system reads it. The malware survives. **Hard to detect** because the model itself looks completely fine on inspection - traditional model-security checks miss it entirely.

**Your connection:** this is where your _traditional_ pentest/web skills matter. ML09 isn't really an ML attack - it's a classic integrity/interception attack on the _plumbing around_ the model. This is exactly the "chain web exploitation with AI attacks" theme - the vulnerability is at the _interaction point_ between the model and the system consuming its output (recall last section: vulnerabilities arise at component seams).