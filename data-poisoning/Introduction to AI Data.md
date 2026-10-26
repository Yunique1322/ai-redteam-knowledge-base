# Introduction To AI Data

Theory section, kept tight. The one principle to anchor the whole module: **a model cannot distinguish attacker-planted patterns from legitimate ones, so anything that enters the pipeline gets learned as fact.** This entire module is OWASP **LLM04:2025 Data and Model Poisoning**, with the model-file attacks bleeding into **LLM03 Supply Chain**.

## Where the attack surface actually is

The most important framing here: **collection, storage, and processing are the attack surface. Modeling is not.** The model is a faithful downstream consumer, it absorbs whatever it is taught, and it will neither flag nor resist poison. So you inject upstream, and the model is simply where the consequences become visible.

## The pipeline stage by stage (attacker's view)
![The pipeline stage by stage (attacker's view)](/assets/Pasted%20image%2020260917222320.png)

- **Collect:** **every data source is a separate trust boundary, and the pipeline does not validate what it ingests.** Poison introduced here rides forward through every later stage.![Introduction to AI Data](/assets/1%20Introduction%20to%20AI%20Data-20260917222417095.png)

- **Store:** training data and serialized model files (`.pkl`, `.pt`, `ONNX`) live in the same infrastructure. **With write access to storage you can skip data poisoning entirely and just swap the model file, and a `.pkl` can execute arbitrary Python the moment it is loaded.** (This is the pickle section later.)![Introduction to AI Data](/assets/1%20Introduction%20to%20AI%20Data-20260917222422795.png)
- **Process:** cleaning, transformation, and feature scripts are all code. **Compromise the processing logic instead of the data and the corruption enters where no one is looking, because the raw data upstream still looks clean.**![Introduction to AI Data](/assets/1%20Introduction%20to%20AI%20Data-20260917222426357.png)
- **Model:** absorbs the poison faithfully. Consequences, not injection.![Introduction to AI Data](/assets/1%20Introduction%20to%20AI%20Data-20260917222429701.png)
- **Deploy:** a tampered or trojaned model file, or an insecure deserialization load path, reaches production silently. **Wrong predictions are the mild outcome, arbitrary code execution is the severe one.**![The pipeline stage by stage (attacker's view)](/assets/Pasted%20image%2020260917222434.png)
- **Monitor / retrain:** **online poisoning through feedback loops. The system is structurally designed to trust its own inputs, so attacker-injected feedback becomes the next model version. It exploits the feature, not a bug.**![The pipeline stage by stage (attacker's view)](/assets/Pasted%20image%2020260917222444.png)

## The two examples, in one line

The e-commerce recommender (Kafka, S3 data lake, Spark, SageMaker, pickled model, K8s API) and the healthcare diagnostic tool (restricted internal data, PyTorch, internal API) are the same pipeline architecture with the same structural weaknesses. Only the stakes and the access controls differ. Every stage (the stream, the bucket, the jobs, the model file, the retraining loop) is an entry point.

## How this maps to what you'll build

The four attack families ahead each target one of these surfaces:

- **Label flipping** and **clean-label** attacks poison the training data (collection/processing surface).
- **Trojan/backdoor** attacks poison the model during training.
- **Pickles and tensor steganography** hit the model-file surface (storage/deployment, supply-chain and deserialization RCE).

Exam anchor: know which pipeline stage each attack targets, the trust-boundary framing, and that **modeling itself is never the attack surface, only the amplifier.**

